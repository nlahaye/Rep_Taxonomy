from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dataclasses import asdict, dataclass, field, is_dataclass

from reptaxonomy.util.general_utils import read_yaml

from reptaxonomy.projections.run_projections import run_knn_gen
from reptaxonomy.robustness.embedding_robustness import analyze_robustness
from reptaxonomy.intercompare.data_kernel_analysis import run_data_kernel_analysis
from reptaxonomy.intercompare.geodesic_compare import run_geodesic_compare
from reptaxonomy.resources.compute_resource_reqs import compute_resource_reqs
from reptaxonomy.robustness.weight_matrix_analysis import run_weight_mtx_analysis
from reptaxonomy.summarize.demsar_wrapper import demsar_wrapper
from reptaxonomy.summarize.cross_perspective_synthesis import run_cross_perspective_synthesis
from reptaxonomy.intercompare.reference_similarity import run_reference_similarity
from reptaxonomy.robustness.invariance_checks import run_invariance_checks

from reptaxonomy.util.experiment_init_utils import (
    build_dataset_bundle,
    build_dataset_spec,
    build_model_bundle,
    resolve_model_spec,
    serialize_dataset_bundle,
    serialize_model_bundle,
)


def ensure_dict(cfg: Any, name: str = "config") -> Dict[str, Any]:
    if cfg is None:
        raise ValueError(f"{name} is empty or unreadable")
    if not isinstance(cfg, dict):
        raise ValueError(f"{name} must be a dictionary")
    return cfg


def load_pipeline_config(path: str | Path) -> Dict[str, Any]:
    cfg = read_yaml(str(path))
    return ensure_dict(cfg, "pipeline config")


def load_model_names(model_names_cfg: Any) -> List[str]:
    if isinstance(model_names_cfg, list):
        model_names = model_names_cfg
    elif isinstance(model_names_cfg, (str, Path)):
        with open(model_names_cfg, "r") as f:
            model_names = json.load(f)
    else:
        raise ValueError("model_names must be a list or path to a JSON file")

    if not isinstance(model_names, list) or not model_names:
        raise ValueError("model_names must resolve to a non-empty list")

    return [str(x) for x in model_names]


def merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def should_run(step_cfg: Dict[str, Any]) -> bool:
    return bool(step_cfg.get("enabled", True))


def validate_pipeline_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = ensure_dict(cfg)

    if "dataset" not in cfg:
        raise ValueError("Missing required top-level config key: 'dataset'")
    if "model_names" not in cfg:
        raise ValueError("Missing required top-level config key: 'model_names'")

    cfg["dataset"] = ensure_dict(cfg["dataset"], "dataset")
    if "dataset_name" not in cfg["dataset"]:
        if "name" in cfg["dataset"]:
            cfg["dataset"]["dataset_name"] = cfg["dataset"]["name"]
        else:
            raise ValueError("dataset must define either 'dataset_name' or 'name'")

    cfg["model_names"] = load_model_names(cfg["model_names"])
    cfg["steps"] = ensure_dict(cfg.get("steps", {}), "steps")
    cfg["per_model_overrides"] = ensure_dict(cfg.get("per_model_overrides", {}), "per_model_overrides")

    cfg.setdefault("work_dir", "./work")
    cfg.setdefault("embed_dir", "./embeddings")
    cfg.setdefault("out_dir", "./outputs")
    cfg.setdefault("n_runs", 3)
    cfg.setdefault("seed", 42)
    cfg.setdefault("projections", ["native", "umap", "tsne", "pca"])
    cfg.setdefault("upsample_methods", [cfg.get("upsample_method", "pixelshuffle")])
    cfg.setdefault("dataset_init", {})
    cfg.setdefault("model_init", {})
    cfg.setdefault("analysis", {})

    return cfg


def attach_bundles_to_cfg(
    cfg: Dict[str, Any],
    dataset_bundle: DatasetBundle,
    model_bundle: Optional[ModelBundle] = None,
) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)

    dataset_spec_dict = asdict(dataset_bundle.dataset_spec)
    dataset_cls_name = dataset_bundle.dataset_cls.__name__
    dataset_kwargs = copy.deepcopy(dataset_bundle.dataset_kwargs)
    sample_schema = copy.deepcopy(dataset_bundle.sample_schema)
    dataset_metadata = copy.deepcopy(dataset_bundle.metadata)

    collate_fn_name = None
    if getattr(dataset_bundle, "collate_fn", None) is not None:
        collate_fn_name = getattr(dataset_bundle.collate_fn, "__name__", None)

    out["dataset"] = copy.deepcopy(dataset_spec_dict)
    out["dataset_bundle"] = {
        "dataset_spec": dataset_spec_dict,
        "dataset_cls_name": dataset_cls_name,
        "dataset_kwargs": dataset_kwargs,
        "sample_schema": sample_schema,
        "metadata": dataset_metadata,
        "collate_fn_name": collate_fn_name,
    }

    if model_bundle is not None:
        model_spec_dict = asdict(model_bundle.model_spec)
        model_dataset_spec_dict = asdict(model_bundle.dataset_spec)
        model_cls_name = model_bundle.model_cls.__name__
        model_kwargs = copy.deepcopy(model_bundle.model_kwargs)
        model_metadata = copy.deepcopy(model_bundle.metadata)

        out["model_bundle"] = {
            "model_name": model_bundle.model_name,
            "model_spec": model_spec_dict,
            "dataset_spec": model_dataset_spec_dict,
            "model_cls_name": model_cls_name,
            "model_kwargs": model_kwargs,
            "metadata": model_metadata,
        }

        out["encoder"] = model_bundle.model_name

    return out


def call_step(step_name: str, fn, cfg: Dict[str, Any]) -> None:
    dataset_name = cfg["dataset_bundle"]["dataset_spec"]["dataset_name"]
    model_name = cfg.get("model_bundle", {}).get("model_name", "global")
    print(f"[PIPELINE] Running step={step_name} dataset={dataset_name} model={model_name}")
    fn(cfg)


# Steps whose outputs do not depend on how token grids are upsampled to pixels.
UPSAMPLE_INDEPENDENT_STEPS = {"compute_resource_reqs", "weight_matrix_analysis"}

# Global steps that are evaluated once on the primary variant only.
PRIMARY_ONLY_GLOBAL_STEPS = {"invariance_checks"}


def variant_out_dir(base_out_dir: str, method: str, primary: str) -> str:
    """Primary upsampling variant writes to out_dir; others to out_dir/upsample_<method>."""
    return base_out_dir if method == primary else os.path.join(base_out_dir, f"upsample_{method}")


def run_step_block(
    steps: List[Any],
    cfg: Dict[str, Any],
    base_cfg: Dict[str, Any],
    model_bundle: Any = None,
    skip: set | None = None,
) -> None:
    for item in steps:
        step_name, fn = item[0], item[1]
        capability = item[2] if len(item) > 2 else None
        if skip and step_name in skip:
            continue
        step_cfg = ensure_dict(cfg["steps"].get(step_name, {}), f"steps.{step_name}")
        if not should_run(step_cfg):
            continue
        if capability is not None and model_bundle is not None and not model_bundle.metadata.get(capability, True):
            print(
                f"[PIPELINE] Skipping step={step_name} "
                f"model={base_cfg.get('encoder')}; capability {capability}=False"
            )
            continue
        final_cfg = merge_dicts(base_cfg, step_cfg.get("config_overrides", {}))
        call_step(step_name, fn, final_cfg)


def run_full_pipeline(cfg: Dict[str, Any]) -> None:
    cfg = validate_pipeline_cfg(cfg)

    dataset_spec = build_dataset_spec(
        dataset_cfg=cfg["dataset"],
        dataset_init=ensure_dict(cfg.get("dataset_init", {}), "dataset_init"),
    )
    dataset_bundle = build_dataset_bundle(dataset_spec)

    common_cfg = copy.deepcopy(cfg)
    common_cfg["dataset"] = vars(dataset_spec)
    common_cfg["model_names"] = cfg["model_names"]
    common_cfg["projections"] = cfg["projections"]
    common_cfg = attach_bundles_to_cfg(common_cfg, dataset_bundle)

    per_model_steps = [
        ("compute_resource_reqs", compute_resource_reqs, "supports_resource_requirements"),
        ("weight_matrix_analysis", run_weight_mtx_analysis, "supports_weight_matrix_analysis"),
        ("run_projections", run_knn_gen, "supports_projection_analysis"),
        ("embedding_robustness", analyze_robustness, "supports_embedding_robustness"),
    ]

    global_steps = [
        ("data_kernel_analysis", run_data_kernel_analysis),
        ("reference_similarity", run_reference_similarity),
        ("invariance_checks", run_invariance_checks),
        ("geodesic_compare", run_geodesic_compare),
        ("demsar_wrapper", demsar_wrapper),
    ]

    # Final, cross-variant synthesis (runs once after every variant is complete).
    final_steps = [("cross_perspective_synthesis", run_cross_perspective_synthesis)]

    shared_model_init = ensure_dict(cfg.get("model_init", {}), "model_init")
    base_out_dir = str(cfg["out_dir"])
    methods = [str(m).lower() for m in cfg["upsample_methods"]]
    primary = methods[0]
    variants: Dict[str, str] = {}

    model_bundles = {}
    for model_name in cfg["model_names"]:
        per_model_cfg = ensure_dict(
            cfg.get("per_model_overrides", {}).get(model_name, {}),
            f"per_model_overrides.{model_name}",
        )
        model_spec = resolve_model_spec(model_name, shared_model_init, per_model_cfg)
        model_bundles[model_name] = build_model_bundle(model_spec, dataset_bundle)

    for method in methods:
        out_dir = variant_out_dir(base_out_dir, method, primary)
        variants[method] = out_dir
        variant_cfg = copy.deepcopy(common_cfg)
        variant_cfg["out_dir"] = out_dir
        variant_cfg["upsample_method"] = method
        # Every variant samples the same pixels as the primary variant.
        variant_cfg["index_cache_root"] = base_out_dir
        print(f"[PIPELINE] Upsampling variant={method} out_dir={out_dir}")

        for model_name, model_bundle in model_bundles.items():
            model_cfg = attach_bundles_to_cfg(variant_cfg, dataset_bundle, model_bundle)
            model_cfg["encoder"] = model_name
            run_step_block(
                per_model_steps, cfg, model_cfg, model_bundle=model_bundle,
                skip=None if method == primary else UPSAMPLE_INDEPENDENT_STEPS,
            )

        run_step_block(
            global_steps, cfg, copy.deepcopy(variant_cfg),
            skip=None if method == primary else PRIMARY_ONLY_GLOBAL_STEPS,
        )

    final_cfg = copy.deepcopy(common_cfg)
    final_cfg["out_dir"] = base_out_dir
    syn = dict(final_cfg.get("synthesis", {}) or {})
    syn.setdefault("variants", variants)
    syn.setdefault("primary_variant", primary)
    final_cfg["synthesis"] = syn
    run_step_block(final_steps, cfg, final_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline from a YAML config.")
    parser.add_argument("yaml", help="Pipeline YAML config file.")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.yaml)
    run_full_pipeline(cfg)


if __name__ == "__main__":
    main()
