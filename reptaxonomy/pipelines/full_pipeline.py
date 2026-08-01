from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from reptaxonomy.util.general_utils import read_yaml

from reptaxonomy.projections.run_projections import run_knn_gen
from reptaxonomy.robustness.embedding_robustness import analyze_robustness
from reptaxonomy.intercompare.data_kernel_analysis import run_data_kernel_analysis
from reptaxonomy.intercompare.geodesic_compare import run_geodesic_compare
from reptaxonomy.resources.compute_resource_reqs import compute_resource_reqs
from reptaxonomy.robustness.weight_matrix_analysis import run_weight_mtx_analysis
from reptaxonomy.summarize.demsar_wrapper import demsar_wrapper

from reptaxonomy.experiment_init_utils import (
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
    cfg.setdefault("projections", ["umap", "tsne", "pca"])
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
        ("compute_resource_reqs", compute_resource_reqs),
        ("weight_matrix_analysis", run_weight_mtx_analysis),
        ("run_projections", run_knn_gen),
        ("embedding_robustness", analyze_robustness),
    ]

    global_steps = [
        ("data_kernel_analysis", run_data_kernel_analysis),
        ("geodesic_compare", run_geodesic_compare),
        ("demsar_wrapper", demsar_wrapper),
    ]

    shared_model_init = ensure_dict(cfg.get("model_init", {}), "model_init")

    for model_name in cfg["model_names"]:
        per_model_cfg = ensure_dict(
            cfg.get("per_model_overrides", {}).get(model_name, {}),
            f"per_model_overrides.{model_name}",
        )
        model_spec = resolve_model_spec(model_name, shared_model_init, per_model_cfg)
        model_bundle = build_model_bundle(model_spec, dataset_bundle)

        model_cfg = attach_bundles_to_cfg(common_cfg, dataset_bundle, model_bundle)
        model_cfg["encoder"] = model_name

        for step_name, fn in per_model_steps:
            step_cfg = ensure_dict(cfg["steps"].get(step_name, {}), f"steps.{step_name}")
            if not should_run(step_cfg):
                continue
            final_cfg = merge_dicts(model_cfg, step_cfg.get("config_overrides", {}))
            call_step(step_name, fn, final_cfg)

    global_cfg = copy.deepcopy(common_cfg)

    for step_name, fn in global_steps:
        step_cfg = ensure_dict(cfg["steps"].get(step_name, {}), f"steps.{step_name}")
        if not should_run(step_cfg):
            continue
        final_cfg = merge_dicts(global_cfg, step_cfg.get("config_overrides", {}))
        call_step(step_name, fn, final_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline from a YAML config.")
    parser.add_argument("yaml", help="Pipeline YAML config file.")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.yaml)
    run_full_pipeline(cfg)


if __name__ == "__main__":
    main()
