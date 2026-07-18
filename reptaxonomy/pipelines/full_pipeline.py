from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from reptaxonomy.util.general_utils import read_yaml

from reptaxonomy.projections.run_projections import run_knn_gen
from reptaxonomy.robustness.embedding_robustness import analyze_robustness
from reptaxonomy.intercomapre.data_kernel_analysis import run_data_kernel_analysis
from reptaxonomy.intercompare.geodesic_compare import run_geodesic_compare
from reptaxonomy.resources.compute_resource_reqs import compute_resource_reqs
from reptaxonomy.robustness.weight_matrix_analysis import run_weight_mtx_analysis
from reptaxonomy.summarize.demsar_wrapper import demsar_wrapper


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
    elif isinstance(model_names_cfg, str):
        with open(model_names_cfg, "r") as f:
            model_names = json.load(f)
    elif isinstance(model_names_cfg, Path):
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


def with_model_cfg(base_cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["encoder"] = model_name
    return cfg


def with_shared_runtime(base_cfg: Dict[str, Any], pipeline_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    for key in ["work_dir", "embed_dir", "out_dir", "n_runs", "seed", "projections"]:
        if key in pipeline_cfg:
            cfg[key] = copy.deepcopy(pipeline_cfg[key])

    if "dataset" in pipeline_cfg:
        cfg["dataset"] = copy.deepcopy(pipeline_cfg["dataset"])

    return cfg


def should_run(step_cfg: Dict[str, Any]) -> bool:
    return bool(step_cfg.get("enabled", True))


def call_step(step_name: str, fn, cfg: Dict[str, Any]) -> None:
    print(f"[PIPELINE] Running step: {step_name}")
    fn(cfg)


def run_full_pipeline(cfg: Dict[str, Any]) -> None:
    cfg = ensure_dict(cfg)

    model_names = load_model_names(cfg["model_names"])
    projections = cfg.get("projections", ["umap", "tsne", "pca"])
    cfg["model_names"] = model_names
    cfg["projections"] = projections

    steps = cfg.get("steps", {})
    common_cfg = with_shared_runtime(cfg, cfg)

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

    for model_name in model_names:
        model_cfg = with_model_cfg(common_cfg, model_name)

        model_overrides = cfg.get("per_model_overrides", {}).get(model_name, {})
        model_cfg = merge_dicts(model_cfg, model_overrides)

        for step_name, fn in per_model_steps:
            step_cfg = ensure_dict(steps.get(step_name, {}), f"steps.{step_name}")
            if not should_run(step_cfg):
                continue

            final_cfg = merge_dicts(model_cfg, step_cfg.get("config_overrides", {}))
            call_step(f"{step_name} [{model_name}]", fn, final_cfg)

    global_cfg = copy.deepcopy(common_cfg)
    global_cfg["model_names"] = model_names
    global_cfg["projections"] = projections

    for step_name, fn in global_steps:
        step_cfg = ensure_dict(steps.get(step_name, {}), f"steps.{step_name}")
        if not should_run(step_cfg):
            continue

        final_cfg = merge_dicts(global_cfg, step_cfg.get("config_overrides", {}))
        call_step(step_name, fn, final_cfg)


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline from a YAML config.")
    parser.add_argument("-y", "--yaml", required=True, help="Pipeline YAML config file.")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.yaml)
    run_full_pipeline(cfg)


if __name__ == "__main__":
    main()
