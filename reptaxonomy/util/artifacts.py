from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_dict(cfg: Any, name: str = "config") -> Dict[str, Any]:
    if cfg is None:
        raise ValueError(f"{name} is empty or unreadable")
    if not isinstance(cfg, dict):
        raise ValueError(f"{name} must be a dictionary")
    return cfg


def get_dataset_bundle_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    bundle_cfg = cfg.get("dataset_bundle")
    if not isinstance(bundle_cfg, dict):
        raise ValueError("cfg['dataset_bundle'] is required and must be a serialized bundle dict")
    return bundle_cfg


def get_model_bundle_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    bundle_cfg = cfg.get("model_bundle")
    if bundle_cfg is None:
        return {}
    if not isinstance(bundle_cfg, dict):
        raise ValueError("cfg['model_bundle'] must be a serialized bundle dict when provided")
    return bundle_cfg


def get_dataset_spec_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    dataset_bundle_cfg = get_dataset_bundle_cfg(cfg)
    dataset_spec = dataset_bundle_cfg.get("dataset_spec")
    if not isinstance(dataset_spec, dict):
        raise ValueError("cfg['dataset_bundle']['dataset_spec'] is required")
    return dataset_spec


def get_dataset_name(cfg: Dict[str, Any]) -> str:
    dataset_spec = get_dataset_spec_cfg(cfg)
    dataset_name = dataset_spec.get("dataset_name") or dataset_spec.get("name")
    if not dataset_name:
        raise ValueError("Could not resolve dataset name from dataset bundle")
    return str(dataset_name)


def get_model_name(cfg: Dict[str, Any]) -> str:
    model_bundle_cfg = get_model_bundle_cfg(cfg)
    model_name = model_bundle_cfg.get("model_name") or cfg.get("encoder")
    if not model_name:
        raise ValueError("Could not resolve model name from cfg['model_bundle'] or cfg['encoder']")
    return str(model_name)


def get_model_names(cfg: Dict[str, Any]) -> List[str]:
    model_names = cfg.get("model_names")
    if not isinstance(model_names, list) or not model_names:
        raise ValueError("cfg['model_names'] must be a non-empty list")
    return [str(x) for x in model_names]


def get_projections(cfg: Dict[str, Any]) -> List[str]:
    projections = cfg.get("projections", ["umap", "tsne", "pca"])
    if not isinstance(projections, list) or not projections:
        raise ValueError("cfg['projections'] must be a non-empty list")
    return [str(x).lower() for x in projections]


def dataset_out_root(cfg: Dict[str, Any]) -> str:
    return ensure_dir(os.path.join(cfg["out_dir"], get_dataset_name(cfg)))


def dataset_embed_root(cfg: Dict[str, Any]) -> str:
    return os.path.join(cfg["embed_dir"], get_dataset_name(cfg))


def model_out_root(cfg: Dict[str, Any], model_name: Optional[str] = None) -> str:
    name = model_name or get_model_name(cfg)
    return ensure_dir(os.path.join(dataset_out_root(cfg), name))


def model_embed_root(cfg: Dict[str, Any], model_name: Optional[str] = None, split: str = "test") -> str:
    name = model_name or get_model_name(cfg)
    return os.path.join(dataset_embed_root(cfg), name, split)


def run_dir(cfg: Dict[str, Any], run_idx: int, model_name: Optional[str] = None) -> str:
    name = model_name or get_model_name(cfg)
    return ensure_dir(os.path.join(model_out_root(cfg, name), f"run_{run_idx}"))


def dataset_run_dir(cfg: Dict[str, Any], run_idx: int) -> str:
    return ensure_dir(os.path.join(dataset_out_root(cfg), f"run_{run_idx}"))


def projection_analysis_dir(cfg: Dict[str, Any], run_idx: int, projection: str) -> str:
    return ensure_dir(os.path.join(dataset_out_root(cfg), f"run_{run_idx}", projection.lower()))


def embed_subset_path(cfg: Dict[str, Any], run_idx: int, model_name: Optional[str] = None) -> str:
    name = model_name or get_model_name(cfg)
    dataset_name = get_dataset_name(cfg)
    return os.path.join(run_dir(cfg, run_idx, name), f"{name}.{dataset_name}.embed_subset.npy")


def target_subset_path(cfg: Dict[str, Any], run_idx: int, model_name: Optional[str] = None) -> str:
    name = model_name or get_model_name(cfg)
    dataset_name = get_dataset_name(cfg)
    return os.path.join(run_dir(cfg, run_idx, name), f"{name}.{dataset_name}.target_subset.npy")


def projection_path(
    cfg: Dict[str, Any],
    run_idx: int,
    projection: str,
    model_name: Optional[str] = None,
) -> str:
    name = model_name or get_model_name(cfg)
    dataset_name = get_dataset_name(cfg)
    return os.path.join(run_dir(cfg, run_idx, name), f"{name}.{dataset_name}.{projection.upper()}.projection.npy")


def projection_label_tif_path(
    cfg: Dict[str, Any],
    run_idx: int,
    projection: str,
    model_name: Optional[str] = None,
) -> str:
    name = model_name or get_model_name(cfg)
    return os.path.join(run_dir(cfg, run_idx, name), f"{name}.{projection.upper()}_Labels.tif")


def knn_graph_path(
    cfg: Dict[str, Any],
    run_idx: int,
    projection: str,
    model_name: Optional[str] = None,
) -> str:
    name = model_name or get_model_name(cfg)
    dataset_name = get_dataset_name(cfg)
    return os.path.join(run_dir(cfg, run_idx, name), f"{name}.{dataset_name}.{projection.upper()}.knn_graph.npz")


def reducer_model_path(
    cfg: Dict[str, Any],
    run_idx: int,
    projection: str,
    model_name: Optional[str] = None,
) -> str:
    name = model_name or get_model_name(cfg)
    return os.path.join(run_dir(cfg, run_idx, name), f"{projection.lower()}_model.joblib")


def silhouette_stats_path(cfg: Dict[str, Any], model_name: Optional[str] = None) -> str:
    name = model_name or get_model_name(cfg)
    dataset_name = get_dataset_name(cfg)
    return os.path.join(model_out_root(cfg, name), f"silhouette_stats.{name}.{dataset_name}.csv")


def index_cache_path(cfg: Dict[str, Any], run_idx: int, image_fname: str) -> str:
    return os.path.join(dataset_run_dir(cfg, run_idx), f"{os.path.splitext(image_fname)[0]}.indices.zarr")


def embedding_file_path(cfg: Dict[str, Any], image_fname: str, model_name: Optional[str] = None, split: str = "test") -> str:
    root = model_embed_root(cfg, model_name=model_name, split=split)
    return os.path.join(root, "embd_" + os.path.splitext(image_fname)[0] + ".npy")


def crop_info_file_path(cfg: Dict[str, Any], image_fname: str, model_name: Optional[str] = None, split: str = "test") -> str:
    root = model_embed_root(cfg, model_name=model_name, split=split)
    return os.path.join(root, "crop_info_" + os.path.splitext(image_fname)[0] + ".npy")


def run_manifest_path(cfg: Dict[str, Any], run_idx: int, model_name: Optional[str] = None) -> str:
    name = model_name or get_model_name(cfg)
    return os.path.join(run_dir(cfg, run_idx, name), "artifact_manifest.json")


def analysis_manifest_path(cfg: Dict[str, Any]) -> str:
    return os.path.join(dataset_out_root(cfg), "data_kernel_manifest.json")


def save_json(data: Dict[str, Any], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def build_run_manifest(
    cfg: Dict[str, Any],
    run_idx: int,
    n_subset_samples: int,
    embedding_dim: int,
    projections: Optional[List[str]] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    name = model_name or get_model_name(cfg)
    projection_names = projections or get_projections(cfg)
    return {
        "dataset": get_dataset_name(cfg),
        "model": name,
        "run_idx": run_idx,
        "embed_subset_path": embed_subset_path(cfg, run_idx, name),
        "target_subset_path": target_subset_path(cfg, run_idx, name),
        "projection_paths": {
            projection: projection_path(cfg, run_idx, projection, name)
            for projection in projection_names
        },
        "knn_graph_paths": {
            projection: knn_graph_path(cfg, run_idx, projection, name)
            for projection in projection_names
        },
        "label_tif_paths": {
            projection: projection_label_tif_path(cfg, run_idx, projection, name)
            for projection in projection_names
        },
        "n_subset_samples": int(n_subset_samples),
        "embedding_dim": int(embedding_dim),
        "projections": projection_names,
    }


def build_analysis_manifest(cfg: Dict[str, Any], n_runs: int) -> Dict[str, Any]:
    return {
        "dataset": get_dataset_name(cfg),
        "model_names": get_model_names(cfg),
        "projections": get_projections(cfg),
        "n_runs": int(n_runs),
        "out_root": dataset_out_root(cfg),
    }


