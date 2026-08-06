from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from reptaxonomy.util.general_utils import read_yaml
from reptaxonomy.util.experiment_init_utils import (
    build_dataset_bundle,
    build_dataset_spec,
    build_model_bundle,
    resolve_model_spec,
    build_test_loader
)
from reptaxonomy.util.artifacts import ensure_dict



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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
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
    cfg["per_model_overrides"] = ensure_dict(cfg.get("per_model_overrides", {}), "per_model_overrides")

    cfg.setdefault("embed_dir", "./embeddings")
    cfg.setdefault("work_dir", "./work")
    cfg.setdefault("dataset_init", {})
    cfg.setdefault("model_init", {})
    cfg.setdefault("split", "test")
    cfg.setdefault("batch_size", 1)
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("pin_memory", True)
    cfg.setdefault("save_crop_info", True)
    cfg.setdefault("overwrite", False)
    cfg.setdefault("fail_fast", False)
    cfg.setdefault("write_manifest", True)
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")

    return cfg


def _bundle_attr(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _safe_asdict(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return copy.deepcopy(x)
    return x


def attach_bundles_to_cfg(cfg: Dict[str, Any], dataset_bundle: Any, model_bundle: Optional[Any] = None) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)

    dataset_spec = _bundle_attr(dataset_bundle, "dataset_spec")
    dataset_spec_dict = _safe_asdict(dataset_spec)
    dataset_cls = _bundle_attr(dataset_bundle, "dataset_cls")
    dataset_kwargs = _bundle_attr(dataset_bundle, "dataset_kwargs", default={})
    sample_schema = _bundle_attr(dataset_bundle, "sample_schema", default={})
    dataset_metadata = _bundle_attr(dataset_bundle, "metadata", default={})
    collate_fn = _bundle_attr(dataset_bundle, "collate_fn", default=None)

    out["dataset"] = copy.deepcopy(dataset_spec_dict)
    out["dataset_bundle"] = {
        "dataset_spec": copy.deepcopy(dataset_spec_dict),
        "dataset_cls_name": getattr(dataset_cls, "__name__", None),
        "dataset_kwargs": copy.deepcopy(dataset_kwargs),
        "sample_schema": copy.deepcopy(sample_schema),
        "metadata": copy.deepcopy(dataset_metadata),
        "collate_fn_name": getattr(collate_fn, "__name__", None),
    }

    if model_bundle is not None:
        model_spec = _bundle_attr(model_bundle, "model_spec")
        model_spec_dict = _safe_asdict(model_spec)
        model_dataset_spec = _bundle_attr(model_bundle, "dataset_spec", default=dataset_spec)
        model_dataset_spec_dict = _safe_asdict(model_dataset_spec)
        model_cls = _bundle_attr(model_bundle, "model_cls")
        model_kwargs = _bundle_attr(model_bundle, "model_kwargs", default={})
        model_metadata = _bundle_attr(model_bundle, "metadata", default={})
        model_name = _bundle_attr(model_bundle, "model_name", default=cfg.get("encoder"))

        out["model_bundle"] = {
            "model_name": model_name,
            "model_spec": copy.deepcopy(model_spec_dict),
            "dataset_spec": copy.deepcopy(model_dataset_spec_dict),
            "model_cls_name": getattr(model_cls, "__name__", None),
            "model_kwargs": copy.deepcopy(model_kwargs),
            "metadata": copy.deepcopy(model_metadata),
        }
        out["encoder"] = model_name

    return out


def build_dataset_instance(dataset_bundle: Any) -> Any:
    dataset_obj = _bundle_attr(dataset_bundle, "dataset")
    if dataset_obj is not None and not isinstance(dataset_obj, type):
        return dataset_obj

    dataset_cls = _bundle_attr(dataset_bundle, "dataset_cls")
    dataset_kwargs = _bundle_attr(dataset_bundle, "dataset_kwargs", default={})
    if dataset_cls is None:
        raise ValueError("Dataset bundle does not expose dataset_cls or a prebuilt dataset instance")
    return dataset_cls(**dataset_kwargs)


def build_model_instance(model_bundle: Any, device: str) -> Any:
    model_obj = _bundle_attr(model_bundle, "model")
    if model_obj is not None and not isinstance(model_obj, type):
        model = model_obj
    else:
        model_cls = _bundle_attr(model_bundle, "model_cls")
        model_kwargs = _bundle_attr(model_bundle, "model_kwargs", default={})
        if model_cls is None:
            raise ValueError("Model bundle does not expose model_cls or a prebuilt model instance")
        model = model_cls(**model_kwargs)

    if hasattr(model, "load_encoder_weights"):
        try:
            model.load_encoder_weights(None)
        except TypeError:
            model.load_encoder_weights()

    model = model.to(device)
    model.eval()
    return model


def build_collate_fn(dataset_bundle: Any, model: Any) -> Any:
    collate_fn = _bundle_attr(dataset_bundle, "collate_fn", default=None)
    if collate_fn is not None:
        return collate_fn

    if get_collate_fn is not None and hasattr(model, "input_bands"):
        try:
            return get_collate_fn(list(model.input_bands.keys()), return_meta=True)
        except Exception:
            return None
    return None


def get_image_fname(batch: Dict[str, Any], batch_index: int) -> str:
    if "filename" in batch:
        value = batch["filename"]
        if isinstance(value, (list, tuple)):
            return str(value[0])
        return str(value)

    meta = batch.get("metadata", {})
    if isinstance(meta, dict):
        for key in ("image_filename", "filename", "image_fname"):
            if key in meta:
                value = meta[key]
                if isinstance(value, (list, tuple)):
                    return str(value[0])
                return str(value)

    return f"sample_{batch_index:06d}.tif"


def get_crop_info(batch: Dict[str, Any]) -> Optional[Any]:
    return batch.get("crop")


def move_image_to_device(image: Any, device: str) -> Any:
    if isinstance(image, dict):
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in image.items()}
    if torch.is_tensor(image):
        return image.to(device)
    raise TypeError(f"Unsupported image container type: {type(image)}")


def select_encoder_output(feat: Any) -> torch.Tensor:
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]
    if not torch.is_tensor(feat):
        raise TypeError(f"Unsupported model output type: {type(feat)}")
    if feat.ndim > 0 and feat.shape[0] == 1:
        feat = feat[0]
    return feat


def run_model_forward(model: Any, image: Any, dataset_instance: Any) -> np.ndarray:
    with torch.no_grad():
        if getattr(model, "multi_temporal", False):
            feat = model.forward_instruments(image)
            if getattr(model, "multi_temporal_output", False) and isinstance(feat, (list, tuple)):
                feat = [f.squeeze(-3) if torch.is_tensor(f) and f.ndim >= 3 else f for f in feat]
            feat = select_encoder_output(feat)
            return feat.detach().cpu().numpy()

        else:
            feat = model.forward_instruments(image)

        feat = select_encoder_output(feat)
        return feat.detach().cpu().numpy()


def save_embedding_artifacts(
    out_dir: Path,
    image_fname: str,
    feat_np: np.ndarray,
    crop_info: Optional[Any],
    save_crop_info: bool,
    overwrite: bool,
) -> Tuple[Path, Optional[Path]]:
    stem = Path(image_fname).stem
    emb_path = out_dir / f"embd_{stem}.npy"
    crop_path = out_dir / f"crop_info_{stem}.npy"

    if overwrite or not emb_path.exists():
        np.save(emb_path, feat_np)

    crop_written = None
    if save_crop_info and crop_info is not None:
        if overwrite or not crop_path.exists():
            np.save(crop_path, crop_info, allow_pickle=True)
        crop_written = crop_path

    return emb_path, crop_written


def init_manifest(path: Path) -> None:
    if path.exists():
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "split",
                "batch_index",
                "image_filename",
                "embedding_path",
                "crop_info_path",
                "embedding_shape",
                "embedding_dtype",
                "status",
                "error",
            ],
        )
        writer.writeheader()


def append_manifest(path: Path, row: Dict[str, Any]) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "split",
                "batch_index",
                "image_filename",
                "embedding_path",
                "crop_info_path",
                "embedding_shape",
                "embedding_dtype",
                "status",
                "error",
            ],
        )
        writer.writerow(row)


def run_embedding_export(cfg: Dict[str, Any]) -> None:
    cfg = validate_cfg(cfg)
    set_global_seed(int(cfg["seed"]))

    dataset_spec = build_dataset_spec(
        dataset_cfg=cfg["dataset"],
        dataset_init=ensure_dict(cfg.get("dataset_init", {}), "dataset_init"),
    )
    dataset_bundle = build_dataset_bundle(dataset_spec)
    dataset_instance = build_dataset_instance(dataset_bundle)

    dataset_name = _safe_asdict(_bundle_attr(dataset_bundle, "dataset_spec")).get("dataset_name", cfg["dataset"]["dataset_name"])
    device = str(cfg["device"])
    shared_model_init = ensure_dict(cfg.get("model_init", {}), "model_init")

    for model_name in cfg["model_names"]:
        per_model_cfg = ensure_dict(
            cfg.get("per_model_overrides", {}).get(model_name, {}),
            f"per_model_overrides.{model_name}",
        )
        model_spec = resolve_model_spec(model_name, shared_model_init, per_model_cfg)
        model_bundle = build_model_bundle(model_spec, dataset_bundle)
        model_cfg = attach_bundles_to_cfg(cfg, dataset_bundle, model_bundle)

        out_dir = Path(model_cfg["embed_dir"]) / dataset_name / model_name / str(model_cfg["split"])
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = out_dir / "embedding_manifest.csv"
        if bool(model_cfg["write_manifest"]):
            init_manifest(manifest_path)

        print(f"[EMBED] dataset={dataset_name} model={model_name} split={model_cfg['split']} out={out_dir}")

        model = build_model_instance(model_bundle, device=device)


        success_count = 0
        fail_count = 0

        loader = build_test_loader(model_cfg)

        for batch_idx, batch in enumerate(loader):
            image_fname = get_image_fname(batch, batch_idx)
            emb_out = out_dir / f"embd_{Path(image_fname).stem}.npy"

            if emb_out.exists() and not bool(model_cfg["overwrite"]):
                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "split": model_cfg["split"],
                    "batch_index": batch_idx,
                    "image_filename": image_fname,
                    "embedding_path": str(emb_out),
                    "crop_info_path": str(out_dir / f"crop_info_{Path(image_fname).stem}.npy"),
                    "embedding_shape": "",
                    "embedding_dtype": "",
                    "status": "skipped_exists",
                    "error": "",
                }
                if bool(model_cfg["write_manifest"]):
                    append_manifest(manifest_path, row)
                continue

            try:
                #image = move_image_to_device(batch["image"], device)
                crop_info = get_crop_info(batch)
                feat_np = run_model_forward(model, batch, dataset_instance)

                emb_path, crop_path = save_embedding_artifacts(
                    out_dir=out_dir,
                    image_fname=image_fname,
                    feat_np=feat_np,
                    crop_info=crop_info,
                    save_crop_info=bool(model_cfg["save_crop_info"]),
                    overwrite=bool(model_cfg["overwrite"]),
                )

                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "split": model_cfg["split"],
                    "batch_index": batch_idx,
                    "image_filename": image_fname,
                    "embedding_path": str(emb_path),
                    "crop_info_path": "" if crop_path is None else str(crop_path),
                    "embedding_shape": json.dumps(list(feat_np.shape)),
                    "embedding_dtype": str(feat_np.dtype),
                    "status": "ok",
                    "error": "",
                }
                if bool(model_cfg["write_manifest"]):
                    append_manifest(manifest_path, row)
                success_count += 1

            except Exception as e:
                fail_count += 1
                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "split": model_cfg["split"],
                    "batch_index": batch_idx,
                    "image_filename": image_fname,
                    "embedding_path": str(emb_out),
                    "crop_info_path": "",
                    "embedding_shape": "",
                    "embedding_dtype": "",
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                }
                if bool(model_cfg["write_manifest"]):
                    append_manifest(manifest_path, row)

                print(f"[EMBED][ERROR] dataset={dataset_name} model={model_name} file={image_fname}")
                print(traceback.format_exc())

                if bool(model_cfg["fail_fast"]):
                    raise

        print(f"[EMBED][DONE] dataset={dataset_name} model={model_name} success={success_count} failed={fail_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle-aware standalone embedding export pipeline.")
    parser.add_argument("-y", "--yaml", help="Pipeline YAML config file.")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.yaml)
    run_embedding_export(cfg)


if __name__ == "__main__":
    main()
