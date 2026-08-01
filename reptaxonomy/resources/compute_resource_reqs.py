from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import pathlib
import pickle
import re
import time
from collections import Counter
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from calflops import calculate_flops
from codecarbon import EmissionsTracker

from reptaxonomy.util.general_utils import read_yaml
from reptaxonomy.experiment_init_utils import MODEL_CLASS_REGISTRY, DATASET_CLASS_REGISTRY


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_dataset_name(cfg: Dict[str, Any]) -> str:
    if "dataset_bundle" in cfg and "dataset_spec" in cfg["dataset_bundle"]:
        spec = cfg["dataset_bundle"]["dataset_spec"]
        return spec.get("dataset_name", spec.get("name"))
    if "dataset" in cfg:
        return cfg["dataset"].get("dataset_name", cfg["dataset"].get("name"))
    raise ValueError("Could not resolve dataset name from config")


def get_model_name(cfg: Dict[str, Any]) -> str:
    if "model_bundle" in cfg:
        return cfg["model_bundle"]["model_name"]
    if "encoder" in cfg:
        return cfg["encoder"]
    raise ValueError("Could not resolve model name from config")


def build_model_from_bundle(cfg: Dict[str, Any], device: str):
    bundle = cfg["model_bundle"]
    cls_name = bundle["model_cls_name"]
    if cls_name not in MODEL_CLASS_REGISTRY:
        raise ValueError(f"Unknown model class in bundle: {cls_name}")
    model_cls = MODEL_CLASS_REGISTRY[cls_name]
    model_kwargs = dict(bundle.get("model_kwargs", {}))
    model = model_cls(**model_kwargs)
    if hasattr(model, "load_encoder_weights"):
        model.load_encoder_weights(logging.getLogger(__name__))
    model.to(device)
    model.eval()
    return model


def build_test_dataset_from_bundle(cfg: Dict[str, Any]):
    bundle = cfg["dataset_bundle"]
    cls_name = bundle["dataset_cls_name"]
    if cls_name not in DATASET_CLASS_REGISTRY:
        raise ValueError(f"Unknown dataset class in bundle: {cls_name}")
    dataset_cls = DATASET_CLASS_REGISTRY[cls_name]
    dataset_kwargs = dict(bundle.get("dataset_kwargs", {}))
    return dataset_cls(**dataset_kwargs)


def normalize_image_dict(image: Any) -> Dict[str, torch.Tensor]:
    if isinstance(image, dict):
        return image
    if torch.is_tensor(image):
        return {"image": image}
    raise TypeError(f"Unsupported image payload type: {type(image)}")


def add_batch_dim(image_dict: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    out = {}
    for modality, value in image_dict.items():
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        out[modality] = value.unsqueeze(0).to(device)
    return out


def build_flops_input(model, test_dataset, image_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if getattr(model, "multi_temporal", False) and getattr(test_dataset, "multi_temporal", False):
        trimmed = {}
        for k, v in image_dict.items():
            if v.ndim >= 5:
                trimmed[k] = v[:, :, 0, :, :]
            else:
                trimmed[k] = v
        return trimmed
    return image_dict


def tensor_shape_map(inpt: Dict[str, torch.Tensor]) -> Dict[str, Tuple[int, ...]]:
    return {k: tuple(int(x) for x in v.shape) for k, v in inpt.items()}


def input_signature(inpt: Dict[str, torch.Tensor]) -> Tuple[Tuple[str, Tuple[int, ...], str], ...]:
    return tuple(sorted(
        (k, tuple(int(x) for x in v.shape), str(v.dtype))
        for k, v in inpt.items()
    ))


def serialize_signature(sig: Tuple[Tuple[str, Tuple[int, ...], str], ...]) -> str:
    return json.dumps(
        [{"modality": k, "shape": list(shape), "dtype": dtype} for k, shape, dtype in sig],
        sort_keys=True,
    )


def parse_numeric_flops(flops_value: Any) -> float:
    if isinstance(flops_value, (int, float)):
        return float(flops_value)

    s = str(flops_value).strip().replace(",", "")
    match = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]+)?\s*$", s)
    if not match:
        raise ValueError(f"Could not parse FLOPs value: {flops_value}")

    value = float(match.group(1))
    suffix = (match.group(2) or "").upper()

    scale = {
        "": 1.0,
        "K": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
        "P": 1e15,
    }
    if suffix not in scale:
        raise ValueError(f"Unsupported FLOPs suffix: {suffix} from {flops_value}")
    return value * scale[suffix]


def compute_resource_reqs(cfg: Dict[str, Any]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = get_dataset_name(cfg)
    model_name = get_model_name(cfg)

    exp_dir = pathlib.Path(cfg["work_dir"]) / "calflops"
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_bundle(cfg, device)
    test_dataset = build_test_dataset_from_bundle(cfg)

    signature_counter: Counter = Counter()
    signature_examples: Dict[str, Dict[str, torch.Tensor]] = {}
    sample_total_bytes = []

    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        image_dict = normalize_image_dict(sample["image"])
        batched = add_batch_dim(image_dict, device)
        flops_input = build_flops_input(model, test_dataset, batched)

        sig = input_signature(flops_input)
        sig_key = serialize_signature(sig)
        signature_counter[sig_key] += 1

        if sig_key not in signature_examples:
            signature_examples[sig_key] = flops_input

        sample_total_bytes.append(
            int(sum(v.numel() * v.element_size() for v in flops_input.values()))
        )

    out_dir = ensure_dir(os.path.join(cfg["embed_dir"], dataset_name, model_name, "test"))
    stats_dir = ensure_dir(os.path.join(out_dir, "calflops_stats"))

    tracker = EmissionsTracker(output_dir=stats_dir, output_file=f"{model_name}.emissions.csv")
    per_signature_rows = []

    tracker.start()
    try:
        with torch.no_grad():
            for sig_key, count in signature_counter.items():
                example_input = signature_examples[sig_key]
                flops, macs, params = calculate_flops(
                    model=model,
                    args=example_input,
                    output_as_string=True,
                    output_precision=4,
                )
                flops_numeric = parse_numeric_flops(flops)
                macs_numeric = parse_numeric_flops(macs)

                per_signature_rows.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "signature": sig_key,
                    "n_samples": int(count),
                    "fraction_of_test_set": float(count / len(test_dataset)),
                    "input_shapes": json.dumps(tensor_shape_map(example_input), sort_keys=True),
                    "sample_input_bytes": int(sum(v.numel() * v.element_size() for v in example_input.values())),
                    "flops": flops,
                    "macs": macs,
                    "params": params,
                    "flops_numeric": flops_numeric,
                    "macs_numeric": macs_numeric,
                })
    finally:
        emissions = tracker.stop()

    sig_df = pd.DataFrame(per_signature_rows).sort_values(["n_samples", "signature"], ascending=[False, True])

    mean_flops_numeric = float(np.average(sig_df["flops_numeric"], weights=sig_df["n_samples"])) if not sig_df.empty else np.nan
    mean_macs_numeric = float(np.average(sig_df["macs_numeric"], weights=sig_df["n_samples"])) if not sig_df.empty else np.nan
    min_flops_numeric = float(sig_df["flops_numeric"].min()) if not sig_df.empty else np.nan
    max_flops_numeric = float(sig_df["flops_numeric"].max()) if not sig_df.empty else np.nan

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "encoder": model_name,
        "dataset": dataset_name,
        "device": device,
        "n_test_samples": int(len(test_dataset)),
        "n_unique_input_signatures": int(len(signature_counter)),
        "mean_flops_numeric": mean_flops_numeric,
        "mean_macs_numeric": mean_macs_numeric,
        "min_flops_numeric": min_flops_numeric,
        "max_flops_numeric": max_flops_numeric,
        "params": None if sig_df.empty else sig_df.iloc[0]["params"],
        "sample_total_input_bytes_mean": float(np.mean(sample_total_bytes)) if sample_total_bytes else 0.0,
        "sample_total_input_bytes_median": float(np.median(sample_total_bytes)) if sample_total_bytes else 0.0,
        "sample_total_input_bytes_min": int(min(sample_total_bytes)) if sample_total_bytes else 0,
        "sample_total_input_bytes_max": int(max(sample_total_bytes)) if sample_total_bytes else 0,
        "emissions_kg_co2eq": emissions,
        "emissions_g_co2eq": None if emissions is None else emissions * 1000.0,
        "dataset_bundle": cfg.get("dataset_bundle"),
        "model_bundle": cfg.get("model_bundle"),
    }

    sig_csv = os.path.join(stats_dir, f"{model_name}.flops_by_signature.csv")
    sig_df.to_csv(sig_csv, index=False)

    summary_pkl = os.path.join(stats_dir, f"{model_name}.flops_summary.pkl")
    with open(summary_pkl, "wb") as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary_csv = os.path.join(stats_dir, f"{model_name}.flops_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print({
        "dataset": dataset_name,
        "model": model_name,
        "n_test_samples": summary["n_test_samples"],
        "n_unique_input_signatures": summary["n_unique_input_signatures"],
        "mean_flops_numeric": summary["mean_flops_numeric"],
        "min_flops_numeric": summary["min_flops_numeric"],
        "max_flops_numeric": summary["max_flops_numeric"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config file.")
    args = parser.parse_args()
    cfg = read_yaml(args.yaml)
    compute_resource_reqs(cfg)




