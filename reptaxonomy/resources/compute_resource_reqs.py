from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.metadata
import json
import logging
import os
import pathlib
import pickle
import platform
import re
import resource
import socket
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from calflops import calculate_flops
from codecarbon import EmissionsTracker

from reptaxonomy.util.general_utils import read_yaml
from reptaxonomy.util.experiment_init_utils import (
    build_test_dataset_from_bundle,
    build_model_from_bundle,
    build_test_loader,
)
from reptaxonomy.pipelines.generate_embeddings import run_model_forward


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




def input_signature(inpt: Dict[str, torch.Tensor]) -> Tuple[Tuple[str, Tuple[int, ...], str], ...]:
    return tuple(sorted(
        (k, type(v).__name__)
        for k, v in inpt.items()
    ))


def serialize_signature(sig: Tuple[Tuple[str, Tuple[int, ...], str], ...]) -> str:
    return json.dumps(
        [{"modality": k, "dtype": dtype} for k, dtype in sig],
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
        "FLOPS": 1.0,
        "KFLOPS": 1e3,
        "MFLOPS": 1e6,
        "GFLOPS": 1e9,
        "TFLOPS": 1e12,
        "PFLOPS": 1e15,

        "MACS": 1.0,
        "KMACS": 1e3,
        "MMACS": 1e6,
        "GMACS": 1e9,
        "TMACS": 1e12,
        "PMACS": 1e15,
    }
    if suffix not in scale:
        raise ValueError(f"Unsupported FLOPs suffix: {suffix} from {flops_value}")
    return value * scale[suffix]


def count_input_elements(batch: Dict[str, Any]) -> int:
    """Number of input values the encoder consumes for one batch.

    Walks batch["instruments"] -> batch[inst]["bands"] -> batch[inst][band]["data"],
    i.e. the same structure consumed by forward_instruments. Targets are excluded.
    """
    total = 0
    for inst in batch.get("instruments", []):
        inst = inst[0] if isinstance(inst, (list, tuple)) else inst
        inst_block = batch.get(inst, {})
        for band in inst_block.get("bands", []):
            band = band[0] if isinstance(band, (list, tuple)) else band
            data = inst_block.get(band, {}).get("data") if isinstance(inst_block.get(band), dict) else None
            if torch.is_tensor(data):
                total += int(data.numel())
            elif isinstance(data, np.ndarray):
                total += int(data.size)
    return total


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _tracker_energy_kwh(tracker: EmissionsTracker, csv_path: str) -> float:
    data = getattr(tracker, "final_emissions_data", None)
    if data is not None and getattr(data, "energy_consumed", None) is not None:
        return float(data.energy_consumed)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "energy_consumed" in df.columns and len(df):
            return float(df["energy_consumed"].iloc[-1])
    return float("nan")


def _tracker_breakdown(tracker: EmissionsTracker) -> Dict[str, Any]:
    """CodeCarbon component energies and location metadata when available."""
    data = getattr(tracker, "final_emissions_data", None)
    keys = ["cpu_energy", "gpu_energy", "ram_energy", "cpu_power", "gpu_power", "ram_power",
            "country_name", "country_iso_code", "region", "cloud_provider", "cloud_region",
            "tracking_mode", "cpu_model", "gpu_model", "codecarbon_version", "duration"]
    return {f"cc_{k}": getattr(data, k, None) for k in keys} if data is not None else {}


def _pkg_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def environment_report(device: str) -> Dict[str, Any]:
    """Hardware / software provenance for resource measurements."""
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_model": platform.processor() or None,
        "cpu_count_logical": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "allow_tf32_matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "torch_num_threads": torch.get_num_threads(),
        "device": device,
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    env["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        env["ram_total_gb"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        env["ram_total_gb"] = None
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        env.update({
            "gpu_name": props.name,
            "gpu_count": torch.cuda.device_count(),
            "gpu_total_memory_gb": props.total_memory / 1e9,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
        })
        try:
            import subprocess
            env["nvidia_driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()[0]
        except Exception:
            env["nvidia_driver"] = None
    for pkg in ["numpy", "scipy", "scikit-learn", "codecarbon", "calflops", "umap-learn", "openTSNE",
                "graspologic", "weightwatcher", "timm", "torchgeo", "pangaea"]:
        env[f"pkg_{pkg}"] = _pkg_version(pkg)
    return env


_PRECISIONS = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def _autocast(device: str, precision: str):
    dtype = _PRECISIONS.get(precision)
    if precision not in _PRECISIONS:
        raise ValueError(f"inference_precision must be one of {list(_PRECISIONS)}")
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype)


def _normalize_batch_keys(batch: Dict[str, Any]) -> None:
    """Same in-place key clean-up performed in run_model_forward (collate tuple artefacts)."""
    for v in range(len(batch["instruments"])):
        if isinstance(batch["instruments"][v], (tuple, list)):
            batch["instruments"][v] = batch["instruments"][v][0]
        inst = batch["instruments"][v]
        for v2 in range(len(batch[inst]["bands"])):
            if isinstance(batch[inst]["bands"][v2], (tuple, list)):
                batch[inst]["bands"][v2] = batch[inst]["bands"][v2][0]
    for v in range(len(batch.get("targets", []))):
        if isinstance(batch["targets"][v], (tuple, list)):
            batch["targets"][v] = batch["targets"][v][0]


def forward_encoder_only(model: Any, batch: Dict[str, Any], device: str) -> Any:
    """Encoder forward pass without output post-processing (valid for any batch size)."""
    with torch.no_grad():
        _normalize_batch_keys(batch)
        if getattr(model, "wavelengths", False):
            return model.forward_instruments(batch, device=device, permissive=True)
        return model.forward_instruments(batch, device=device)


def _batch_size_of(batch: Dict[str, Any]) -> int:
    for inst in batch.get("instruments", []):
        inst = inst[0] if isinstance(inst, (list, tuple)) else inst
        for band in batch.get(inst, {}).get("bands", []):
            band = band[0] if isinstance(band, (list, tuple)) else band
            data = batch[inst].get(band, {}).get("data")
            # Collated band tensors are (B, H, W) (or (B, T, H, W)); see run_projections.
            if torch.is_tensor(data) and data.ndim >= 3:
                return int(data.shape[0])
    return 1


def measure_compression_and_inference_energy(
    cfg: Dict[str, Any],
    model: Any,
    device: str,
    stats_dir: str,
    model_name: str,
    dataset_name: str,
) -> Dict[str, Any]:
    """Compression ratio, latency / throughput / peak memory, and amortized inference energy.

    Protocol
      * `energy_n_samples` test batches are pre-loaded (data loading excluded).
      * `energy_warmup` untimed passes over those batches.
      * `energy_n_repeats` timed passes inside a dedicated CodeCarbon tracker.
        torch.cuda.synchronize() is called before and after every timed batch, so the
        latency of each batch includes all queued GPU work.
      * Precision: `inference_precision` (fp32 | fp16 | bf16 autocast).
      * Batch size: `test_batch_size` (encoder-only forward; outputs are discarded).
    Reported per image: latency, energy (kWh and J), emissions; plus throughput and
    peak device / host memory.
    """
    n_batches = int(cfg.get("energy_n_samples", 32))
    n_repeats = int(cfg.get("energy_n_repeats", 8))
    n_warmup = int(cfg.get("energy_warmup", 2))
    precision = str(cfg.get("inference_precision", "fp32")).lower()
    do_energy = bool(cfg.get("measure_inference_energy", True))
    batch_size_cfg = int(cfg.get("test_batch_size", 1))

    env = environment_report(device)
    save_json_file(env, os.path.join(stats_dir, f"{model_name}.environment.json"))

    loader = build_test_loader(cfg)
    batches = []
    for b_idx, batch in enumerate(loader):
        if b_idx >= max(1, n_batches):
            break
        batches.append(batch)
    if not batches:
        raise ValueError("Test loader returned no batches for inference measurement")

    # Compression ratio uses the same path as the embedding export (batch size 1 semantics).
    feat = run_model_forward(model, batches[0], loader.dataset, device)
    in_elems = count_input_elements(batches[0]) // max(1, _batch_size_of(batches[0]))
    out_elems = int(np.asarray(feat).size)
    compression = {
        "dataset": dataset_name,
        "model": model_name,
        "input_elements": in_elems,
        "output_elements": out_elems,
        "output_shape": json.dumps(list(np.asarray(feat).shape)),
        "compression_ratio": float(in_elems / out_elems) if out_elems else float("nan"),
    }
    pd.DataFrame([compression]).to_csv(os.path.join(stats_dir, f"{model_name}.compression.csv"), index=False)

    if not do_energy:
        return {"compression": compression, "environment": env}

    with _autocast(device, precision):
        for _ in range(n_warmup):
            for batch in batches:
                forward_encoder_only(model, batch, device)
    _sync(device)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    energy_csv = f"{model_name}.inference_emissions.csv"
    tracker = EmissionsTracker(
        output_dir=stats_dir,
        output_file=energy_csv,
        measure_power_secs=float(cfg.get("measure_power_secs", 1)),
        log_level="error",
    )
    latencies: List[float] = []
    n_images = 0
    tracker.start()
    t0 = time.perf_counter()
    try:
        with _autocast(device, precision):
            for _ in range(n_repeats):
                for batch in batches:
                    _sync(device)
                    tb = time.perf_counter()
                    forward_encoder_only(model, batch, device)
                    _sync(device)
                    latencies.append(time.perf_counter() - tb)
                    n_images += _batch_size_of(batch)
    finally:
        elapsed = time.perf_counter() - t0
        kg = tracker.stop()

    lat = np.asarray(latencies, dtype=float)
    kwh = _tracker_energy_kwh(tracker, os.path.join(stats_dir, energy_csv))
    kg = float("nan") if kg is None else float(kg)
    peak_alloc = peak_reserved = None
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
        peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 20
    host_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB

    imgs_per_batch = n_images / max(1, len(latencies))
    energy = {
        "dataset": dataset_name,
        "model": model_name,
        "device": device,
        "gpu_name": env.get("gpu_name"),
        "precision": precision,
        "batch_size": batch_size_cfg,
        "images_per_batch_observed": float(imgs_per_batch),
        "synchronization": "torch.cuda.synchronize before/after each batch" if device.startswith("cuda") else "none (cpu)",
        "n_warmup_passes": n_warmup,
        "n_repeats": n_repeats,
        "n_unique_batches": int(len(batches)),
        "n_images_measured": int(n_images),
        "n_samples_measured": int(n_images),
        "total_seconds": float(elapsed),
        "throughput_images_per_s": float(n_images / elapsed) if elapsed > 0 else float("nan"),
        "latency_batch_mean_ms": float(lat.mean() * 1e3),
        "latency_batch_median_ms": float(np.median(lat) * 1e3),
        "latency_batch_p95_ms": float(np.percentile(lat, 95) * 1e3),
        "latency_batch_std_ms": float(lat.std(ddof=1) * 1e3) if lat.size > 1 else 0.0,
        "latency_per_image_ms": float(lat.sum() / n_images * 1e3),
        "seconds_per_sample": float(elapsed / n_images),
        "peak_gpu_memory_allocated_mb": peak_alloc,
        "peak_gpu_memory_reserved_mb": peak_reserved,
        "peak_host_rss_mb_process": float(host_peak_mb),
        "total_kwh": kwh,
        "total_kg_co2eq": kg,
        "kwh_per_sample": float(kwh / n_images),
        "joules_per_image": float(kwh / n_images * 3.6e6),
        "kg_co2eq_per_sample": float(kg / n_images),
        "implied_kg_co2eq_per_kwh": float(kg / kwh) if kwh and kwh > 0 else float("nan"),
    }
    energy.update(_tracker_breakdown(tracker))
    energy.update({k: env.get(k) for k in ["torch", "cuda_runtime", "cudnn", "nvidia_driver", "cpu_model", "python"]})
    pd.DataFrame([energy]).to_csv(os.path.join(stats_dir, f"{model_name}.inference_energy.csv"), index=False)
    pd.DataFrame({"latency_s": lat}).to_csv(os.path.join(stats_dir, f"{model_name}.latencies.csv"), index=False)
    return {"compression": compression, "energy": energy, "environment": env}


def save_json_file(data: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def compute_resource_reqs(cfg: Dict[str, Any]):
    step_t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = get_dataset_name(cfg)
    model_name = get_model_name(cfg)

    exp_dir = pathlib.Path(cfg["work_dir"]) / "calflops"
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_bundle(cfg, device)
    test_dataset = build_test_dataset_from_bundle(cfg)

    signature_counter: Counter = Counter()
    signature_examples: Dict[str, Dict[str, torch.Tensor]] = {}
    #sample_total_bytes = []

    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]

        sig = input_signature(sample)
        sig_key = serialize_signature(sig)
        signature_counter[sig_key] += 1

        if sig_key not in signature_examples:
            signature_examples[sig_key] = sample

        #sample_total_bytes.append(
        #    int(sum(v.numel() * v.element_size() for v in sample.values()))
        #)

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
        #"sample_total_input_bytes_mean": float(np.mean(sample_total_bytes)) if sample_total_bytes else 0.0,
        #"sample_total_input_bytes_median": float(np.median(sample_total_bytes)) if sample_total_bytes else 0.0,
        #"sample_total_input_bytes_min": int(min(sample_total_bytes)) if sample_total_bytes else 0,
        #"sample_total_input_bytes_max": int(max(sample_total_bytes)) if sample_total_bytes else 0,
        "emissions_kg_co2eq": emissions,
        "emissions_g_co2eq": None if emissions is None else emissions * 1000.0,
        "dataset_bundle": cfg.get("dataset_bundle"),
        "model_bundle": cfg.get("model_bundle"),
    }

    if bool(cfg.get("measure_compression_energy", True)):
        extra = measure_compression_and_inference_energy(
            cfg, model, device, stats_dir, model_name, dataset_name
        )
        summary["compression_ratio"] = extra["compression"]["compression_ratio"]
        for k in ["gpu_name", "cpu_model", "torch", "cuda_runtime", "cudnn", "python"]:
            summary[f"env_{k}"] = extra["environment"].get(k)
        if "energy" in extra:
            e = extra["energy"]
            summary["inference_kwh_per_sample"] = e["kwh_per_sample"]
            summary["inference_kg_co2eq_per_sample"] = e["kg_co2eq_per_sample"]
            summary["inference_kwh_per_gflop"] = (
                e["kwh_per_sample"] / (mean_flops_numeric / 1e9)
                if mean_flops_numeric and np.isfinite(mean_flops_numeric) else np.nan
            )
            summary["implied_kg_co2eq_per_kwh"] = e["implied_kg_co2eq_per_kwh"]
            for k in ["precision", "batch_size", "throughput_images_per_s", "latency_per_image_ms",
                      "latency_batch_p95_ms", "peak_gpu_memory_allocated_mb", "joules_per_image"]:
                summary[f"inference_{k}"] = e[k]

    summary["step_duration_s"] = float(time.perf_counter() - step_t0)

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




