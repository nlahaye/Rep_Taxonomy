from __future__ import annotations

import argparse
import logging
import os
import pickle
from random import randint
from typing import Any, Dict

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import weightwatcher as ww

from reptaxonomy.util.general_utils import read_yaml
from reptaxonomy.util.experiment_init_utils import build_model_from_bundle


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def safe_pickle_dump(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def dataframe_to_pickle(df, path: str) -> None:
    if hasattr(df, "to_pickle"):
        df.to_pickle(path)
    else:
        safe_pickle_dump(df, path)

def get_dataset_name(cfg: Dict[str, Any]) -> str:
    if "dataset_bundle" in cfg:
        spec = cfg["dataset_bundle"]["dataset_spec"]
        return spec.get("dataset_name", spec.get("name"))
    return cfg["dataset"].get("dataset_name", cfg["dataset"].get("name"))

def get_model_name(cfg: Dict[str, Any]) -> str:
    if "model_bundle" in cfg:
        return cfg["model_bundle"]["model_name"]
    return cfg["encoder"]


def get_colors(n):
    return ["#%06X" % randint(0, 0xFFFFFF) for _ in range(n)]

def plot_metrics_depth(metric, ylabel, title, series_name, all_details, colors, out_dir, log=False, valid_ids=None):
    valid_ids = [] if valid_ids is None else valid_ids
    plt.figure()
    idname = "all" if len(valid_ids) == 0 else "fnl"
    keys_to_plot = list(all_details.keys()) if len(valid_ids) == 0 else valid_ids

    ind = 0
    for key in keys_to_plot:
        if key not in all_details or metric not in all_details[key]:
            continue
        y = all_details[key][metric].to_numpy()
        x = np.arange(len(y))
        if log:
            y = np.log10(np.asarray(y + 1e-6, dtype=float))
        plt.scatter(x, y, label=key, color=colors[ind % len(colors)], s=12)
        ind += 1

    plt.title(f"Depth vs {title} {ylabel}")
    plt.xlabel("Layer id")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{series_name}_{idname}_{metric}_depth.png"))
    plt.close()

def run_weight_mtx_analysis(cfg: Dict[str, Any]):
    dataset_name = get_dataset_name(cfg)
    model_name = get_model_name(cfg)
    analysis_cfg = cfg.get("analysis", {})

    model = build_model_from_bundle(cfg)

    out_dir = ensure_dir(os.path.join(cfg["embed_dir"], dataset_name, model_name, "test"))
    plot_dir = ensure_dir(os.path.join(out_dir, "ww_plots"))
    stats_dir = ensure_dir(os.path.join(out_dir, "ww_stats"))

    watcher = ww.WeightWatcher(model=model, log_level=logging.INFO)
    details = watcher.analyze(
        model=model,
        plot=False,
        min_evals=analysis_cfg.get("ww_min_evals", 50),
        max_evals=analysis_cfg.get("ww_max_evals", 5000),
        randomize=analysis_cfg.get("ww_randomize", True),
        mp_fit=analysis_cfg.get("ww_mp_fit", True),
        pool=analysis_cfg.get("ww_pool", True),
        savefig=False,
        layers=analysis_cfg.get("ww_layers", []),
    )
    summaries = watcher.get_summary(details)

    prefix = model_name
    dataframe_to_pickle(details, os.path.join(stats_dir, f"{prefix}.details.pkl"))
    dataframe_to_pickle(summaries, os.path.join(stats_dir, f"{prefix}.summary.pkl"))
    safe_pickle_dump(cfg, os.path.join(stats_dir, f"{prefix}.config.pkl"))
    safe_pickle_dump(
        {
            "dataset": dataset_name,
            "model": model_name,
            "dataset_bundle": cfg.get("dataset_bundle"),
            "model_bundle": cfg.get("model_bundle"),
            "details_columns": list(details.columns) if hasattr(details, "columns") else None,
            "summary_columns": list(summaries.columns) if hasattr(summaries, "columns") else None,
        },
        os.path.join(stats_dir, f"{prefix}.meta.pkl"),
    )

    details.to_csv(os.path.join(stats_dir, f"{prefix}.details.csv"), index=False)
    summaries.to_csv(os.path.join(stats_dir, f"{prefix}.summary.csv"), index=False)

    colors = get_colors(4)
    details_dict = {model_name: details}
    plot_metrics_depth(
        "log_norm",
        r"Log Frobenius Norm",
        model_name,
        model_name,
        details_dict,
        colors,
        plot_dir,
        log=False,
        valid_ids=[],
    )
    plot_metrics_depth(
        "alpha",
        r"Alpha",
        model_name,
        model_name,
        details_dict,
        colors,
        plot_dir,
        log=False,
        valid_ids=[],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config file.")
    args = parser.parse_args()
    cfg = read_yaml(args.yaml)
    run_weight_mtx_analysis(cfg)


