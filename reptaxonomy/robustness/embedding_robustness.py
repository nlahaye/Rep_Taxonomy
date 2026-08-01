from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import skdim
from sklearn import metrics

from reptaxonomy.util.general_utils import read_yaml

def get_dataset_name(cfg: Dict[str, Any]) -> str:
    if "dataset_bundle" in cfg:
        spec = cfg["dataset_bundle"]["dataset_spec"]
        return spec.get("dataset_name", spec.get("name"))
    return cfg["dataset"].get("dataset_name", cfg["dataset"].get("name"))

def get_model_name(cfg: Dict[str, Any]) -> str:
    if "model_bundle" in cfg:
        return cfg["model_bundle"]["model_name"]
    return cfg["encoder"]

def get_id_estimators():
    return {
        "MLE": lambda k: skdim.id.MLE(neighborhood_based=True, n_neighbors=k),
        "TwoNN": lambda k: skdim.id.TwoNN(),
        "FisherS": lambda k: skdim.id.FisherS(project_on_sphere=False),
        "MOM": lambda k: skdim.id.MOM(),
        "TLE": lambda k: skdim.id.TLE(),
        "CorrInt": lambda k: skdim.id.CorrInt(),
        "DANCo": lambda k: skdim.id.DANCo(k=k),
        "ESS": lambda k: skdim.id.ESS(),
        "MiND_ML": lambda k: skdim.id.MiND_ML(ver="ML"),
        "MiND_KL": lambda k: skdim.id.MiND_ML(ver="KL"),
        "MADA": lambda k: skdim.id.MADA(),
    }

def estimate_id(embed: np.ndarray, methods: List[str], k_values: List[int]) -> List[Dict[str, Any]]:
    results = []
    estimators = get_id_estimators()
    for method in methods:
        for k in k_values:
            try:
                estimator = estimators[method](k)
                estimator.fit(embed)
                if hasattr(estimator, "dimension_"):
                    val = float(estimator.dimension_)
                elif hasattr(estimator, "dimension_pw_"):
                    val = float(np.nanmean(estimator.dimension_pw_))
                else:
                    raise RuntimeError(f"No dimension attribute for {method}")
                results.append({"method": method, "k": k, "global_id": val, "error": ""})
            except Exception as e:
                results.append({"method": method, "k": k, "global_id": np.nan, "error": str(e)})
    return results

def subsample(X: np.ndarray, y: np.ndarray, max_samples: int, seed: int):
    if X.shape[0] <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[idx], y[idx]

def analyze_robustness(cfg: Dict[str, Any]):
    dataset_name = get_dataset_name(cfg)
    encoder_name = get_model_name(cfg)
    analysis_cfg = cfg.get("analysis", {})

    out_dir = os.path.join(cfg["out_dir"], dataset_name, encoder_name)
    n_runs = int(cfg.get("n_runs", 3))
    methods = analysis_cfg.get("id_methods", ["MLE", "TwoNN", "TLE", "DANCo", "FisherS", "MOM", "CorrInt", "ESS", "MiND_ML", "MiND_KL", "MADA"])
    k_values = analysis_cfg.get("id_k_values", [5, 10, 20])
    max_id_samples = int(analysis_cfg.get("max_id_samples", 20000))
    projections = cfg.get("projections", ["umap", "tsne", "pca"])

    all_id_rows = []
    all_sil_rows = []

    for run_idx in range(n_runs):
        run_name = f"run_{run_idx}"
        run_dir = os.path.join(out_dir, run_name)
        embed_path = os.path.join(run_dir, f"{encoder_name}.{dataset_name}.embed_subset.npy")
        target_path = os.path.join(run_dir, f"{encoder_name}.{dataset_name}.target_subset.npy")

        if not (os.path.exists(embed_path) and os.path.exists(target_path)):
            continue

        embed_full = np.load(embed_path, mmap_mode="r")
        target_full = np.load(target_path, mmap_mode="r")
        X_id, y_id = subsample(np.asarray(embed_full), np.asarray(target_full), max_id_samples, int(cfg.get("seed", 42)) + run_idx)

        id_rows = estimate_id(X_id, methods, k_values)
        for row in id_rows:
            row.update({
                "dataset": dataset_name,
                "model": encoder_name,
                "run": run_idx,
                "n_samples": int(X_id.shape[0]),
                "ambient_dim": int(X_id.shape[1]),
                "full_n_samples": int(embed_full.shape[0]),
            })
            all_id_rows.append(row)

        for projection in projections:
            proj_path = os.path.join(run_dir, f"{encoder_name}.{dataset_name}.{projection.upper()}.projection.npy")
            if not os.path.exists(proj_path):
                continue
            proj = np.load(proj_path, mmap_mode="r")
            sil = metrics.silhouette_score(np.asarray(proj), np.asarray(target_full))
            all_sil_rows.append({
                "dataset": dataset_name,
                "projection": projection,
                "model": encoder_name,
                "run": run_idx,
                "silhouette": float(sil),
            })

    if all_id_rows:
        pd.DataFrame(all_id_rows).to_csv(
            os.path.join(out_dir, f"intrinsic_dimension_summary.{encoder_name}.{dataset_name}.csv"),
            index=False,
        )

    if all_sil_rows:
        df = pd.DataFrame(all_sil_rows)
        df.to_csv(
            os.path.join(out_dir, f"silhouette_stats_recomputed.by_run.{encoder_name}.{dataset_name}.csv"),
            index=False,
        )
        summary = (
            df.groupby(["dataset", "projection", "model"], as_index=False)["silhouette"]
            .agg(mean_sil="mean", std_sil="std")
        )
        summary.to_csv(
            os.path.join(out_dir, f"silhouette_stats_recomputed.{encoder_name}.{dataset_name}.csv"),
            index=False,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config file.")
    args = parser.parse_args()
    cfg = read_yaml(args.yaml)
    analyze_robustness(cfg)


