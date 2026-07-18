import os
import pathlib
import pprint
from typing import Dict, List

import argparse

import numpy as np
import pandas as pd
import skdim
from sklearn import metrics

from reptaxonomy.util.general_utils import read_yaml

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


def estimate_id(embed: np.ndarray, methods: List[str], k_values: List[int]) -> List[Dict]:
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


def analyze_robustness(cfg):

    encoder_name = cfg["encoder"]
    out_dir = os.path.join(cfg["out_dir"], cfg["dataset"]["dataset_name"], encoder_name)

    n_runs = getattr(cfg, "n_runs", 3)
    methods = getattr(cfg, "id_methods", ["MLE", "TwoNN", "TLE", "DANCo", "FisherS", "MOM", "CorrInt", "ESS", "MiND_ML", "MiND_KL", "MADA"])
    k_values = getattr(cfg, "id_k_values", [5, 10, 20])
    max_id_samples = getattr(cfg, "max_id_samples", 20000)

    all_id_rows = []
    all_sil_rows = []

    for run_idx in range(n_runs):
        run_name = f"run_{run_idx}"
        run_dir = os.path.join(out_dir, run_name)
        embed_path = os.path.join(run_dir, f"{encoder_name}.{cfg["dataset"]["dataset_name"]}.embed_subset.npy")
        target_path = os.path.join(run_dir, f"{encoder_name}.{cfg["dataset"]["dataset_name"]}.target_subset.npy")

        if not (os.path.exists(embed_path) and os.path.exists(target_path)):
            logger.warning("Skipping %s because cached subset files are missing", run_name)
            continue

        embed_full = np.load(embed_path, mmap_mode="r")
        target_full = np.load(target_path, mmap_mode="r")
        #X_id, y_id = subsample(np.asarray(embed_full), np.asarray(target_full), max_id_samples, cfg["seed"] + run_idx)

        id_rows = estimate_id(embed_full, methods, k_values)
        for row in id_rows:
            row.update({
                "model": encoder_name,
                "run": run_idx,
                "n_samples": int(embed_full.shape[0]),
                "ambient_dim": int(embed_full.shape[1]),
            })
            all_id_rows.append(row)

        for projection in ["tsne", "umap", "pca"]:
            proj_path = os.path.join(run_dir, f"{encoder_name}.{cfg["dataset"]["dataset_name"]}.{projection.upper()}.projection.npy")
            if not os.path.exists(proj_path):
                continue
            proj = np.load(proj_path, mmap_mode="r")
            sil = metrics.silhouette_score(np.asarray(proj), np.asarray(target_full))
            all_sil_rows.append({
                "projection": projection,
                "model": encoder_name,
                "run": run_idx,
                "silhouette": float(sil),
            })

    if all_id_rows:
        pd.DataFrame(all_id_rows).to_csv(
            os.path.join(out_dir, f"intrinsic_dimension_summary.{encoder_name}.{cfg["dataset"]["dataset_name"]}.csv"),
            index=False,
        )
    if all_sil_rows:
        df = pd.DataFrame(all_sil_rows)
        summary = df.groupby(["projection", "model"], as_index=False)["silhouette"].agg(["mean", "std"]).reset_index()
        summary.columns = ["projection", "model", "mean_sil", "std_sil"]
        summary.to_csv(
            os.path.join(out_dir, f"silhouette_stats_recomputed.{encoder_name}.{cfg["dataset"]["dataset_name"]}.csv"),
            index=False,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config file.")
    args = parser.parse_args()
    cfg = read_yaml(args.yaml)
    analyze_robustness(cfg)


