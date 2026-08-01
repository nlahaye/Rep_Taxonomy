from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from reptaxonomy.util.general_utils import read_yaml, resolve_model_names


REPRESENTATIONS = {"euc", "cos", "abs", "geo_euc", "geo_ig", "geo_sphere"}


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_dataset_name(cfg: Dict[str, Any]) -> str:
    if "dataset_bundle" in cfg and "dataset_spec" in cfg["dataset_bundle"]:
        return cfg["dataset_bundle"]["dataset_spec"].get(
            "dataset_name",
            cfg["dataset_bundle"]["dataset_spec"].get("name"),
        )
    if "dataset" in cfg:
        return cfg["dataset"].get("dataset_name", cfg["dataset"].get("name"))
    raise ValueError("Could not resolve dataset name from config")


def get_representation_name(cfg: Dict[str, Any]) -> str:
    rep = cfg.get("representation", "geo_euc")
    if rep not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {rep}")
    return rep


def get_n_runs(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("n_runs", 3))


def get_analysis_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("analysis", {}) if isinstance(cfg.get("analysis", {}), dict) else {}


def load_representation_builder(rep_name: str):
    if rep_name == "euc":
        import representations.euc as mod
    elif rep_name == "cos":
        import representations.cos as mod
    elif rep_name == "abs":
        import representations.abs as mod
    elif rep_name == "geo_euc":
        import representations.geo_euc as mod
    elif rep_name == "geo_ig":
        import representations.geo_ig as mod
    elif rep_name == "geo_sphere":
        import representations.geo_sphere as mod
    else:
        raise ValueError(f"Unknown representation: {rep_name}")

    for name in ["build_representation", "compute_representation", "get_representation", "transform"]:
        if hasattr(mod, name):
            return getattr(mod, name), mod

    raise AttributeError(
        f"Could not find a representation entry point in module '{rep_name}'."
    )


def build_relative_geodesic_representation(
    embeddings: np.ndarray,
    builder,
    module,
    anchors: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    if anchors is not None:
        try:
            rep = builder(embeddings, anchors=anchors, **kwargs)
            return np.asarray(rep)
        except TypeError:
            pass

    try:
        rep = builder(embeddings, **kwargs)
        return np.asarray(rep)
    except TypeError:
        for cls_name in ["Representation", "RelativeRepresentation", "GeodesicRepresentation"]:
            if hasattr(module, cls_name):
                cls = getattr(module, cls_name)
                obj = cls(**kwargs)
                for method_name in ["fit_transform", "transform", "compute"]:
                    if hasattr(obj, method_name):
                        method = getattr(obj, method_name)
                        if anchors is not None:
                            try:
                                return np.asarray(method(embeddings, anchors=anchors))
                            except TypeError:
                                pass
                        return np.asarray(method(embeddings))
        raise


def pairwise_geometry_similarity(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]

    n = min(len(a), len(b))
    if n < 2:
        return {
            "mean_cosine_between_representations": np.nan,
            "pairwise_distance_spearman": np.nan,
            "pairwise_distance_spearman_p": np.nan,
        }

    a = a[:n]
    b = b[:n]

    denom = (np.linalg.norm(a, axis=1) + 1e-12) * (np.linalg.norm(b, axis=1) + 1e-12)
    mean_cos = float(np.mean(np.sum(a * b, axis=1) / denom))

    da = cdist(a, a, metric="euclidean")
    db = cdist(b, b, metric="euclidean")
    iu = np.triu_indices_from(da, k=1)
    rho, p = spearmanr(da[iu], db[iu])

    return {
        "mean_cosine_between_representations": mean_cos,
        "pairwise_distance_spearman": float(rho),
        "pairwise_distance_spearman_p": float(p),
    }


def subset_artifact_paths(
    out_root: str,
    dataset_name: str,
    model_name: str,
    run_idx: int,
) -> Tuple[str, str]:
    run_dir = os.path.join(out_root, dataset_name, model_name, f"run_{run_idx}")
    embed_path = os.path.join(run_dir, f"{model_name}.{dataset_name}.embed_subset.npy")
    target_path = os.path.join(run_dir, f"{model_name}.{dataset_name}.target_subset.npy")
    return embed_path, target_path


def load_cached_embeddings(
    cfg: Dict[str, Any],
    model_names: List[str],
) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    dataset_name = get_dataset_name(cfg)
    out_root = cfg["out_dir"]
    n_runs = get_n_runs(cfg)

    cached: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}

    for run_idx in range(n_runs):
        run_cache: Dict[str, Dict[str, np.ndarray]] = {}
        for model_name in model_names:
            embed_path, target_path = subset_artifact_paths(out_root, dataset_name, model_name, run_idx)
            if not os.path.exists(embed_path):
                raise FileNotFoundError(
                    f"Missing embedding subset for model '{model_name}', run {run_idx}: {embed_path}"
                )
            if not os.path.exists(target_path):
                raise FileNotFoundError(
                    f"Missing target subset for model '{model_name}', run {run_idx}: {target_path}"
                )

            run_cache[model_name] = {
                "embed": np.load(embed_path),
                "target": np.load(target_path),
                "embed_path": embed_path,
                "target_path": target_path,
            }
        cached[run_idx] = run_cache

    return cached


def align_for_comparison(
    a_embed: np.ndarray,
    a_target: np.ndarray,
    b_embed: np.ndarray,
    b_target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(a_embed), len(b_embed), len(a_target), len(b_target))
    a_embed = a_embed[:n]
    b_embed = b_embed[:n]
    a_target = a_target[:n]
    b_target = b_target[:n]

    same_target_mask = (a_target == b_target)
    if same_target_mask.sum() >= 2:
        return a_embed[same_target_mask], b_embed[same_target_mask], a_target[same_target_mask]

    return a_embed, b_embed, a_target


def run_geodesic_compare(cfg: Dict[str, Any]) -> None:
    model_names = resolve_model_names(cfg)
    dataset_name = get_dataset_name(cfg)
    representation = get_representation_name(cfg)
    analysis_cfg = get_analysis_cfg(cfg)
    n_runs = get_n_runs(cfg)

    out_root = cfg["out_dir"]
    compare_root = ensure_dir(os.path.join(out_root, dataset_name, "geodesic_compare"))
    builder, module = load_representation_builder(representation)

    cached = load_cached_embeddings(cfg, model_names)

    anchor_mode = analysis_cfg.get("anchor_mode", "per_model")
    anchor_model_name = analysis_cfg.get("anchor_model")
    representation_kwargs = analysis_cfg.get("representation_kwargs", {})

    all_pair_rows: List[Dict[str, Any]] = []
    all_manifest_rows: List[Dict[str, Any]] = []

    for run_idx in range(n_runs):
        run_name = f"run_{run_idx}"
        run_out_dir = ensure_dir(os.path.join(compare_root, run_name))
        run_cache = cached[run_idx]

        if anchor_mode == "fixed":
            if not anchor_model_name:
                raise ValueError("analysis.anchor_model must be set when analysis.anchor_mode='fixed'")
            anchor_models = [anchor_model_name]
        else:
            anchor_models = list(model_names)

        for anchor_name in anchor_models:
            if anchor_name not in run_cache:
                raise ValueError(f"Anchor model '{anchor_name}' not found in cached run {run_idx}")

            anchors = run_cache[anchor_name]["embed"]
            transformed: Dict[str, np.ndarray] = []

            transformed = {}
            manifest_rows = []

            anchor_out_dir = ensure_dir(os.path.join(run_out_dir, anchor_name))

            for model_name in model_names:
                x = run_cache[model_name]["embed"]
                rep = build_relative_geodesic_representation(
                    embeddings=x,
                    builder=builder,
                    module=module,
                    anchors=anchors,
                    **representation_kwargs,
                )
                rep = np.asarray(rep)

                out_path = os.path.join(
                    anchor_out_dir,
                    f"{anchor_name}_anchored_{model_name}_{representation}.npy",
                )
                np.save(out_path, rep)
                transformed[model_name] = rep

                row = {
                    "run": run_name,
                    "dataset": dataset_name,
                    "model": model_name,
                    "anchor": anchor_name,
                    "representation": representation,
                    "input_samples": int(x.shape[0]),
                    "input_dim": int(x.shape[1]) if x.ndim > 1 else 1,
                    "output_samples": int(rep.shape[0]),
                    "output_dim": int(rep.shape[1]) if rep.ndim > 1 else 1,
                    "embed_subset_path": run_cache[model_name]["embed_path"],
                    "target_subset_path": run_cache[model_name]["target_path"],
                    "path": out_path,
                }
                manifest_rows.append(row)
                all_manifest_rows.append(row)

            manifest_df = pd.DataFrame(manifest_rows)
            manifest_df.to_csv(
                os.path.join(
                    anchor_out_dir,
                    f"{anchor_name}_anchored_{representation}_representation_manifest.csv",
                ),
                index=False,
            )

            pair_rows = []
            ordered_models = list(transformed.keys())
            for i, m1 in enumerate(ordered_models):
                for j, m2 in enumerate(ordered_models):
                    if j <= i:
                        continue

                    a_rep = transformed[m1]
                    b_rep = transformed[m2]
                    a_target = run_cache[m1]["target"]
                    b_target = run_cache[m2]["target"]

                    a_rep_aligned, b_rep_aligned, aligned_target = align_for_comparison(
                        a_rep, a_target, b_rep, b_target
                    )

                    metrics_row = pairwise_geometry_similarity(a_rep_aligned, b_rep_aligned)
                    row = {
                        "run": run_name,
                        "dataset": dataset_name,
                        "anchor": anchor_name,
                        "model_a": m1,
                        "model_b": m2,
                        "representation": representation,
                        "n_compared": int(len(a_rep_aligned)),
                        "target_agreement_fraction": float(
                            np.mean(a_target[: min(len(a_target), len(b_target))] == b_target[: min(len(a_target), len(b_target))])
                        ),
                        **metrics_row,
                    }
                    pair_rows.append(row)
                    all_pair_rows.append(row)

            pair_df = pd.DataFrame(pair_rows)
            pair_df.to_csv(
                os.path.join(
                    anchor_out_dir,
                    f"{anchor_name}_anchored_{representation}_pairwise_comparison.csv",
                ),
                index=False,
            )

    if all_manifest_rows:
        pd.DataFrame(all_manifest_rows).to_csv(
            os.path.join(compare_root, f"{representation}_representation_manifest.all_runs.csv"),
            index=False,
        )

    if all_pair_rows:
        pair_df = pd.DataFrame(all_pair_rows)
        pair_df.to_csv(
            os.path.join(compare_root, f"{representation}_pairwise_comparison.all_runs.csv"),
            index=False,
        )

        summary = (
            pair_df.groupby(["dataset", "anchor", "model_a", "model_b", "representation"], dropna=False)
            .agg(
                n_runs=("run", "nunique"),
                mean_n_compared=("n_compared", "mean"),
                mean_target_agreement_fraction=("target_agreement_fraction", "mean"),
                mean_cosine_between_representations=("mean_cosine_between_representations", "mean"),
                std_cosine_between_representations=("mean_cosine_between_representations", "std"),
                mean_pairwise_distance_spearman=("pairwise_distance_spearman", "mean"),
                std_pairwise_distance_spearman=("pairwise_distance_spearman", "std"),
            )
            .reset_index()
        )
        summary.to_csv(
            os.path.join(compare_root, f"{representation}_pairwise_comparison.summary.csv"),
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config file.")
    args = parser.parse_args()
    cfg = read_yaml(args.yaml)
    run_geodesic_compare(cfg)
