"""Anchor-based relative (geodesic) representations across encoders.

Each encoder's embeddings are re-expressed relative to a shared set of anchor *samples* (the
same pixels for every encoder, each encoder using its own embeddings of them), so encoders with
different embedding dimensions become directly comparable. Representations (``abs``, ``cos``,
``euc``, ``geo_euc``, ``geo_sphere``, ``geo_ig``) come from
:mod:`reptaxonomy.intercompare.relative_geodesics`, a re-implementation of RelativeGeodesics
(Yu et al., NeurIPS 2025, https://github.com/marc0git/RelativeGeodesics; see that module for
attribution). The geodesic variants measure straight latent paths through a decoder; as in
RelativeGeodesics a small per-encoder classifier head is trained on the frozen embeddings and
labels (fitted on the non-anchor samples of each run).

Outputs: ``<out_dir>/<dataset>/geodesic_compare/run_<i>/<anchor>/`` with per-model ``.npy``
relative representations, ``anchor_rows.npy``, a manifest and ``*_pairwise_comparison.csv``
(read by demsar_rank), plus all-run and summary CSVs in ``geodesic_compare/``.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from reptaxonomy.intercompare import relative_geodesics as RG
from reptaxonomy.util.general_utils import read_yaml, resolve_model_names

logger = logging.getLogger(__name__)

REPRESENTATIONS = set(RG.REPRESENTATIONS)


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


def get_representation_names(cfg: Dict[str, Any]) -> List[str]:
    rep = cfg.get("representation", "geo_euc")
    reps = [r.strip() for r in rep.split(",")] if isinstance(rep, str) else [str(r) for r in rep]
    bad = [r for r in reps if r not in REPRESENTATIONS]
    if bad:
        raise ValueError(f"Unknown representation(s) {bad}; choose from {sorted(REPRESENTATIONS)}")
    return reps


def get_representation_name(cfg: Dict[str, Any]) -> str:   # backwards compatible
    return get_representation_names(cfg)[0]


def get_n_runs(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("n_runs", 3))


def get_analysis_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    a = cfg.get("analysis", {}) if isinstance(cfg.get("analysis", {}), dict) else {}
    g = cfg.get("geodesic", {}) if isinstance(cfg.get("geodesic", {}), dict) else {}
    return {**a, **g}


def _torch_device(cfg: Dict[str, Any]):
    import torch
    dev = str(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev.startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"
    return torch.device(dev)


# ----------------------------------------------------------------------------- comparison
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

    mean_cos = np.nan
    if a.shape[1] == b.shape[1]:   # 'abs' keeps each encoder's own (different) dimension
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


def subset_artifact_paths(out_root: str, dataset_name: str, model_name: str, run_idx: int) -> Tuple[str, str]:
    run_dir = os.path.join(out_root, dataset_name, model_name, f"run_{run_idx}")
    embed_path = os.path.join(run_dir, f"{model_name}.{dataset_name}.embed_subset.npy")
    target_path = os.path.join(run_dir, f"{model_name}.{dataset_name}.target_subset.npy")
    return embed_path, target_path


def load_cached_embeddings(cfg: Dict[str, Any], model_names: List[str]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    dataset_name = get_dataset_name(cfg)
    out_root = cfg["out_dir"]
    cached: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for run_idx in range(get_n_runs(cfg)):
        run_cache: Dict[str, Dict[str, Any]] = {}
        for model_name in model_names:
            embed_path, target_path = subset_artifact_paths(out_root, dataset_name, model_name, run_idx)
            for p, what in ((embed_path, "embedding"), (target_path, "target")):
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Missing {what} subset for model '{model_name}', run {run_idx}: {p}")
            run_cache[model_name] = {
                "embed": np.load(embed_path, mmap_mode="r"),
                "target": np.load(target_path, mmap_mode="r"),
                "embed_path": embed_path,
                "target_path": target_path,
            }
        cached[run_idx] = run_cache
    return cached


def shared_rows(run_cache: Dict[str, Dict[str, Any]], models: Sequence[str], max_samples: int,
                seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Row indices used for every model (subsets are row-aligned across models).

    Returns (rows, targets, label agreement across models). Rows whose labels disagree
    between models are dropped when at least ``2`` agreeing rows remain.
    """
    n = min(min(len(run_cache[m]["embed"]), len(run_cache[m]["target"])) for m in models)
    ys = np.stack([np.asarray(run_cache[m]["target"][:n]).reshape(n, -1)[:, 0] for m in models])
    agree = np.all(ys == ys[0], axis=0)
    rows = np.flatnonzero(agree) if agree.sum() >= 2 else np.arange(n)
    if max_samples and len(rows) > max_samples:
        rows = np.sort(np.random.default_rng(int(seed)).choice(rows, size=int(max_samples), replace=False))
    return rows, ys[0][rows], float(agree.mean()) if n else float("nan")


# ----------------------------------------------------------------------------- step
def run_geodesic_compare(cfg: Dict[str, Any]) -> None:
    """Pipeline step. Config keys (top level or under ``analysis`` / ``geodesic``):

    representation: geo_euc | [geo_euc, cos, ...]   n_anchors: 256   anchor_scheme: uniform|fps|kmeans
    anchor_mode: per_model (fps/kmeans anchors chosen in each model's space in turn) | fixed
    anchor_model: <model> (fixed mode)   steps: 30   max_samples: 2000   split_steps: 8
    decoder: {hidden_dim: 500, epochs: 10, lr: 5.0e-4, batch_size: 64}   normalize: true
    energy: false (path length; true = path energy)   save_representations: true
    """
    model_names = resolve_model_names(cfg)
    dataset_name = get_dataset_name(cfg)
    reps = get_representation_names(cfg)
    a = get_analysis_cfg(cfg)
    n_runs = get_n_runs(cfg)
    seed = int(cfg.get("seed", 42))
    steps = int(a.get("steps", 30))
    n_anchors = int(a.get("n_anchors", 256))
    scheme = str(a.get("anchor_scheme", "uniform")).lower()
    anchor_mode = str(a.get("anchor_mode", "per_model")).lower()
    anchor_model_name = a.get("anchor_model")
    max_samples = int(a.get("max_samples", 2000))
    split_steps = int(a.get("split_steps", 8))
    energy = bool(a.get("energy", False))
    normalize = bool(a.get("normalize", True))
    save_reps = bool(a.get("save_representations", True))
    dcfg = dict(a.get("decoder", {}) or {})
    device = _torch_device(cfg)

    out_root = cfg["out_dir"]
    compare_root = ensure_dir(os.path.join(out_root, dataset_name, "geodesic_compare"))
    cached = load_cached_embeddings(cfg, model_names)

    all_pair_rows: List[Dict[str, Any]] = []
    all_manifest_rows: List[Dict[str, Any]] = []
    for run_idx in range(n_runs):
        run_name = f"run_{run_idx}"
        run_out_dir = ensure_dir(os.path.join(compare_root, run_name))
        run_cache = cached[run_idx]
        rows, y, label_agreement = shared_rows(run_cache, model_names, max_samples, seed + run_idx)
        X = {m: np.asarray(run_cache[m]["embed"][rows], dtype=np.float32).reshape(len(rows), -1)
             for m in model_names}

        if scheme == "uniform":   # anchors do not depend on any model's geometry
            anchor_sets = {"shared": RG.select_anchors(X[model_names[0]], n_anchors, "uniform", seed + run_idx)}
        else:
            if anchor_mode == "fixed":
                if not anchor_model_name or anchor_model_name not in X:
                    raise ValueError("anchor_model must name one of the models when anchor_mode='fixed'")
                pickers = [anchor_model_name]
            else:
                pickers = list(model_names)
            anchor_sets = {m: RG.select_anchors(X[m], n_anchors, scheme, seed + run_idx) for m in pickers}

        decoders: Dict[Tuple[str, str], Any] = {}
        for anchor_name, aidx in anchor_sets.items():
            anchor_out_dir = ensure_dir(os.path.join(run_out_dir, anchor_name))
            np.save(os.path.join(anchor_out_dir, "anchor_rows.npy"), rows[aidx])
            fit_mask = np.ones(len(rows), dtype=bool)
            fit_mask[aidx] = False
            for rep_name in reps:
                needs_dec = rep_name in RG.NEEDS_DECODER
                transformed: Dict[str, np.ndarray] = {}
                manifest_rows = []
                for m in model_names:
                    dec, dinfo = None, {}
                    if needs_dec:
                        key = (m, anchor_name)
                        if key not in decoders:
                            decoders[key] = RG.train_decoder(X[m][fit_mask], y[fit_mask], device=device,
                                                          seed=seed + run_idx, **dcfg)
                        dec, dinfo = decoders[key]
                    rep, std = RG.relative_representation(rep_name, X[m], aidx, dec, device=device, steps=steps,
                                                          energy=energy, chunk=split_steps, normalize=normalize)
                    out_path = os.path.join(anchor_out_dir, f"{anchor_name}_anchored_{m}_{rep_name}.npy")
                    if save_reps:
                        np.save(out_path, rep.astype(np.float32))
                    transformed[m] = rep
                    row = {
                        "run": run_name, "dataset": dataset_name, "model": m, "anchor": anchor_name,
                        "representation": rep_name, "energy": energy if rep_name != "abs" else None,
                        "anchor_scheme": scheme, "n_anchors": int(len(aidx)), "steps": steps if needs_dec else None,
                        "input_samples": int(X[m].shape[0]), "input_dim": int(X[m].shape[1]),
                        "output_samples": int(rep.shape[0]), "output_dim": int(rep.shape[1]) if rep.ndim > 1 else 1,
                        "anchor_std": std, "label_agreement": label_agreement, **dinfo,
                        "embed_subset_path": run_cache[m]["embed_path"],
                        "target_subset_path": run_cache[m]["target_path"],
                        "path": out_path if save_reps else "",
                    }
                    manifest_rows.append(row)
                    all_manifest_rows.append(row)
                pd.DataFrame(manifest_rows).to_csv(os.path.join(
                    anchor_out_dir, f"{anchor_name}_anchored_{rep_name}_representation_manifest.csv"), index=False)

                pair_rows = []
                ordered = list(transformed)
                for i, m1 in enumerate(ordered):
                    for m2 in ordered[i + 1:]:
                        metrics = pairwise_geometry_similarity(transformed[m1], transformed[m2])
                        row = {"run": run_name, "dataset": dataset_name, "anchor": anchor_name,
                               "model_a": m1, "model_b": m2, "representation": rep_name,
                               "n_compared": int(len(rows)), "target_agreement_fraction": label_agreement,
                               **metrics}
                        pair_rows.append(row)
                        all_pair_rows.append(row)
                pd.DataFrame(pair_rows).to_csv(os.path.join(
                    anchor_out_dir, f"{anchor_name}_anchored_{rep_name}_pairwise_comparison.csv"), index=False)

    tag = "_".join(reps)
    if all_manifest_rows:
        pd.DataFrame(all_manifest_rows).to_csv(
            os.path.join(compare_root, f"{tag}_representation_manifest.all_runs.csv"), index=False)
    if all_pair_rows:
        pair_df = pd.DataFrame(all_pair_rows)
        pair_df.to_csv(os.path.join(compare_root, f"{tag}_pairwise_comparison.all_runs.csv"), index=False)
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
        summary.to_csv(os.path.join(compare_root, f"{tag}_pairwise_comparison.summary.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config file.")
    args = parser.parse_args()
    run_geodesic_compare(read_yaml(args.yaml))
