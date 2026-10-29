"""Empirical invariance checks for P1 (silhouette / lens geometry) and P2 (k-NN graphs,
data-kernel statistics, linear CKA).

For each model, a subsample of the run-0 native embeddings is transformed by

  translation  X + 1 t^T          (t ~ N(0, s_t^2 I), s_t = translation_scale * mean feature std)
  rotation     X Q                (Q Haar-random orthogonal, d x d)
  scaling      c X                (isotropic, c = scale_factor)
  ordering     P X                (random row permutation; results are un-permuted before comparison)

and each lens (native, PCA, optionally UMAP / t-SNE) is recomputed. We compare
silhouette, the k-NN graph (mean per-node Jaccard after un-permuting), and linear CKA
between the baseline and transformed lens. For P2, data-kernel statistics for a model
pair are recomputed after (a) a joint sample permutation and (b) a transform applied
to one model.

Each row carries `expected`: "invariant", "not_invariant", or "approximate"
(stochastic lenses). `passed` is evaluated only for deterministic expectations.

Outputs (out_dir/<ds>/):
  invariance_checks.csv
  invariance_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sklearn import metrics
from sklearn.decomposition import PCA

from reptaxonomy.util.general_utils import read_yaml
from reptaxonomy.util.artifacts import (
    ensure_dict,
    get_model_names,
    dataset_out_root,
    embed_subset_path,
    target_subset_path,
)
from reptaxonomy.projections.run_projections import build_sparse_knn_graph, normalize_native, knn_k
from reptaxonomy.intercompare.reference_similarity import knn_jaccard, linear_cka

logger = logging.getLogger(__name__)

TRANSFORMS = ("translation", "rotation", "scaling", "ordering")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))[None, :]


def apply_transform(X: np.ndarray, name: str, rng: np.random.Generator, cfg: Dict[str, Any]
                    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (X_transformed, perm). perm is the row permutation for `ordering`, else None."""
    X = np.asarray(X, dtype=np.float64)
    if name == "translation":
        s = float(cfg.get("translation_scale", 5.0)) * float(X.std(axis=0).mean() + 1e-12)
        return X + rng.normal(0.0, s, size=(1, X.shape[1])), None
    if name == "rotation":
        return X @ random_orthogonal(X.shape[1], rng), None
    if name == "scaling":
        return float(cfg.get("scale_factor", 3.7)) * X, None
    if name == "ordering":
        perm = rng.permutation(X.shape[0])
        return X[perm], perm
    raise ValueError(f"Unknown transform {name}")


def unpermute_graph(graph: csr_array, perm: np.ndarray) -> csr_array:
    """Graph built on X[perm] -> graph in the original node order."""
    inv = np.argsort(perm)
    return csr_array(graph[inv][:, inv])


# ---------------------------------------------------------------------------
# Expected invariances (analytic)
# ---------------------------------------------------------------------------

def expected_invariance(lens: str, transform: str, metric: str, normalization: str) -> str:
    """Analytic expectation for silhouette and k-NN graph under a transform."""
    if lens in ("umap", "tsne"):
        if transform == "translation" and metric == "cosine":
            return "not_invariant"
        return "approximate"
    if transform == "ordering":
        return "invariant"
    if lens == "pca":
        # PCA centers the data; components are rotation-equivariant (sign flips do not
        # change distances); isotropic scaling rescales all coordinates equally.
        return "invariant"
    # native lens
    norm = normalization.lower()
    if transform == "scaling":
        return "invariant"
    if transform == "rotation":
        return "not_invariant" if norm == "standardize" else "invariant"
    if transform == "translation":
        if norm in ("center", "standardize"):
            return "invariant"
        if norm == "l2" or metric == "cosine":
            return "not_invariant"
        return "invariant"
    return "approximate"


# ---------------------------------------------------------------------------
# Lenses
# ---------------------------------------------------------------------------

def fit_lens(X: np.ndarray, lens: str, cfg: Dict[str, Any]) -> np.ndarray:
    seed = int(cfg.get("seed", 42))
    if lens == "native":
        return normalize_native(X, cfg.get("native_normalization", "l2")).astype(np.float64)
    if lens == "pca":
        # Full SVD so the check isolates the transform rather than randomized-solver noise.
        return PCA(n_components=min(int(cfg.get("pca_max_components", 64)), X.shape[1]),
                   svd_solver="full").fit_transform(X)[:, :2]
    if lens == "umap":
        import umap
        return umap.UMAP(metric="cosine", n_neighbors=int(cfg.get("umap_n_neighbors", 15)),
                         min_dist=float(cfg.get("umap_min_dist", 0.1)), n_components=2,
                         spread=float(cfg.get("umap_spread", 1.0)), random_state=seed).fit_transform(X)
    if lens == "tsne":
        import openTSNE
        emb = openTSNE.TSNE(n_jobs=int(cfg.get("tsne_n_jobs", 8)), metric="cosine",
                            exaggeration=4, random_state=seed).fit(X)
        return np.asarray(emb)
    raise ValueError(f"Unsupported lens {lens}")


def lens_metric(cfg: Dict[str, Any], lens: str) -> str:
    return str(cfg.get("native_metric", "cosine")) if lens == "native" else str(cfg.get("knn_metric", "euclidean"))


def _lens_stats(L: np.ndarray, y: np.ndarray, metric: str, k: int) -> Tuple[float, csr_array]:
    sil = float(metrics.silhouette_score(L, y, metric=metric))
    g = build_sparse_knn_graph(L, k=k, metric=metric)
    return sil, g


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _subsample(X: np.ndarray, y: np.ndarray, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= n:
        return np.asarray(X, dtype=np.float64), np.asarray(y)
    idx = np.sort(rng.choice(X.shape[0], size=n, replace=False))
    return np.asarray(X[idx], dtype=np.float64), np.asarray(y[idx])


def p1_checks(model: str, X: np.ndarray, y: np.ndarray, cfg: Dict[str, Any],
              lenses: List[str], rng: np.random.Generator) -> List[Dict[str, Any]]:
    rows = []
    norm = str(cfg.get("native_normalization", "l2"))
    sil_tol = float(cfg.get("invariance_tol", 1e-6))
    graph_tol = float(cfg.get("graph_jaccard_tol", 0.01))
    k = knn_k(cfg, X.shape[0])
    for lens in lenses:
        metric = lens_metric(cfg, lens)
        base_L = fit_lens(X, lens, cfg)
        base_sil, base_g = _lens_stats(base_L, y, metric, k)
        for t in TRANSFORMS:
            Xt, perm = apply_transform(X, t, rng, cfg)
            yt = y[perm] if perm is not None else y
            Lt = fit_lens(Xt, lens, cfg)
            sil_t, g_t = _lens_stats(Lt, yt, metric, k)
            if perm is not None:
                g_t = unpermute_graph(g_t, perm)
                Lt = Lt[np.argsort(perm)]
            jac = knn_jaccard(base_g, g_t)
            cka = linear_cka(base_L, Lt)
            exp = expected_invariance(lens, t, metric, norm)
            d_sil = abs(sil_t - base_sil)
            passed = None
            if exp == "invariant":
                # k-NN graphs are invariant only up to tie-breaking among equidistant
                # neighbors (frequent with nearest-token upsampling, where all pixels under
                # one token share an embedding), hence a separate graph tolerance.
                passed = bool(d_sil <= sil_tol and jac >= 1.0 - graph_tol)
            elif exp == "not_invariant":
                passed = None  # informative only: a change is expected but not required
            rows.append({
                "perspective": "P1", "model": model, "lens": lens, "metric": metric,
                "native_normalization": norm if lens == "native" else None, "transform": t,
                "expected": exp, "silhouette_base": base_sil, "silhouette_transformed": sil_t,
                "abs_delta_silhouette": d_sil, "knn_jaccard": jac, "lens_linear_cka": cka,
                "passed": passed, "n_samples": int(X.shape[0]), "k": int(k),
            })
    return rows


def p2_checks(pair: Tuple[str, str], data: Dict[str, Tuple[np.ndarray, np.ndarray]],
              cfg: Dict[str, Any], rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Data-kernel invariance on native-lens graphs for one model pair."""
    try:
        from reptaxonomy.intercompare.data_kernel_analysis import pair_kernel_statistics
    except Exception as exc:  # graspologic missing
        logger.warning("Skipping P2 data-kernel invariance (%s)", exc)
        return []

    a, b = pair
    Xa, Xb = data[a][0], data[b][0]
    metric = lens_metric(cfg, "native")
    k = knn_k(cfg, Xa.shape[0])
    kw = dict(number_of_bootstraps=int(cfg.get("number_of_bootstraps", 10)),
              n_components=int(cfg.get("ase_n_components", 1)), seed=int(cfg.get("seed", 42)),
              pvalue_plus_one=bool(cfg.get("pvalue_plus_one", False)))
    ga = build_sparse_knn_graph(fit_lens(Xa, "native", cfg), k=k, metric=metric).astype(np.float64)
    gb = build_sparse_knn_graph(fit_lens(Xb, "native", cfg), k=k, metric=metric).astype(np.float64)
    base = pair_kernel_statistics(ga, gb, **kw)
    base_cka = linear_cka(Xa, Xb)
    rows = []
    tol = float(cfg.get("invariance_tol", 1e-6))
    graph_tol = float(cfg.get("graph_jaccard_tol", 0.01))
    res_tol = float(cfg.get("residual_tol", 0.05))
    # OmnibusEmbed uses randomized SVD, so the omnibus distance is invariant only up to
    # solver noise (~0.1-1% relative in our tests); compare on a relative scale.
    dist_rel_tol = float(cfg.get("distance_rel_tol", 0.01))
    norm = str(cfg.get("native_normalization", "l2"))

    for t in TRANSFORMS:
        if t == "ordering":
            # Joint permutation of both models' samples (alignment preserved).
            perm = rng.permutation(Xa.shape[0])
            ga_t = build_sparse_knn_graph(fit_lens(Xa[perm], "native", cfg), k=k, metric=metric).astype(np.float64)
            gb_t = build_sparse_knn_graph(fit_lens(Xb[perm], "native", cfg), k=k, metric=metric).astype(np.float64)
            cka_t = linear_cka(Xa[perm], Xb[perm])
            exp = "invariant_in_distribution"
        else:
            Xb_t, _ = apply_transform(Xb, t, rng, cfg)
            ga_t = ga
            gb_t = build_sparse_knn_graph(fit_lens(Xb_t, "native", cfg), k=k, metric=metric).astype(np.float64)
            cka_t = linear_cka(Xa, Xb_t)
            exp = expected_invariance("native", t, metric, norm)
        st = pair_kernel_statistics(ga_t, gb_t, **kw)
        d_res = abs(st["residual"] - base["residual"])
        d_dist = abs(st["rel_frobenius"] - base["rel_frobenius"])
        d_dist_rel = d_dist / max(abs(base["rel_frobenius"]), 1e-12)
        graph_jac = knn_jaccard(gb, gb_t) if t != "ordering" else np.nan
        if exp == "invariant":
            # Identical graphs give identical (seeded) statistics; tie-induced edge changes
            # propagate, so the pass criterion is on the graph and a residual tolerance.
            passed = bool(graph_jac >= 1.0 - graph_tol and d_res <= res_tol and d_dist_rel <= dist_rel_tol)
        elif exp == "invariant_in_distribution":
            # Bootstrap draws are re-assigned to nodes under permutation, so the residual is
            # invariant only in distribution; the omnibus distance up to randomized-SVD noise.
            passed = bool(d_res <= res_tol and d_dist_rel <= dist_rel_tol)
        else:
            passed = None
        rows.append({
            "perspective": "P2", "model": f"{a}|{b}", "lens": "native", "metric": metric,
            "native_normalization": norm, "transform": t, "expected": exp,
            "residual_base": base["residual"], "residual_transformed": st["residual"],
            "abs_delta_residual": d_res, "rel_frobenius_base": base["rel_frobenius"],
            "rel_frobenius_transformed": st["rel_frobenius"], "abs_delta_rel_frobenius": d_dist, "rel_delta_rel_frobenius": d_dist_rel,
            "linear_cka_base": base_cka, "linear_cka_transformed": cka_t,
            "abs_delta_linear_cka": abs(cka_t - base_cka), "knn_jaccard": graph_jac, "passed": passed,
            "n_samples": int(Xa.shape[0]), "k": int(k),
        })
    return rows


def run_invariance_checks(cfg: Dict[str, Any]) -> pd.DataFrame:
    cfg = ensure_dict(cfg, "invariance_checks cfg")
    models = get_model_names(cfg)
    run_idx = int(cfg.get("invariance_run", 0))
    n_sub = int(cfg.get("invariance_n_samples", 2000))
    lenses = [str(x).lower() for x in cfg.get("invariance_lenses", ["native", "pca"])]
    pairs = cfg.get("invariance_pairs") or ([models[:2]] if len(models) >= 2 else [])
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    shared_idx = None
    rows: List[Dict[str, Any]] = []
    for m in models:
        ep, tp = embed_subset_path(cfg, run_idx, m), target_subset_path(cfg, run_idx, m)
        if not (os.path.exists(ep) and os.path.exists(tp)):
            logger.warning("Missing embed subset for %s run %s", m, run_idx)
            continue
        X = np.load(ep, mmap_mode="r")
        y = np.load(tp, mmap_mode="r")
        if shared_idx is None:  # identical rows for every model keeps P2 pairs aligned
            shared_idx = np.sort(rng.choice(X.shape[0], size=min(n_sub, X.shape[0]), replace=False))
        Xs, ys = np.asarray(X[shared_idx], dtype=np.float64), np.asarray(y[shared_idx])
        data[m] = (Xs, ys)
        rows.extend(p1_checks(m, Xs, ys, cfg, lenses, rng))

    for pair in pairs:
        if len(pair) == 2 and pair[0] in data and pair[1] in data:
            rows.extend(p2_checks((str(pair[0]), str(pair[1])), data, cfg, rng))

    df = pd.DataFrame(rows)
    out_root = dataset_out_root(cfg)
    if not df.empty:
        df.to_csv(os.path.join(out_root, "invariance_checks.csv"), index=False)
        summ = (df.assign(passed_num=df["passed"].map({True: 1.0, False: 0.0}))
                  .groupby(["perspective", "lens", "transform", "expected"], dropna=False)
                  .agg(n=("model", "count"), pass_rate=("passed_num", "mean"))
                  .reset_index())
        summ.to_csv(os.path.join(out_root, "invariance_summary.csv"), index=False)
        n_fail = int((df["passed"] == False).sum())  # noqa: E712
        if n_fail:
            logger.warning("%d invariance checks failed; see invariance_checks.csv", n_fail)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True, help="YAML config file.")
    args = parser.parse_args()
    run_invariance_checks(read_yaml(args.yaml))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
