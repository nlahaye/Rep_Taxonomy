"""Conventional inter-model similarity references for P2.

Computes, for every run and lens (native embeddings and each projection):
  * linear CKA (Kornblith et al., 2019) between aligned per-pixel representations
  * debiased linear CKA (optional; unbiased HSIC estimator, Song et al., 2012)
  * k-NN neighborhood overlap: mean Jaccard index of each sample's neighbor set,
    read from the same symmetrized k-NN graphs that feed the data-kernel analysis

All models in a run are evaluated on identical pixel indices (shared index cache),
which `validate_run_alignment` checks.

Outputs (out_dir/<ds>/):
  pairwise_reference_summary.csv  (run, projection, model_a, model_b, linear_cka,
                                   linear_cka_debiased, knn_jaccard, n_samples)
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_array, load_npz

from reptaxonomy.util.general_utils import read_yaml
from reptaxonomy.util.artifacts import (
    ensure_dict,
    get_model_names,
    get_projections,
    dataset_out_root,
    projection_path,
    knn_graph_path,
    embed_subset_path,
)

logger = logging.getLogger(__name__)


def _center(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA via the feature-space identity (O(n d^2), no n x n Gram matrix).

    CKA(X, Y) = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F), with column-centered Xc, Yc.
    Invariant to orthogonal transforms and isotropic scaling of either input, and to
    translation (because of centering); equivariant to a joint permutation of samples.
    """
    Xc, Yc = _center(X), _center(Y)
    num = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    den = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    return float(num / den) if den > 0 else float("nan")


def _unbiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    n = K.shape[0]
    K = K.copy()
    L = L.copy()
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    ones = np.ones(n)
    term1 = np.sum(K * L)
    term2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
    term3 = 2.0 * (ones @ K @ L @ ones) / (n - 2)
    return float((term1 + term2 - term3) / (n * (n - 3)))


def linear_cka_debiased(X: np.ndarray, Y: np.ndarray) -> float:
    """Debiased linear CKA (uses n x n Gram matrices; call on a subsample)."""
    Xc, Yc = _center(X), _center(Y)
    K, L = Xc @ Xc.T, Yc @ Yc.T
    hxy = _unbiased_hsic(K, L)
    hxx = _unbiased_hsic(K, K)
    hyy = _unbiased_hsic(L, L)
    den = np.sqrt(max(hxx, 0.0) * max(hyy, 0.0))
    return float(hxy / den) if den > 0 else float("nan")


def knn_jaccard(graph_a: csr_array, graph_b: csr_array) -> float:
    """Mean over nodes of |N_a(i) & N_b(i)| / |N_a(i) | N_b(i)| for binary adjacency graphs."""
    A = csr_array(graph_a, dtype=np.int8)
    B = csr_array(graph_b, dtype=np.int8)
    A.data[:] = 1
    B.data[:] = 1
    inter = np.asarray(A.multiply(B).sum(axis=1)).ravel()
    deg_a = np.asarray(A.sum(axis=1)).ravel()
    deg_b = np.asarray(B.sum(axis=1)).ravel()
    union = deg_a + deg_b - inter
    ok = union > 0
    return float(np.mean(inter[ok] / union[ok])) if np.any(ok) else float("nan")


def run_reference_similarity(cfg: Dict[str, Any]) -> None:
    cfg = ensure_dict(cfg, "reference_similarity cfg")
    model_names = get_model_names(cfg)
    projections = get_projections(cfg)
    n_runs = int(cfg.get("n_runs", 3))
    max_n = int(cfg.get("cka_max_samples", 20000))
    debiased_n = int(cfg.get("cka_debiased_samples", 0))   # 0 disables debiased CKA
    seed = int(cfg.get("seed", 42))
    out_root = dataset_out_root(cfg)

    rows: List[Dict[str, Any]] = []
    for run_idx in range(n_runs):
        for projection in projections:
            reps: Dict[str, np.ndarray] = {}
            graphs: Dict[str, csr_array] = {}
            for m in model_names:
                pp = projection_path(cfg, run_idx, projection, model_name=m)
                if projection == "native" and bool(cfg.get("cka_native_raw", True)):
                    # CKA on raw (un-normalized) per-pixel embeddings; CKA centers internally.
                    pp = embed_subset_path(cfg, run_idx, model_name=m)
                gp = knn_graph_path(cfg, run_idx, projection, model_name=m)
                if os.path.exists(pp):
                    reps[m] = np.asarray(np.load(pp, mmap_mode="r"), dtype=np.float32)
                if os.path.exists(gp):
                    graphs[m] = load_npz(gp).tocsr()
            if not reps:
                logger.warning("run=%s projection=%s: no representations found", run_idx, projection)
                continue

            n = min(r.shape[0] for r in reps.values())
            if len({r.shape[0] for r in reps.values()}) != 1:
                raise ValueError(f"Misaligned sample counts for run={run_idx} projection={projection}")
            rng = np.random.default_rng(seed + run_idx)
            idx = np.sort(rng.choice(n, size=min(n, max_n), replace=False))
            idx_db = np.sort(rng.choice(n, size=min(n, debiased_n), replace=False)) if debiased_n else None

            names = [m for m in model_names if m in reps]
            for i, a in enumerate(names):
                for b in names[i:]:
                    cka = 1.0 if a == b else linear_cka(reps[a][idx], reps[b][idx])
                    cka_db = np.nan
                    if idx_db is not None:
                        cka_db = 1.0 if a == b else linear_cka_debiased(reps[a][idx_db], reps[b][idx_db])
                    jac = np.nan
                    if a in graphs and b in graphs:
                        jac = 1.0 if a == b else knn_jaccard(graphs[a], graphs[b])
                    for x, y in ((a, b), (b, a)) if a != b else ((a, b),):
                        rows.append({
                            "run": run_idx, "projection": projection, "model_a": x, "model_b": y,
                            "linear_cka": cka, "linear_cka_debiased": cka_db, "knn_jaccard": jac,
                            "n_samples": int(idx.size),
                        })

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_root, "pairwise_reference_summary.csv"), index=False)
    logger.info("Saved reference similarity summaries under %s", out_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True, help="YAML config file.")
    args = parser.parse_args()
    run_reference_similarity(read_yaml(args.yaml))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
