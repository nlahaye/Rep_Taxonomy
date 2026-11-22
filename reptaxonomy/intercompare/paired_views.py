"""Paired-view representation comparison with block-bootstrap uncertainty.

Generic counterpart of reference_similarity for representations that are not produced by
the Rep_Taxonomy embedding pipeline: any set of aligned per-sample views (encoder embeddings
of two sensors at co-located pixels, two encoders on the same inputs, an embedding and a
table of physical variables, ...). Rows are aligned by key across bundles; uncertainty comes
from a bootstrap over dependence blocks (e.g. date block x spatial tile), never over rows.

Views bundle (.npz; one or more per run)
  keys          (n,)     str   row identity used for alignment across bundles
  groups        (n,)     str   bootstrap / cross-validation block of each row
  domain        (n,)     str   optional (sensor, platform, region, ...)
  view__<name>  (n, d)   float continuous representation
  labels__<name>(n,)     int   discrete assignment (cluster id); negative = missing
  target__<name>(n,)     float probe target (physical variable); NaN = missing

Metrics for each requested pair (A, B) of views
  linear_cka           feature-space linear CKA (reference_similarity.linear_cka)
  linear_cka_debiased  unbiased-HSIC CKA on a subsample (reference_similarity.linear_cka_debiased)
  knn_jaccard          mean k-NN neighbourhood Jaccard (reference_similarity.knn_jaccard)
  procrustes           orthogonal Procrustes similarity ||Xc^T Yc||_* / (||Xc||_F ||Yc||_F)
  ridge_r2_ab / _ba    grouped-CV ridge cross-predictability (variance-weighted R^2)
  ari / ami / nmi      for label pairs (or label vs label)
Probes
  ridge R^2 / Spearman of each target from each view, grouped CV
  cross_view transfer: probe fit on view A, applied to view B (same row space; meaningful when
    A and B share an encoder, e.g. one encoder on two platforms)
  cross_domain transfer: probe fit on rows of domain a, scored on domain b, vs within-b CV
Contrasts
  paired bootstrap of metric(pair_1) - metric(pair_2) over the same blocks, e.g.
  CKA(s3a, s3b | shared encoder) - CKA(s3a, s3b | per-platform encoders)

Config (YAML, top-level key paired_views):
  out_dir: /path
  inputs: {name: /path/bundle.npz, ...}   # views are addressed as "<input>:<view>"
  align: intersect                        # rows present in every input used by a pair
  max_rows: 20000                         # deterministic subsample (by group) before metrics
  metrics: [linear_cka, linear_cka_debiased, knn_jaccard, procrustes, ridge_r2]
  pairs: [[a:view__x, b:view__y], ...]    # or "all" (every pair of views across inputs)
  label_pairs: [[a:labels__x, b:labels__y], ...]
  knn: {k: 10, metric: euclidean, standardize: true}
  debiased_max_rows: 3000
  ridge: {alphas: [0.1, 1, 10, 100, 1000], n_folds: 5}
  bootstrap: {n: 200, seed: 0}
  probes: {views: [a:view__x, ...], targets: [a:target__chl, ...]}
  transfer:
    cross_view: [{fit: a:view__x, apply: a:view__y, targets: [a:target__chl]}]
    cross_domain: [{view: a:view__x, fit_domain: s3a, apply_domain: s3b, targets: [...]}]
  contrasts: [{name: shared_vs_separate, metric: linear_cka, pair_1: [..], pair_2: [..]}]   # linear_cka | procrustes
  resume: {check_config: true}

Outputs (out_dir): pairwise.csv, label_agreement.csv, probes.csv, transfer.csv, contrasts.csv,
bootstrap_draws.npz (per-draw values, for later contrasts) and run_state markers
(one unit per output table; a rerun recomputes only missing or changed units).

  python -m reptaxonomy.intercompare.paired_views -y paired_views.yaml
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from reptaxonomy.intercompare.reference_similarity import knn_jaccard, linear_cka, linear_cka_debiased
from reptaxonomy.util import run_state as rs
from reptaxonomy.util.general_utils import read_yaml

logger = logging.getLogger(__name__)

VIEW, LABELS, TARGET = "view__", "labels__", "target__"


# ----------------------------------------------------------------------------- bundles
@dataclass
class Bundle:
    name: str
    keys: np.ndarray
    groups: np.ndarray
    domain: Optional[np.ndarray]
    arrays: Dict[str, np.ndarray]

    @classmethod
    def load(cls, name: str, path: str) -> "Bundle":
        with np.load(path, allow_pickle=False) as z:
            keys = z["keys"].astype(str)
            groups = z["groups"].astype(str) if "groups" in z else keys.copy()
            domain = z["domain"].astype(str) if "domain" in z else None
            arrays = {k: z[k] for k in z.files if k.startswith((VIEW, LABELS, TARGET))}
        if len(np.unique(keys)) != len(keys):
            raise ValueError(f"{path}: duplicate keys")
        return cls(name, keys, groups, domain, arrays)


def save_bundle(path: str, keys, groups, views: Optional[Dict[str, np.ndarray]] = None,
                labels: Optional[Dict[str, np.ndarray]] = None, targets: Optional[Dict[str, np.ndarray]] = None,
                domain=None) -> str:
    """Write a views bundle (atomic). Producers in other repositories can call this or write the same keys."""
    arrs: Dict[str, Any] = {"keys": np.asarray(keys).astype(str), "groups": np.asarray(groups).astype(str)}
    if domain is not None:
        arrs["domain"] = np.asarray(domain).astype(str)
    for pre, d in ((VIEW, views), (LABELS, labels), (TARGET, targets)):
        for k, v in (d or {}).items():
            arrs[k if k.startswith(pre) else pre + k] = np.asarray(v)
    rs.savez_compressed(path, **arrs)
    return path


class Registry:
    """Bundles by name; resolves "<input>:<array>" references to aligned arrays."""

    def __init__(self, inputs: Dict[str, str]):
        self.b = {n: Bundle.load(n, p) for n, p in inputs.items()}
        self.paths = dict(inputs)

    @staticmethod
    def split(ref: str) -> Tuple[str, str]:
        if ":" not in ref:
            raise ValueError(f"view reference must be '<input>:<array>', got {ref!r}")
        a, b = ref.split(":", 1)
        return a, b

    def all_views(self, prefix: str = VIEW) -> List[str]:
        return [f"{n}:{k}" for n, b in self.b.items() for k in b.arrays if k.startswith(prefix)]

    def aligned(self, refs: Sequence[str]) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Arrays for refs on their common keys (order of the first input); returns (arrays, keys, groups, domain)."""
        names = [self.split(r)[0] for r in refs]
        base = self.b[names[0]]
        common = pd.Index(base.keys)
        for n in names[1:]:
            common = common.intersection(pd.Index(self.b[n].keys), sort=False)
        if len(common) == 0:
            raise ValueError(f"no common keys for {refs}")
        out = []
        for r in refs:
            n, k = self.split(r)
            b = self.b[n]
            if k not in b.arrays:
                raise KeyError(f"{k} not in {self.paths[n]}")
            pos = pd.Index(b.keys).get_indexer(common)
            out.append(np.asarray(b.arrays[k])[pos])
        pos0 = pd.Index(base.keys).get_indexer(common)
        dom = base.domain[pos0] if base.domain is not None else None
        return out, np.asarray(common), base.groups[pos0], dom


# ----------------------------------------------------------------------------- helpers
def finite_rows(*arrs: np.ndarray) -> np.ndarray:
    ok = None
    for a in arrs:
        a = np.asarray(a)
        if a.dtype.kind in "iu":
            m = a >= 0 if a.ndim == 1 else np.ones(len(a), bool)
        else:
            m = np.isfinite(a.astype(np.float64)).all(axis=1) if a.ndim > 1 else np.isfinite(a.astype(np.float64))
        ok = m if ok is None else ok & m
    return ok


def subsample_by_group(groups: np.ndarray, max_rows: Optional[int], seed: int = 0) -> np.ndarray:
    """Row indices: whole groups in a seeded random order until max_rows (keeps the block structure)."""
    n = len(groups)
    if not max_rows or n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    order = rng.permutation(len(uniq))
    srt = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[srt], np.arange(len(uniq) + 1))
    take, tot = [], 0
    for g in order:
        ix = srt[bounds[g]:bounds[g + 1]]
        take.append(ix)
        tot += len(ix)
        if tot >= max_rows:
            break
    return np.sort(np.concatenate(take))[:max_rows] if take else np.arange(0)


def block_bootstrap(groups: np.ndarray, n_boot: int, seed: int = 0) -> Iterable[np.ndarray]:
    """Row indices of n_boot draws: groups resampled with replacement (rows of a group stay together)."""
    codes, uniq = pd.factorize(groups)
    srt = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[srt], np.arange(len(uniq) + 1))
    members = [srt[bounds[g]:bounds[g + 1]] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[g] for g in pick])


def group_folds(groups: np.ndarray, n_folds: int, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    codes, uniq = pd.factorize(groups)
    k = int(min(n_folds, len(uniq)))
    if k < 2:
        raise ValueError("grouped CV needs at least two groups")
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = np.empty(len(uniq), int)
    fold_of[perm] = np.arange(len(uniq)) % k
    f = fold_of[codes]
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64)
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)


def knn_graph(X: np.ndarray, k: int = 10, metric: str = "euclidean", standardize: bool = True) -> csr_array:
    """Symmetrized binary k-NN adjacency (self excluded)."""
    from sklearn.neighbors import NearestNeighbors
    X = _standardize(X) if standardize else np.asarray(X, np.float64)
    k = int(min(k, len(X) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    ind = nn.kneighbors(X, return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(len(X)), k)
    A = csr_array((np.ones(rows.size, np.int8), (rows, ind.ravel())), shape=(len(X), len(X)))
    A = ((A + A.T) > 0).astype(np.int8)
    return csr_array(A)


def procrustes_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    """max_R <Xc R, Yc> / (||Xc|| ||Yc||) over orthogonal R (pads the smaller width with zeros)."""
    Xc = np.asarray(X, np.float64) - np.mean(X, 0)
    Yc = np.asarray(Y, np.float64) - np.mean(Y, 0)
    den = np.linalg.norm(Xc) * np.linalg.norm(Yc)
    if den == 0:
        return float("nan")
    s = np.linalg.svd(Xc.T @ Yc, compute_uv=False)
    return float(s.sum() / den)


def _ridge_fit(X, Y, alpha):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    d = X.shape[1]
    if d <= len(X):
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ Yc)
    else:   # dual form for wide X
        W = Xc.T @ np.linalg.solve(Xc @ Xc.T + alpha * np.eye(len(X)), Yc)
    return W, mx, my


def ridge_cv_predict(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, alphas=(0.1, 1, 10, 100, 1000),
                     n_folds: int = 5, seed: int = 0, inner_folds: int = 3) -> np.ndarray:
    """Out-of-fold ridge predictions of Y from X (grouped outer CV; alpha by grouped inner CV)."""
    X = _standardize(X)
    Y = np.asarray(Y, np.float64)
    Y2 = Y[:, None] if Y.ndim == 1 else Y
    P = np.full_like(Y2, np.nan, dtype=np.float64)
    for tr, te in group_folds(groups, n_folds, seed):
        best, best_err = alphas[0], np.inf
        if len(alphas) > 1:
            try:
                inner = group_folds(groups[tr], inner_folds, seed + 1)
            except ValueError:
                inner = []
            for a in alphas:
                err = 0.0
                for itr, ite in inner:
                    W, mx, my = _ridge_fit(X[tr][itr], Y2[tr][itr], a)
                    err += float(np.sum(((X[tr][ite] - mx) @ W + my - Y2[tr][ite]) ** 2))
                if inner and err < best_err:
                    best, best_err = a, err
        W, mx, my = _ridge_fit(X[tr], Y2[tr], best)
        P[te] = (X[te] - mx) @ W + my
    return P[:, 0] if Y.ndim == 1 else P


def r2_weighted(Y: np.ndarray, P: np.ndarray) -> float:
    """1 - SSE/SST summed over output dims (variance-weighted R^2)."""
    Y = np.asarray(Y, np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    P = P[:, None] if P.ndim == 1 else P
    sst = np.sum((Y - Y.mean(0)) ** 2)
    return float(1 - np.sum((Y - P) ** 2) / sst) if sst > 0 else float("nan")


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).correlation) if ok.sum() > 2 else float("nan")


def label_agreement(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
    return {"ari": float(adjusted_rand_score(a, b)), "ami": float(adjusted_mutual_info_score(a, b)),
            "nmi": float(normalized_mutual_info_score(a, b))}


# ----------------------------------------------------------------------------- metrics
def pair_metrics(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, metrics: Sequence[str],
                 cfg: Dict[str, Any], seed: int = 0) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "linear_cka" in metrics:
        out["linear_cka"] = linear_cka(X, Y)
    if "procrustes" in metrics:
        out["procrustes"] = procrustes_similarity(X, Y)
    if "linear_cka_debiased" in metrics:
        sub = subsample_by_group(groups, int(cfg.get("debiased_max_rows", 3000)), seed)
        out["linear_cka_debiased"] = linear_cka_debiased(X[sub], Y[sub]) if len(sub) > 4 else float("nan")
    if "knn_jaccard" in metrics:
        kc = cfg.get("knn") or {}
        sub = subsample_by_group(groups, int(kc.get("max_rows", 10000)), seed)
        ga = knn_graph(X[sub], int(kc.get("k", 10)), kc.get("metric", "euclidean"), bool(kc.get("standardize", True)))
        gb = knn_graph(Y[sub], int(kc.get("k", 10)), kc.get("metric", "euclidean"), bool(kc.get("standardize", True)))
        out["knn_jaccard"] = knn_jaccard(ga, gb)
    if "ridge_r2" in metrics:
        rc = cfg.get("ridge") or {}
        al, nf = tuple(rc.get("alphas", (0.1, 1, 10, 100, 1000))), int(rc.get("n_folds", 5))
        out["ridge_r2_ab"] = r2_weighted(Y, ridge_cv_predict(X, Y, groups, al, nf, seed))
        out["ridge_r2_ba"] = r2_weighted(X, ridge_cv_predict(Y, X, groups, al, nf, seed))
    return out


BOOTSTRAPPABLE = ("linear_cka", "procrustes")    # cheap closed-form; others report point values


def _ci(draws: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size < 10:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a))


def _pairs(cfg: Dict[str, Any], reg: Registry) -> List[Tuple[str, str]]:
    p = cfg.get("pairs", "all")
    if p == "all":
        return list(itertools.combinations(reg.all_views(), 2))
    return [tuple(x) for x in p]


def run_pairwise(cfg, reg, seed, n_boot) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    metrics = list(cfg.get("metrics", ["linear_cka", "linear_cka_debiased", "knn_jaccard", "procrustes", "ridge_r2"]))
    rows, draws = [], {}
    for a, b in _pairs(cfg, reg):
        (X, Y), keys, groups, _ = reg.aligned([a, b])
        ok = finite_rows(X, Y)
        X, Y, groups = X[ok].astype(np.float64), Y[ok].astype(np.float64), groups[ok]
        sub = subsample_by_group(groups, cfg.get("max_rows", 20000), seed)
        X, Y, groups = X[sub], Y[sub], groups[sub]
        if len(X) < 10:
            rows.append({"view_a": a, "view_b": b, "n_rows": len(X), "error": "too few rows"})
            continue
        m = pair_metrics(X, Y, groups, metrics, cfg, seed)
        rec = {"view_a": a, "view_b": b, "n_rows": len(X), "n_groups": int(pd.Series(groups).nunique()),
               "dim_a": X.shape[1], "dim_b": Y.shape[1], **m}
        for met in [x for x in BOOTSTRAPPABLE if x in metrics]:
            fn = linear_cka if met == "linear_cka" else procrustes_similarity
            d = np.array([fn(X[ix], Y[ix]) for ix in block_bootstrap(groups, n_boot, seed)])
            draws[f"{met}|{a}|{b}"] = d
            rec[f"{met}_lo95"], rec[f"{met}_hi95"] = _ci(d)
        rows.append(rec)
        logger.info("pair %s vs %s: %s", a, b, {k: round(v, 3) for k, v in m.items()})
    return pd.DataFrame(rows), draws


def run_label_pairs(cfg, reg, seed, n_boot) -> pd.DataFrame:
    rows = []
    for a, b in [tuple(x) for x in (cfg.get("label_pairs") or [])]:
        (la, lb), _, groups, _ = reg.aligned([a, b])
        ok = (la >= 0) & (lb >= 0)
        la, lb, groups = la[ok], lb[ok], groups[ok]
        rec = {"labels_a": a, "labels_b": b, "n_rows": int(ok.sum()), **label_agreement(la, lb)}
        d = np.array([label_agreement(la[ix], lb[ix])["ari"] for ix in block_bootstrap(groups, n_boot, seed)])
        rec["ari_lo95"], rec["ari_hi95"] = _ci(d)
        rows.append(rec)
    return pd.DataFrame(rows)


def run_probes(cfg, reg, seed) -> pd.DataFrame:
    pc = cfg.get("probes") or {}
    rc = cfg.get("ridge") or {}
    al, nf = tuple(rc.get("alphas", (0.1, 1, 10, 100, 1000))), int(rc.get("n_folds", 5))
    rows = []
    for v in pc.get("views", []):
        for t in pc.get("targets", []):
            (X, y), _, groups, _ = reg.aligned([v, t])
            ok = finite_rows(X, y)
            X, y, groups = X[ok].astype(np.float64), y[ok].astype(np.float64), groups[ok]
            sub = subsample_by_group(groups, cfg.get("max_rows", 20000), seed)
            X, y, groups = X[sub], y[sub], groups[sub]
            if len(y) < 20:
                rows.append({"view": v, "target": t, "n_rows": len(y), "error": "too few rows"})
                continue
            p = ridge_cv_predict(X, y, groups, al, nf, seed)
            rows.append({"view": v, "target": t, "n_rows": len(y), "r2": r2_weighted(y, p), "spearman": spearman(y, p)})
    return pd.DataFrame(rows)


def run_transfer(cfg, reg, seed) -> pd.DataFrame:
    tc = cfg.get("transfer") or {}
    rc = cfg.get("ridge") or {}
    al, nf = tuple(rc.get("alphas", (0.1, 1, 10, 100, 1000))), int(rc.get("n_folds", 5))
    rows = []
    for spec in tc.get("cross_view", []) or []:
        for t in spec["targets"]:
            (Xa, Xb, y), _, groups, _ = reg.aligned([spec["fit"], spec["apply"], t])
            if Xa.shape[1] != Xb.shape[1]:
                rows.append({"kind": "cross_view", "fit": spec["fit"], "apply": spec["apply"], "target": t,
                             "error": "views have different widths (different encoders)"})
                continue
            ok = finite_rows(Xa, Xb, y)
            Xa, Xb, y, g = Xa[ok].astype(float), Xb[ok].astype(float), y[ok].astype(float), groups[ok]
            mu, sd = Xa.mean(0), np.where(Xa.std(0) > 0, Xa.std(0), 1)
            pa = np.full(len(y), np.nan)
            pb = np.full(len(y), np.nan)
            for tr, te in group_folds(g, nf, seed):
                W, mx, my = _ridge_fit((Xa[tr] - mu) / sd, y[tr, None], al[len(al) // 2])
                pa[te] = (((Xa[te] - mu) / sd - mx) @ W + my)[:, 0]
                pb[te] = (((Xb[te] - mu) / sd - mx) @ W + my)[:, 0]   # same standardization as the fit view
            rows.append({"kind": "cross_view", "fit": spec["fit"], "apply": spec["apply"], "target": t,
                         "n_rows": len(y), "r2_within": r2_weighted(y, pa), "r2_transfer": r2_weighted(y, pb),
                         "r2_gap": r2_weighted(y, pa) - r2_weighted(y, pb)})
    for spec in tc.get("cross_domain", []) or []:
        for t in spec["targets"]:
            (X, y), _, groups, dom = reg.aligned([spec["view"], t])
            if dom is None:
                raise ValueError("cross_domain transfer needs a 'domain' array in the bundle")
            ok = finite_rows(X, y)
            X, y, g, dom = X[ok].astype(float), y[ok].astype(float), groups[ok], dom[ok]
            a, b = dom == str(spec["fit_domain"]), dom == str(spec["apply_domain"])
            if a.sum() < 20 or b.sum() < 20:
                rows.append({"kind": "cross_domain", "view": spec["view"], "target": t, "error": "too few rows"})
                continue
            mu, sd = X[a].mean(0), np.where(X[a].std(0) > 0, X[a].std(0), 1)
            W, mx, my = _ridge_fit((X[a] - mu) / sd, y[a, None], al[len(al) // 2])
            pt = (((X[b] - mu) / sd - mx) @ W + my)[:, 0]
            pw = ridge_cv_predict(X[b], y[b], g[b], al, nf, seed)
            rows.append({"kind": "cross_domain", "view": spec["view"], "fit": spec["fit_domain"],
                         "apply": spec["apply_domain"], "target": t, "n_fit": int(a.sum()), "n_apply": int(b.sum()),
                         "r2_within": r2_weighted(y[b], pw), "r2_transfer": r2_weighted(y[b], pt),
                         "r2_gap": r2_weighted(y[b], pw) - r2_weighted(y[b], pt)})
    return pd.DataFrame(rows)


METRIC_FN: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "linear_cka": linear_cka, "procrustes": procrustes_similarity}


def run_contrasts(cfg, reg, seed, n_boot) -> pd.DataFrame:
    """metric(pair_1) - metric(pair_2), rows restricted to keys common to all four views, same draws."""
    rows = []
    for c in cfg.get("contrasts") or []:
        if c.get("metric", "linear_cka") not in METRIC_FN:
            raise ValueError(f"contrast {c.get('name')}: metric must be one of {sorted(METRIC_FN)} (bootstrapped)")
        fn = METRIC_FN[c.get("metric", "linear_cka")]
        refs = list(c["pair_1"]) + list(c["pair_2"])
        (A1, B1, A2, B2), _, groups, _ = reg.aligned(refs)
        ok = finite_rows(A1, B1, A2, B2)
        A1, B1, A2, B2 = (x[ok].astype(float) for x in (A1, B1, A2, B2))
        g = groups[ok]
        sub = subsample_by_group(g, cfg.get("max_rows", 20000), seed)
        A1, B1, A2, B2, g = A1[sub], B1[sub], A2[sub], B2[sub], g[sub]
        m1, m2 = fn(A1, B1), fn(A2, B2)
        d = np.array([fn(A1[ix], B1[ix]) - fn(A2[ix], B2[ix]) for ix in block_bootstrap(g, n_boot, seed)])
        lo, hi = _ci(d)
        rows.append({"name": c.get("name", "contrast"), "metric": c.get("metric", "linear_cka"),
                     "pair_1": "|".join(c["pair_1"]), "pair_2": "|".join(c["pair_2"]), "n_rows": len(g),
                     "value_1": m1, "value_2": m2, "diff": m1 - m2, "diff_lo95": lo, "diff_hi95": hi,
                     "p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))) if d.size else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- driver
UNITS = ("pairwise", "label_agreement", "probes", "transfer", "contrasts")


def run_paired_views(cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    pv = cfg.get("paired_views", cfg)
    out_dir = pv["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    seed = int((pv.get("bootstrap") or {}).get("seed", 0))
    n_boot = int((pv.get("bootstrap") or {}).get("n", 200))
    check = bool((pv.get("resume") or {}).get("check_config", True))
    reg = Registry(pv["inputs"])
    stamps = {n: rs.file_stamp(p) for n, p in pv["inputs"].items()}
    results = {}
    for unit in UNITS:
        if unit == "label_agreement" and not pv.get("label_pairs"):
            continue
        if unit == "probes" and not pv.get("probes"):
            continue
        if unit == "transfer" and not pv.get("transfer"):
            continue
        if unit == "contrasts" and not pv.get("contrasts"):
            continue
        fp = rs.fingerprint(unit, {k: v for k, v in pv.items() if k not in ("out_dir", "resume")}, stamps)
        path = os.path.join(out_dir, f"{unit}.csv")
        mk = rs.read_marker(out_dir, unit)
        if mk and os.path.exists(path) and (not check or mk.get("fingerprint") == fp):
            logger.info("%s: done (marker)", unit)
            results[unit] = pd.read_csv(path)
            continue
        try:
            if unit == "pairwise":
                df, draws = run_pairwise(pv, reg, seed, n_boot)
                if draws:
                    rs.savez_compressed(os.path.join(out_dir, "bootstrap_draws.npz"),
                                        **{k.replace(":", "~").replace("|", "__"): v for k, v in draws.items()})
            elif unit == "label_agreement":
                df = run_label_pairs(pv, reg, seed, n_boot)
            elif unit == "probes":
                df = run_probes(pv, reg, seed)
            elif unit == "transfer":
                df = run_transfer(pv, reg, seed)
            else:
                df = run_contrasts(pv, reg, seed, n_boot)
        except Exception as e:
            rs.mark_failed(out_dir, unit, error=repr(e))
            raise
        rs.to_csv(df, path, index=False)
        rs.mark_done(out_dir, unit, fp, rows=len(df))
        results[unit] = df
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-y", "--yaml", required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_paired_views(read_yaml(a.yaml))


if __name__ == "__main__":
    main()
