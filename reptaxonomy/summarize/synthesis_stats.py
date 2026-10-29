"""Pure statistical helpers for the cross-perspective synthesis.

Every function here operates on pandas / numpy objects only (no file IO), so the
pieces can be unit tested and reused from notebooks.

Implements the analyses introduced in the journal version of the benchmark:
  * robustness summaries (mean / median / IQR / min / projection sensitivity)
  * three-way variance decomposition (model x dataset x projection)
  * rank concordance (Kendall tau-b between projections, Kendall W)
  * normalized / averaged inter-model dissimilarity matrices, centrality,
    nearest neighbours, Mantel test, and average-linkage clustering
  * WeightWatcher alpha summaries
  * multi-axis ranking, cross-axis Kendall tau, and Pareto fronts
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Robustness / silhouette summaries
# ---------------------------------------------------------------------------

def silhouette_condition_table(sil_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse runs: one row per (dataset, projection, model) with run mean/std."""
    return (
        sil_df.groupby(["dataset", "projection", "model"], as_index=False)["silhouette"]
        .agg(silhouette="mean", silhouette_std="std", n_runs="count")
        .fillna({"silhouette_std": 0.0})
    )


def robustness_summary(cond_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model distribution statistics over all (dataset, projection) conditions.

    delta_proj is the mean over datasets of (max - min) silhouette across
    projections, i.e. the projection sensitivity of a model.
    """
    rows = []
    for model, g in cond_df.groupby("model"):
        s = g["silhouette"].to_numpy(dtype=float)
        q1, q3 = np.percentile(s, [25, 75])
        per_ds_range = g.groupby("dataset")["silhouette"].agg(lambda x: float(x.max() - x.min()))
        rows.append({
            "model": model,
            "n_conditions": int(s.size),
            "mean": float(s.mean()),
            "median": float(np.median(s)),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "min": float(s.min()),
            "max": float(s.max()),
            "delta_proj": float(per_ds_range.mean()),
        })
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def training_pair_deltas(cond_df: pd.DataFrame, pairs: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Per-condition silhouette difference (pretrained - scratch) for configured pairs."""
    rows = []
    piv = cond_df.pivot_table(index=["dataset", "projection"], columns="model", values="silhouette")
    for pair in pairs:
        if len(pair) != 2:
            continue
        a, b = str(pair[0]), str(pair[1])
        if a not in piv.columns or b not in piv.columns:
            continue
        diff = (piv[a] - piv[b]).dropna()
        for (ds, proj), val in diff.items():
            rows.append({"model_pretrained": a, "model_scratch": b, "dataset": ds,
                         "projection": proj, "delta_silhouette": float(val)})
        if len(diff):
            rows.append({"model_pretrained": a, "model_scratch": b, "dataset": "ALL",
                         "projection": "ALL", "delta_silhouette": float(diff.mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------

def three_way_variance_decomposition(cond_df: pd.DataFrame, value: str = "silhouette") -> pd.DataFrame:
    """Sum-of-squares decomposition for a fully crossed model x dataset x projection design.

    With one observation per cell (run-averaged), the three-way interaction is
    confounded with error and reported as the residual. Shares are fractions of
    the total sum of squares. If run-level replicates are passed instead (multiple
    rows per cell), they are averaged first so the design stays balanced.
    """
    factors = ["model", "dataset", "projection"]
    cell = cond_df.groupby(factors, as_index=False)[value].mean()
    cube = cell.pivot_table(index=["model", "dataset"], columns="projection", values=value)
    n_missing = int(cube.isna().sum().sum())
    if n_missing:
        raise ValueError(
            f"Variance decomposition requires a fully crossed design; {n_missing} "
            "(model, dataset, projection) cells are missing."
        )

    y = cell.set_index(factors)[value]
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())

    def main_effect(f: str) -> Tuple[pd.Series, float]:
        m = y.groupby(level=f).mean() - grand
        n_per = y.groupby(level=f).size()
        return m, float((n_per * m ** 2).sum())

    eff = {}
    ss = {}
    for f in factors:
        eff[f], ss[f] = main_effect(f)

    for f1, f2 in itertools.combinations(factors, 2):
        cm = y.groupby(level=[f1, f2]).mean()
        n_per = y.groupby(level=[f1, f2]).size()
        idx = cm.index
        inter = (
            cm.to_numpy()
            - eff[f1].reindex(idx.get_level_values(f1)).to_numpy()
            - eff[f2].reindex(idx.get_level_values(f2)).to_numpy()
            - grand
        )
        ss[f"{f1} x {f2}"] = float((n_per.to_numpy() * inter ** 2).sum())

    ss["residual"] = ss_total - sum(ss.values())

    levels = {f: y.index.get_level_values(f).nunique() for f in factors}
    dof = {f: levels[f] - 1 for f in factors}
    for f1, f2 in itertools.combinations(factors, 2):
        dof[f"{f1} x {f2}"] = (levels[f1] - 1) * (levels[f2] - 1)
    dof["residual"] = int(np.prod([levels[f] - 1 for f in factors]))

    rows = [{"source": k, "sum_sq": v, "dof": dof[k],
             "share": v / ss_total if ss_total > 0 else np.nan} for k, v in ss.items()]
    rows.append({"source": "total", "sum_sq": ss_total, "dof": len(y) - 1, "share": 1.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rank concordance
# ---------------------------------------------------------------------------

def kendall_w(rank_matrix: np.ndarray) -> float:
    """Kendall's coefficient of concordance (with tie correction).

    rank_matrix: shape (n_raters, n_items); each row is a ranking of the items.
    """
    r = np.asarray(rank_matrix, dtype=float)
    m, n = r.shape
    if m < 2 or n < 2:
        return np.nan
    rs = r.sum(axis=0)
    s = float(((rs - rs.mean()) ** 2).sum())
    ties = 0.0
    for row in r:
        _, counts = np.unique(row, return_counts=True)
        ties += float((counts ** 3 - counts).sum())
    denom = m ** 2 * (n ** 3 - n) - m * ties
    return 12.0 * s / denom if denom > 0 else np.nan


def condition_rankings(cond_df: pd.DataFrame, value: str = "silhouette") -> pd.DataFrame:
    """Rank models within every (dataset, projection) condition (1 = best)."""
    out = cond_df.copy()
    out["rank"] = out.groupby(["dataset", "projection"])[value].rank(ascending=False, method="average")
    return out


def rank_concordance(cond_df: pd.DataFrame, value: str = "silhouette") -> Dict[str, pd.DataFrame]:
    """Kendall tau-b between projection rankings (per dataset and pooled) + Kendall W."""
    tau_rows = []
    w_rows = []
    projections = sorted(cond_df["projection"].unique())
    for ds, g in cond_df.groupby("dataset"):
        piv = g.pivot_table(index="model", columns="projection", values=value).dropna()
        for p1, p2 in itertools.combinations([p for p in projections if p in piv.columns], 2):
            tau, pval = stats.kendalltau(piv[p1], piv[p2], variant="b")
            tau_rows.append({"dataset": ds, "proj_a": p1, "proj_b": p2,
                             "tau_b": float(tau), "p_value": float(pval), "n_models": int(len(piv))})
        ranks = piv.rank(ascending=False, axis=0).T.to_numpy()
        w_rows.append({"scope": ds, "kendall_w": kendall_w(ranks),
                       "n_raters": int(ranks.shape[0]), "n_models": int(ranks.shape[1])})

    # Pooled tau: average the per-dataset tau for each projection pair.
    tau_df = pd.DataFrame(tau_rows)
    if not tau_df.empty:
        pooled = tau_df.groupby(["proj_a", "proj_b"], as_index=False)["tau_b"].mean()
        pooled["dataset"] = "MEAN"
        tau_df = pd.concat([tau_df, pooled], ignore_index=True)

    piv_all = cond_df.pivot_table(index="model", columns=["dataset", "projection"], values=value).dropna()
    ranks_all = piv_all.rank(ascending=False, axis=0).T.to_numpy()
    w_rows.append({"scope": "ALL", "kendall_w": kendall_w(ranks_all),
                   "n_raters": int(ranks_all.shape[0]), "n_models": int(ranks_all.shape[1])})
    return {"tau": tau_df, "w": pd.DataFrame(w_rows)}


# ---------------------------------------------------------------------------
# Inter-model dissimilarity synthesis
# ---------------------------------------------------------------------------

def pairwise_long_to_cube(
    df: pd.DataFrame,
    value: str,
    models: Sequence[str],
    cond_cols: Sequence[str] = ("dataset", "projection", "run"),
) -> Tuple[np.ndarray, List[Tuple]]:
    """Long pairwise table -> array (n_conditions, M, M); missing pairs become NaN."""
    models = list(models)
    idx = {m: i for i, m in enumerate(models)}
    conds = []
    mats = []
    for key, g in df.groupby(list(cond_cols)):
        mat = np.full((len(models), len(models)), np.nan)
        for a, b, v in g[["model_a", "model_b", value]].itertuples(index=False):
            if a in idx and b in idx:
                mat[idx[a], idx[b]] = float(v)
        conds.append(key if isinstance(key, tuple) else (key,))
        mats.append(mat)
    return np.stack(mats, axis=0) if mats else np.empty((0, len(models), len(models))), conds


def normalize_residual_cube(cube: np.ndarray) -> np.ndarray:
    """Jointly min-max normalize residuals over all conditions and cells to [0, 1]."""
    lo, hi = np.nanmin(cube), np.nanmax(cube)
    if not np.isfinite(lo) or hi - lo <= 0:
        return np.zeros_like(cube)
    return (cube - lo) / (hi - lo)


def normalize_distance_cube(cube: np.ndarray) -> np.ndarray:
    """Divide each condition's distance matrix by its maximum off-diagonal value."""
    out = np.array(cube, dtype=float, copy=True)
    m = cube.shape[-1]
    off = ~np.eye(m, dtype=bool)
    for k in range(out.shape[0]):
        mx = np.nanmax(out[k][off]) if np.any(np.isfinite(out[k][off])) else np.nan
        if np.isfinite(mx) and mx > 0:
            out[k] = out[k] / mx
    return out


def symmetrize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def centrality(norm_residual: np.ndarray, norm_distance: np.ndarray, models: Sequence[str]) -> pd.DataFrame:
    """c_m = mean over j != m of the symmetrized normalized residual and distance.

    Lower c_m means a model is, on average, closer to all other models (more central).
    """
    m = len(models)
    off = ~np.eye(m, dtype=bool)
    r = symmetrize(norm_residual)
    d = symmetrize(norm_distance)
    rows = []
    for i, name in enumerate(models):
        mask = off[i]
        r_bar = float(np.nanmean(r[i, mask]))
        d_bar = float(np.nanmean(d[i, mask]))
        rows.append({"model": name, "mean_norm_residual": r_bar,
                     "mean_norm_distance": d_bar, "centrality": 0.5 * (r_bar + d_bar)})
    out = pd.DataFrame(rows)
    out["centrality_rank"] = out["centrality"].rank(method="average").astype(float)
    return out.sort_values("centrality").reset_index(drop=True)


def nearest_neighbors(norm_residual: np.ndarray, norm_distance: np.ndarray,
                      models: Sequence[str], k: int = 1) -> pd.DataFrame:
    m = len(models)
    r = symmetrize(norm_residual).copy()
    d = symmetrize(norm_distance).copy()
    np.fill_diagonal(r, np.inf)
    np.fill_diagonal(d, np.inf)
    rows = []
    for i, name in enumerate(models):
        nr = np.argsort(r[i])[:k]
        nd = np.argsort(d[i])[:k]
        rows.append({
            "model": name,
            "nn_residual": ";".join(models[j] for j in nr),
            "nn_residual_value": ";".join(f"{r[i, j]:.4f}" for j in nr),
            "nn_distance": ";".join(models[j] for j in nd),
            "nn_distance_value": ";".join(f"{d[i, j]:.4f}" for j in nd),
            "agree": set(models[j] for j in nr) == set(models[j] for j in nd),
        })
    return pd.DataFrame(rows)


def mantel_test(a: np.ndarray, b: np.ndarray, n_perm: int = 10000,
                method: str = "spearman", seed: int = 42) -> Dict[str, float]:
    """Permutation Mantel test on the strict upper triangles of two symmetric matrices."""
    a = symmetrize(np.asarray(a, dtype=float))
    b = symmetrize(np.asarray(b, dtype=float))
    n = a.shape[0]
    iu = np.triu_indices(n, k=1)
    corr = stats.spearmanr if method == "spearman" else stats.pearsonr

    def _r(x, y):
        return float(corr(x, y)[0])

    obs = _r(a[iu], b[iu])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(int(n_perm)):
        p = rng.permutation(n)
        if abs(_r(a[np.ix_(p, p)][iu], b[iu])) >= abs(obs):
            count += 1
    return {"statistic": obs, "p_value": (count + 1) / (n_perm + 1),
            "n_perm": int(n_perm), "method": method, "n_pairs": int(iu[0].size)}


def average_linkage(mat: np.ndarray) -> np.ndarray:
    d = symmetrize(np.asarray(mat, dtype=float)).copy()
    np.fill_diagonal(d, 0.0)
    d = np.clip(d, 0.0, None)
    return linkage(squareform(d, checks=False), method="average")


# ---------------------------------------------------------------------------
# Spectral (WeightWatcher) summaries
# ---------------------------------------------------------------------------

def alpha_summary(alpha_df: pd.DataFrame, lo: float = 2.0, hi: float = 5.0) -> pd.DataFrame:
    """Per-model alpha distribution summary and distance of the median from [lo, hi]."""
    rows = []
    for model, g in alpha_df.groupby("model"):
        a = g["alpha"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if a.size == 0:
            continue
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        rows.append({
            "model": model,
            "n_layers": int(a.size),
            "median_alpha": float(med),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "frac_in_range": float(np.mean((a >= lo) & (a <= hi))),
            "frac_under_fit": float(np.mean(a > hi)),
            "frac_over_fit": float(np.mean(a < lo)),
            "median_distance_from_range": float(max(lo - med, 0.0, med - hi)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-axis ranking and trade-offs
# ---------------------------------------------------------------------------

def rank_series(values: pd.Series, ascending: bool) -> pd.Series:
    """Rank 1 = best; ties receive the average rank."""
    return values.rank(ascending=ascending, method="average")


def kendall_tau_matrix(rank_df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    cols = list(cols)
    out = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for a, b in itertools.combinations(cols, 2):
        sub = rank_df[[a, b]].dropna()
        tau = stats.kendalltau(sub[a], sub[b], variant="b")[0] if len(sub) > 2 else np.nan
        out.loc[a, b] = out.loc[b, a] = tau
    return out


def pareto_front(x: np.ndarray, y: np.ndarray, minimize_x: bool = True, maximize_y: bool = True) -> np.ndarray:
    """Boolean mask of non-dominated points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xs = x if minimize_x else -x
    ys = -y if maximize_y else y
    n = len(x)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not (np.isfinite(xs[i]) and np.isfinite(ys[i])):
            mask[i] = False
            continue
        dominated = (xs <= xs[i]) & (ys <= ys[i]) & ((xs < xs[i]) | (ys < ys[i]))
        if np.any(dominated):
            mask[i] = False
    return mask


def spearman(x: Iterable[float], y: Iterable[float]) -> Dict[str, float]:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"rho": np.nan, "p_value": np.nan, "n": int(ok.sum())}
    rho, p = stats.spearmanr(x[ok], y[ok])
    return {"rho": float(rho), "p_value": float(p), "n": int(ok.sum())}
