"""Geolocated intrinsic-dimension (ID) analysis of Earth-observation representations.

Follows the GeoINRID protocol (Rao, Russwurm, Klemmer, Rolf, "Measuring the Intrinsic
Dimension of Earth Representations", ICLR 2026; https://github.com/arjunarao619/GeoINRID)
and extends it from location encoders to image encoders whose chips carry coordinates.

For every (scope, model) where a *scope* is one analysed condition or the pooled set of
all geolocated chips of a dataset family, the analysis computes

1. **Global ID** -- skdim estimators (MLE, TwoNN, MOM, TLE, FisherS, CorrInt, DANCo, ESS, ...)
   for several k on ``n_runs`` seeded subsamples (mean / std reported).
2. **Local ID (LID)** -- pointwise MLE / TwoNN / TLE / MOM (raw and kNN-smoothed) for each
   chip, either on mean-pooled chip embeddings (``unit: chip``) or on a pooled token cloud
   aggregated back to chips by the median (``unit: token``).
3. **Spatial statistics of LID**
   * Moran's I with row-standardised haversine kNN weights + permutation p-value,
   * an empirical LID semivariogram over great-circle distance bins,
   * region summaries (country / ecoregion / split tag) + Kruskal-Wallis H,
   * a sampling-density control: Spearman(LID, log distance to k-th geographic neighbour),
   * geographic-vs-embedding neighbourhood agreement (kNN Jaccard) and a Mantel test
     (Spearman between great-circle and embedding cosine distances, permutation p).
4. **Cross-model agreement** of chip LID maps (Spearman matrix) and, if location encoders
   are configured, Spearman(image LID, location-encoder LID) at the same sites.
5. lon/lat scatter maps of LID per (scope, model).

Inputs follow the gfm experiment layout::

    embeddings/<cond>/<model>/test/embd_<stem>.npy        (C x h x w or C)
    work/manifests/<cond>.json                            (stems, lats, lons, times, tags)

or any directory with the same structure plus an optional ``coords_csv`` (stem, lat, lon[, group]).

Outputs (``<out_dir>/geo_id/<dataset>/``)::

    global_id.csv  local_id_summary.csv  spatial_stats.csv  region_summary.csv
    lid_points.parquet|csv  lid_model_agreement.csv  location_encoder_agreement.csv
    variogram.csv  maps/<scope>__<model>__<method>.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EARTH_R_KM = 6371.0088
LOCAL_METHODS = ("MLE", "TwoNN", "TLE", "MOM")
SLOW_ESTIMATORS = ("ESS", "DANCo", "MiND_ML", "MiND_KL")
K_DEPENDENT = ("MLE", "DANCo", "TLE", "MOM", "MADA")   # others ignore k (reported once, at ks[0])
DEFAULTS: Dict[str, Any] = {
    "unit": "chip",                    # chip | token
    "max_tokens_per_chip": 16,
    "global_estimators": ["MLE", "TwoNN", "MOM", "TLE", "FisherS", "CorrInt", "DANCo", "ESS"],
    "ks_global": [10, 20, 100],
    "local_estimators": ["MLE", "TwoNN", "TLE"],
    "ks_local": [20, 50],
    "smooth": True,
    "n_runs": 3,
    "max_points": 5000,
    "slow_max_points": 1000,           # cap for ESS / DANCo / MiND
    "min_points": 50,
    "moran_k": 8,
    "n_perm": 499,
    "density_k": 8,
    "knn_jaccard_k": 10,
    "mantel_max": 1500,
    "mantel_perm": 99,
    "variogram_bins_km": [0, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20040],
    "group_keys": ["country", "ecoregion", "region", "split", "scope"],
    "pool": True,
    "per_condition": True,
    "make_maps": True,
    "map_method": "MLE",
    "seed": 0,
}


# ----------------------------------------------------------------------------- data
def _manifest(work_dir: str, cond: str) -> Dict[str, Any]:
    p = Path(work_dir, "manifests", f"{cond}.json")
    return json.loads(p.read_text()) if p.exists() else {}


def condition_coordinates(work_dir: str, cond: str, coords_csv: Optional[str] = None) -> pd.DataFrame:
    """stem -> lat, lon, time, group columns for one condition (NaNs dropped)."""
    man = _manifest(work_dir, cond)
    stems = list(man.get("stems", []))
    df = pd.DataFrame({"stem": stems})
    if "lats" in man and "lons" in man:
        df["lat"], df["lon"] = man["lats"], man["lons"]
        df["time"] = man.get("times", [None] * len(stems))
    if man.get("sample_keys"):
        df["sample_key"] = man["sample_keys"][: len(stems)]
    for k, v in (man.get("tags") or {}).items():
        if isinstance(v, (str, int, float)):
            df[k] = v
    if man.get("point_tags"):
        for k, vals in man["point_tags"].items():
            df[k] = vals
    if coords_csv:
        ext = pd.read_csv(coords_csv)
        ext["stem"] = ext["stem"].astype(str)
        df = df.drop(columns=[c for c in ("lat", "lon") if c in df], errors="ignore").merge(
            ext, on="stem", how="left", suffixes=("", "_csv"))
    if "lat" not in df:
        df["lat"] = df["lon"] = np.nan
    df["condition"] = cond
    df = df[np.isfinite(df["lat"].astype(float)) & np.isfinite(df["lon"].astype(float))]
    return df.reset_index(drop=True)


def _load_embedding(path: str, unit: str, max_tokens: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    try:
        a = np.load(path, mmap_mode="r")
    except Exception:
        return None
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return a[None] if unit == "token" else a
    tok = a.reshape(a.shape[0], -1).T  # (h*w, C)
    tok = tok[np.isfinite(tok).all(1)]
    if not len(tok):
        return None
    if unit == "chip":
        return tok.mean(0)
    if len(tok) > max_tokens:
        tok = tok[rng.choice(len(tok), max_tokens, replace=False)]
    return tok


def load_points(embed_dir: str, cond: str, model: str, coords: pd.DataFrame, unit: str = "chip",
                max_tokens: int = 16, split: str = "test", seed: int = 0) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """-> X (n x C) or token cloud, owner (index of chip per row), chip table (rows with embeddings)."""
    rng = np.random.default_rng(seed)
    xs, own, keep = [], [], []
    for i, stem in enumerate(coords["stem"]):
        e = _load_embedding(os.path.join(embed_dir, cond, model, split, f"embd_{stem}.npy"), unit, max_tokens, rng)
        if e is None:
            continue
        j = len(keep)
        keep.append(i)
        if unit == "token":
            xs.append(e); own.append(np.full(len(e), j))
        else:
            xs.append(e[None]); own.append(np.array([j]))
    if not keep:
        return np.zeros((0, 0)), np.zeros(0, int), coords.iloc[[]]
    return np.concatenate(xs).astype(np.float64), np.concatenate(own), coords.iloc[keep].reset_index(drop=True)


def clean_for_id(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Remove exact duplicates' zero distances (skdim returns inf/NaN otherwise) with a tiny jitter."""
    X = np.asarray(X, dtype=np.float64)
    sd = float(np.nanstd(X)) or 1.0
    _, inv, cnt = np.unique(np.round(X / sd, 8), axis=0, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        rng = np.random.default_rng(seed)
        dup = cnt[inv.ravel()] > 1
        X = X.copy()
        X[dup] += rng.normal(scale=1e-6 * sd, size=(int(dup.sum()), X.shape[1]))
    return X


# ----------------------------------------------------------------------------- ID
def _estimators():
    from reptaxonomy.robustness.embedding_robustness import get_id_estimators
    est = get_id_estimators()
    import skdim
    est["FisherS"] = lambda k: skdim.id.FisherS(conditional_number=10)
    return est


def global_id_runs(X: np.ndarray, methods: Sequence[str], ks: Sequence[int], n_runs: int, max_points: int,
                   seed: int = 0, slow_max: int = 1000) -> pd.DataFrame:
    """Global ID per (method, k, run). ESS / DANCo / MiND are O(n^2)-O(n^3) in skdim and are
    evaluated on the first ``slow_max`` points of each run's subsample."""
    est = _estimators()
    rows = []
    for r in range(int(n_runs)):
        rng = np.random.default_rng(seed + r)
        idx = rng.choice(len(X), min(len(X), max_points), replace=False) if len(X) > max_points else (
            rng.permutation(len(X)) if r else np.arange(len(X)))
        Xr_full = clean_for_id(X[idx], seed + r)
        for m in methods:
            Xr = Xr_full[: int(slow_max)] if m in SLOW_ESTIMATORS else Xr_full
            k_list = ks if m in K_DEPENDENT else [ks[0]]
            for k in k_list:
                if k >= len(Xr):
                    continue
                try:
                    e = est[m](k)
                    if m == "MLE":
                        e.fit(Xr, n_neighbors=k)
                        v = float(e.dimension_)
                    elif m in ("TLE", "MOM", "MADA") and hasattr(e, "fit_pw"):
                        e.fit_pw(Xr, n_neighbors=k)
                        v = float(np.nanmean(e.dimension_pw_))
                    else:
                        e.fit(Xr)
                        v = float(e.dimension_) if hasattr(e, "dimension_") else float(np.nanmean(e.dimension_pw_))
                    err = ""
                    if m == "DANCo" and np.isfinite(v) and v >= Xr.shape[1] - 1e-6:
                        v, err = np.nan, "DANCo returned the ambient dimension (failed fit)"
                except Exception as exc:
                    v, err = np.nan, str(exc)[:200]
                rows.append({"method": m, "k": k, "run": r, "n": len(Xr), "global_id": v, "error": err})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.loc[~np.isfinite(df["global_id"].astype(float)), "global_id"] = np.nan
    return df


def local_id(X: np.ndarray, methods: Sequence[str], ks: Sequence[int], smooth: bool = True,
             n_jobs: int = 1, seed: int = 0) -> Dict[str, np.ndarray]:
    """{'<method>_k<k>[_smooth]': (n,) pointwise ID}."""
    import skdim
    cls = {"MLE": skdim.id.MLE, "TwoNN": skdim.id.TwoNN, "TLE": skdim.id.TLE, "MOM": skdim.id.MOM}
    Xc = clean_for_id(X, seed)
    out: Dict[str, np.ndarray] = {}
    for m in methods:
        if m not in cls:
            logger.warning("local ID: %s has no pointwise estimator; skipped", m)
            continue
        for k in ks:
            if k >= len(Xc):
                continue
            try:
                res = cls[m]().fit_transform_pw(Xc, n_neighbors=int(k), smooth=bool(smooth), n_jobs=n_jobs)
            except Exception as exc:
                logger.warning("local %s k=%d failed: %s", m, k, exc)
                continue
            raw, sm = (res if isinstance(res, tuple) else (res, None))
            out[f"{m}_k{k}"] = _finite(raw)
            if sm is not None:
                out[f"{m}_k{k}_smooth"] = _finite(sm)
    return out


def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).ravel()
    a[~np.isfinite(a)] = np.nan
    return a


def aggregate_tokens(values: np.ndarray, owner: np.ndarray, n_chips: int) -> np.ndarray:
    s = pd.Series(values).groupby(owner).median()
    out = np.full(n_chips, np.nan)
    out[s.index.values.astype(int)] = s.values
    return out


# ----------------------------------------------------------------------------- spatial statistics
def _rad(lonlat: np.ndarray) -> np.ndarray:
    return np.deg2rad(np.asarray(lonlat, dtype=np.float64)[:, [1, 0]])  # (lat, lon) radians


def haversine_knn(lonlat: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import BallTree
    k = int(min(k, len(lonlat) - 1))
    d, i = BallTree(_rad(lonlat), metric="haversine").query(_rad(lonlat), k=k + 1)
    return i[:, 1:], d[:, 1:] * EARTH_R_KM


def great_circle_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = _rad(a), _rad(b)
    dlat, dlon = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[:, 0]) * np.cos(b[:, 0]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def morans_i(values: np.ndarray, nbr: np.ndarray, n_perm: int = 499, seed: int = 0) -> Dict[str, float]:
    """Global Moran's I with row-standardised binary kNN weights; two-sided permutation p."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return {"moran_I": np.nan, "moran_z": np.nan, "moran_p": np.nan, "moran_EI": np.nan}
    v = np.where(ok, v, np.nanmean(v))
    n = len(v)

    def _I(x):
        z = x - x.mean()
        return float((z * z[nbr].mean(1)).sum() / (z ** 2).sum())

    I = _I(v)
    rng = np.random.default_rng(seed)
    perm = np.array([_I(rng.permutation(v)) for _ in range(int(n_perm))]) if n_perm else np.array([])
    EI = -1.0 / (n - 1)
    if len(perm):
        z = (I - perm.mean()) / (perm.std() + 1e-12)
        p = (1 + (np.abs(perm - perm.mean()) >= abs(I - perm.mean())).sum()) / (len(perm) + 1)
    else:
        z = p = np.nan
    return {"moran_I": I, "moran_z": float(z), "moran_p": float(p), "moran_EI": EI}


def variogram(values: np.ndarray, lonlat: np.ndarray, bins_km: Sequence[float], max_pairs: int = 200000,
              seed: int = 0) -> pd.DataFrame:
    v = np.asarray(values, float)
    ok = np.where(np.isfinite(v))[0]
    if len(ok) < 10:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    m = min(max_pairs, len(ok) * (len(ok) - 1) // 2)
    a, b = rng.choice(ok, m), rng.choice(ok, m)
    keep = a != b
    a, b = a[keep], b[keep]
    d = great_circle_km(lonlat[a], lonlat[b])
    g = 0.5 * (v[a] - v[b]) ** 2
    cut = pd.cut(d, bins=list(bins_km), include_lowest=True)
    df = pd.DataFrame({"bin": cut, "d": d, "gamma": g}).groupby("bin", observed=True).agg(
        dist_km=("d", "mean"), semivariance=("gamma", "mean"), n_pairs=("gamma", "size")).reset_index()
    df["bin"] = df["bin"].astype(str)
    df["sill_ratio"] = df["semivariance"] / (np.nanvar(v[ok]) + 1e-12)
    return df


def density_control(values: np.ndarray, lonlat: np.ndarray, k: int) -> Dict[str, float]:
    """Spearman(LID, log distance to k-th geographic neighbour): LID that merely tracks sampling
    density is an artefact of the benchmark's spatial design rather than of the representation."""
    from scipy.stats import spearmanr
    _, d = haversine_knn(lonlat, k)
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return {"density_rho": np.nan, "density_p": np.nan}
    r = spearmanr(v[ok], np.log1p(d[ok, -1]))
    return {"density_rho": float(r.statistic), "density_p": float(r.pvalue)}


def knn_jaccard(lonlat: np.ndarray, X: np.ndarray, k: int) -> float:
    from sklearn.neighbors import NearestNeighbors
    k = int(min(k, len(X) - 1))
    gi, _ = haversine_knn(lonlat, k)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    ei = NearestNeighbors(n_neighbors=k + 1).fit(Xn).kneighbors(Xn, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(gi, ei)]))


def mantel(lonlat: np.ndarray, X: np.ndarray, max_n: int = 1500, n_perm: int = 199, seed: int = 0) -> Dict[str, float]:
    from scipy.stats import rankdata
    from sklearn.metrics.pairwise import cosine_distances, haversine_distances
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), max_n, replace=False) if len(X) > max_n else np.arange(len(X))
    G = haversine_distances(_rad(lonlat[idx])) * EARTH_R_KM
    E = cosine_distances(X[idx])
    iu = np.triu_indices(len(idx), 1)
    g, e = rankdata(G[iu]), rankdata(E[iu])
    g = (g - g.mean()) / g.std(); e = (e - e.mean()) / e.std()
    r = float((g * e).mean())
    if not n_perm:
        return {"mantel_r": r, "mantel_p": np.nan}
    # permuting sites permutes the (standardised) ranks, so rank once and index per permutation
    Rg = np.zeros_like(G)
    Rg[iu] = g
    Rg = Rg + Rg.T
    cnt = 0
    for _ in range(int(n_perm)):
        p = rng.permutation(len(idx))
        gp = Rg[np.ix_(p, p)][iu]
        cnt += float((gp * e).mean()) >= r
    return {"mantel_r": r, "mantel_p": (1 + cnt) / (1 + n_perm)}


def region_summary(values: np.ndarray, groups: pd.Series) -> Tuple[pd.DataFrame, Dict[str, float]]:
    from scipy.stats import kruskal
    df = pd.DataFrame({"lid": values, "group": groups.astype(str).values}).dropna()
    if df.empty:
        return pd.DataFrame(), {"kw_H": np.nan, "kw_p": np.nan}
    s = df.groupby("group")["lid"].agg(n="size", mean="mean", median="median", std="std",
                                       q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75)).reset_index()
    samples = [g["lid"].values for _, g in df.groupby("group") if len(g) >= 3]
    if len(samples) >= 2:
        r = kruskal(*samples)
        return s, {"kw_H": float(r.statistic), "kw_p": float(r.pvalue)}
    return s, {"kw_H": np.nan, "kw_p": np.nan}


# ----------------------------------------------------------------------------- plotting
def plot_lid_map(lonlat: np.ndarray, values: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    try:
        import geodatasets  # noqa: F401
        import geopandas as gpd
        gpd.read_file(geodatasets.get_path("naturalearth.land")).boundary.plot(ax=ax, lw=.3, color="0.5")
    except Exception:
        pass
    ok = np.isfinite(values)
    lo, hi = np.nanpercentile(values[ok], [2, 98]) if ok.any() else (0, 1)
    sc = ax.scatter(lonlat[ok, 0], lonlat[ok, 1], c=values[ok], s=6, cmap="viridis", vmin=lo, vmax=hi, lw=0)
    pad = 3
    ax.set_xlim(max(-180, lonlat[:, 0].min() - pad), min(180, lonlat[:, 0].max() + pad))
    ax.set_ylim(max(-90, lonlat[:, 1].min() - pad), min(90, lonlat[:, 1].max() + pad))
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_title(title, fontsize=9)
    fig.colorbar(sc, ax=ax, label="local ID", shrink=.8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- one scope
def analyse_points(X: np.ndarray, owner: np.ndarray, chips: pd.DataFrame, scope: str, model: str,
                   p: Mapping[str, Any], maps_dir: Optional[Path] = None, Xchip: Optional[np.ndarray] = None
                   ) -> Dict[str, Any]:
    """Global + local ID + spatial statistics for one point set. Returns frames + per-chip LID."""
    seed = int(p["seed"])
    n_all = len(chips)
    g = global_id_runs(X, p["global_estimators"], p["ks_global"], p["n_runs"], p["max_points"], seed,
                       int(p.get("slow_max_points", 1000)))
    if not g.empty:
        g.insert(0, "model", model); g.insert(0, "scope", scope)
    # local ID + spatial statistics on a fixed, seeded chip subsample when the scope is large
    # (tokens follow their chip); chips outside the subsample get NaN LID in the points table.
    full_chips = chips
    n_tok_per_chip = max(1.0, len(X) / max(n_all, 1))
    cap = max(1, int(p["max_points"] / n_tok_per_chip))
    if n_all > cap:
        keep = np.sort(np.random.default_rng(seed).choice(n_all, cap, replace=False))
        remap = -np.ones(n_all, int); remap[keep] = np.arange(len(keep))
        sel = np.isin(owner, keep)
        X, owner = X[sel], remap[owner[sel]]
        if Xchip is not None:
            Xchip = Xchip[keep]
        chips = chips.iloc[keep].reset_index(drop=True)
    else:
        keep = np.arange(n_all)
    lonlat = chips[["lon", "lat"]].to_numpy(float)
    n = len(chips)
    lids_raw = local_id(X, p["local_estimators"], p["ks_local"], p["smooth"], int(p.get("n_jobs", 1)), seed)
    lids = {name: aggregate_tokens(v, owner, n) for name, v in lids_raw.items()}
    Xc = Xchip if Xchip is not None else (X if len(X) == n else
                                          pd.DataFrame(X).groupby(owner).mean().reindex(range(n)).to_numpy())
    nbr, _ = haversine_knn(lonlat, p["moran_k"]) if n > p["moran_k"] + 1 else (None, None)
    base = {"scope": scope, "model": model, "n_chips_total": n_all, "n_chips": n, "n_points": len(X)}
    try:
        base.update({"geo_embed_knn_jaccard": knn_jaccard(lonlat, Xc, p["knn_jaccard_k"]),
                     **mantel(lonlat, Xc, p["mantel_max"], int(p.get("mantel_perm", 99)), seed)})
    except Exception as exc:
        logger.warning("mantel/jaccard failed %s/%s: %s", scope, model, exc)
    spat, local_rows, regions, vgs = [], [], [], []
    for name, v in lids.items():
        ok = np.isfinite(v)
        local_rows.append({**base, "estimator": name, "lid_mean": np.nanmean(v), "lid_median": np.nanmedian(v),
                           "lid_std": np.nanstd(v), "lid_q05": np.nanpercentile(v, 5) if ok.any() else np.nan,
                           "lid_q95": np.nanpercentile(v, 95) if ok.any() else np.nan, "frac_finite": ok.mean()})
        row = {**base, "estimator": name}
        if nbr is not None:
            row.update(morans_i(v, nbr, int(p["n_perm"]), seed))
            row.update(density_control(v, lonlat, p["density_k"]))
        for gk in p["group_keys"]:
            if gk in chips and chips[gk].nunique() > 1:
                rs, kw = region_summary(v, chips[gk])
                if not rs.empty:
                    rs.insert(0, "group_key", gk); rs.insert(0, "estimator", name)
                    rs.insert(0, "model", model); rs.insert(0, "scope", scope)
                    regions.append(rs)
                row[f"kw_H_{gk}"], row[f"kw_p_{gk}"] = kw["kw_H"], kw["kw_p"]
        spat.append(row)
        vg = variogram(v, lonlat, p["variogram_bins_km"], seed=seed)
        if not vg.empty:
            vg.insert(0, "estimator", name); vg.insert(0, "model", model); vg.insert(0, "scope", scope)
            vgs.append(vg)
    if maps_dir is not None and p.get("make_maps", True):
        key = next((k for k in lids if k.startswith(p["map_method"]) and k.endswith("smooth")), None) or next(iter(lids), None)
        if key:
            try:
                plot_lid_map(lonlat, lids[key], maps_dir / f"{_safe(scope)}__{_safe(model)}__{key}.png",
                             f"{model} | {scope} | {key}")
            except Exception as exc:
                logger.warning("map failed: %s", exc)
    pts = full_chips[["stem", "condition", "lat", "lon"] + [c for c in p["group_keys"] if c in full_chips and c != "scope"]].copy()
    pts.insert(0, "model", model); pts.insert(0, "scope", scope)
    for name, v in lids.items():
        col = np.full(n_all, np.nan); col[keep] = v
        pts[name] = col
    pts = pts.iloc[keep].reset_index(drop=True)   # only chips with LID
    return {"global": g, "local": pd.DataFrame(local_rows), "spatial": pd.DataFrame(spat),
            "regions": pd.concat(regions) if regions else pd.DataFrame(),
            "variogram": pd.concat(vgs) if vgs else pd.DataFrame(), "points": pts}


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def _cat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    fr = [f for f in frames if f is not None and not f.empty]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


# ----------------------------------------------------------------------------- agreement
def model_agreement(points: pd.DataFrame, estimator: str) -> pd.DataFrame:
    """Spearman correlation of chip LID maps between every pair of models within each scope."""
    from scipy.stats import spearmanr
    rows = []
    if points.empty or estimator not in points:
        return pd.DataFrame()
    for scope, g in points.groupby("scope"):
        wide = g.pivot_table(index="stem", columns="model", values=estimator)
        cols = list(wide.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                d = wide[[a, b]].dropna()
                if len(d) >= 10:
                    r = spearmanr(d[a], d[b])
                    rows.append({"scope": scope, "estimator": estimator, "model_a": a, "model_b": b,
                                 "n": len(d), "spearman": float(r.statistic), "p": float(r.pvalue)})
    return pd.DataFrame(rows)


def location_encoder_agreement(points: pd.DataFrame, loc_models: Sequence[str], estimator: str) -> pd.DataFrame:
    from scipy.stats import spearmanr
    rows = []
    if points.empty or estimator not in points:
        return pd.DataFrame()
    for scope, g in points.groupby("scope"):
        wide = g.pivot_table(index="stem", columns="model", values=estimator)
        for L in [m for m in loc_models if m in wide]:
            for M in [m for m in wide.columns if m not in loc_models]:
                d = wide[[L, M]].dropna()
                if len(d) >= 10:
                    r = spearmanr(d[L], d[M])
                    rows.append({"scope": scope, "estimator": estimator, "location_encoder": L, "image_encoder": M,
                                 "n": len(d), "spearman": float(r.statistic), "p": float(r.pvalue)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- drivers
def location_encoder_embeddings(lonlat: np.ndarray, cfg: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    from reptaxonomy.robustness.location_encoders import available_location_encoders, load_location_encoder
    out = {}
    for name, kw in available_location_encoders(dict(cfg)).items():
        try:
            enc = load_location_encoder(name, kw.get("geoinrid_root"), kw.get("device"), kw.get("checkpoint"),
                                        kw.get("satclip_repo"))
            out[f"loc__{name}"] = enc(lonlat)
            logger.info("location encoder %s -> %s", name, out[f"loc__{name}"].shape)
        except Exception as exc:
            logger.warning("location encoder %s unavailable: %s", name, exc)
    return out


def run_geo_id(embed_dir: str, work_dir: str, out_dir: str, dataset: str, conditions: Sequence[str],
               models: Sequence[str], params: Optional[Mapping[str, Any]] = None, split: str = "test",
               coords_csv: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Run the geolocated ID analysis for one dataset family; writes CSVs and maps, returns frames."""
    p = {**DEFAULTS, **dict(params or {})}
    out = Path(out_dir, "geo_id", dataset)
    maps = out / "maps"
    out.mkdir(parents=True, exist_ok=True)
    coords = {c: condition_coordinates(work_dir, c, coords_csv) for c in conditions}
    coords = {c: df for c, df in coords.items() if len(df) >= p["min_points"]}
    if not coords:
        logger.warning("[geo_id:%s] no condition has >= %d geolocated chips; nothing to do", dataset, p["min_points"])
        return {}
    res: List[Dict[str, Any]] = []
    loc_names: List[str] = []

    def _scopes():
        if p["per_condition"]:
            for c, df in coords.items():
                yield c, {c: df}
        if p["pool"] and len(coords) > 1:
            yield f"{dataset}__pooled", coords

    for scope, cdict in _scopes():
        chip_tab = _cat(df.assign(scope=c) for c, df in cdict.items())
        for m in models:
            Xs, owns, tabs, off = [], [], [], 0
            for c, df in cdict.items():
                X, own, tab = load_points(embed_dir, c, m, df, p["unit"], p["max_tokens_per_chip"], split, p["seed"])
                if not len(tab):
                    continue
                Xs.append(X); owns.append(own + off); tabs.append(tab.assign(scope=c)); off += len(tab)
            if not tabs or off < p["min_points"]:
                continue
            tab = pd.concat(tabs, ignore_index=True)
            logger.info("[geo_id:%s] %s / %s: %d chips", dataset, scope, m, len(tab))
            res.append(analyse_points(np.concatenate(Xs), np.concatenate(owns), tab, scope, m, p, maps))
        le_cfg = p.get("location_encoders")
        if le_cfg:
            # location-only reference at the same sites (one row per unique stem)
            tab = chip_tab.drop_duplicates("stem").reset_index(drop=True)
            if len(tab) >= p["min_points"]:
                for L, E in location_encoder_embeddings(tab[["lon", "lat"]].to_numpy(float), {"location_encoders": le_cfg}).items():
                    loc_names.append(L)
                    res.append(analyse_points(E, np.arange(len(tab)), tab, scope, L, p, maps))
    frames = {k: _cat(r[k] for r in res) for k in ("global", "local", "spatial", "regions", "variogram", "points")}
    if not frames["global"].empty:
        gs = frames["global"].groupby(["scope", "model", "method", "k"], dropna=False)["global_id"].agg(
            ["mean", "std", "count"]).reset_index().rename(columns={"mean": "global_id_mean", "std": "global_id_std",
                                                                   "count": "n_runs"})
        frames["global_summary"] = gs
    est = next((c for c in frames["points"].columns if c.startswith(p["map_method"]) and c.endswith("smooth")), None) \
        if not frames["points"].empty else None
    if est:
        frames["model_agreement"] = model_agreement(frames["points"][~frames["points"]["model"].isin(loc_names)], est)
        if loc_names:
            frames["location_encoder_agreement"] = location_encoder_agreement(frames["points"], sorted(set(loc_names)), est)
    names = {"global": "global_id_runs", "global_summary": "global_id", "local": "local_id_summary",
             "spatial": "spatial_stats", "regions": "region_summary", "variogram": "variogram",
             "model_agreement": "lid_model_agreement", "location_encoder_agreement": "location_encoder_agreement"}
    for k, fn in names.items():
        if k in frames and not frames[k].empty:
            frames[k].to_csv(out / f"{fn}.csv", index=False)
    if not frames["points"].empty:
        try:
            frames["points"].to_parquet(out / "lid_points.parquet", index=False)
        except Exception:
            frames["points"].to_csv(out / "lid_points.csv", index=False)
    (out / "params.json").write_text(json.dumps({k: v for k, v in p.items()}, indent=1, default=str))
    logger.info("[geo_id:%s] wrote %s", dataset, out)
    return frames


def run_location_only(sites: pd.DataFrame, out_dir: str, name: str, params: Optional[Mapping[str, Any]] = None,
                      scope_col: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Location-encoder ID at arbitrary site tables (columns lat, lon, stem, optional group/scope columns).

    Used where imagery is unavailable but footprints are (e.g. all FTW chips, EarthShift ID/OOD sets).
    ``scope_col`` splits the table into scopes (e.g. 'scope' with values 'all', 'train', 'CountryA'...)."""
    p = {**DEFAULTS, **dict(params or {})}
    out = Path(out_dir, "geo_id", name)
    maps = out / "maps"
    out.mkdir(parents=True, exist_ok=True)
    sites = sites.copy()
    sites["condition"] = sites.get("condition", name)
    res, loc_names = [], []
    uniq = sites.drop_duplicates("stem").reset_index(drop=True)
    embs = location_encoder_embeddings(uniq[["lon", "lat"]].to_numpy(float), {"location_encoders": p["location_encoders"]})
    row_of = {s: i for i, s in enumerate(uniq["stem"])}
    scopes = sites.groupby(scope_col) if scope_col else [("all", sites)]
    for scope, tab in scopes:
        tab = tab.drop_duplicates("stem").reset_index(drop=True)
        if len(tab) < p["min_points"]:
            continue
        ix = tab["stem"].map(row_of).to_numpy()
        for L, E in embs.items():
            loc_names.append(L)
            logger.info("[geo_id:%s] %s / %s: %d sites", name, scope, L, len(tab))
            res.append(analyse_points(E[ix], np.arange(len(tab)), tab, str(scope), L, p, maps))
    frames = {k: _cat(r[k] for r in res) for k in ("global", "local", "spatial", "regions", "variogram", "points")}
    if not frames["global"].empty:
        frames["global_summary"] = frames["global"].groupby(["scope", "model", "method", "k"])["global_id"].agg(
            global_id_mean="mean", global_id_std="std", n_runs="count").reset_index()
    est = next((c for c in frames["points"].columns if c.startswith(p["map_method"]) and c.endswith("smooth")), None) \
        if not frames["points"].empty else None
    if est:
        frames["model_agreement"] = model_agreement(frames["points"], est)
    for k, fn in {"global": "global_id_runs", "global_summary": "global_id", "local": "local_id_summary",
                  "spatial": "spatial_stats", "regions": "region_summary", "variogram": "variogram",
                  "model_agreement": "lid_model_agreement"}.items():
        if k in frames and not frames[k].empty:
            frames[k].to_csv(out / f"{fn}.csv", index=False)
    if not frames["points"].empty:
        frames["points"].to_parquet(out / "lid_points.parquet", index=False)
    (out / "params.json").write_text(json.dumps(p, indent=1, default=str))
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Geolocated intrinsic-dimension analysis (GeoINRID-style)")
    ap.add_argument("--embed-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--coords-csv")
    ap.add_argument("--params", help="YAML/JSON file with a geo_id parameter block")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    params = {}
    if a.params:
        import yaml
        params = yaml.safe_load(Path(a.params).read_text()) or {}
        params = params.get("geo_id", params)
    run_geo_id(a.embed_dir, a.work_dir, a.out_dir, a.dataset, a.conditions, a.models, params, coords_csv=a.coords_csv)


if __name__ == "__main__":
    main()
