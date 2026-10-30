"""Paper tables and figures from ``geo_intrinsic_dimension`` outputs.

Reads one or more ``outputs/geo_id/<name>/`` directories (location-only runs from
``pipelines.geo_id_locations`` and/or image-encoder runs from the ``geo_id`` stage) and writes

    <out>/geo_id_global_table.csv|.tex      global ID per (scope, model), selected estimators / k
    <out>/geo_id_spatial_table.csv|.tex     LID mean, Moran's I (p), density rho, Mantel r, KW H per (scope, model)
    <out>/geo_id_shift_sets.csv             EarthShift ID vs OOD site sets (location encoders)
    <out>/fig_geo_id_global.{png,pdf}       global ID per scope / model (estimator median, range)
    <out>/fig_geo_id_shift.{png,pdf}        ID vs OOD site-set global ID and LID per task
    <out>/fig_geo_id_lid_maps.{png,pdf}     lon/lat LID maps of one scope for every model
    <out>/fig_geo_id_variogram.{png,pdf}    LID sill ratio vs great-circle distance
    <out>/fig_geo_id_agreement.{png,pdf}    LID Spearman: image x location encoders (if present)

Usage::

    python -m reptaxonomy.summarize.geo_id_report --runs outputs/geo_id/ftw_locations outputs/geo_id/ftw \
        --out paper/geo_id --map-scope all --estimator MLE_k20_smooth
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRETTY = {
    "loc__geoclip": "GeoCLIP", "loc__csp_fmow": "CSP-fMoW", "loc__csp_inat": "CSP-iNat",
    "loc__satclip_l10": "SatCLIP-L10", "loc__satclip_l40": "SatCLIP-L40",
    "resnet18__sentinel2_all_moco": "RN18-MoCo", "resnet50__sentinel2_all_dino": "RN50-DINO",
    "resnet50__sentinel2_all_moco": "RN50-MoCo", "swin_v2_t__sentinel2_si_ms_satlas": "SwinV2T-Satlas",
    "dofa_base__dofa_mae": "DOFA-B", "croma_base__croma_vit": "CROMA-B",
    "prithvi_eo1__eo_1_0_100m": "Prithvi-EO1", "olmoearth_v1_nano": "OlmoEarth-nano",
    "unet__sentinel2_3class_ftw": "FTW-UNet", "resnet50__scratch": "RN50-scratch",
}


def pretty(m: str) -> str:
    return PRETTY.get(m, m)


def _read(run: Path, name: str) -> pd.DataFrame:
    p = run / name
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df.insert(0, "run", run.name)
    return df


def load_runs(runs: Sequence[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, List[pd.DataFrame]] = {}
    for r in runs:
        r = Path(r)
        for key, fn in {"global": "global_id.csv", "local": "local_id_summary.csv", "spatial": "spatial_stats.csv",
                        "variogram": "variogram.csv", "points": "lid_points.parquet",
                        "agree": "location_encoder_agreement.csv", "model_agree": "lid_model_agreement.csv"}.items():
            df = _read(r, fn)
            if not df.empty:
                out.setdefault(key, []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def _tex(df: pd.DataFrame, path: Path, float_fmt: str = "%.2f") -> None:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace("_", r"\_", regex=False)
    path.write_text(df.to_latex(index=False, float_format=lambda v: float_fmt % v, na_rep="--", escape=False))


def global_table(g: pd.DataFrame, methods: Sequence[str], k: int) -> pd.DataFrame:
    g = g[g["method"].isin(methods)].copy()
    kdep = g.groupby("method")["k"].transform("nunique") > 1
    g = g[~kdep | (g["k"] == k)]
    g = g.groupby(["run", "scope", "model", "method"], as_index=False)["global_id_mean"].mean()
    t = g.pivot_table(index=["run", "scope", "model"], columns="method", values="global_id_mean").reset_index()
    t["median"] = t[[m for m in methods if m in t]].median(axis=1)
    return t


def spatial_table(spatial: pd.DataFrame, local: pd.DataFrame, estimator: str) -> pd.DataFrame:
    s = spatial[spatial["estimator"] == estimator]
    l = local[local["estimator"] == estimator][["run", "scope", "model", "lid_mean", "lid_std"]]
    cols = [c for c in ["run", "scope", "model", "n_chips_total", "n_chips", "moran_I", "moran_p", "density_rho",
                        "geo_embed_knn_jaccard", "mantel_r", "mantel_p"] if c in s]
    cols += [c for c in s.columns if c.startswith("kw_")]
    return l.merge(s[cols], on=["run", "scope", "model"], how="right")


def shift_sets(t: pd.DataFrame, sp: pd.DataFrame, aliases: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """ID vs OOD site sets per task; ``aliases`` maps skipped duplicate scopes to the analysed one."""
    aliases = aliases or {}
    rows = []
    for (run, model), d in t.groupby(["run", "model"]):
        d = d.set_index("scope")
        for ood in d.index:
            m = re.match(r"(.+)__ood$", str(ood))
            if not m:
                continue
            task = m.group(1)
            sc = f"{task}__id"
            sc = sc if sc in d.index else aliases.get(sc)
            if sc is None or sc not in d.index:
                continue
            r = {"run": run, "model": model, "task": task, "id_scope": sc, "gid_id": d.loc[sc, "median"], "gid_ood": d.loc[ood, "median"]}
            for tag, scope in (("id", sc), ("ood", ood)):
                q = sp[(sp["run"] == run) & (sp["model"] == model) & (sp["scope"] == scope)]
                if not q.empty:
                    r[f"lid_{tag}"] = float(q["lid_mean"].iloc[0])
                    r[f"n_{tag}"] = int(q["n_chips_total"].iloc[0]) if "n_chips_total" in q else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gid_ratio_ood_id"] = df["gid_ood"] / df["gid_id"]
    return df


def _save(fig, out: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")


def fig_global(t: pd.DataFrame, methods: Sequence[str], scopes: Optional[Sequence[str]], out: Path,
               stem: str = "fig_geo_id_global") -> None:
    import matplotlib.pyplot as plt
    d = t if not scopes else t[t["scope"].isin(scopes)]
    if d.empty:
        return
    ms = [m for m in methods if m in d]
    d = d.assign(lo=d[ms].min(axis=1), hi=d[ms].max(axis=1)).sort_values("median")
    models = list(dict.fromkeys(d["model"]))
    sc = list(dict.fromkeys(d["scope"]))
    fig, ax = plt.subplots(figsize=(3.5, 0.28 * len(models) * max(1, len(sc)) ** 0.5 + 1.0))
    for j, s in enumerate(sc):
        q = d[d["scope"] == s].set_index("model").reindex(models)
        y = np.arange(len(models)) + (j - (len(sc) - 1) / 2) * 0.8 / max(len(sc), 1)
        ax.errorbar(q["median"], y, xerr=[q["median"] - q["lo"], q["hi"] - q["median"]], fmt="o", ms=3.5,
                    capsize=2, lw=0.8, label=s)
    ax.set_yticks(range(len(models)), [pretty(m) for m in models], fontsize=7)
    ax.set_xlabel("Global ID (median over estimators; bar = range)", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    if len(sc) > 1:
        ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, out, stem)
    plt.close(fig)


def fig_shift(ss: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    if ss.empty:
        return
    tasks = list(dict.fromkeys(ss["task"]))
    models = list(dict.fromkeys(ss["model"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.32 * len(tasks) + 1.3), sharey=True)
    for ax, (a, b, lab) in zip(axes, [("gid_id", "gid_ood", "Global ID (median over estimators)"),
                                      ("lid_id", "lid_ood", "Mean local ID")]):
        if a not in ss:
            continue
        for j, m in enumerate(models):
            q = ss[ss["model"] == m].set_index("task").reindex(tasks)
            y = np.arange(len(tasks)) + (j - (len(models) - 1) / 2) * 0.22
            ax.hlines(y, q[a], q[b], color=f"C{j}", lw=1)
            ax.plot(q[a], y, "o", color=f"C{j}", ms=3.5, mfc="white", label=f"{pretty(m)} ID" if ax is axes[0] else None)
            ax.plot(q[b], y, "o", color=f"C{j}", ms=3.5, label=f"{pretty(m)} OOD" if ax is axes[0] else None)
        ax.set_xlabel(lab, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(tasks)), tasks, fontsize=7)
    axes[0].legend(fontsize=5.5, frameon=False, ncol=1, loc="best")
    fig.tight_layout()
    _save(fig, out, "fig_geo_id_shift")
    plt.close(fig)


def fig_size_matched(sm: pd.DataFrame, out: Path) -> None:
    """OOD site-set global ID relative to same-size random ID subsets (shift_size_matched.csv)."""
    import matplotlib.pyplot as plt
    if sm.empty:
        return
    tasks = list(dict.fromkeys(sm["task"]))
    methods = list(dict.fromkeys(sm["method"]))
    models = list(dict.fromkeys(sm["model"]))
    fig, axes = plt.subplots(1, len(methods), figsize=(7.0, 0.3 * len(tasks) + 1.2), sharey=True, squeeze=False)
    for ax, me in zip(axes[0], methods):
        for j, m in enumerate(models):
            q = sm[(sm["model"] == m) & (sm["method"] == me)].set_index("task").reindex(tasks)
            y = np.arange(len(tasks)) + (j - (len(models) - 1) / 2) * 0.22
            z = (q["id_ood"] - q["id_matched_mean"]) / q["id_matched_sd"]
            ax.plot(np.clip(z, -12, 12), y, "o", color=f"C{j}", ms=4, label=pretty(m))
        ax.axvline(0, color="k", lw=0.6)
        for v in (-2, 2):
            ax.axvline(v, color="k", lw=0.5, ls=":")
        ax.set_xlabel(f"{me}: (OOD $-$ size-matched ID) / sd", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.3)
    axes[0, 0].set_yticks(range(len(tasks)), tasks, fontsize=7)
    axes[0, -1].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, out, "fig_geo_id_shift_size_matched")
    plt.close(fig)


def fig_maps(pts: pd.DataFrame, scope: str, estimator: str, out: Path, models: Optional[Sequence[str]] = None,
             stem: str = "fig_geo_id_lid_maps", ncol: int = 3, run: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt
    d = pts[(pts["scope"] == scope)]
    if run:
        d = d[d["run"] == run]
    if estimator not in d or d.empty:
        return
    models = list(models or dict.fromkeys(d["model"]))
    nrow = int(np.ceil(len(models) / ncol))
    lon, lat = d["lon"], d["lat"]
    pad = 3
    ext = [lon.min() - pad, lon.max() + pad, lat.min() - pad, lat.max() + pad]
    w = 7.0 / ncol
    h = w * (ext[3] - ext[2]) / max(ext[1] - ext[0], 1e-6) * 0.85 + 0.45
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, h * nrow), squeeze=False)
    for ax, m in zip(axes.ravel(), models):
        q = d[d["model"] == m].dropna(subset=[estimator])
        v = q[estimator].to_numpy()
        lo, hi = np.nanpercentile(v, [2, 98]) if len(v) else (0, 1)
        sca = ax.scatter(q["lon"], q["lat"], c=np.clip(v, lo, hi), s=2 if len(q) > 3000 else 7, cmap="viridis", lw=0, rasterized=True)
        ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); ax.set_aspect("equal")
        ax.set_title(pretty(m), fontsize=7, pad=2)
        ax.set_xlabel("Longitude", fontsize=6, labelpad=1)
        ax.set_ylabel("Latitude", fontsize=6, labelpad=1)
        ax.tick_params(labelsize=5.5)
        ax.grid(alpha=0.25, lw=0.4)
        cb = fig.colorbar(sca, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=5.5)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, out, stem)
    plt.close(fig)


def fig_variogram(vg: pd.DataFrame, scope: str, estimator: str, out: Path, run: Optional[str] = None,
                  stem: str = "fig_geo_id_variogram") -> None:
    import matplotlib.pyplot as plt
    d = vg[(vg["scope"] == scope) & (vg["estimator"] == estimator)] if "estimator" in vg else pd.DataFrame()
    if run is not None and not d.empty:
        d = d[d["run"] == run]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for j, (m, q) in enumerate(d.groupby("model", sort=False)):
        q = q[q["n_pairs"] >= 30].sort_values("dist_km")
        ax.plot(q["dist_km"], q["sill_ratio"], "o-", ms=3, lw=1, label=pretty(m), color=f"C{j % 10}")
    ax.set_xscale("log")
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("Great-circle distance (km)", fontsize=7)
    ax.set_ylabel(r"$\hat\gamma(h)/\mathrm{Var}(\mathrm{LID})$", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.5, frameon=False, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out, stem)
    plt.close(fig)


def fig_agreement(agree: pd.DataFrame, scope: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    d = agree[agree["scope"] == scope] if not agree.empty else agree
    if d.empty:
        return
    t = d.pivot_table(index="image_encoder", columns="location_encoder", values="spearman")
    fig, ax = plt.subplots(figsize=(3.5, 0.3 * len(t) + 0.9))
    im = ax.imshow(t.to_numpy(), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(t.shape[1]), [pretty(c) for c in t.columns], fontsize=7)
    ax.set_yticks(range(t.shape[0]), [pretty(c) for c in t.index], fontsize=7)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.05)
    cb.ax.tick_params(labelsize=6)
    cb.set_label(r"Spearman $\rho$ (LID)", fontsize=7)
    fig.tight_layout()
    _save(fig, out, "fig_geo_id_agreement")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["MLE", "TwoNN", "MOM", "TLE", "FisherS", "CorrInt", "ESS"])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--estimator", default="MLE_k20_smooth")
    ap.add_argument("--map-scope", default="all")
    ap.add_argument("--map-run", default=None)
    ap.add_argument("--map-models", nargs="*", default=None, help="subset / order of models in the LID maps")
    ap.add_argument("--global-scopes", nargs="*", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R = load_runs(a.runs)
    if "global" in R:
        t = global_table(R["global"], a.methods, a.k)
        t.to_csv(out / "geo_id_global_table.csv", index=False)
        _tex(t.drop(columns=["run"]), out / "geo_id_global_table.tex")
        fig_global(t, a.methods, a.global_scopes, out)
    if "spatial" in R and "local" in R:
        sp = spatial_table(R["spatial"], R["local"], a.estimator)
        sp.to_csv(out / "geo_id_spatial_table.csv", index=False)
        _tex(sp.drop(columns=["run"]), out / "geo_id_spatial_table.tex", "%.3g")
        if "global" in R:
            aliases: Dict[str, str] = {}
            for r in a.runs:
                f = Path(r) / "scope_aliases.json"
                if f.exists():
                    aliases.update(json.loads(f.read_text()))
            ss = shift_sets(t, sp, aliases)
            ss.to_csv(out / "geo_id_shift_sets.csv", index=False)
            fig_shift(ss, out)
    sm_frames = [pd.read_csv(Path(r) / "shift_size_matched.csv") for r in a.runs
                 if (Path(r) / "shift_size_matched.csv").exists()]
    if sm_frames:
        sm = pd.concat(sm_frames, ignore_index=True)
        sm.to_csv(out / "geo_id_shift_size_matched.csv", index=False)
        fig_size_matched(sm, out)
    if "points" in R:
        fig_maps(R["points"], a.map_scope, a.estimator, out, models=a.map_models, run=a.map_run)
    if "variogram" in R:
        fig_variogram(R["variogram"], a.map_scope, a.estimator, out, run=a.map_run)
    if "points" in R:
        # recompute image x location agreement for the raw and smoothed variant of --estimator
        from reptaxonomy.robustness.geo_intrinsic_dimension import location_encoder_agreement
        pts = R["points"]
        locs = sorted(m for m in pts["model"].unique() if str(m).startswith("loc__"))
        base = a.estimator.replace("_smooth", "")
        ag = [location_encoder_agreement(pts, locs, e) for e in (base, base + "_smooth") if e in pts]
        ag = pd.concat([x for x in ag if not x.empty], ignore_index=True) if ag else pd.DataFrame()
        if not ag.empty:
            ag.to_csv(out / "geo_id_location_agreement.csv", index=False)
            fig_agreement(ag[ag["estimator"] == base], a.map_scope, out)
    elif "agree" in R:
        fig_agreement(R["agree"], a.map_scope, out)


if __name__ == "__main__":
    main()
