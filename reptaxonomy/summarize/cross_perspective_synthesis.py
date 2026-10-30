"""Cross-perspective synthesis across datasets, projections, and analysis families.

Reads artifacts already written by the per-dataset pipeline and produces the
journal-level analyses:

  robustness   : silhouette summaries, pretrained-vs-scratch deltas,
                 3-way variance decomposition, Kendall tau-b / W concordance
  similarity   : jointly normalized residual / distance heatmaps averaged over
                 datasets x projections x runs, centrality, nearest neighbours,
                 Mantel test, average-linkage dendrograms
  spectral     : WeightWatcher alpha summaries and figures
  resources    : compute table, compression ratio, inference energy
  synthesis    : 5-axis ranking table, cross-axis Kendall tau, parallel-coordinate
                 rank profiles, Pareto fronts (silhouette / GFLOPs vs energy)
  sensitivity  : native lens (primary) vs 2-D projections (rank tau, Mantel),
                 upsampling variants (pixelshuffle / nearest / bilinear) vs primary
  references   : linear CKA and k-NN overlap matrices, Mantel vs data-kernel matrices
  invariance   : aggregation of invariance_checks.csv

Inputs (per dataset <ds>, model <m>):
  out_dir/<ds>/<m>/silhouette_stats_recomputed.by_run.<m>.<ds>.csv   (embedding_robustness)
  out_dir/<ds>/pairwise_residual_summary.csv                         (data_kernel_analysis, signed)
  out_dir/<ds>/pairwise_ece_summary.csv                              (fallback, unsigned)
  out_dir/<ds>/pairwise_distance_summary.csv                         (data_kernel_analysis)
  embed_dir/<ds>/<m>/test/ww_stats/<m>.details.csv                   (weight_matrix_analysis)
  embed_dir/<ds>/<m>/test/calflops_stats/<m>.flops_summary.csv       (compute_resource_reqs)
  embed_dir/<ds>/<m>/test/calflops_stats/<m>.inference_energy.csv    (compute_resource_reqs, new)
  embed_dir/<ds>/<m>/test/calflops_stats/<m>.compression.csv         (compute_resource_reqs, new)

Usage:
  As the `cross_perspective_synthesis` global pipeline step (single dataset by
  default, or the list in `synthesis.datasets`), or standalone:
      python -m reptaxonomy.summarize.cross_perspective_synthesis -y synthesis.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from reptaxonomy.util.general_utils import read_yaml
from reptaxonomy.util.artifacts import ensure_dir, save_json
from reptaxonomy.summarize import synthesis_stats as S
from reptaxonomy.summarize import synthesis_plots as P

logger = logging.getLogger(__name__)


DEFAULT_SYNTHESIS_CFG: Dict[str, Any] = {
    "datasets": None,                    # None -> dataset of the current pipeline cfg
    "out_subdir": "synthesis",           # written to out_dir/<out_subdir>/
    "display_names": {},                 # model_name -> label used in figures
    "training_pairs": [],                # [[pretrained, scratch], ...]
    "residual_source": "auto",           # auto | signed | ece
    "mantel_permutations": 10000,
    "mantel_method": "spearman",
    "alpha_range": [2.0, 5.0],
    "alpha_dataset": None,               # dataset whose ww_stats to read (weights are dataset agnostic)
    "resource_dataset": None,            # None -> average over datasets that have the artifact
    "robustness_rank_by": "mean",        # mean | median | median_minus_iqr | min
    "layer_fit_rank_by": "frac_in_range",  # frac_in_range | median_distance
    "compute_rank_by": "gflops",         # gflops | gmacs | params
    "emissions_rank_by": "kg_co2eq_per_sample",  # kg_co2eq_per_sample | kwh_per_sample
    "figure_formats": ["pdf", "png"],
    "require_full_design": False,        # if True, raise when the model x dataset x projection grid is incomplete
    "primary_lens": "native",            # primary analysis lens; 2-D projections are treated as sensitivity lenses
    "reference_metrics": ["linear_cka", "knn_jaccard"],
    "variants": None,                    # {upsample_method: out_dir}; filled automatically by full_pipeline
    "primary_variant": None,             # key in `variants` used for the primary analysis
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _dataset_from_cfg(cfg: Dict[str, Any]) -> Optional[str]:
    bundle = cfg.get("dataset_bundle")
    if isinstance(bundle, dict):
        spec = bundle.get("dataset_spec", {})
        name = spec.get("dataset_name") or spec.get("name")
        if name:
            return str(name)
    ds = cfg.get("dataset")
    if isinstance(ds, dict):
        name = ds.get("dataset_name") or ds.get("name")
        return str(name) if name else None
    if isinstance(ds, str):
        return ds
    return None


def resolve_synthesis_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    syn = dict(DEFAULT_SYNTHESIS_CFG)
    syn.update(cfg.get("synthesis", {}) or {})
    if not syn["datasets"]:
        ds = _dataset_from_cfg(cfg)
        if ds is None:
            raise ValueError("synthesis.datasets is empty and no dataset could be resolved from cfg")
        syn["datasets"] = [ds]
    syn["datasets"] = [str(d) for d in syn["datasets"]]
    return syn


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_silhouettes(out_dir: str, datasets: Sequence[str], models: Sequence[str]) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        for m in models:
            by_run = os.path.join(out_dir, ds, m, f"silhouette_stats_recomputed.by_run.{m}.{ds}.csv")
            fallback = os.path.join(out_dir, ds, m, f"silhouette_stats.{m}.{ds}.csv")
            if os.path.exists(by_run):
                df = pd.read_csv(by_run)
            elif os.path.exists(fallback):
                df = pd.read_csv(fallback).rename(columns={"mean_sil": "silhouette"})
                df["run"] = df.get("run", -1)
                df["dataset"] = ds
                df["model"] = m
                logger.warning("Using run_projections silhouette file for %s/%s", ds, m)
            else:
                logger.warning("No silhouette stats for dataset=%s model=%s", ds, m)
                continue
            frames.append(df[["dataset", "projection", "model", "run", "silhouette"]])
    if not frames:
        return pd.DataFrame(columns=["dataset", "projection", "model", "run", "silhouette"])
    out = pd.concat(frames, ignore_index=True)
    out["projection"] = out["projection"].str.lower()
    return out


def load_pairwise(out_dir: str, datasets: Sequence[str], residual_source: str = "auto") -> Dict[str, pd.DataFrame]:
    res_frames, dist_frames = [], []
    used_source = set()
    for ds in datasets:
        root = os.path.join(out_dir, ds)
        signed = os.path.join(root, "pairwise_residual_summary.csv")
        ece = os.path.join(root, "pairwise_ece_summary.csv")
        dist = os.path.join(root, "pairwise_distance_summary.csv")

        if residual_source in ("auto", "signed") and os.path.exists(signed):
            df = pd.read_csv(signed)[["run", "projection", "model_a", "model_b", "residual"]]
            used_source.add("signed")
        elif residual_source in ("auto", "ece") and os.path.exists(ece):
            df = pd.read_csv(ece).rename(columns={"ece": "residual"})
            used_source.add("ece")
            logger.warning("dataset=%s: using unsigned ECE as residual (re-run data_kernel_analysis "
                           "to produce pairwise_residual_summary.csv)", ds)
        else:
            logger.warning("dataset=%s: no residual summary found", ds)
            df = None
        if df is not None:
            df["dataset"] = ds
            res_frames.append(df)

        if os.path.exists(dist):
            d = pd.read_csv(dist)
            d["dataset"] = ds
            dist_frames.append(d)
        else:
            logger.warning("dataset=%s: no pairwise_distance_summary.csv", ds)

    if len(used_source) > 1:
        logger.warning("Mixing signed residuals and unsigned ECE across datasets; results are not comparable.")
    return {
        "residual": pd.concat(res_frames, ignore_index=True) if res_frames else pd.DataFrame(),
        "distance": pd.concat(dist_frames, ignore_index=True) if dist_frames else pd.DataFrame(),
        "residual_source": ",".join(sorted(used_source)) or "none",
    }


def load_alphas(embed_dir: str, datasets: Sequence[str], models: Sequence[str],
                alpha_dataset: Optional[str] = None) -> pd.DataFrame:
    order = [alpha_dataset] if alpha_dataset else list(datasets)
    frames = []
    for m in models:
        for ds in order:
            p = os.path.join(embed_dir, ds, m, "test", "ww_stats", f"{m}.details.csv")
            if os.path.exists(p):
                d = pd.read_csv(p)
                if "alpha" not in d.columns:
                    logger.warning("No alpha column in %s", p)
                    break
                d = d.reset_index(drop=True)
                frames.append(pd.DataFrame({
                    "model": m, "dataset": ds,
                    "layer_order": np.arange(len(d)),
                    "layer_id": d.get("layer_id", pd.Series(np.arange(len(d)))),
                    "name": d.get("name", pd.Series([""] * len(d))),
                    "alpha": pd.to_numeric(d["alpha"], errors="coerce"),
                }))
                break
        else:
            logger.warning("No WeightWatcher details for model=%s", m)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["model", "alpha"])


_PARAM_SCALE = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9, "T": 1e12}


def parse_param_count(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([KMBGT]?)", str(value).replace(",", "").upper())
    return float(m.group(1)) * _PARAM_SCALE[m.group(2)] if m else np.nan


def load_resources(embed_dir: str, datasets: Sequence[str], models: Sequence[str],
                   resource_dataset: Optional[str] = None) -> pd.DataFrame:
    """One row per model: compute, compression, and inference-energy metrics.

    If resource_dataset is None, numeric metrics are averaged over datasets
    that provide them (FLOPs depend on input size / band count).
    """
    use = [resource_dataset] if resource_dataset else list(datasets)
    rows = []
    for m in models:
        for ds in use:
            root = os.path.join(embed_dir, ds, m, "test", "calflops_stats")
            row: Dict[str, Any] = {"model": m, "dataset": ds}
            fs = os.path.join(root, f"{m}.flops_summary.csv")
            if os.path.exists(fs):
                s = pd.read_csv(fs).iloc[0]
                row["gflops"] = float(s.get("mean_flops_numeric", np.nan)) / 1e9
                row["gmacs"] = float(s.get("mean_macs_numeric", np.nan)) / 1e9
                row["params"] = parse_param_count(s.get("params"))
                row["profiling_kg_co2eq"] = float(s.get("emissions_kg_co2eq", np.nan))
            en = os.path.join(root, f"{m}.inference_energy.csv")
            if os.path.exists(en):
                e = pd.read_csv(en).iloc[0]
                for k in ["kwh_per_sample", "kg_co2eq_per_sample", "seconds_per_sample",
                          "implied_kg_co2eq_per_kwh", "n_samples_measured", "joules_per_image",
                          "throughput_images_per_s", "latency_per_image_ms", "latency_batch_median_ms",
                          "latency_batch_p95_ms", "peak_gpu_memory_allocated_mb",
                          "peak_gpu_memory_reserved_mb", "peak_host_rss_mb_process", "batch_size",
                          "total_seconds"]:
                    row[k] = pd.to_numeric(e.get(k, np.nan), errors="coerce")
                for k in ["precision", "gpu_name", "torch", "cuda_runtime", "synchronization"]:
                    row[f"meta_{k}"] = e.get(k)
            co = os.path.join(root, f"{m}.compression.csv")
            if os.path.exists(co):
                c = pd.read_csv(co).iloc[0]
                for k in ["input_elements", "output_elements", "compression_ratio"]:
                    row[k] = float(c.get(k, np.nan))
            if len(row) > 2:
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["model"])
    df = pd.DataFrame(rows)
    out = df.drop(columns=["dataset"]).groupby("model", as_index=False).mean(numeric_only=True)
    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    if meta_cols:
        meta = df.groupby("model", as_index=False)[meta_cols].agg(
            lambda x: ";".join(sorted({str(v) for v in x if pd.notna(v)})))
        out = out.merge(meta, on="model", how="left")
    if "kg_co2eq_per_sample" not in out.columns and "profiling_kg_co2eq" in out.columns:
        logger.warning("No inference_energy.csv found; emissions axis falls back to calflops profiling emissions.")
        out["kg_co2eq_per_sample"] = out["profiling_kg_co2eq"]
    if {"kwh_per_sample", "gflops"}.issubset(out.columns):
        out["kwh_per_gflop"] = out["kwh_per_sample"] / out["gflops"]
    return out


# ---------------------------------------------------------------------------
# Analysis blocks
# ---------------------------------------------------------------------------

def split_lenses(df: pd.DataFrame, primary: str, col: str = "projection") -> Dict[str, pd.DataFrame]:
    """primary-lens rows, projection-lens rows, and all rows."""
    if df.empty or col not in df.columns:
        return {"primary": df, "projections": df, "all": df}
    prim = df[df[col] == primary]
    proj = df[df[col] != primary]
    if prim.empty:
        logger.warning("Primary lens '%s' absent; using projection lenses as primary.", primary)
        prim = proj
    return {"primary": prim, "projections": proj, "all": df}


def run_robustness_block(sil: pd.DataFrame, syn: Dict[str, Any], out: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    if sil.empty:
        logger.warning("Skipping robustness block: no silhouette data")
        return results
    cond_all = S.silhouette_condition_table(sil)
    cond_all.to_csv(os.path.join(out, "silhouette_by_condition.csv"), index=False)
    lenses = split_lenses(cond_all, syn["primary_lens"])

    # Primary robustness summary: native lens over datasets (falls back to projections).
    summ = S.robustness_summary(lenses["primary"])
    summ["median_minus_iqr"] = summ["median"] - summ["iqr"]
    if not lenses["projections"].empty:
        ps = S.robustness_summary(lenses["projections"])
        ps.to_csv(os.path.join(out, "robustness_summary_projections.csv"), index=False)
        summ = summ.drop(columns=["delta_proj"]).merge(
            ps[["model", "mean", "delta_proj"]].rename(columns={"mean": "mean_projections"}),
            on="model", how="left")
    summ.to_csv(os.path.join(out, "robustness_summary.csv"), index=False)
    S.robustness_summary(cond_all).to_csv(os.path.join(out, "robustness_summary_all_lenses.csv"), index=False)
    results["robustness"] = summ

    # Lens sensitivity: does each 2-D projection reproduce the native-lens ordering?
    if syn["primary_lens"] in set(cond_all["projection"]) and not lenses["projections"].empty:
        rows = []
        for ds, g in cond_all.groupby("dataset"):
            piv = g.pivot_table(index="model", columns="projection", values="silhouette")
            if syn["primary_lens"] not in piv.columns:
                continue
            for lens in [c for c in piv.columns if c != syn["primary_lens"]]:
                sub = piv[[syn["primary_lens"], lens]].dropna()
                if len(sub) > 2:
                    tau, p = stats.kendalltau(sub[syn["primary_lens"]], sub[lens], variant="b")
                    rows.append({"dataset": ds, "lens": lens, "tau_b_vs_primary": float(tau), "p_value": float(p),
                                 "mean_abs_delta_silhouette": float((sub[lens] - sub[syn["primary_lens"]]).abs().mean()),
                                 "n_models": int(len(sub))})
        sens = pd.DataFrame(rows)
        if not sens.empty:
            sens.to_csv(os.path.join(out, "lens_sensitivity_silhouette.csv"), index=False)
            P.plot_grid(sens, "dataset", "lens", "tau_b_vs_primary", out, "fig_lens_sensitivity",
                        f"Kendall tau-b vs {syn['primary_lens']} lens ranking", -1, 1, "RdBu_r",
                        formats=syn["figure_formats"])
            results["lens_sensitivity"] = sens

    cond = lenses["projections"] if not lenses["projections"].empty else cond_all

    deltas = S.training_pair_deltas(cond, syn["training_pairs"])
    if not deltas.empty:
        deltas.to_csv(os.path.join(out, "pretrained_minus_scratch.csv"), index=False)

    n_ds, n_proj = cond["dataset"].nunique(), cond["projection"].nunique()
    counts = cond.groupby("model").size()
    full_models = counts[counts == n_ds * n_proj].index.tolist()
    if len(full_models) < cond["model"].nunique():
        msg = f"Incomplete design; variance/concordance restricted to models {full_models}"
        if syn["require_full_design"]:
            raise ValueError(msg)
        logger.warning(msg)
    cond_full = cond[cond["model"].isin(full_models)]

    if n_ds > 1 and n_proj > 1 and len(full_models) > 1:
        # Sensitivity design: model x dataset x 2-D projection lens.
        var = S.three_way_variance_decomposition(cond_full)
        var.to_csv(os.path.join(out, "variance_decomposition.csv"), index=False)
        P.plot_variance_decomposition(var, out, syn["figure_formats"])
        results["variance"] = var
        all_full = cond_all[cond_all["model"].isin(full_models)]
        if all_full["projection"].nunique() > n_proj:
            S.three_way_variance_decomposition(all_full).to_csv(
                os.path.join(out, "variance_decomposition_all_lenses.csv"), index=False)
    else:
        logger.info("Variance decomposition needs >1 dataset and >1 projection; skipped.")

    if len(full_models) > 2:
        conc = S.rank_concordance(cond_all[cond_all["model"].isin(full_models)])
        conc["tau"].to_csv(os.path.join(out, "rank_concordance_tau.csv"), index=False)
        conc["w"].to_csv(os.path.join(out, "rank_concordance_w.csv"), index=False)
        results["concordance"] = conc

    S.condition_rankings(cond_all).to_csv(os.path.join(out, "silhouette_ranks_by_condition.csv"), index=False)
    P.plot_silhouette_distribution(cond_all, summ["model"].tolist(), out, syn["display_names"], syn["figure_formats"])
    return results


def run_similarity_block(pair: Dict[str, Any], models: Sequence[str], syn: Dict[str, Any],
                         out: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    res_df, dist_df = pair["residual"], pair["distance"]
    if res_df.empty or dist_df.empty:
        logger.warning("Skipping similarity block: missing residual or distance summaries")
        return results

    present = sorted(set(res_df["model_a"]) & set(dist_df["model_a"]))
    models = [m for m in models if m in present]
    primary = syn["primary_lens"]
    res_l = split_lenses(res_df, primary)
    dist_l = split_lenses(dist_df, primary)

    res_mean, dist_mean, res_conds, dist_conds, res_cube = mean_normalized_matrices(
        res_l["primary"], dist_l["primary"], models)
    results["matrices"] = {"models": models, "residual": res_mean, "distance": dist_mean}

    # Lens sensitivity for P2: Mantel between primary-lens matrices and each projection lens.
    sens_rows = []
    for lens in sorted(set(res_l["projections"].get("projection", pd.Series(dtype=str)))):
        r_l, d_l, *_ = mean_normalized_matrices(res_df[res_df["projection"] == lens],
                                                dist_df[dist_df["projection"] == lens], models)
        if len(models) > 3 and (res_l["primary"] is not res_l["projections"]):
            for name, A, B in (("residual", res_mean, r_l), ("distance", dist_mean, d_l)):
                mt = S.mantel_test(A, B, n_perm=int(syn["mantel_permutations"]), method=syn["mantel_method"])
                sens_rows.append({"lens": lens, "matrix": name, "mantel_r": mt["statistic"], "p_value": mt["p_value"]})
            cl = S.centrality(r_l, d_l, models)
            cp = S.centrality(res_mean, dist_mean, models)
            m_ = cp.merge(cl, on="model", suffixes=("_primary", "_lens"))
            sens_rows.append({"lens": lens, "matrix": "centrality_rank_tau",
                              "mantel_r": float(stats.kendalltau(m_["centrality_primary"], m_["centrality_lens"])[0]),
                              "p_value": float(stats.kendalltau(m_["centrality_primary"], m_["centrality_lens"])[1])})
    if sens_rows:
        sens = pd.DataFrame(sens_rows)
        sens.to_csv(os.path.join(out, "lens_sensitivity_similarity.csv"), index=False)
        P.plot_grid(sens, "matrix", "lens", "mantel_r", out, "fig_lens_sensitivity_p2",
                    f"Agreement with {primary}-lens P2 matrices", -1, 1, "RdBu_r",
                    formats=syn["figure_formats"])

    pd.DataFrame(res_mean, index=models, columns=models).to_csv(os.path.join(out, "residual_norm_mean.csv"))
    pd.DataFrame(dist_mean, index=models, columns=models).to_csv(os.path.join(out, "distance_norm_mean.csv"))
    pd.DataFrame(np.nanmean(res_cube, axis=0), index=models, columns=models).to_csv(
        os.path.join(out, "residual_raw_mean.csv"))

    lbl = "signed residual" if pair["residual_source"] == "signed" else f"residual ({pair['residual_source']})"
    P.plot_heatmap(res_mean, models, out, "fig_heatmap_residual",
                   f"Normalized {lbl}, mean over {len(res_conds)} conditions",
                   syn["display_names"], "viridis", syn["figure_formats"])
    P.plot_heatmap(dist_mean, models, out, "fig_heatmap_distance",
                   f"Normalized distance, mean over {len(dist_conds)} conditions",
                   syn["display_names"], "magma", syn["figure_formats"])

    cent = S.centrality(res_mean, dist_mean, models)
    cent.to_csv(os.path.join(out, "centrality.csv"), index=False)
    results["centrality"] = cent

    S.nearest_neighbors(res_mean, dist_mean, models, k=1).to_csv(
        os.path.join(out, "nearest_neighbors.csv"), index=False)

    if len(models) > 3:
        mantel = S.mantel_test(res_mean, dist_mean, n_perm=int(syn["mantel_permutations"]),
                               method=syn["mantel_method"], seed=42)
        mantel["residual_source"] = pair["residual_source"]
        save_json(mantel, os.path.join(out, "mantel_test.json"))
        results["mantel"] = mantel

        P.plot_dendrograms(S.average_linkage(res_mean), S.average_linkage(dist_mean), models, out,
                           syn["display_names"], syn["figure_formats"])
    return results


def mean_normalized_matrices(res_df: pd.DataFrame, dist_df: pd.DataFrame, models: Sequence[str]):
    """Normalize residuals jointly (min-max) and distances per condition (max off-diagonal), then average."""
    res_cube, res_conds = S.pairwise_long_to_cube(res_df, "residual", models)
    dist_cube, dist_conds = S.pairwise_long_to_cube(dist_df, "distance", models)
    res_mean = np.nanmean(S.normalize_residual_cube(res_cube), axis=0) if len(res_cube) else np.full((len(models),) * 2, np.nan)
    dist_mean = np.nanmean(S.normalize_distance_cube(dist_cube), axis=0) if len(dist_cube) else np.full((len(models),) * 2, np.nan)
    return res_mean, dist_mean, res_conds, dist_conds, res_cube


def load_reference(out_dir: str, datasets: Sequence[str]) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        p = os.path.join(out_dir, ds, "pairwise_reference_summary.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            d["dataset"] = ds
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_reference_block(ref: pd.DataFrame, blocks: Dict[str, Any], models: Sequence[str],
                        syn: Dict[str, Any], out: str) -> Dict[str, Any]:
    """Conventional references (linear CKA, k-NN Jaccard) vs data-kernel matrices."""
    if ref.empty:
        logger.warning("Skipping reference block: no pairwise_reference_summary.csv")
        return {}
    mats = blocks.get("matrices")
    primary = split_lenses(ref, syn["primary_lens"])["primary"]
    models = [m for m in models if m in set(primary["model_a"])]
    rows, cents = [], []
    for metric in syn["reference_metrics"]:
        if metric not in primary.columns or primary[metric].isna().all():
            continue
        cube, _ = S.pairwise_long_to_cube(primary, metric, models)
        sim = np.nanmean(cube, axis=0)
        dis = 1.0 - sim
        np.fill_diagonal(dis, 0.0)
        pd.DataFrame(sim, index=models, columns=models).to_csv(os.path.join(out, f"reference_{metric}_mean.csv"))
        P.plot_heatmap(sim, models, out, f"fig_heatmap_{metric}",
                       f"{metric} ({syn['primary_lens']} lens), mean over conditions",
                       syn["display_names"], "cividis", syn["figure_formats"])
        c = pd.DataFrame({"model": models,
                          f"{metric}_centrality": [float(np.nanmean(np.delete(dis[i], i))) for i in range(len(models))]})
        cents.append(c)
        if mats is not None and len(models) > 3:
            idx = [mats["models"].index(m) for m in models]
            for name in ("residual", "distance"):
                M = mats[name][np.ix_(idx, idx)]
                mt = S.mantel_test(dis, M, n_perm=int(syn["mantel_permutations"]), method=syn["mantel_method"])
                rows.append({"reference": metric, "data_kernel_matrix": name,
                             "mantel_r": mt["statistic"], "p_value": mt["p_value"]})
    res: Dict[str, Any] = {}
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, "reference_vs_data_kernel_mantel.csv"), index=False)
        res["reference_mantel"] = df
    if cents:
        cdf = cents[0]
        for c in cents[1:]:
            cdf = cdf.merge(c, on="model")
        if blocks.get("centrality") is not None:
            cdf = cdf.merge(blocks["centrality"][["model", "centrality"]], on="model", how="left")
        cdf.to_csv(os.path.join(out, "reference_centrality.csv"), index=False)
        res["reference_centrality"] = cdf
    return res


def run_invariance_block(out_dir: str, datasets: Sequence[str], syn: Dict[str, Any], out: str) -> Dict[str, Any]:
    frames = []
    for ds in datasets:
        p = os.path.join(out_dir, ds, "invariance_checks.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            d["dataset"] = ds
            frames.append(d)
    if not frames:
        logger.warning("Skipping invariance block: no invariance_checks.csv")
        return {}
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(out, "invariance_checks_all.csv"), index=False)
    df["passed_num"] = df["passed"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    delta = np.where(df["perspective"] == "P1", df.get("abs_delta_silhouette"), df.get("abs_delta_residual"))
    df["abs_delta_primary_stat"] = pd.to_numeric(pd.Series(delta), errors="coerce")
    summ = (df.groupby(["perspective", "lens", "transform", "expected"], dropna=False)
              .agg(n=("model", "count"), pass_rate=("passed_num", "mean"),
                   max_abs_delta=("abs_delta_primary_stat", "max"),
                   min_knn_jaccard=("knn_jaccard", "min"))
              .reset_index())
    summ.to_csv(os.path.join(out, "invariance_summary.csv"), index=False)
    p1 = df[df["perspective"] == "P1"]
    if not p1.empty:
        P.plot_grid(p1.assign(row=p1["lens"]), "row", "transform", "knn_jaccard", out, "fig_invariance",
                    "P1 k-NN graph Jaccard (baseline vs transformed)", 0, 1, "viridis",
                    formats=syn["figure_formats"])
    return {"invariance": summ}


def run_variant_block(syn: Dict[str, Any], datasets: Sequence[str], models: Sequence[str],
                      primary_sil: pd.DataFrame, primary_pair: Dict[str, Any], out: str) -> Dict[str, Any]:
    """Upsampling sensitivity: compare each variant's rankings / matrices with the primary variant."""
    variants = syn.get("variants") or {}
    pv = syn.get("primary_variant")
    if len(variants) < 2:
        return {}
    rows, mrows = [], []
    base_cond = S.silhouette_condition_table(primary_sil) if not primary_sil.empty else pd.DataFrame()
    base_lenses = split_lenses(primary_pair["residual"], syn["primary_lens"]) if not primary_pair["residual"].empty else None
    for name, vdir in variants.items():
        if name == pv:
            continue
        vs = load_silhouettes(vdir, datasets, models)
        if vs.empty or base_cond.empty:
            continue
        vc = S.silhouette_condition_table(vs)
        merged = base_cond.merge(vc, on=["dataset", "projection", "model"], suffixes=("_primary", "_variant"))
        for (ds, lens), g in merged.groupby(["dataset", "projection"]):
            if len(g) > 2:
                tau, p = stats.kendalltau(g["silhouette_primary"], g["silhouette_variant"], variant="b")
                rows.append({"variant": name, "dataset": ds, "lens": lens, "tau_b_vs_primary": float(tau),
                             "p_value": float(p),
                             "mean_abs_delta_silhouette": float((g["silhouette_variant"] - g["silhouette_primary"]).abs().mean()),
                             "mean_delta_silhouette": float((g["silhouette_variant"] - g["silhouette_primary"]).mean())})
        vp = load_pairwise(vdir, datasets, syn["residual_source"])
        if base_lenses is not None and not vp["residual"].empty and len(models) > 3:
            ms = [m for m in models if m in set(vp["residual"]["model_a"])]
            r0, d0, *_ = mean_normalized_matrices(base_lenses["primary"],
                                                  split_lenses(primary_pair["distance"], syn["primary_lens"])["primary"], ms)
            r1, d1, *_ = mean_normalized_matrices(split_lenses(vp["residual"], syn["primary_lens"])["primary"],
                                                  split_lenses(vp["distance"], syn["primary_lens"])["primary"], ms)
            for mname, A, B in (("residual", r0, r1), ("distance", d0, d1)):
                mt = S.mantel_test(A, B, n_perm=int(syn["mantel_permutations"]), method=syn["mantel_method"])
                mrows.append({"variant": name, "matrix": mname, "mantel_r": mt["statistic"], "p_value": mt["p_value"]})
    res: Dict[str, Any] = {}
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out, "upsampling_sensitivity_silhouette.csv"), index=False)
        P.plot_grid(df.assign(row=df["variant"] + " / " + df["lens"]), "row", "dataset", "tau_b_vs_primary", out,
                    "fig_upsampling_sensitivity", f"Kendall tau-b vs primary upsampling ({pv})", -1, 1, "RdBu_r",
                    formats=syn["figure_formats"])
        res["upsampling"] = df
    if mrows:
        pd.DataFrame(mrows).to_csv(os.path.join(out, "upsampling_sensitivity_similarity.csv"), index=False)
    return res


def run_spectral_block(alphas: pd.DataFrame, models: Sequence[str], syn: Dict[str, Any], out: str) -> Dict[str, Any]:
    if alphas.empty:
        logger.warning("Skipping spectral block: no WeightWatcher details")
        return {}
    lo, hi = [float(x) for x in syn["alpha_range"]]
    summ = S.alpha_summary(alphas, lo, hi)
    summ.to_csv(os.path.join(out, "alpha_summary.csv"), index=False)
    order = [m for m in summ.sort_values("median_alpha")["model"]]
    P.plot_alpha_boxplot(alphas, order, out, lo, hi, syn["display_names"], syn["figure_formats"])
    P.plot_alpha_pairs(alphas, syn["training_pairs"], out, lo, hi, syn["display_names"], syn["figure_formats"])
    return {"alpha": summ}


def build_ranking_table(blocks: Dict[str, Any], resources: pd.DataFrame, syn: Dict[str, Any],
                        models: Sequence[str]) -> pd.DataFrame:
    df = pd.DataFrame({"model": list(models)})

    rob = blocks.get("robustness")
    if rob is not None:
        key = syn["robustness_rank_by"]
        df = df.merge(rob[["model", "mean", "median", "iqr", "min", "median_minus_iqr"]]
                      .rename(columns={"mean": "robustness_mean"}), on="model", how="left")
        col = "robustness_mean" if key == "mean" else key
        df["Robustness"] = S.rank_series(df[col], ascending=False)

    cent = blocks.get("centrality")
    if cent is not None:
        df = df.merge(cent[["model", "centrality"]], on="model", how="left")
        df["Centrality"] = S.rank_series(df["centrality"], ascending=True)

    alpha = blocks.get("alpha")
    if alpha is not None:
        df = df.merge(alpha[["model", "median_alpha", "frac_in_range", "median_distance_from_range"]],
                      on="model", how="left")
        if syn["layer_fit_rank_by"] == "median_distance":
            df["Layer fit"] = S.rank_series(df["median_distance_from_range"], ascending=True)
        else:
            df["Layer fit"] = S.rank_series(df["frac_in_range"], ascending=False)

    if not resources.empty:
        df = df.merge(resources, on="model", how="left")
        c = syn["compute_rank_by"]
        if c in df.columns:
            df["Compute"] = S.rank_series(df[c], ascending=True)
        e = syn["emissions_rank_by"]
        if e in df.columns:
            df["Emissions"] = S.rank_series(df[e], ascending=True)
    return df


def run_synthesis_block(blocks: Dict[str, Any], resources: pd.DataFrame, models: Sequence[str],
                        syn: Dict[str, Any], out: str) -> Dict[str, Any]:
    table = build_ranking_table(blocks, resources, syn, models)
    axes = [a for a in ["Robustness", "Centrality", "Layer fit", "Compute", "Emissions"] if a in table.columns]
    if axes:
        table["mean_rank"] = table[axes].mean(axis=1)
    table.to_csv(os.path.join(out, "ranking_table.csv"), index=False)
    if not resources.empty:
        resources.to_csv(os.path.join(out, "resource_table.csv"), index=False)

    results: Dict[str, Any] = {"ranking": table}
    if len(axes) >= 2 and len(table) > 2:
        tau = S.kendall_tau_matrix(table, axes)
        tau.to_csv(os.path.join(out, "rank_axis_tau.csv"))
        P.plot_rank_tau(tau, out, syn["figure_formats"])
        P.plot_rank_parallel(table.sort_values(axes[0]), axes, out, syn["display_names"], syn["figure_formats"])
        results["axis_tau"] = tau

    energy_col = "kwh_per_sample" if "kwh_per_sample" in table.columns else (
        "kg_co2eq_per_sample" if "kg_co2eq_per_sample" in table.columns else None)
    if energy_col and {"robustness_mean", "gflops"}.issubset(table.columns):
        t = table.copy()
        t["pareto_sil_energy"] = S.pareto_front(t[energy_col], t["robustness_mean"], True, True)
        t["pareto_flops_energy"] = S.pareto_front(t[energy_col], t["gflops"], True, False)
        rho_sil = S.spearman(t[energy_col], t["robustness_mean"])
        rho_flops = S.spearman(t[energy_col], t["gflops"])
        t[["model", energy_col, "robustness_mean", "gflops", "pareto_sil_energy", "pareto_flops_energy"]].to_csv(
            os.path.join(out, "pareto_fronts.csv"), index=False)
        save_json({"energy_column": energy_col, "silhouette_vs_energy": rho_sil,
                   "gflops_vs_energy": rho_flops}, os.path.join(out, "tradeoff_spearman.json"))
        label = "Energy per sample (kWh)" if energy_col == "kwh_per_sample" else "kg CO2eq per sample"
        P.plot_tradeoff(t, t["pareto_sil_energy"].to_numpy(), t["pareto_flops_energy"].to_numpy(),
                        rho_sil, rho_flops, out, energy_col, label, syn["display_names"], syn["figure_formats"])
    return results


def write_environment_table(embed_dir: str, datasets: Sequence[str], models: Sequence[str], out: str) -> None:
    rows = []
    for ds in datasets:
        for m in models:
            p = os.path.join(embed_dir, ds, m, "test", "calflops_stats", f"{m}.environment.json")
            if os.path.exists(p):
                with open(p) as f:
                    env = json.load(f)
                env.update({"dataset": ds, "model": m})
                rows.append(env)
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out, "environment_table.csv"), index=False)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_cross_perspective_synthesis(cfg: Dict[str, Any]) -> Dict[str, Any]:
    syn = resolve_synthesis_cfg(cfg)
    out_dir, embed_dir = cfg.get("out_dir", "./outputs"), cfg.get("embed_dir", "./embeddings")
    model_names_cfg = cfg["model_names"]
    if isinstance(model_names_cfg, str):
        with open(model_names_cfg, "r") as f:
            model_names_cfg = json.load(f)
    models = [str(m) for m in model_names_cfg]
    datasets = syn["datasets"]

    # synthesis.tag overrides the output folder name (needed when many datasets /
    # conditions are pooled, e.g. dozens of EPA ecoregions -> filename too long).
    tag = syn.get("tag") or (datasets[0] if len(datasets) == 1 else "_".join(datasets))
    if len(tag) > 120:
        import hashlib
        tag = f"{len(datasets)}ds_" + hashlib.sha1(tag.encode()).hexdigest()[:10]
    out = ensure_dir(os.path.join(out_dir, syn["out_subdir"], tag))
    logger.info("Cross-perspective synthesis datasets=%s models=%s -> %s", datasets, models, out)

    sil = load_silhouettes(out_dir, datasets, models)
    if not sil.empty and cfg.get("projections"):
        sil = sil[sil["projection"].isin([p.lower() for p in cfg["projections"]])]
    pair = load_pairwise(out_dir, datasets, syn["residual_source"])
    alphas = load_alphas(embed_dir, datasets, models, syn["alpha_dataset"])
    resources = load_resources(embed_dir, datasets, models, syn["resource_dataset"])

    blocks: Dict[str, Any] = {}
    blocks.update(run_robustness_block(sil, syn, out))
    blocks.update(run_similarity_block(pair, models, syn, out))
    blocks.update(run_reference_block(load_reference(out_dir, datasets), blocks, models, syn, out))
    blocks.update(run_invariance_block(out_dir, datasets, syn, out))
    blocks.update(run_variant_block(syn, datasets, models, sil, pair, out))
    blocks.update(run_spectral_block(alphas, models, syn, out))
    write_environment_table(embed_dir, datasets, models, out)
    blocks.update(run_synthesis_block(blocks, resources, models, syn, out))

    save_json({"datasets": datasets, "models": models, "residual_source": pair["residual_source"],
               "config": {k: v for k, v in syn.items() if k != "display_names"}},
              os.path.join(out, "synthesis_manifest.json"))
    print(f"[SYNTHESIS] wrote outputs to {out}")
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset, cross-perspective synthesis.")
    parser.add_argument("-y", "--yaml", required=True, help="YAML config file.")
    args = parser.parse_args()
    run_cross_perspective_synthesis(read_yaml(args.yaml))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
