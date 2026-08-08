from __future__ import annotations

from typing import Any

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from reptaxonomy.util.general_utils import read_yaml

try:
    from aeon.visualisation import plot_critical_difference
    HAVE_AEON = True
except Exception:
    HAVE_AEON = False

ANALYSIS_SPECS = {
    "flops": {"pattern": "**/calflops_stats/*.flops.pkl", "higher_is_better": False, "block_cols": ["dataset"]},
    "weightwatcher": {"pattern": "**/ww_stats/*.summary.pkl", "higher_is_better": True, "block_cols": ["dataset"]},
    "silhouette": {"pattern": "**/silhouette_stats*.csv", "higher_is_better": True, "block_cols": ["dataset", "projection"]},
    "intrinsic_dimension": {"pattern": "**/intrinsic_dimension_summary.*.csv", "higher_is_better": False, "block_cols": ["dataset", "method", "k"]},
    "geodesic_pairwise": {"pattern": "**/*_pairwise_comparison.csv", "higher_is_better": True, "block_cols": ["dataset", "anchor", "representation"]},
    "geodesic_ece": {"pattern": "**/Full_ECE_data.npz", "higher_is_better": False, "block_cols": ["dataset", "projection", "run", "opponent"]},
    "geodesic_dist": {"pattern": "**/Full_Dist_data.npz", "higher_is_better": False, "block_cols": ["dataset", "projection", "run", "opponent"]},
    "dist_diffs": {"pattern": "**/Dist_Diffs_data.npz", "higher_is_better": False, "block_cols": ["dataset", "diff_scope", "diff_key", "run_model"]},
}
WEIGHTWATCHER_METRIC_CANDIDATES = ["alpha", "log_norm", "rand_distance", "mp_softrank", "stable_rank", "num_spikes"]

def _safe_model_name_from_path(path: Path) -> Optional[str]:
    parts = path.parts
    if "test" in parts:
        idx = parts.index("test")
        if idx >= 1:
            return parts[idx - 1]
    if len(parts) >= 3:
        return parts[-3]
    return None

def _safe_dataset_name_from_path(path: Path) -> Optional[str]:
    parts = path.parts
    if len(parts) >= 4:
        return parts[-4]
    return None

def benjamini_hochberg(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out

def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def make_score_matrix(df: pd.DataFrame, block_col: str, model_col: str, score_col: str) -> pd.DataFrame:
    mat = df.pivot_table(index=block_col, columns=model_col, values=score_col, aggfunc="mean")
    return mat.dropna(axis=0, how="any")

def scores_to_ranks(score_matrix: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    arr = score_matrix.to_numpy(dtype=float)
    ranks = np.zeros_like(arr, dtype=float)
    for i in range(arr.shape[0]):
        ranks[i] = rankdata(-arr[i], method="average") if higher_is_better else rankdata(arr[i], method="average")
    return pd.DataFrame(ranks, index=score_matrix.index, columns=score_matrix.columns)

def run_friedman_test(rank_matrix: pd.DataFrame) -> Dict:
    stat, p = friedmanchisquare(*[rank_matrix[c].values for c in rank_matrix.columns])
    return {"friedman_statistic": float(stat), "friedman_pvalue": float(p), "n_blocks": int(rank_matrix.shape[0]), "n_models": int(rank_matrix.shape[1])}

def run_nemenyi(score_matrix: pd.DataFrame) -> pd.DataFrame:
    pvals = sp.posthoc_nemenyi_friedman(score_matrix.to_numpy())
    pvals.index = score_matrix.columns
    pvals.columns = score_matrix.columns
    return pvals

def build_pairwise_table(avg_ranks: pd.Series, nemenyi_p: pd.DataFrame, score_matrix: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    models = list(avg_ranks.index)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if j <= i:
                continue
            try:
                _, p_w = wilcoxon(score_matrix[a].to_numpy(dtype=float), score_matrix[b].to_numpy(dtype=float), zero_method="wilcox", alternative="two-sided", mode="auto")
            except Exception:
                p_w = np.nan
            rows.append({"model_a": a, "model_b": b, "avg_rank_a": float(avg_ranks[a]), "avg_rank_b": float(avg_ranks[b]), "rank_diff": float(abs(avg_ranks[a]-avg_ranks[b])), "nemenyi_p": float(nemenyi_p.loc[a,b]), "wilcoxon_p": float(p_w) if not pd.isna(p_w) else np.nan, "significant": bool(nemenyi_p.loc[a,b] < alpha)})
    pairwise = pd.DataFrame(rows).sort_values(["nemenyi_p", "rank_diff"], ascending=[True, False])
    if not pairwise.empty:
        pairwise["wilcoxon_q"] = benjamini_hochberg(pairwise["wilcoxon_p"].fillna(1.0).to_numpy())
    return pairwise

def summarize_best_group(avg_ranks: pd.Series, pairwise_df: pd.DataFrame) -> pd.DataFrame:
    best_model = avg_ranks.sort_values().index[0]
    rows = []
    for model in avg_ranks.index:
        if model == best_model:
            rows.append({"model": model, "average_rank": float(avg_ranks[model]), "best_group_member": True, "p_vs_best": 1.0})
            continue
        pair = pairwise_df[((pairwise_df["model_a"] == best_model) & (pairwise_df["model_b"] == model)) | ((pairwise_df["model_b"] == best_model) & (pairwise_df["model_a"] == model))]
        p_vs_best = float(pair.iloc[0]["nemenyi_p"]) if len(pair) else np.nan
        sig_vs_best = bool(pair.iloc[0]["significant"]) if len(pair) else False
        rows.append({"model": model, "average_rank": float(avg_ranks[model]), "best_group_member": not sig_vs_best, "p_vs_best": p_vs_best})
    return pd.DataFrame(rows).sort_values("average_rank")

def save_cd_diagram(score_matrix: pd.DataFrame, output_path: Path, higher_is_better: bool, alpha: float) -> bool:
    if not HAVE_AEON:
        return False
    fig = plt.figure(figsize=(10, 4))
    plot_critical_difference(scores=score_matrix.to_numpy(), labels=list(score_matrix.columns), alpha=alpha, reverse=higher_is_better)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True

def normalize_block_columns(df: pd.DataFrame, block_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in block_cols:
        if c not in out.columns:
            out[c] = "unknown"
        out[c] = out[c].astype(str)
    out["block_id"] = out[block_cols].agg("|".join, axis=1)
    return out

def rank_analysis(df: pd.DataFrame, metric_name: str, higher_is_better: bool, alpha: float, output_dir: Path) -> Dict:
    validate_columns(df, ["block_id", "model", "score"])
    score_matrix = make_score_matrix(df, "block_id", "model", "score")
    if score_matrix.shape[0] < 2 or score_matrix.shape[1] < 3:
        raise ValueError(f"Need at least 2 blocks and 3 models for {metric_name}; got {score_matrix.shape}")
    rank_matrix = scores_to_ranks(score_matrix, higher_is_better)
    avg_ranks = rank_matrix.mean(axis=0).sort_values()
    friedman = run_friedman_test(rank_matrix)
    nemenyi_p = run_nemenyi(score_matrix)
    pairwise_df = build_pairwise_table(avg_ranks, nemenyi_p, score_matrix, alpha)
    best_group_df = summarize_best_group(avg_ranks, pairwise_df)
    score_matrix.to_csv(output_dir / f"{metric_name}.score_matrix.csv")
    rank_matrix.to_csv(output_dir / f"{metric_name}.rank_matrix.csv")
    avg_ranks.rename("average_rank").to_csv(output_dir / f"{metric_name}.avg_ranks.csv", header=True)
    nemenyi_p.to_csv(output_dir / f"{metric_name}.nemenyi_pvalues.csv")
    pairwise_df.to_csv(output_dir / f"{metric_name}.pairwise_tests.csv", index=False)
    best_group_df.to_csv(output_dir / f"{metric_name}.best_group.csv", index=False)
    cd_saved = save_cd_diagram(score_matrix, output_dir / f"{metric_name}.critical_difference.png", higher_is_better, alpha)
    return {"metric_name": metric_name, "higher_is_better": higher_is_better, "friedman": friedman, "n_blocks_after_dropna": int(score_matrix.shape[0]), "n_models": int(score_matrix.shape[1]), "best_model": str(avg_ranks.index[0]), "cd_diagram_saved": bool(cd_saved)}

def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _coerce_numeric(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", "")
    mult = 1.0
    for suf, m in [("K", 1e3), ("M", 1e6), ("G", 1e9), ("T", 1e12)]:
        if s.endswith(suf):
            mult = m
            s = s[:-1]
            break
    try:
        return float(s) * mult
    except Exception:
        return np.nan

def collect_flops(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.glob(ANALYSIS_SPECS["flops"]["pattern"]):
        obj = _load_pickle(path)
        rows.append({"analysis": "flops", "dataset": obj.get("dataset") or _safe_dataset_name_from_path(path), "model": obj.get("encoder") or _safe_model_name_from_path(path), "flops": _coerce_numeric(obj.get("flops")), "macs": _coerce_numeric(obj.get("macs")), "params": _coerce_numeric(obj.get("params")), "emissions_g_co2eq": obj.get("emissions_g_co2eq"), "source": str(path)})
    return pd.DataFrame(rows)

def collect_weightwatcher(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.glob(ANALYSIS_SPECS["weightwatcher"]["pattern"]):
        obj = _load_pickle(path)
        df = obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        model = _safe_model_name_from_path(path)
        dataset = _safe_dataset_name_from_path(path)
        available = [c for c in WEIGHTWATCHER_METRIC_CANDIDATES if c in df.columns]
        if not available:
            available = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:5]
        for metric in available:
            vals = pd.to_numeric(df[metric], errors="coerce").dropna()
            if len(vals):
                rows.append({"analysis": "weightwatcher", "dataset": dataset, "model": model, "metric": metric, "score": float(vals.mean()), "source": str(path)})
    return pd.DataFrame(rows)

def collect_silhouette(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.glob(ANALYSIS_SPECS["silhouette"]["pattern"]):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if {"projection", "model", "mean_sil"}.issubset(df.columns):
            tmp = df[["projection", "model", "mean_sil"]].copy()
            tmp["dataset"] = _safe_dataset_name_from_path(path)
            tmp["score"] = pd.to_numeric(tmp["mean_sil"], errors="coerce")
            rows.extend(tmp.assign(analysis="silhouette", source=str(path))[["analysis","dataset","projection","model","score","source"]].to_dict("records"))
        elif {"projection", "model", "silhouette"}.issubset(df.columns):
            tmp = df.groupby(["projection", "model"], as_index=False)["silhouette"].mean()
            tmp["dataset"] = _safe_dataset_name_from_path(path)
            tmp["score"] = tmp["silhouette"]
            rows.extend(tmp.assign(analysis="silhouette", source=str(path))[["analysis","dataset","projection","model","score","source"]].to_dict("records"))
    return pd.DataFrame(rows)

def collect_intrinsic_dimension(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.glob(ANALYSIS_SPECS["intrinsic_dimension"]["pattern"]):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not {"method", "k", "global_id"}.issubset(df.columns):
            continue
        model = df["model"].iloc[0] if "model" in df.columns and len(df) else _safe_model_name_from_path(path)
        dataset = _safe_dataset_name_from_path(path)
        for _, r in df.groupby(["method", "k"], as_index=False)["global_id"].mean().iterrows():
            rows.append({"analysis": "intrinsic_dimension", "dataset": dataset, "model": model, "method": str(r["method"]), "k": str(r["k"]), "score": float(r["global_id"]), "source": str(path)})
    return pd.DataFrame(rows)

def collect_geodesic_pairwise(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.glob(ANALYSIS_SPECS["geodesic_pairwise"]["pattern"]):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not {"model_a", "model_b"}.issubset(df.columns):
            continue
        dataset = _safe_dataset_name_from_path(path)
        anchor = path.stem.split("_anchored_")[0]
        representation = df["representation"].iloc[0] if "representation" in df.columns and len(df) else "unknown"
        metric_col = "pairwise_distance_spearman" if "pairwise_distance_spearman" in df.columns else None
        if metric_col is None:
            continue
        for _, r in df.iterrows():
            rows.append({"analysis": "geodesic_pairwise", "dataset": dataset, "anchor": anchor, "representation": representation, "model": r["model_a"], "opponent": r["model_b"], "score": float(r[metric_col]), "source": str(path)})
            rows.append({"analysis": "geodesic_pairwise", "dataset": dataset, "anchor": anchor, "representation": representation, "model": r["model_b"], "opponent": r["model_a"], "score": float(r[metric_col]), "source": str(path)})
    return pd.DataFrame(rows)

def _flatten_nested_metric_npz(npz_path: Path, key: str, analysis_name: str) -> pd.DataFrame:
    obj = np.load(npz_path, allow_pickle=True)
    arr = obj[key]
    rows = []
    dataset = _safe_dataset_name_from_path(npz_path)
    for run_idx, run_obj in enumerate(arr):
        if isinstance(run_obj, np.ndarray) and run_obj.shape == ():
            run_obj = run_obj.item()
        if not isinstance(run_obj, dict):
            continue
        for projection, model_dict in run_obj.items():
            if not isinstance(model_dict, dict):
                continue
            for model, opp_dict in model_dict.items():
                if not isinstance(opp_dict, dict):
                    continue
                for opponent, value in opp_dict.items():
                    rows.append({"analysis": analysis_name, "dataset": dataset, "projection": projection, "run": str(run_idx), "model": model, "opponent": opponent, "score": float(value), "source": str(npz_path)})
    return pd.DataFrame(rows)

def collect_geodesic_ece(root: Path) -> pd.DataFrame:
    dfs = []
    for path in root.glob(ANALYSIS_SPECS["geodesic_ece"]["pattern"]):
        df = _flatten_nested_metric_npz(path, "ece_dicts", "geodesic_ece")
        if len(df):
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def collect_geodesic_dist(root: Path) -> pd.DataFrame:
    dfs = []
    for path in root.glob(ANALYSIS_SPECS["geodesic_dist"]["pattern"]):
        df = _flatten_nested_metric_npz(path, "dist_dicts", "geodesic_dist")
        if len(df):
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _flatten_dist_diffs_npz(npz_path: Path) -> pd.DataFrame:
    obj = np.load(npz_path, allow_pickle=True)
    arr = obj["dist_diffs"]
    rows = []
    dataset = _safe_dataset_name_from_path(npz_path)
    if isinstance(arr, np.ndarray) and arr.shape == ():
        arr = arr.item()
    if isinstance(arr, dict):
        arr = [arr]
    for run_model_idx, item in enumerate(arr):
        if isinstance(item, np.ndarray) and item.shape == ():
            item = item.item()
        if not isinstance(item, dict):
            continue
        for scope in ["same_proj", "cross_proj"]:
            scope_obj = item.get(scope, {})
            if isinstance(scope_obj, list):
                iterable = enumerate(scope_obj)
                for j, sub in iterable:
                    if isinstance(sub, np.ndarray) and sub.shape == ():
                        sub = sub.item()
                    if not isinstance(sub, dict):
                        continue
                    for diff_key, value in sub.items():
                        rows.append({
                            "analysis": "dist_diffs",
                            "dataset": dataset,
                            "diff_scope": scope,
                            "diff_key": str(diff_key),
                            "run_model": str(run_model_idx),
                            "model": str(run_model_idx),
                            "score": float(value),
                            "source": str(npz_path),
                        })
            elif isinstance(scope_obj, dict):
                for diff_key, value in scope_obj.items():
                    rows.append({
                        "analysis": "dist_diffs",
                        "dataset": dataset,
                        "diff_scope": scope,
                        "diff_key": str(diff_key),
                        "run_model": str(run_model_idx),
                        "model": str(run_model_idx),
                        "score": float(value),
                        "source": str(npz_path),
                    })
    return pd.DataFrame(rows)

def collect_dist_diffs(root: Path) -> pd.DataFrame:
    dfs = []
    for path in root.glob(ANALYSIS_SPECS["dist_diffs"]["pattern"]):
        df = _flatten_dist_diffs_npz(path)
        if len(df):
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def collect_all_analyses(root: Path) -> Dict[str, pd.DataFrame]:
    return {"flops": collect_flops(root), "weightwatcher": collect_weightwatcher(root), "silhouette": collect_silhouette(root), "intrinsic_dimension": collect_intrinsic_dimension(root), "geodesic_pairwise": collect_geodesic_pairwise(root), "geodesic_ece": collect_geodesic_ece(root), "geodesic_dist": collect_geodesic_dist(root), "dist_diffs": collect_dist_diffs(root)}

def build_meta_ranking(metric_summaries: List[Dict], output_dir: Path) -> pd.DataFrame:
    rows = []
    for item in metric_summaries:
        avg_path = output_dir / f"{item['metric_name']}.avg_ranks.csv"
        if not avg_path.exists():
            continue
        avg_df = pd.read_csv(avg_path)
        for _, r in avg_df.iterrows():
            rows.append({"metric_name": item["metric_name"], "model": r.iloc[0], "average_rank": float(r.iloc[1])})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    meta = df.pivot_table(index="metric_name", columns="model", values="average_rank", aggfunc="mean").dropna(axis=0, how="any")
    if meta.shape[0] < 2 or meta.shape[1] < 3:
        return pd.DataFrame()
    avg_ranks = meta.mean(axis=0).sort_values()
    stat, p = friedmanchisquare(*[meta[c].values for c in meta.columns])
    out = avg_ranks.rename("meta_average_rank").reset_index().rename(columns={"index": "model"})
    out["friedman_statistic"] = float(stat)
    out["friedman_pvalue"] = float(p)
    out.to_csv(output_dir / "meta_ranking.csv", index=False)
    meta.to_csv(output_dir / "meta_rank_matrix.csv")
    return out



def _coerce_path(value: Any, field_name: str) -> Path:
    if value is None:
        raise ValueError(f"Missing required config field: {field_name}")
    return Path(value)

def load_config(config_path: Path) -> Dict[str, Any]:
    cfg = read_yaml(str(config_path))
    if cfg is None:
        raise ValueError(f"Config file is empty or unreadable: {config_path}")
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML config must parse to a mapping/dictionary")
    return cfg

def run_demsar(cfg):
 
    root_dir = _coerce_path(cfg.get("root_dir"), "root_dir")
    output_dir = Path(cfg.get("output_dir", "output/demsar_analysis"))
    alpha = float(cfg.get("alpha", 0.05))
    metric_overrides = cfg.get("metric_overrides", {}) or {}

    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_frames = collect_all_analyses(root_dir)
    metric_summaries = []

    flops_df = analysis_frames["flops"]
    if not flops_df.empty:
        flops_df = normalize_block_columns(flops_df, ["dataset"])
        for metric in ["flops", "macs", "params", "emissions_g_co2eq"]:
            if metric in flops_df.columns and flops_df[metric].notna().any():
                tmp = flops_df[["block_id", "model", metric]].dropna().rename(columns={metric: "score"})
                metric_summaries.append(rank_analysis(tmp, f"flops__{metric}", False, alpha, output_dir))

    ww = analysis_frames["weightwatcher"]
    if not ww.empty:
        for metric, sub in ww.groupby("metric"):
            hib = bool(metric_overrides.get(f"weightwatcher::{metric}", True))
            sub = normalize_block_columns(sub, ["dataset"])
            tmp = sub[["block_id", "model", "score"]].dropna()
            if len(tmp):
                metric_summaries.append(rank_analysis(tmp, f"weightwatcher__{metric}", hib, alpha, output_dir))

    sil = analysis_frames["silhouette"]
    if not sil.empty:
        sil = normalize_block_columns(sil, ["dataset", "projection"])
        metric_summaries.append(rank_analysis(sil[["block_id", "model", "score"]].dropna(), "silhouette__mean_sil", True, alpha, output_dir))

    ide = analysis_frames["intrinsic_dimension"]
    if not ide.empty:
        ide = normalize_block_columns(ide, ["dataset", "method", "k"])
        metric_summaries.append(rank_analysis(ide[["block_id", "model", "score"]].dropna(), "intrinsic_dimension__global_id", False, alpha, output_dir))

    geo = analysis_frames["geodesic_pairwise"]
    if not geo.empty:
        geo = normalize_block_columns(geo, ["dataset", "anchor", "representation"])
        metric_summaries.append(rank_analysis(geo[["block_id", "model", "score"]].dropna(), "geodesic_pairwise__pairwise_distance_spearman", True, alpha, output_dir))

    ece = analysis_frames["geodesic_ece"]
    if not ece.empty:
        ece = normalize_block_columns(ece, ["dataset", "projection", "run", "opponent"])
        metric_summaries.append(rank_analysis(ece[["block_id", "model", "score"]].dropna(), "geodesic_ece__ece", False, alpha, output_dir))

    dist = analysis_frames["geodesic_dist"]
    if not dist.empty:
        dist = normalize_block_columns(dist, ["dataset", "projection", "run", "opponent"])
        metric_summaries.append(rank_analysis(dist[["block_id", "model", "score"]].dropna(), "geodesic_dist__cmds_distance", False, alpha, output_dir))

    dist_diffs = analysis_frames["dist_diffs"]
    if not dist_diffs.empty:
        dist_diffs = normalize_block_columns(dist_diffs, ["dataset", "diff_scope", "diff_key", "run_model"])
        metric_summaries.append(rank_analysis(dist_diffs[["block_id", "model", "score"]].dropna(), "dist_diffs__rmsd", False, alpha, output_dir))

    pd.DataFrame(metric_summaries).to_csv(output_dir / "metric_analysis_summary.csv", index=False)
    build_meta_ranking(metric_summaries, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config file.")
    args = parser.parse_args()
    cfg = read_yaml(args.yaml)
    run_demsar(cfg)

