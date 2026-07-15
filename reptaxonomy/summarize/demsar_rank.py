from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, rankdata

try:
    from aeon.visualisation import plot_critical_difference
    HAVE_AEON = True
except Exception:
    HAVE_AEON = False


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def make_score_matrix(
    df: pd.DataFrame,
    dataset_col: str,
    model_col: str,
    score_col: str,
) -> pd.DataFrame:
    """
    Wide matrix: rows=datasets, cols=models, values=scores.
    """
    mat = df.pivot_table(
        index=dataset_col,
        columns=model_col,
        values=score_col,
        aggfunc="mean",
    )
    mat = mat.dropna(axis=0, how="any")
    return mat


def scores_to_ranks(score_matrix: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    arr = score_matrix.to_numpy(dtype=float)
    ranks = np.zeros_like(arr, dtype=float)

    for i in range(arr.shape[0]):
        row = arr[i]
        if higher_is_better:
            ranks[i] = rankdata(-row, method="average")
        else:
            ranks[i] = rankdata(row, method="average")

    return pd.DataFrame(ranks, index=score_matrix.index, columns=score_matrix.columns)


def run_friedman_test(rank_matrix: pd.DataFrame) -> dict:
    samples = [rank_matrix[c].values for c in rank_matrix.columns]
    stat, p = friedmanchisquare(*samples)
    return {
        "friedman_statistic": float(stat),
        "friedman_pvalue": float(p),
        "n_datasets": int(rank_matrix.shape[0]),
        "n_models": int(rank_matrix.shape[1]),
    }


def run_nemenyi(score_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    scikit-posthocs expects rows=blocks/datasets, cols=groups/models.
    """
    pvals = sp.posthoc_nemenyi_friedman(score_matrix.to_numpy())
    pvals.index = score_matrix.columns
    pvals.columns = score_matrix.columns
    return pvals


def build_pairwise_table(avg_ranks: pd.Series, nemenyi_p: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    models = list(avg_ranks.index)

    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if j <= i:
                continue
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "avg_rank_a": float(avg_ranks[a]),
                    "avg_rank_b": float(avg_ranks[b]),
                    "rank_diff": float(abs(avg_ranks[a] - avg_ranks[b])),
                    "nemenyi_p": float(nemenyi_p.loc[a, b]),
                    "significant": bool(nemenyi_p.loc[a, b] < alpha),
                }
            )

    return pd.DataFrame(rows).sort_values(["nemenyi_p", "rank_diff"], ascending=[True, False])


def summarize_best_group(avg_ranks: pd.Series, pairwise_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Heuristic summary: identify models not significantly worse than the best-ranked model.
    """
    best_model = avg_ranks.sort_values().index[0]
    rows = []

    for model in avg_ranks.index:
        if model == best_model:
            sig_vs_best = False
            p_vs_best = 1.0
        else:
            pair = pairwise_df[
                ((pairwise_df["model_a"] == best_model) & (pairwise_df["model_b"] == model)) |
                ((pairwise_df["model_b"] == best_model) & (pairwise_df["model_a"] == model))
            ]
            if len(pair) == 0:
                sig_vs_best = False
                p_vs_best = np.nan
            else:
                sig_vs_best = bool(pair.iloc[0]["significant"])
                p_vs_best = float(pair.iloc[0]["nemenyi_p"])

        rows.append(
            {
                "model": model,
                "average_rank": float(avg_ranks[model]),
                "best_group_member": not sig_vs_best,
                "p_vs_best": p_vs_best,
            }
        )

    return pd.DataFrame(rows).sort_values("average_rank")


def save_cd_diagram(
    score_matrix: pd.DataFrame,
    output_path: Path,
    higher_is_better: bool,
    alpha: float,
) -> bool:
    """
    Use aeon if available. Its CD function computes average ranks, Friedman,
    and Nemenyi-style cliques for visualization.[web:113]
    """
    if not HAVE_AEON:
        return False

    fig = plt.figure(figsize=(10, 4))
    labels = list(score_matrix.columns)
    scores = score_matrix.to_numpy()

    plot_critical_difference(
        scores=scores,
        labels=labels,
        alpha=alpha,
        reverse=higher_is_better,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Demšar-style Friedman + Nemenyi ranking analysis."
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--dataset_col", type=str, default="dataset")
    parser.add_argument("--model_col", type=str, default="model")
    parser.add_argument("--score_col", type=str, required=True)
    parser.add_argument("--metric_name", type=str, required=True)
    parser.add_argument("--higher_is_better", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output_dir", type=Path, default=Path("output/demsar_analysis"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    validate_columns(df, [args.dataset_col, args.model_col, args.score_col])

    score_matrix = make_score_matrix(
        df=df,
        dataset_col=args.dataset_col,
        model_col=args.model_col,
        score_col=args.score_col,
    )

    if score_matrix.shape[0] < 2 or score_matrix.shape[1] < 3:
        raise ValueError(
            "Need at least 2 datasets and 3 models for a meaningful Friedman/Nemenyi analysis."
        )

    
