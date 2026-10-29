"""Figures for the cross-perspective synthesis (matplotlib, headless)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram


def _label(name: str, display_names: Optional[Dict[str, str]]) -> str:
    return (display_names or {}).get(name, name)


def _text_color(im, value: float) -> str:
    """Black or white annotation depending on the luminance of the rendered cell."""
    r, g, b, _ = im.cmap(im.norm(value))
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else "white"


def _save(fig, out_dir: str, stem: str, formats: Sequence[str]) -> None:
    for fmt in formats:
        fig.savefig(os.path.join(out_dir, f"{stem}.{fmt}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_variance_decomposition(var_df: pd.DataFrame, out_dir: str, formats=("pdf", "png")) -> None:
    df = var_df[var_df["source"] != "total"].sort_values("share", ascending=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.barh(df["source"], 100.0 * df["share"], color="#3b6ea5")
    for y, v in enumerate(100.0 * df["share"]):
        ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Share of total sum of squares (%)")
    ax.set_xlim(0, max(5.0, 100.0 * df["share"].max() * 1.18))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir, "fig_variance", formats)


def plot_silhouette_distribution(cond_df: pd.DataFrame, order: Sequence[str], out_dir: str,
                                 display_names=None, formats=("pdf", "png")) -> None:
    data = [cond_df.loc[cond_df["model"] == m, "silhouette"].to_numpy() for m in order]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.55 * len(order)), 3.2))
    ax.boxplot(data, showfliers=False, widths=0.6)
    markers = {p: mk for p, mk in zip(sorted(cond_df["projection"].unique()), "os^dvP")}
    for i, m in enumerate(order, start=1):
        sub = cond_df[cond_df["model"] == m]
        for p, g in sub.groupby("projection"):
            jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(g))
            ax.scatter(i + jitter, g["silhouette"], s=10, marker=markers[p], alpha=0.7,
                       label=p.upper() if i == 1 else None)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([_label(m, display_names) for m in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Silhouette score")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir, "fig_robustness_distribution", formats)


def plot_heatmap(mat: np.ndarray, models: Sequence[str], out_dir: str, stem: str, title: str,
                 display_names=None, cmap: str = "viridis", formats=("pdf", "png")) -> None:
    labels = [_label(m, display_names) for m in models]
    fig, ax = plt.subplots(figsize=(0.45 * len(models) + 2.0, 0.45 * len(models) + 1.5))
    im = ax.imshow(mat, cmap=cmap, vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    if len(models) <= 14:
        for i in range(len(models)):
            for j in range(len(models)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                            color=_text_color(im, mat[i, j]))
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out_dir, stem, formats)


def plot_dendrograms(link_res: np.ndarray, link_dist: np.ndarray, models: Sequence[str], out_dir: str,
                     display_names=None, formats=("pdf", "png")) -> None:
    labels = [_label(m, display_names) for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for ax, link, title in zip(axes, [link_res, link_dist],
                               ["Normalized residual", "Normalized distance"]):
        dendrogram(link, labels=labels, ax=ax, leaf_rotation=60, leaf_font_size=7,
                   color_threshold=0.7 * np.max(link[:, 2]))
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Average-linkage height")
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig_dendrograms", formats)


def plot_alpha_boxplot(alpha_df: pd.DataFrame, order: Sequence[str], out_dir: str,
                       lo: float = 2.0, hi: float = 5.0, display_names=None, formats=("pdf", "png")) -> None:
    data = [alpha_df.loc[alpha_df["model"] == m, "alpha"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            for m in order]
    fig, ax = plt.subplots(figsize=(max(5.0, 0.55 * len(order)), 3.2))
    ax.axhspan(lo, hi, color="#cfe8cf", alpha=0.6, zorder=0, label=f"{lo:g} <= alpha <= {hi:g}")
    ax.boxplot(data, showfliers=True, widths=0.6, flierprops={"markersize": 2})
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([_label(m, display_names) for m in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"Layer power-law exponent $\alpha$")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir, "fig_alpha_box", formats)


def plot_alpha_pairs(alpha_df: pd.DataFrame, pairs: Sequence[Sequence[str]], out_dir: str,
                     lo: float = 2.0, hi: float = 5.0, display_names=None, formats=("pdf", "png")) -> None:
    pairs = [p for p in pairs if len(p) == 2
             and p[0] in set(alpha_df["model"]) and p[1] in set(alpha_df["model"])]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.0 * len(pairs), 3.0), squeeze=False)
    for ax, (a, b) in zip(axes[0], pairs):
        for m, c in [(a, "#3b6ea5"), (b, "#d07a2d")]:
            g = alpha_df[alpha_df["model"] == m].reset_index(drop=True)
            ax.plot(np.arange(len(g)), g["alpha"], marker="o", ms=2.5, lw=0.8, color=c,
                    label=_label(m, display_names))
        ax.axhspan(lo, hi, color="#cfe8cf", alpha=0.6, zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("Layer index")
        ax.set_ylabel(r"$\alpha$")
        ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig_alpha_pairs", formats)


def plot_rank_tau(tau_df: pd.DataFrame, out_dir: str, formats=("pdf", "png")) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    im = ax.imshow(tau_df.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(tau_df)))
    ax.set_yticks(range(len(tau_df)))
    ax.set_xticklabels(tau_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tau_df.index, fontsize=8)
    for i in range(len(tau_df)):
        for j in range(len(tau_df)):
            v = tau_df.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color=_text_color(im, v))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"Kendall $\tau_b$")
    _save(fig, out_dir, "fig_rank_tau", formats)


def plot_rank_parallel(rank_df: pd.DataFrame, axes_cols: Sequence[str], out_dir: str,
                       display_names=None, formats=("pdf", "png")) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    cmap = plt.get_cmap("tab20")
    x = np.arange(len(axes_cols))
    for k, (_, row) in enumerate(rank_df.iterrows()):
        y = row[list(axes_cols)].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", ms=3, lw=1.0, color=cmap(k % 20), label=_label(row["model"], display_names))
    ax.set_xticks(x)
    ax.set_xticklabels(axes_cols, fontsize=8)
    ax.set_ylabel("Rank (1 = best)")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_dir, "fig_rank_parallel", formats)


def plot_tradeoff(df: pd.DataFrame, front_sil: np.ndarray, front_flops: np.ndarray,
                  rho_sil: Dict[str, float], rho_flops: Dict[str, float], out_dir: str,
                  energy_col: str, energy_label: str, display_names=None, formats=("pdf", "png")) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    panels = [
        (axes[0], "robustness_mean", "Mean silhouette", front_sil, rho_sil),
        (axes[1], "gflops", "GFLOPs / sample", front_flops, rho_flops),
    ]
    for ax, ycol, ylabel, front, rho in panels:
        sub = df[[energy_col, ycol, "model"]].dropna()
        ax.scatter(df[energy_col], df[ycol], s=18, color="#3b6ea5", zorder=3)
        f = df[front].sort_values(energy_col)
        if len(f) > 1 and ycol == "robustness_mean":
            ax.step(f[energy_col], f[ycol], where="post", color="#d07a2d", lw=1.0, label="Pareto front")
        ax.scatter(f[energy_col], f[ycol], s=40, facecolor="none", edgecolor="#d07a2d", zorder=4)
        for _, r in sub.iterrows():
            ax.annotate(_label(r["model"], display_names), (r[energy_col], r[ycol]),
                        fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.set_xscale("log")
        yv = sub[ycol].to_numpy(dtype=float)
        if ycol == "gflops" and len(yv) and np.nanmin(yv) > 0 and np.nanmax(yv) / np.nanmin(yv) > 20:
            ax.set_yscale("log")
        ax.set_xlabel(energy_label)
        ax.set_ylabel(ylabel)
        ax.set_title(rf"Spearman $\rho$ = {rho['rho']:.2f} (p = {rho['p_value']:.2g})", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_dir, "fig_tradeoff", formats)


def plot_grid(df: pd.DataFrame, row: str, col: str, value: str, out_dir: str, stem: str, title: str,
              vmin: Optional[float] = None, vmax: Optional[float] = None, cmap: str = "viridis",
              fmt: str = "{:.2f}", formats=("pdf", "png")) -> None:
    """Annotated heatmap of a long table (row x col -> value)."""
    if df.empty:
        return
    piv = df.pivot_table(index=row, columns=col, values=value, aggfunc="mean")
    arr = piv.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(0.9 * piv.shape[1] + 2.4, 0.42 * piv.shape[0] + 1.4))
    im = ax.imshow(arr, cmap=cmap, aspect="auto",
                   vmin=np.nanmin(arr) if vmin is None else vmin,
                   vmax=np.nanmax(arr) if vmax is None else vmax)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_yticks(range(piv.shape[0]))
    ax.set_xticklabels([str(c) for c in piv.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels([str(r) for r in piv.index], fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, fmt.format(arr[i, j]), ha="center", va="center", fontsize=7,
                        color=_text_color(im, arr[i, j]))
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out_dir, stem, formats)
