from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.sparse import csr_array, load_npz
from graspologic.embed import AdjacencySpectralEmbed, ClassicalMDS, OmnibusEmbed
from graspologic.simulations import rdpg

from reptaxonomy.util.general_utils import read_yaml
from reptaxonomy.util.artifacts import (
    ensure_dict,
    get_dataset_name,
    get_model_names,
    get_projections,
    dataset_out_root,
    projection_analysis_dir,
    knn_graph_path,
    run_manifest_path,
    analysis_manifest_path,
    build_analysis_manifest,
    load_json,
    save_json,
)

logger = logging.getLogger(__name__)

COLORS = [
    "red", "black", "blue", "orange", "green", "magenta", "cyan", "olive", "purple",
    "gray", "pink", "brown", "darkcyan", "chocolate", "lightgreen", "gold", "deeppink",
    "lightgrey", "rosybrown", "maroon", "coral", "sandybrown",
]


def expected_calibration_error(expected: np.ndarray, observed: np.ndarray, total_count: np.ndarray | None = None) -> float:
    abs_diff = np.abs(observed - expected)
    if total_count is not None:
        weights = total_count / total_count.sum()
        return float((abs_diff * weights).sum())
    return float(abs_diff.mean())


def plot_calibration(
    cdf: np.ndarray,
    out_path: str,
    show_ece: bool = True,
    show_ideal: bool = True,
    show_area: bool = True,
) -> float:
    fig, ax = plt.subplots(figsize=(5, 5))
    linspace = np.linspace(0, 1, num=len(cdf))
    ax.plot(linspace, cdf, label="observed")

    if show_ideal:
        ax.plot(linspace, linspace, label="null / uniform (y=x)")
    if show_area:
        ax.fill_between(linspace, linspace, cdf, alpha=0.2)

    ece = expected_calibration_error(linspace, cdf) if show_ece else -1.0
    if show_ece:
        ax.text(0.95, 0.05, f"Sum of Residuals: {ece:.3f}", ha="right", va="center", transform=ax.transAxes)

    ax.set_xlabel("Expected Proportion", fontsize=12)
    ax.set_ylabel("Observed Proportion", fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(frameon=False)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return ece


def get_cdf(pvalues: np.ndarray, num: int = 500) -> np.ndarray:
    linspace = np.linspace(0, 1, num=num)
    pvalues = np.asarray(pvalues)
    return np.searchsorted(np.sort(pvalues), linspace, side="right") / pvalues.size


def bootstrap_null_from_latents(
    graph: csr_array,
    ase_latents: np.ndarray,
    number_of_bootstraps: int = 10,
    n_components: int | None = None,
    acorn: int | None = None,
) -> np.ndarray:
    if acorn is not None:
        np.random.seed(acorn)

    if n_components is None:
        n_components = ase_latents.shape[1]

    n = ase_latents.shape[0]
    distances = np.zeros((n, number_of_bootstraps), dtype=np.float32)
    omni = OmnibusEmbed(n_components=n_components)

    for b in range(number_of_bootstraps):
        graph_b = csr_array(rdpg(ase_latents, directed=False))
        boot_latents = omni.fit_transform([graph, graph_b])
        distances[:, b] = np.linalg.norm(boot_latents[0] - boot_latents[1], axis=1)

    return distances


def compute_pairwise_node_stats(omni_embds: np.ndarray, model_names: List[str]) -> Dict[Tuple[str, str], np.ndarray]:
    stats: Dict[Tuple[str, str], np.ndarray] = {}
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if j < i:
                continue
            vals = np.linalg.norm(omni_embds[i] - omni_embds[j], axis=1)
            stats[(m1, m2)] = vals
            stats[(m2, m1)] = vals
    return stats


def build_dist_mtx_cached(model_names: List[str], knn_graphs: Dict[str, csr_array]) -> np.ndarray:
    m = len(model_names)
    dist_matrix = np.zeros((m, m), dtype=np.float32)
    omni = OmnibusEmbed(n_components=2)

    for i, name_i in enumerate(model_names):
        for j in range(i + 1, m):
            name_j = model_names[j]
            omni_embds = omni.fit_transform([knn_graphs[name_i], knn_graphs[name_j]])
            num = np.linalg.norm(omni_embds[0] - omni_embds[1])
            den = np.linalg.norm(omni_embds[0] + omni_embds[1]) + 1e-8
            temp_dist = num / den
            dist_matrix[i, j] = temp_dist
            dist_matrix[j, i] = temp_dist

    return dist_matrix


def plot_multi_embed(omni_embds: np.ndarray, model_names: List[str], out_dir: str) -> None:
    fig, ax = plt.subplots()
    for i, model_name in enumerate(model_names):
        ax.scatter(
            omni_embds[i, :, 0],
            omni_embds[i, :, 1],
            c=COLORS[i % len(COLORS)],
            label=model_name,
            s=8,
        )
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(out_dir, "Embed_Scatter_multi.png"), bbox_inches="tight")
    plt.close(fig)


def plot_single_embeds(omni_embds: np.ndarray, model_names: List[str], out_dir: str) -> None:
    for i, model_name in enumerate(model_names):
        fig, ax = plt.subplots()
        ax.scatter(
            omni_embds[i, :, 0],
            omni_embds[i, :, 1],
            c=COLORS[i % len(COLORS)],
            s=8,
        )
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.savefig(os.path.join(out_dir, f"Embed_Scatter_{model_name}.png"), bbox_inches="tight")
        plt.close(fig)


def plot_null_hyp_scatter(
    omni_embds: np.ndarray,
    p_values: np.ndarray,
    idx: np.ndarray,
    i: int,
    j: int,
    model_name: str,
    model_name2: str,
    out_dir: str,
) -> None:
    fig, ax = plt.subplots()
    tmp = omni_embds[:, idx, :]

    for d in range(tmp.shape[1]):
        ax.scatter(tmp[j, d, 0], tmp[j, d, 1], color=COLORS[j % len(COLORS)], s=8)
        ax.scatter(
            tmp[i, d, 0],
            tmp[i, d, 1],
            color=COLORS[i % len(COLORS)],
            alpha=max(0.05, 1 - p_values[d]),
            s=8,
        )

    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.savefig(
        os.path.join(out_dir, f"Embed_Scatter_Null_Hyp_{model_name}_{model_name2}.png"),
        bbox_inches="tight",
    )
    plt.close(fig)


def run_taxonomic_analysis(
    model_names: List[str],
    knn_graphs: Dict[str, csr_array],
    out_dir: str,
    number_of_bootstraps: int = 10,
    n_components: int = 1,
    top_frac: float = 1.0,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    os.makedirs(out_dir, exist_ok=True)

    ordered_graphs = [knn_graphs[m] for m in model_names]
    omni_embds = OmnibusEmbed(n_components=2).fit_transform(ordered_graphs)
    omni_embds = np.squeeze(omni_embds)

    plot_multi_embed(omni_embds, model_names, out_dir)
    plot_single_embeds(omni_embds, model_names, out_dir)

    pairwise_stats = compute_pairwise_node_stats(omni_embds, model_names)

    null_cache: Dict[str, np.ndarray] = {}
    for model_name in model_names:
        ase_latents = AdjacencySpectralEmbed(
            n_components=n_components,
            svd_seed=seed,
        ).fit_transform(knn_graphs[model_name])

        null_cache[model_name] = bootstrap_null_from_latents(
            knn_graphs[model_name],
            ase_latents,
            number_of_bootstraps=number_of_bootstraps,
            n_components=n_components,
            acorn=seed,
        )

    ece_dict = {m: {} for m in model_names}
    dist_dict = {m: {} for m in model_names}

    for i, model_name in enumerate(model_names):
        for j, model_name2 in enumerate(model_names):
            test_statistics = pairwise_stats[(model_name, model_name2)]
            null_dist = null_cache[model_name]
            p_values = (null_dist >= test_statistics[:, None]).mean(axis=1)

            if top_frac < 1.0:
                top_k = max(1, int(len(test_statistics) * top_frac))
                idx = np.argsort(test_statistics)[-top_k:]
                plot_p = p_values[idx]
                plot_idx = idx
            else:
                plot_p = p_values
                plot_idx = np.arange(len(test_statistics))

            plot_null_hyp_scatter(omni_embds, plot_p, plot_idx, i, j, model_name, model_name2, out_dir)

            cdf = get_cdf(p_values, num=500)
            ece = plot_calibration(
                cdf,
                os.path.join(out_dir, f"P_Value_Dist_{model_name}_{model_name2}.png"),
            )
            ece_dict[model_name][model_name2] = ece

    dist_matrix = build_dist_mtx_cached(model_names, knn_graphs)
    cmds_embds = ClassicalMDS(n_components=2).fit_transform(dist_matrix)

    fig, ax = plt.subplots()
    for i, cmds in enumerate(cmds_embds):
        ax.scatter(cmds[0], cmds[1], label=model_names[i], c=COLORS[i % len(COLORS)])
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(out_dir, "Embed_Space_Dist_Mtx.png"), bbox_inches="tight")
    plt.close(fig)

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            dist_dict[m1][m2] = float(np.linalg.norm(cmds_embds[i] - cmds_embds[j]))

    return ece_dict, dist_dict


def summarize_ece_diffs(
    ece_dict_full: Dict[str, Dict[str, Dict[str, float]]],
    model_name: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    diffs_cross_proj: Dict[str, float] = {}
    proj_order = ["umap", "tsne", "pca"]

    valid_proj_order = [p for p in proj_order if p in ece_dict_full]
    for i in range(0, len(valid_proj_order) - 1):
        for j in range(i + 1, len(valid_proj_order)):
            key = valid_proj_order[i] + "_" + valid_proj_order[j]
            vals = []
            for model in ece_dict_full[valid_proj_order[i]].keys():
                for model2 in ece_dict_full[valid_proj_order[i]][model].keys():
                    vals.append(
                        (
                            ece_dict_full[valid_proj_order[i]][model][model2]
                            - ece_dict_full[valid_proj_order[j]][model][model2]
                        ) ** 2
                    )
            diffs_cross_proj[key] = math.sqrt(np.mean(vals)) if vals else np.nan

    diffs_same_proj: Dict[str, float] = {}
    for proj in ece_dict_full.keys():
        tmp = []
        for model in ece_dict_full[proj].keys():
            for model2 in ece_dict_full[proj][model].keys():
                tmp.append(ece_dict_full[proj][model][model2])

        vals = []
        for i in range(0, len(tmp) - 1):
            for j in range(i + 1, len(tmp)):
                vals.append((tmp[i] - tmp[j]) ** 2)
        diffs_same_proj[proj] = math.sqrt(np.mean(vals)) if vals else np.nan

    logger.info("MODEL %s cross_proj_rmsd %s", model_name, diffs_cross_proj)
    logger.info("MODEL %s same_proj_rmsd %s", model_name, diffs_same_proj)
    return diffs_cross_proj, diffs_same_proj


def flatten_nested_matrix(
    run_idx: int,
    projection: str,
    matrix_dict: Dict[str, Dict[str, float]],
    value_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_a, inner in matrix_dict.items():
        for model_b, val in inner.items():
            rows.append({
                "run": run_idx,
                "projection": projection,
                "model_a": model_a,
                "model_b": model_b,
                value_name: float(val),
            })
    return rows


def validate_run_alignment(cfg: Dict[str, Any], run_idx: int, model_names: List[str], projections: List[str]) -> None:
    for model_name in model_names:
        manifest_path = run_manifest_path(cfg, run_idx, model_name=model_name)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing run manifest for model '{model_name}' at: {manifest_path}")

    for projection in projections:
        n_points = []
        for model_name in model_names:
            manifest = load_json(run_manifest_path(cfg, run_idx, model_name=model_name))
            projection_stats = manifest.get("projection_stats", {})
            if projection not in projection_stats:
                raise ValueError(f"Projection '{projection}' missing from manifest for model '{model_name}', run {run_idx}")
            n_points.append(int(projection_stats[projection]["n_points"]))

        if len(set(n_points)) != 1:
            raise ValueError(
                f"Projection '{projection}' run {run_idx} has mismatched point counts across models: {dict(zip(model_names, n_points))}"
            )


def load_knn_graphs_for_projection(
    cfg: Dict[str, Any],
    model_names: List[str],
    run_idx: int,
    projection: str,
) -> Dict[str, csr_array]:
    knn_graphs: Dict[str, csr_array] = {}
    for model_name in model_names:
        path = knn_graph_path(cfg, run_idx, projection, model_name=model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing KNN graph for model '{model_name}' at: {path}")
        knn_graphs[model_name] = load_npz(path).tocsr()
    return knn_graphs


def run_data_kernel_analysis(cfg: Dict[str, Any]) -> None:
    cfg = ensure_dict(cfg, "data_kernel_analysis cfg")

    dataset_name = get_dataset_name(cfg)
    model_names = get_model_names(cfg)
    projections = get_projections(cfg)
    n_runs = int(cfg.get("n_runs", 3))

    out_root = dataset_out_root(cfg)

    dist_dicts = []
    ece_dicts = []

    pairwise_ece_rows: List[Dict[str, Any]] = []
    pairwise_dist_rows: List[Dict[str, Any]] = []
    diff_rows: List[Dict[str, Any]] = []

    for run_idx in range(n_runs):
        validate_run_alignment(cfg, run_idx, model_names, projections)

        ece_dict_full: Dict[str, Dict[str, Dict[str, float]]] = {}
        dist_dict_full: Dict[str, Dict[str, Dict[str, float]]] = {}

        for projection in projections:
            knn_graphs = load_knn_graphs_for_projection(
                cfg=cfg,
                model_names=model_names,
                run_idx=run_idx,
                projection=projection,
            )

            projection_out_dir = projection_analysis_dir(cfg, run_idx, projection)
            ece_dict, dist_dict = run_taxonomic_analysis(
                model_names=model_names,
                knn_graphs=knn_graphs,
                out_dir=projection_out_dir,
                number_of_bootstraps=int(cfg.get("number_of_bootstraps", 10)),
                n_components=int(cfg.get("ase_n_components", 1)),
                top_frac=float(cfg.get("top_frac", 1.0)),
                seed=int(cfg.get("seed", 42)),
            )

            ece_dict_full[projection] = ece_dict
            dist_dict_full[projection] = dist_dict

            pairwise_ece_rows.extend(flatten_nested_matrix(run_idx, projection, ece_dict, "ece"))
            pairwise_dist_rows.extend(flatten_nested_matrix(run_idx, projection, dist_dict, "distance"))

        dist_dicts.append(dist_dict_full)
        ece_dicts.append(ece_dict_full)

        for model_name in model_names:
            diffs_cross_proj, diffs_same_proj = summarize_ece_diffs(ece_dict_full, model_name)

            for metric_name, metric_value in diffs_same_proj.items():
                diff_rows.append({
                    "run": run_idx,
                    "model": model_name,
                    "diff_type": "same_proj",
                    "metric": metric_name,
                    "value": float(metric_value) if not np.isnan(metric_value) else np.nan,
                })

            for metric_name, metric_value in diffs_cross_proj.items():
                diff_rows.append({
                    "run": run_idx,
                    "model": model_name,
                    "diff_type": "cross_proj",
                    "metric": metric_name,
                    "value": float(metric_value) if not np.isnan(metric_value) else np.nan,
                })

    np.savez(os.path.join(out_root, "Full_ECE_data.npz"), ece_dicts=np.array(ece_dicts, dtype=object))
    np.savez(os.path.join(out_root, "Full_Dist_data.npz"), dist_dicts=np.array(dist_dicts, dtype=object))

    if pairwise_ece_rows:
        pd.DataFrame(pairwise_ece_rows).to_csv(os.path.join(out_root, "pairwise_ece_summary.csv"), index=False)
    if pairwise_dist_rows:
        pd.DataFrame(pairwise_dist_rows).to_csv(os.path.join(out_root, "pairwise_distance_summary.csv"), index=False)
    if diff_rows:
        pd.DataFrame(diff_rows).to_csv(os.path.join(out_root, "projection_difference_summary.csv"), index=False)

    analysis_manifest = build_analysis_manifest(cfg, n_runs=n_runs)
    save_json(analysis_manifest, analysis_manifest_path(cfg))

    logger.info("Saved data kernel analysis outputs under %s", out_root)
    logger.info("Completed analysis for dataset=%s models=%s projections=%s", dataset_name, model_names, projections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True, help="YAML config file.")
    args = parser.parse_args()

    cfg = read_yaml(args.yaml)
    run_data_kernel_analysis(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()


