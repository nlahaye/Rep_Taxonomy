from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
import umap
import zarr
import openTSNE

from scipy.sparse import csr_array, save_npz
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from reptaxonomy.util.general_utils import read_yaml
from reptaxonomy.util.experiment_init_utils import build_test_loader
from reptaxonomy.util.artifacts import (
    ensure_dict,
    get_dataset_name,
    get_model_name,
    get_projections,
    dataset_out_root,
    run_dir,
    dataset_run_dir,
    embedding_file_path,
    crop_info_file_path,
    embed_subset_path,
    target_subset_path,
    projection_path,
    projection_label_tif_path,
    knn_graph_path,
    reducer_model_path,
    silhouette_stats_path,
    index_cache_path,
    run_manifest_path,
    build_run_manifest,
    save_json,
)

logger = logging.getLogger(__name__)


def center_mask(shape_info: Tuple[int, int], frac_low: float = 0.15, frac_high: float = 0.85) -> np.ndarray:
    h, w = shape_info
    rows = np.arange(h * w) // w
    cols = np.arange(h * w) % w
    return (
        (rows >= int(frac_low * h)) &
        (rows < int(frac_high * h)) &
        (cols >= int(frac_low * w)) &
        (cols < int(frac_high * w))
    )


def infer_hw_from_crop(crop_info: Any) -> Tuple[int, int]:
    if crop_info is None:
        raise ValueError("crop_info is None")
    return int(crop_info[0][-2]), int(crop_info[0][-1])


def build_sparse_knn_graph(
    X: np.ndarray,
    k: int,
    metric: str = "euclidean",
    symmetrize: bool = True,
) -> csr_array:
    n_neighbors = min(k + 1, X.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X, return_distance=True)

    if indices.shape[1] > 1:
        indices = indices[:, 1:]

    rows = np.repeat(np.arange(X.shape[0]), indices.shape[1])
    cols = indices.reshape(-1)
    data = np.ones_like(cols, dtype=np.int8)

    graph = csr_array(
        (data, (rows, cols)),
        shape=(X.shape[0], X.shape[0]),
        dtype=np.int8,
    )
    if symmetrize:
        graph = graph.maximum(graph.T)
    return graph


def save_sparse_graph(graph: csr_array, out_path: str) -> None:
    save_npz(out_path, graph, compressed=True)


def get_indices(
    cfg: Dict[str, Any],
    run_idx: int,
    image_fname: str,
    target: np.ndarray,
    shape_info: Tuple[int, int],
) -> np.ndarray:
    index_fname = index_cache_path(cfg, run_idx, image_fname)
    os.makedirs(os.path.dirname(index_fname), exist_ok=True)

    if os.path.exists(index_fname):
        return zarr.load(index_fname)

    n_classes = int(cfg["dataset"]["num_classes"])
    ignore_class = int(cfg["dataset"]["ignore_index"])
    label_file_subset = int(cfg["label_file_subset"])

    c_mask = center_mask(shape_info)
    selected = []

    for cls in range(n_classes + 1):
        if cls == ignore_class:
            continue
        valid = np.where((target == cls) & c_mask)[0]
        if valid.size == 0:
            continue
        if valid.size > label_file_subset:
            valid = np.random.choice(valid, size=label_file_subset, replace=False)
        selected.append(valid.astype(np.int32, copy=False))

    final_inds = np.concatenate(selected, axis=0) if selected else np.empty((0,), dtype=np.int32)
    zarr.save(index_fname, final_inds)
    return final_inds


def rescale_embed(embed: np.ndarray, image_shape: int, target: Any, crop_info: Any = None) -> Tuple[np.ndarray, np.ndarray]:
    ind = 1 if embed.ndim > 3 else 0
    tensor = torch.from_numpy(embed)

    if int(math.sqrt(embed.shape[ind])) // 8 > 3:
        rescale_factor = int(math.sqrt(embed.shape[ind])) // 8
        while embed.shape[ind] % (rescale_factor ** 2) > 0:
            rescale_factor += 1
        tensor = torch.nn.PixelShuffle(rescale_factor)(tensor)

    if tensor.ndim < 4:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = torch.flatten(tensor, start_dim=0, end_dim=1).unsqueeze(0)

    if tensor.shape[-1] != image_shape:
        tensor = F.interpolate(tensor, size=(image_shape, image_shape), mode="nearest")

    target_np = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)

    if crop_info is not None:
        out_h, out_w = int(crop_info[0][-2]), int(crop_info[0][-1])

        full_tensor = torch.zeros((tensor.shape[0], tensor.shape[1], out_h, out_w), dtype=tensor.dtype)
        full_tensor[:, :, crop_info[1]:crop_info[1] + crop_info[3], crop_info[2]:crop_info[2] + crop_info[4]] = tensor
        tensor = full_tensor

        full_target = np.zeros((1, out_h, out_w), dtype=target_np.dtype)
        full_target[:, crop_info[1]:crop_info[1] + crop_info[3], crop_info[2]:crop_info[2] + crop_info[4]] = target_np
        target_np = full_target

    target_np = target_np.reshape(-1)
    embed_np = tensor.permute(0, 2, 3, 1).reshape(-1, tensor.shape[1]).numpy()
    return embed_np, target_np


def fit_or_load_projection(
    embed: np.ndarray,
    cfg: Dict[str, Any],
    run_idx: int,
    projection: str,
) -> np.ndarray:
    reducer_fname = reducer_model_path(cfg, run_idx, projection)

    if projection == "umap":
        if os.path.exists(reducer_fname):
            reducer = joblib.load(reducer_fname)
            logger.info("UMAP projecting data %s", embed.shape)
            return reducer.transform(embed)

        logger.info("Training UMAP and projecting data %s", embed.shape)
        reducer = umap.UMAP(
            metric="cosine",
            n_neighbors=int(cfg["umap_n_neighbors"]),
            min_dist=float(cfg["umap_min_dist"]),
            n_components=int(cfg["umap_n_components"]),
            spread=float(cfg["umap_spread"]),
            random_state=int(cfg["seed"]),
        )
        proj = reducer.fit_transform(embed)

    elif projection == "pca":
        if os.path.exists(reducer_fname):
            reducer = joblib.load(reducer_fname)
            logger.info("Projecting PCs %s", embed.shape)
            return reducer.transform(embed)

        logger.info("Computing PCs and projecting data %s", embed.shape)
        reducer = PCA(n_components=min(int(cfg.get("pca_max_components", 64)), embed.shape[1]))
        proj = reducer.fit_transform(embed)

    elif projection == "tsne":
        if os.path.exists(reducer_fname):
            reducer = joblib.load(reducer_fname)
            logger.info("TSNE projecting data %s", embed.shape)
            return reducer.transform(embed)

        logger.info("Training TSNE and projecting data %s", embed.shape)
        reducer = openTSNE.TSNE(
            n_jobs=int(cfg.get("tsne_n_jobs", 8)),
            verbose=True,
            metric="cosine",
            exaggeration=4,
            random_state=int(cfg["seed"]),
        )
        reducer = reducer.fit(embed)
        proj = reducer.transform(embed)

    else:
        raise ValueError(f"Unsupported projection: {projection}")

    joblib.dump(reducer, reducer_fname)
    return proj


def normalize_projection_for_raster(projection_data: np.ndarray, scale: float = 10.0) -> np.ndarray:
    proj = projection_data[:, :2].copy()
    proj[:, 0] -= proj[:, 0].min()
    proj[:, 1] -= proj[:, 1].min()
    return np.rint(proj * scale).astype(np.int32)


def write_projection_label_tif(projection_data: np.ndarray, target_full: np.ndarray, out_file: str) -> None:
    y = projection_data[:, 0].astype(np.int32)
    x = projection_data[:, 1].astype(np.int32)

    final_projection = np.full((int(y.max()) + 1, int(x.max()) + 1), -1, dtype=np.int32)
    final_projection[y, x] = target_full

    ras_meta = {
        "driver": "GTiff",
        "dtype": "int32",
        "nodata": -1,
        "width": final_projection.shape[1],
        "height": final_projection.shape[0],
        "count": 1,
        "tiled": False,
        "interleave": "band",
    }

    with rasterio.open(out_file, "w", **ras_meta) as dst:
        dst.write(final_projection, 1)


def build_projection_artifacts(
    cfg: Dict[str, Any],
    run_idx: int,
    embed_full: np.ndarray,
    target_full: np.ndarray,
) -> List[Dict[str, Any]]:
    rows = []
    projection_names = get_projections(cfg)
    model_name = get_model_name(cfg)

    for projection in projection_names:
        projection_data = fit_or_load_projection(embed_full, cfg, run_idx, projection)
        projection_2d = normalize_projection_for_raster(projection_data)

        tif_path = projection_label_tif_path(cfg, run_idx, projection, model_name=model_name)
        write_projection_label_tif(projection_2d, target_full, tif_path)

        projection_save_path = projection_path(cfg, run_idx, projection, model_name=model_name)
        np.save(projection_save_path, projection_data[:, :2])

        silhouette = metrics.silhouette_score(projection_data[:, :2], target_full)

        row = {
            "run": run_idx,
            "projection": projection,
            "model": model_name,
            "mean_sil": float(silhouette),
            "std_sil": 0.0,
            "n_points": int(projection_data.shape[0]),
            "projection_path": projection_save_path,
            "label_tif_path": tif_path,
        }

        if bool(cfg.get("build_knn", True)):
            k = max(2, int(np.log(max(3, projection_2d.shape[0]))))
            graph = build_sparse_knn_graph(
                projection_data[:, :2],
                k=k,
                metric=str(cfg.get("knn_metric", "euclidean")),
            )
            graph_path = knn_graph_path(cfg, run_idx, projection, model_name=model_name)
            save_sparse_graph(graph, graph_path)
            row["knn_graph_path"] = graph_path
            row["knn_k"] = int(k)
            row["knn_metric"] = str(cfg.get("knn_metric", "euclidean"))

        rows.append(row)

    return rows




def run_knn_gen(cfg: Dict[str, Any]) -> None:
    cfg = ensure_dict(cfg, "run_projections cfg")

    model_name = get_model_name(cfg)
    dataset_name = get_dataset_name(cfg)
    n_runs = int(cfg.get("n_runs", 3))
    seed = int(cfg.get("seed", 42))

    np.random.seed(seed)
    dataset_out_root(cfg)
    logger.info("Running projections for model=%s dataset=%s", model_name, dataset_name)

    test_loader = build_test_loader(cfg)
    all_sil_rows: List[Dict[str, Any]] = []

    for run_idx in range(n_runs):
        embed_batches: List[np.ndarray] = []
        target_batches: List[np.ndarray] = []

        run_dir(cfg, run_idx, model_name=model_name)
        dataset_run_dir(cfg, run_idx)

        for data in test_loader:
            if "filename" in data:
                image = data["image"]
                target = data["target"]
                image_fname = data["filename"][0]
            else:
                image = data["image"]
                target = data["target"]
                meta = data["metadata"]
                image_fname = meta["image_filename"][0]

            embed_fname = embedding_file_path(cfg, image_fname, model_name=model_name, split="test")
            crop_fname = crop_info_file_path(cfg, image_fname, model_name=model_name, split="test")

            embed = np.load(embed_fname)
            crop_info_all = np.load(crop_fname, allow_pickle=True).item() if os.path.exists(crop_fname) else None

            crop_info = None
            img_size = None
            for key, value in image.items():
                crop_info = crop_info_all[key] if crop_info_all is not None else None
                img_size = value[:, :, 0, :, :].shape[-1]

            if img_size is None:
                raise ValueError(f"Could not infer image size for {image_fname}")

            embed_rescaled, target_np = rescale_embed(embed, img_size, target, crop_info)
            shape_info = infer_hw_from_crop(crop_info) if crop_info is not None else (img_size, img_size)

            indices = get_indices(cfg, run_idx, image_fname, target_np, shape_info)
            if indices.size == 0:
                continue

            embed_batches.append(embed_rescaled[indices, :].astype(np.float32, copy=False))
            target_batches.append(target_np[indices].astype(np.int32, copy=False))

        if not embed_batches:
            logger.warning("No embeddings selected for run_%s", run_idx)
            continue

        embed_full = np.concatenate(embed_batches, axis=0)
        target_full = np.concatenate(target_batches, axis=0)

        np.save(embed_subset_path(cfg, run_idx, model_name=model_name), embed_full)
        np.save(target_subset_path(cfg, run_idx, model_name=model_name), target_full)

        sil_rows = build_projection_artifacts(cfg, run_idx, embed_full, target_full)
        all_sil_rows.extend(sil_rows)

        manifest = build_run_manifest(
            cfg=cfg,
            run_idx=run_idx,
            n_subset_samples=embed_full.shape[0],
            embedding_dim=embed_full.shape[1],
            projections=get_projections(cfg),
            model_name=model_name,
        )

        by_projection = {
            row["projection"]: {
                "mean_sil": row["mean_sil"],
                "std_sil": row["std_sil"],
                "n_points": row["n_points"],
                "projection_path": row["projection_path"],
                "label_tif_path": row["label_tif_path"],
                "knn_graph_path": row.get("knn_graph_path"),
                "knn_k": row.get("knn_k"),
                "knn_metric": row.get("knn_metric"),
            }
            for row in sil_rows
        }
        manifest["projection_stats"] = by_projection
        save_json(manifest, run_manifest_path(cfg, run_idx, model_name=model_name))

    if all_sil_rows:
        pd.DataFrame(all_sil_rows).to_csv(
            silhouette_stats_path(cfg, model_name=model_name),
            index=False,
        )

    logger.info("Saved projection artifacts under %s", run_dir(cfg, 0, model_name=model_name).rsplit("/run_", 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", required=True, help="YAML config file.")
    args = parser.parse_args()

    cfg = read_yaml(args.yaml)
    run_knn_gen(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()


