
import os
import math
import pathlib
import pprint
from typing import Dict, List, Tuple

import argparse

import joblib
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
import umap
import zarr

from scipy.sparse import csr_array, save_npz, load_npz
import scipy.sparse as sp
from sklearn import metrics

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

import openTSNE

from reptaxonomy.util.general_utils import read_yaml


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


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


def build_sparse_knn_graph(X: np.ndarray, k: int, metric: str = "euclidean", symmetrize: bool = True) -> csr_array:
    n_neighbors = min(k + 1, X.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X, return_distance=True)
    if indices.shape[1] > 1:
        indices = indices[:, 1:]
    rows = np.repeat(np.arange(X.shape[0]), indices.shape[1])
    cols = indices.reshape(-1)
    data = np.ones_like(cols, dtype=np.int8)
    graph = csr_array((data, (rows, cols)), shape=(X.shape[0], X.shape[0]), dtype=np.int8)
    if symmetrize:
        graph = graph.maximum(graph.T)
    return graph


def save_sparse_graph(graph: csr_array, out_path: str) -> None:
    save_npz(out_path, graph, compressed=True)



def get_indices(cfg, image_fname: str, target: np.ndarray, indices_dir: str, shape_info: Tuple[int, int]) -> np.ndarray:
    index_fname = os.path.join(indices_dir, os.path.splitext(image_fname)[0] + ".indices.zarr")
    if os.path.exists(index_fname):
        return zarr.load(index_fname)

    n_classes = cfg["dataset"]["num_classes"]
    ignore_class = cfg["dataset"]["ignore_index"]
    c_mask = center_mask(shape_info)
    selected = []

    for cls in range(n_classes + 1):
        if cls == ignore_class:
            continue
        valid = np.where((target == cls) & c_mask)[0]
        if valid.size == 0:
            continue
        if valid.size > cfg["label_file_subset"]
            valid = np.random.choice(valid, size=cfg["label_file_subset"], replace=False)
        selected.append(valid.astype(np.int32, copy=False))

    final_inds = np.concatenate(selected, axis=0) if selected else np.empty((0,), dtype=np.int32)
    zarr.save(index_fname, final_inds)
    return final_inds




def rescale_embed(embed: np.ndarray, image_shape: int, target, crop_info=None) -> Tuple[np.ndarray, np.ndarray]:
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
 
    #Assumption is currently square images + tiles 
    #Adjusting to account for potential off-by-ones
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


def fit_or_load_projection(embed: np.ndarray, out_dir: str, cfg, projection: str) -> np.ndarray:
    reducer_fname = os.path.join(out_dir, f"{projection}_model.joblib")

    if projection == "umap":
        if os.path.exists(reducer_fname):
            reducer = joblib.load(reducer_fname)
            print("UMAP projecting data", embed.shape)
            return reducer.transform(embed)
        print("Training UMAP and projecting data", embed.shape)
        reducer = umap.UMAP(
            metric="cosine",
            n_neighbors=cfg["umap_n_neighbors"],
            min_dist=cfg["umap_min_dist"],
            n_components=cfg["umap_n_components"],
            spread=cfg["umap_spread"],
            random_state=cfg["seed"],
        )
        proj = reducer.fit_transform(embed)
    elif projection == "pca":
        if os.path.exists(reducer_fname):
            reducer = joblib.load(reducer_fname)
            print("Projecting PCs", embed.shape)    
            return reducer.transform(embed)
        print("Computing PCs and projecting data", embed.shape)
        reducer = PCA(n_components=min(getattr(cfg, "pca_max_components", 64), embed.shape[1]))
        proj = reducer.fit_transform(embed)
    else:
        if os.path.exists(reducer_fname):
            reducer = joblib.load(reducer_fname)
            print("TSNE projecting data", embed.shape)
            return reducer.transform(embed)
        print("Training TSNE and projecting data", embed.shape)
        reducer = openTSNE.TSNE(
            n_jobs=getattr(cfg, "tsne_n_jobs", 8),
            verbose=True,
            metric="cosine",
            exaggeration=4,
            random_state=cfg["seed"],
        )
        reducer = reducer.fit(embed)
        proj = reducer.transform(embed)

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



def build_projection_artifacts(embed_full: np.ndarray, target_full: np.ndarray, out_subdir: str, cfg, dataset_name: str, encoder_name: str) -> List[Dict]:
    rows = []
    for projection in PROJECTIONS:
        projection_data = fit_or_load_projection(embed_full, out_subdir, cfg, projection)
        projection_2d = normalize_projection_for_raster(projection_data)

        out_file = os.path.join(out_subdir, f"{encoder_name}.{projection.upper()}_Labels.tif")
        write_projection_label_tif(projection_2d, target_full, out_file)

        silhouette = metrics.silhouette_score(projection_data[:, :2], target_full)
        rows.append({
            "projection": projection,
            "model": encoder_name,
            "mean_sil": float(silhouette),
            "std_sil": 0.0,
        })

        if getattr(cfg, "build_knn", True):
            k = max(2, int(np.log(max(3, projection_2d.shape[0]))))
            graph = build_sparse_knn_graph(projection_data[:, :2], k=k, metric=getattr(cfg, "knn_metric", "euclidean"))
            knn_fpath = os.path.join(out_subdir, f"{encoder_name}.{dataset_name}.{projection.upper()}.knn_graph.npz")
            save_sparse_graph(graph, knn_fpath)

        np.save(os.path.join(out_subdir, f"{encoder_name}.{dataset_name}.{projection.upper()}.projection.npy"), projection_data[:, :2])

    return rows




def run_knn_gen(cfg):


    #TODO expected input format


    # get datasets
    raw_test_dataset: RawGeoFMDataset = instantiate(cfg["dataset"], split="test")
    test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)

    test_loader = DataLoader(
        test_dataset,
        # sampler=DistributedSampler(test_dataset),
        num_workers=cfg["test_num_workers"],
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn,
    )   

    #TODO - end dataset/model initialiation - to change over to gfmtools style


    encoder_name = cfg["encoder"]
    embed_dir = os.path.join(cfg["embed_dir"], cfg["dataset"]["dataset_name"])
    indices_root = ensure_dir(os.path.join(cfg["out_dir"], cfg["dataset"]["dataset_name"]))
    out_dir = ensure_dir(os.path.join(indices_root, encoder_name))

    n_runs = getattr(cfg, "n_runs", 3)
    all_sil_rows = []

    for run_idx in range(n_runs):
        embed_batches, target_batches = [], []
        run_name = f"run_{run_idx}"
        indices_subdir = ensure_dir(os.path.join(indices_root, run_name))
        out_subdir = ensure_dir(os.path.join(out_dir, run_name))

        for data in test_loader:
            if "filename" in data:
                image, target, image_fname, meta = data["image"], data["target"], data["filename"], data["metadata"]
                image_fname = image_fname[0]
            else:
                image, target, meta = data["image"], data["target"], data["metadata"]
                image_fname = meta["image_filename"][0]

            embed_fname = os.path.join(embed_dir, encoder_name, "test", "embd_" + os.path.splitext(image_fname)[0] + ".npy")
            crop_info_fname = os.path.join(embed_dir, encoder_name, "test", "crop_info_" + os.path.splitext(image_fname)[0] + ".npy")
            embed = np.load(embed_fname)
            crop_info_all = np.load(crop_info_fname, allow_pickle=True).item() if os.path.exists(crop_info_fname) else None

            crop_info = None
            img_size = None
            for k, v in image.items():
                crop_info = crop_info_all[k] if crop_info_all is not None else None
                img_size = v[:, :, 0, :, :].shape[-1]

            embed, target_np = rescale_embed(embed, img_size, target, crop_info)
            shape_info = infer_hw_from_crop(crop_info) if crop_info is not None else (img_size, img_size)
            indices = get_indices(cfg, image_fname, target_np, indices_subdir, shape_info)
            if indices.size == 0:
                continue
            embed_batches.append(embed[indices, :].astype(np.float32, copy=False))
            target_batches.append(target_np[indices].astype(np.int32, copy=False))

        if not embed_batches:
            logger.warning("No embeddings selected for run %s", run_name)
            continue

        embed_full = np.concatenate(embed_batches, axis=0)
        target_full = np.concatenate(target_batches, axis=0)
        np.save(os.path.join(out_subdir, f"{encoder_name}.{cfg["dataset"]["dataset_name"]}.embed_subset.npy"), embed_full)
        np.save(os.path.join(out_subdir, f"{encoder_name}.{cfg["dataset"]["dataset_name"]}.target_subset.npy"), target_full)

        sil_rows = build_projection_artifacts(embed_full, target_full, out_subdir, cfg, cfg["dataset"]["dataset_name"], encoder_name)
        all_sil_rows.extend(sil_rows)

    if all_sil_rows:
        pd.DataFrame(all_sil_rows).to_csv(
            os.path.join(out_dir, f"silhouette_stats.{encoder_name}.{cfg["dataset"]["dataset_name"]}.csv"),
            index=False,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config file.")
    args = parser.parse_args()
    cfg = read_yaml(args.yaml)
    run_knn_gen(cfg)
 


