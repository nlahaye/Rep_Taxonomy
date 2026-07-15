import hashlib
import os as os
import pathlib
import pprint
import time

import numpy as np
import pandas as pd

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pangaea.datasets.base import GeoFMDataset, GeoFMSubset, RawGeoFMDataset
from pangaea.encoders.base import Encoder
from pangaea.engine.evaluator import Evaluator
from pangaea.engine.trainer import Trainer
from pangaea.utils.collate_fn import get_collate_fn
from pangaea.utils.logger import init_logger
from pangaea.utils.subset_sampler import get_subset_indices
from pangaea.utils.utils import (
    fix_seed,
    get_best_model_ckpt_path,
    get_final_model_ckpt_path,
    get_generator,
    seed_worker,
)

import math
import zarr
import umap
import joblib
import rasterio

from sklearn import metrics

import scipy


def pairwise_geometry_similarity(a, b):
    """
    Secondary comparison metrics between two transformed spaces.
    """
    # Mean embedding similarity after transformation
    mean_cos = float(
        np.mean(
            np.sum(a * b, axis=1) /
            ((np.linalg.norm(a, axis=1) + 1e-12) * (np.linalg.norm(b, axis=1) + 1e-12))
        )
    )

    # Compare intra-space geometry through pairwise distances
    da = cdist(a, a, metric="euclidean")
    db = cdist(b, b, metric="euclidean")

    iu = np.triu_indices_from(da, k=1)
    rho, p = spearmanr(da[iu], db[iu])

    return {
        "mean_cosine_between_representations": mean_cos,
        "pairwise_distance_spearman": float(rho),
        "pairwise_distance_spearman_p": float(p),
    }


def build_relative_geodesic_representation(
    embeddings: np.ndarray,
    builder,
    module,
    anchors: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Generic wrapper. RelativeGeodesics appears to expose multiple representation modules,
    each centered on transforming embeddings into a relative representation.[cite:3]
    """
    if anchors is not None:
        try:
            rep = builder(embeddings, anchors=anchors, **kwargs)
            return np.asarray(rep)
        except TypeError:
            pass

    try:
        rep = builder(embeddings, **kwargs)
        return np.asarray(rep)
    except TypeError:
        # Fallback: try common class-based API patterns
        for cls_name in ["Representation", "RelativeRepresentation", "GeodesicRepresentation"]:
            if hasattr(module, cls_name):
                cls = getattr(module, cls_name)
                obj = cls(**kwargs)
                for method_name in ["fit_transform", "transform", "compute"]:
                    if hasattr(obj, method_name):
                        method = getattr(obj, method_name)
                        if anchors is not None:
                            try:
                                return np.asarray(method(embeddings, anchors=anchors))
                            except TypeError:
                                pass
                        return np.asarray(method(embeddings))
        raise

def load_representation_builder(rep_name: str):

    if rep_name == "euc":
        import representations.euc as mod
    elif rep_name == "cos":
        import representations.cos as mod
    elif rep_name == "abs":
        import representations.abs as mod
    elif rep_name == "geo_euc":
        import representations.geo_euc as mod
    elif rep_name == "geo_ig":
        import representations.geo_ig as mod
    elif rep_name == "geo_sphere":
        import representations.geo_sphere as mod
    else:
        raise ValueError(f"Unknown representation: {rep_name}")


    # Try a few common entry-point names.
    for name in ["build_representation", "compute_representation", "get_representation", "transform"]:
        if hasattr(mod, name):
            return getattr(mod, name), mod

    raise AttributeError(
        f"Could not find a representation entry point in module '{rep_name}'. "
        "Inspect the module and set the correct callable."
    )



@hydra.main(version_base=None, config_path="../../configs", config_name="geodesics")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """

    default="geo_euc",
    choices=["euc", "cos", "abs", "geo_euc", "geo_ig", "geo_sphere"],

    # fix all random seeds
    fix_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = 'embed_projection'
    exp_dir = './'
    exp_dir = pathlib.Path(cfg.work_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    task_name='Embed_project'
    logger_path = os.path.join(exp_dir,'embed_project.log')

    logger = init_logger(logger_path, rank=0)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    OmegaConf.save(cfg, "tmp_geodesics.yaml") 

    choices = OmegaConf.to_container(HydraConfig.get().runtime.choices)

    model_names = [
        "croma_optical",
        "dofa",
        #"gfmswin",
        "prithvi",
        "satlasnet_si",
        #"scalemae",
        #"spectralgpt",
        #"ssl4eo_data2vec",
        #"ssl4eo_dino",
        #"ssl4eo_mae_optical",
        #"ssl4eo_moco",
        "unet_encoder",
        "terramind_large",
        "resnet50_scratch",
        "resnet50_pretrained",
        "vit_scratch",
        "vit"
    ]


    #Get/make directories
    embed_dir = os.path.join(cfg.embed_dir,cfg.dataset.dataset_name)
    indices_dir = os.path.join(cfg.out_dir,cfg.dataset.dataset_name)

    embeddings = {}
    targets = {}
 
    for model_name in model_names:

        out_dir = os.path.join(indices_dir, model_name)

        cfg = compose(config_name="tmp_geodesics", overrides=["encoder=" + model_name])

        encoder: Encoder = instantiate(cfg.encoder)
        encoder.load_encoder_weights(logger)
        logger.info("Built {}.".format(encoder.model_name))

        modalities = list(encoder.input_bands.keys())
        collate_fn = get_collate_fn(modalities,return_meta=True)

        # Evaluation
        test_preprocessor = instantiate(
            cfg.preprocessing.test,
            dataset_cfg=cfg.dataset,
            encoder_cfg=model_name,
            _recursive_=False,
        )

        # get datasets
        raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
        test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)

        test_loader = DataLoader(
            test_dataset,
            # sampler=DistributedSampler(test_dataset),
            batch_size=cfg.test_batch_size,
            num_workers=cfg.test_num_workers,
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
 
        #Structure embedding test data
        embed_full = None
        target_full = None

        for batch_idx, data in enumerate(test_loader):

            if "filename" in data:
                image, target, image_fname, meta = data["image"], data["target"],data["filename"], data['metadata']
                image_fname = image_fname[0]
            else:
                image, target, meta = data["image"], data["target"], data['metadata']
                image_fname = meta['image_filename'][0]

            #Load embeddings for single input file
            embed_fname = os.path.join(embed_dir, choices["encoder"], "test", "embd_" + os.path.splitext(image_fname)[0] + ".npy")
            crop_info_fname = os.path.join(embed_dir, choices["encoder"], "test", "crop_info_" + os.path.splitext(image_fname)[0] + ".npy")
            embed = np.load(embed_fname)

            crop_info = None
            if os.path.exists(crop_info_fname):
                crop_info = np.load(crop_info_fname, allow_pickle=True).item()

            #print(crop_info)

            #print(embed_fname,  cfg.dataset.img_size)
            for k, v in image.items():
                crop_info = crop_info[k]
                img_size = v[:, :, 0, :, :].shape
            img_size = img_size[-1]

            #print(crop_info)

            #Rescale embedding to original dimension
            embed, target = rescale_embed(embed, img_size, device, target, crop_info)


            #Flatten dimensions, except feature/channel dim

            #target = target.flatten()

            #Get subset and apply indices, sampling each class available - subsetting done due to compuational complexity of tasks

            indices_subdir = os.path.join(indices_dir, runs_subdir)
            os.makedirs(indices_subdir, exist_ok=True)

            indices = get_indices(cfg, image_fname, target, indices_subdir, (crop_info[0][-2], crop_info[0][-1]))


            sub_embed = embed[indices,:]
            sub_target = target[indices]

            #Merge individual subsets together
            if embed_full is None:
                embed_full = sub_embed
                target_full = sub_target
            else:
                embed_full = np.concatenate((sub_embed, embed_full))
                target_full = np.concatenate((sub_target, target_full))

        embeddings[model_name] = embed_full
        targets[model_name] = target_full



    builder, module = load_representation_builder(cfg.representation)


    for anchor_model in model_names:

        anchors = embeddings[anchor_model]

        transformed = {}
        manifest_rows = []

        for model_name, x in embeddings.items():
            rep = build_relative_geodesic_representation(
                embeddings=x,
                builder=builder,
                module=module,
                anchors=anchors,
            )

            transformed[model_name] = rep

            out_path = os.path.join(out_dir, f"{anchor_name}_anchored_{model_name}_{cfg.representation}.npy")
            np.save(out_path, rep)

            manifest_rows.append(
                {
                    "model": model_name,
                    "anchor": anchor_name,
                    "representation": cfg.representation,
                    "input_samples": int(x.shape[0]),
                    "input_dim": int(x.shape[1]),
                    "output_samples": int(rep.shape[0]),
                    "output_dim": int(rep.shape[1]) if rep.ndim == 2 else 1,
                    "path": str(out_path),
                }
            )

            pd.DataFrame(manifest_rows).to_csv(
                os.path.join(out_dir, f"{anchor_name}_anchored_{cfg.representation}_representation_manifest.csv"),
                index=False,
            )
  
        pair_rows = []
        model_names = list(transformed.keys())
        for i, m1 in enumerate(model_names):
            for j, m2 in enumerate(model_names):
                if j <= i:
                    continue

                a = transformed[m1]
                b = transformed[m2]

                n = min(len(a), len(b))
                a = a[:n]
                b = b[:n]

                metrics = pairwise_geometry_similarity(a, b)
                row = {
                    "model_a": m1,
                    "model_b": m2,
                    "representation": cfg.representation,
                    "n_compared": n,
                    **metrics,
                }
                pair_rows.append(row)
 
        pair_df = pd.DataFrame(pair_rows)
        pair_df.to_csv(os.path.join(out_dir, f"{anchor_name}_anchored_{cfg.representation}_pairwise_comparison.csv"), index=False)
 
        print(pair_df.sort_values("pairwise_distance_spearman", ascending=False))


if __name__ == "__main__":
    main()





