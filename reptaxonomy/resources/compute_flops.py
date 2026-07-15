import hashlib
import os as os
import pathlib
import pprint
import time

import numpy as np

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from calflops import calculate_flops
from codecarbon import EmissionsTracker

from pangaea.datasets.base import GeoFMDataset, GeoFMSubset, RawGeoFMDataset
from pangaea.decoders.base import Decoder
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

@hydra.main(version_base=None, config_path="../../configs", config_name="calflops")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """
    # fix all random seeds
    fix_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = 'calflops'
    exp_dir = './'
    exp_dir = pathlib.Path(cfg.work_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    task_name='calflops'
    logger_path = os.path.join(exp_dir,'calflops.log')

    logger = init_logger(logger_path, rank=0)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")


    with EmissionsTracker() as tracker:

        encoder: Encoder = instantiate(cfg.encoder)
        encoder.load_encoder_weights(logger)
        logger.info("Built {}.".format(encoder.model_name))

        modalities = list(encoder.input_bands.keys())
        collate_fn = get_collate_fn(modalities,return_meta=True)

        n_bands = len(modalities)

        # Evaluation
        test_preprocessor = instantiate(
            cfg.preprocessing.test,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
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

        choices = OmegaConf.to_container(HydraConfig.get().runtime.choices)

        out_dir = os.path.join(cfg.embed_dir,cfg.dataset.dataset_name,choices["encoder"],'test/')

        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok = True)
   
        encoder.to(device)
 
        #embed test data
        for batch_idx, data in enumerate(test_loader):
            if "filename" in data:
                image, target, image_fname, meta = data["image"], data["target"],data["filename"], data['metadata']
                image_fname = image_fname[0]
            else:
                image, target, meta = data["image"], data["target"], data['metadata']
                image_fname = meta['image_filename'][0]



            image = {modality: value.to(device) for modality, value in image.items()}

            batch_size = 1

            input = None
            if encoder.multi_temporal:
                if not test_dataset.multi_temporal:
                    inpt = image
            else:
                inpt = {k: v[:, :, 0, :, :] for k, v in image.items()}        
            print(inpt.keys(), inpt['optical'].shape) 

            encoder.train()
            flops, macs, params = calculate_flops(model=encoder,
                                          args=inpt,
                                          output_as_string=True,
                                          output_precision=4)

            #log_str = f"\nCarbon emissions from computation: {tracker.final_emissions * 1000:.4f} g CO2eq"
            #logger.info(log_str)
            #log_str = "Detailed emissions data:" + tracker.final_emissions_data
            #logger.info(log_str)
            log_str = " FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)
            log_str = choices["encoder"] + log_str
            logger.info(log_str)
      
            break

if __name__ == "__main__":
    main()
