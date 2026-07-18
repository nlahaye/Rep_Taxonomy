import os
import pathlib
import pickle
import pprint
import time

import argparse

import hydra
import torch
from calflops import calculate_flops
from codecarbon import EmissionsTracker

from reptaxonomy.util.general_utils import read_yaml

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_single_input_sample(test_dataset, device: str):
    sample = test_dataset[0]
    image = sample["image"]
    image = {modality: value.unsqueeze(0).to(device) for modality, value in image.items()}
    return image


def build_flops_input(encoder, test_dataset, image_dict):
    if getattr(encoder, "multi_temporal", False) and getattr(test_dataset, "multi_temporal", False):
        return {k: v[:, :, 0, :, :] for k, v in image_dict.items()}
    return image_dict


def tensor_shape_map(inpt):
    return {k: tuple(v.shape) for k, v in inpt.items()}


def compute_resource_reqs(cfg):
    fix_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = "calflops"
    exp_dir = pathlib.Path(cfg.work_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger_path = os.path.join(exp_dir, "calflops.log")

    logger = init_logger(logger_path, rank=0)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s", exp_dir)
    logger.info("Device used: %s", device)

    encoder: Encoder = instantiate(cfg.encoder)
    encoder.load_encoder_weights(logger)
    encoder.to(device)
    encoder.eval()
    logger.info("Built %s.", encoder.model_name)

    test_preprocessor = instantiate(
        cfg.preprocessing.test,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )
    raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
    test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)

    image = get_single_input_sample(test_dataset, device)
    inpt = build_flops_input(encoder, test_dataset, image)
    input_shapes = tensor_shape_map(inpt)
    logger.info("FLOPs input shapes: %s", input_shapes)

    choices = OmegaConf.to_container(HydraConfig.get().runtime.choices)
    out_dir = ensure_dir(os.path.join(cfg.embed_dir, cfg.dataset.dataset_name, choices["encoder"], "test"))
    stats_dir = ensure_dir(os.path.join(out_dir, "calflops_stats"))

    tracker = EmissionsTracker(output_dir=stats_dir, output_file=f"{encoder.model_name}.emissions.csv")
    tracker.start()
    try:
        with torch.no_grad():
            flops, macs, params = calculate_flops(
                model=encoder,
                args=inpt,
                output_as_string=True,
                output_precision=4,
            )
    finally:
        emissions = tracker.stop()

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "encoder": encoder.model_name,
        "dataset": cfg.dataset.dataset_name,
        "device": device,
        "input_shapes": input_shapes,
        "flops": flops,
        "macs": macs,
        "params": params,
        "emissions_kg_co2eq": emissions,
        "emissions_g_co2eq": None if emissions is None else emissions * 1000.0,
        "hydra_choices": choices,
        "resolved_cfg": OmegaConf.to_container(cfg, resolve=True),
    }

    out_pkl = os.path.join(stats_dir, f"{encoder.model_name}.flops.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("%s FLOPs:%s MACs:%s Params:%s", choices["encoder"], flops, macs, params)
    if emissions is not None:
        logger.info("Carbon emissions: %.4f g CO2eq", emissions * 1000.0)
    logger.info("Saved FLOPs summary to %s", out_pkl)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config file.")
    args = parser.parse_args()
    cfg = read_yaml(args.yaml)
    compute_resource_reqs(cfg)


