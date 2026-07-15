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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from random import randint
import weightwatcher as ww
import logging

def get_colors(n):
    color = []

    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))

    return color


#Integrating weight matrix analyses as demonstrated by:
#Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning 
#    Charles H. Martin, Michael W. Mahoney; JMLR 22(165):1−73, 2021
#AND
#Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data 
#    Charles H. Martin, Tongsu (Serena) Peng & Michael W. Mahoney; Nature Communications 12(4122), 2021 
# 
#Plotting code from example in associated repo:
#    https://github.com/CalculatedContent/WeightWatcher/blob/d6fd015dc5ff00e95d5b7612333dfab15966ad0d/examples/ModelPlots.ipynb

def plot_metrics_histogram(metric, xlabel, title, series_name, \
    all_names, all_details, colors, log=False, valid_ids = []):

    transparency = 1.0

    if len(valid_ids) == 0:
        valid_ids = range(0,len(all_details)-1)
        idname='all'
    else:
        idname='fnl'

    ind = 0
    #for im, details in enumerate(all_details):
    for key in all_details.keys():
        if key in valid_ids:
            if metric not in all_details[key]:
                continue
            vals = all_details[key][metric].to_numpy()
            if log:
                vals = np.log10(np.array(vals+0.000001, dtype=np.float))


            plt.hist(vals, bins=100, label=key, alpha=transparency, color=colors[ind], density=True)
            transparency -= 0.15
            ind = ind + 1

    fulltitle = "Histogram: "+title+" "+xlabel

    #plt.legend()
    plt.title(title)
    plt.title(fulltitle)
    plt.xlabel(xlabel)

    figname = "img/{}_{}_{}_hist.png".format(series_name, idname, metric)
    print("saving {}".format(figname))
    plt.savefig(figname)
    plt.tight_layout()
    plt.show()


def plot_metrics_depth(metric, ylabel, title, series_name, \
    all_names, all_details, colors, log=False, valid_ids = []):

    transparency = 1.0

    if len(valid_ids) == 0:
        valid_ids = range(len(all_details)-1)
        idname='all'
    else:
        idname='fnl'
    yout = []

    #for im, details in enumerate(all_details):
    ind = 0
    for key in all_details.keys():
        if key in valid_ids:
            #details = all_details[im]
            name = key #all_names[im]
            #x = details["layer_id"].to_numpy()
            print(metric, all_details)
            if metric not in all_details[key]:
                continue
            y = all_details[key][metric].to_numpy()
            yout.extend(y)
            x = [i for i in range(len(y))]
            if log:
                y = np.log10(np.array(y+0.000001, dtype=np.float))

            plt.scatter(x,y, label=name, color=colors[ind])
            ind = ind + 1

    #plt.legend()
    plt.title("Depth vs "+title+" "+ylabel)
    plt.xlabel("Layer id")
    plt.ylabel(ylabel)

    figname = "img/{}_{}_{}_depth.png".format(series_name, idname, metric)
    print("saving {}".format(figname))
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()

    if idname == 'fnl' and metric == "alpha":
        print("Ys", yout)


def plot_all_metric_histograms(\
    series_name, all_names, colors, all_summaries, all_details,  first_n_last_ids):

    metric = "log_norm"
    xlabel = r"Log Frobenius Norm $\log\Vert W_{F}\Vert$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors)
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, valid_ids = first_n_last_ids)

    metric = "alpha"
    xlabel = r"Alpha $\alpha$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors)
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, valid_ids = first_n_last_ids)


    metric = "alpha_weighted"
    xlabel = r"Weighted Alpha $\hat{\alpha}$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors)
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, valid_ids = first_n_last_ids)


    metric = "stable_rank"
    xlabel = r"Stable Rank $\mathcal{R}_{s}$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors)
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, valid_ids = first_n_last_ids)

    metric = "log_spectral_norm"
    xlabel = r"Log Spectral Norm $\log\Vert\mathbf{W}\Vert_{\infty}$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title,  series_name, \
            all_names, all_details, colors)
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, valid_ids = first_n_last_ids)


    metric = "mp_softrank"
    xlabel = r"Log MP Soft Rank $\mathcal{R}_{mp}$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title,  series_name, \
            all_names, all_details, colors)
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, valid_ids = first_n_last_ids)


    metric = "log_alpha_norm"
    xlabel = r"Log $\alpha$-Norm $\log\Vert\mathbf{X}\Vert^{\alpha}_{\alpha}$"
    title = series_name
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors)
    plot_metrics_histogram(metric, xlabel, title, series_name, \
            all_names, all_details, colors, \
            valid_ids = first_n_last_ids)

def plot_all_metric_vs_depth(\
    series_name, all_names, colors, all_summaries, all_details, first_n_last_ids):

    metric = "log_norm"
    xlabel = r"Log Frobenius Norm $\langle\log\;\Vert\mathbf{W}\Vert\rangle_{F}$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title,series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)

    metric = "alpha"
    xlabel = r"Alpha $\alpha$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)

    return

    metric = "alpha_weighted"
    xlabel = r"Weighted Alpha $\hat{\alpha}$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)
    


    metric = "stable_rank"
    xlabel = r"Stable Rank $\log\;\mathcal{R}_{s}$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)

    metric = "log_spectral_norm"
    xlabel = r"Log Spectral Norm $\log\;\Vert\mathbf{W}\Vert_{\infty}$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)



    metric = "mp_softrank"
    xlabel = r"Log MP Soft Rank $\log\;\mathcal{R}_{mp}$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)

    metric = "log_alpha_norm"
    xlabel = r"Log $\alpha$-Norm $\log\;\Vert\mathbf{X}\Vert^{\alpha}_{\alpha}$"
    title = series_name
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = [])
    plot_metrics_depth(metric, xlabel, title, series_name, \
            all_names, all_details, colors, log=False, valid_ids = first_n_last_ids)



def get_exp_info(hydra_config: HydraConf) -> dict[str, str]:
    """Create a unique experiment name based on the choices made in the config.

    Args:
        hydra_config (HydraConf): hydra config.

    Returns:
        str: experiment information.
    """
    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    cfg_hash = hashlib.sha1(
        OmegaConf.to_yaml(hydra_config).encode(), usedforsecurity=False
    ).hexdigest()[:6]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fm = choices["encoder"]
    decoder = choices["decoder"]
    ds = choices["dataset"]
    task = choices["task"]
    exp_info = {
        "timestamp": timestamp,
        "fm": fm,
        "decoder": decoder,
        "ds": ds,
        "task": task,
        "exp_name": f"{timestamp}_{cfg_hash}_{fm}_{decoder}_{ds}",
    }
    return exp_info


@hydra.main(version_base=None, config_path="../../configs", config_name="ww")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """
    # fix all random seeds
    fix_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = 'ww'
    exp_dir = './'
    exp_dir = pathlib.Path(cfg.work_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    task_name='ww'
    logger_path = os.path.join(exp_dir,'ww.log')

    logger = init_logger(logger_path, rank=0)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    encoder: Encoder = instantiate(cfg.encoder)
    encoder.load_encoder_weights(logger)
    logger.info("Built {}.".format(encoder.model_name))


    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities,return_meta=True)

    out_dir = os.path.join(cfg.embed_dir,cfg.dataset.dataset_name,encoder.model_name,'test/')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok = True)
   
    #initialize weight watcher instance 
    watcher = ww.WeightWatcher(model=encoder, log_level=logging.DEBUG)

    #Run analysis
    details = watcher.analyze(model=encoder, plot=True, min_evals=50, max_evals=5000, \
        randomize=True, mp_fit=True, pool=True, savefig=True, layers=[]) 

    #Extract summary
    summaries = watcher.get_summary(details)

    #Get empirical spectral density from weight matrices
    #watcher.get_ESD()


    #Plot metrics
    metrics = ["log_norm","alpha","alpha_weighted","log_alpha_norm",\
        "log_spectral_norm","stable_rank","mp_softrank"]

    colors = get_colors(len(details))
    all_names = []
    series_name = encoder.model_name #Gaussian DBN 2-Layer" #"Pix-Wise Contrastive CNN" #"JEPA_Local" #"Clay" #"Gaussian DBN 2-Layer"
    first_n_last_ids = [series_name]
    details = {series_name: details}
    for i in range(len(details)):
        all_names.append("TEST" + str(i))


    #plt.rcParams.update({'font.size': 20})
    plot_all_metric_vs_depth(series_name, all_names, colors, summaries, details, first_n_last_ids)
    ##plot_all_metric_histograms(series_name, all_names, colors, summaries, details,  first_n_last_ids)


if __name__ == "__main__":
    main()
