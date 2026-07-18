import hashlib
import os
import pathlib
import pickle
import pprint
import time
import logging
from random import randint
from typing import Any, Dict

import argparse

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import weightwatcher as ww

 from reptaxonomy.util.general_utils import read_yaml

def get_colors(n):
    return ['#%06X' % randint(0, 0xFFFFFF) for _ in range(n)]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_pickle_dump(obj: Any, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def dataframe_to_pickle(df, path: str) -> None:
    if hasattr(df, 'to_pickle'):
        df.to_pickle(path)
    else:
        safe_pickle_dump(df, path)


def normalize_details_for_pickle(details):
    if hasattr(details, 'copy'):
        return details.copy()
    return details


def plot_metrics_histogram(metric, xlabel, title, series_name,
                           all_names, all_details, colors, out_dir,
                           log=False, valid_ids=None):
    valid_ids = [] if valid_ids is None else valid_ids
    plt.figure()
    transparency = 1.0
    idname = 'all' if len(valid_ids) == 0 else 'fnl'
    keys_to_plot = list(all_details.keys()) if len(valid_ids) == 0 else valid_ids

    ind = 0
    for key in keys_to_plot:
        if key not in all_details:
            continue
        if metric not in all_details[key]:
            continue
        vals = all_details[key][metric].to_numpy()
        if log:
            vals = np.log10(np.asarray(vals + 1e-6, dtype=float))
        plt.hist(vals, bins=100, label=key, alpha=max(0.15, transparency), color=colors[ind % len(colors)], density=True)
        transparency -= 0.15
        ind += 1

    fulltitle = f"Histogram: {title} {xlabel}"
    plt.title(fulltitle)
    plt.xlabel(xlabel)
    plt.tight_layout()
    figname = os.path.join(out_dir, f"{series_name}_{idname}_{metric}_hist.png")
    plt.savefig(figname)
    plt.close()


def plot_metrics_depth(metric, ylabel, title, series_name,
                       all_names, all_details, colors, out_dir,
                       log=False, valid_ids=None):
    valid_ids = [] if valid_ids is None else valid_ids
    plt.figure()
    idname = 'all' if len(valid_ids) == 0 else 'fnl'
    keys_to_plot = list(all_details.keys()) if len(valid_ids) == 0 else valid_ids

    ind = 0
    for key in keys_to_plot:
        if key not in all_details or metric not in all_details[key]:
            continue
        y = all_details[key][metric].to_numpy()
        x = np.arange(len(y))
        if log:
            y = np.log10(np.asarray(y + 1e-6, dtype=float))
        plt.scatter(x, y, label=key, color=colors[ind % len(colors)], s=12)
        ind += 1

    plt.title(f"Depth vs {title} {ylabel}")
    plt.xlabel("Layer id")
    plt.ylabel(ylabel)
    plt.tight_layout()
    figname = os.path.join(out_dir, f"{series_name}_{idname}_{metric}_depth.png")
    plt.savefig(figname)
    plt.close()


def plot_all_metric_vs_depth(series_name, all_names, colors, all_summaries, all_details, first_n_last_ids, out_dir):
    plot_metrics_depth("log_norm", r"Log Frobenius Norm $\langle\log\;\Vert\mathbf{W}\Vert\rangle_{F}$", series_name,
                       series_name, all_names, all_details, colors, out_dir, log=False, valid_ids=[])
    plot_metrics_depth("log_norm", r"Log Frobenius Norm $\langle\log\;\Vert\mathbf{W}\Vert\rangle_{F}$", series_name,
                       series_name, all_names, all_details, colors, out_dir, log=False, valid_ids=first_n_last_ids)
    plot_metrics_depth("alpha", r"Alpha $\alpha$", series_name,
                       series_name, all_names, all_details, colors, out_dir, log=False, valid_ids=[])
    plot_metrics_depth("alpha", r"Alpha $\alpha$", series_name,
                       series_name, all_names, all_details, colors, out_dir, log=False, valid_ids=first_n_last_ids)


def run_weight_mtx_analysis(cfg):

    #TODO

    encoder: Encoder = instantiate(cfg["encoder"])
    encoder.load_encoder_weights(logger)
    encoder.eval()
    #END TODO

    out_dir = ensure_dir(os.path.join(cfg["embed_dir"], cfg["dataset"]["dataset_name"], encoder["model_name"], 'test'))
    plot_dir = ensure_dir(os.path.join(out_dir, 'ww_plots'))
    stats_dir = ensure_dir(os.path.join(out_dir, 'ww_stats'))

    watcher = ww.WeightWatcher(model=encoder, log_level=logging.INFO)

    details = watcher.analyze(
        model=encoder,
        plot=False,
        min_evals=getattr(cfg, 'ww_min_evals', 50),
        max_evals=getattr(cfg, 'ww_max_evals', 5000),
        randomize=getattr(cfg, 'ww_randomize', True),
        mp_fit=getattr(cfg, 'ww_mp_fit', True),
        pool=getattr(cfg, 'ww_pool', True),
        savefig=False,
        layers=getattr(cfg, 'ww_layers', []),
    )
    summaries = watcher.get_summary(details)

    details_pkl = os.path.join(stats_dir, f"{prefix}.details.pkl")
    summaries_pkl = os.path.join(stats_dir, f"{prefix}.summary.pkl")
    cfg_pkl = os.path.join(stats_dir, f"{prefix}.config.pkl")
    meta_pkl = os.path.join(stats_dir, f"{prefix}.meta.pkl")

    dataframe_to_pickle(details, details_pkl)
    dataframe_to_pickle(summaries, summaries_pkl)
    safe_pickle_dump({
        "exp_info": exp_info,
        "encoder_model_name": encoder.model_name,
        "device": device,
        "details_columns": list(details.columns) if hasattr(details, 'columns') else None,
        "summary_columns": list(summaries.columns) if hasattr(summaries, 'columns') else None,
    }, meta_pkl)

    details.to_csv(os.path.join(stats_dir, f"{prefix}.details.csv"), index=False)
    summaries.to_csv(os.path.join(stats_dir, f"{prefix}.summary.csv"), index=False)

    colors = get_colors(4)
    all_names = [encoder.model_name]
    series_name = encoder.model_name
    first_n_last_ids = [series_name]
    details_dict = {series_name: details}

    plot_all_metric_vs_depth(series_name, all_names, colors, summaries, details_dict, first_n_last_ids, plot_dir)

    logger.info("Saved WeightWatcher details to %s", details_pkl)
    logger.info("Saved WeightWatcher summaries to %s", summaries_pkl)
    logger.info("Saved config pickle to %s", cfg_pkl)
    logger.info("Saved metadata pickle to %s", meta_pkl)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", help="YAML config file.")
    args = parser.parse_args()
    cfg = read_yaml(args.yaml)
    run_weight_mtx_analysis(cfg)


