import os as os

import numpy as np

import torch

import umap
import openTSNE
import joblib

from sklearn.decomposition import PCA
from sklearn import metrics


def train_and_gen_projection(embed, out_dir, cfg, projection = "umap"):
   
    reducer = None
     
    if projection == "umap":
        reducer_fname = os.path.join(out_dir, 'umap_model.joblib')
 
        if not os.path.exists(reducer_fname):
            print("Training UMAP and projecting data", embed.shape)
            reducer = umap.UMAP(metric="cosine", n_neighbors=cfg.umap_n_neighbors, \
                min_dist=cfg.umap_min_dist, n_components=cfg.umap_n_components, spread=cfg.umap_spread)
            embed = reducer.fit_transform(embed)
            joblib.dump(reducer, reducer_fname)
        else:
            print("UMAP projecting data", embed.shape)
            reducer = joblib.load(reducer_fname)
            embed = reducer.transform(embed)  

    elif projection == "pca":
        reducer_fname = os.path.join(out_dir, 'pca_model.joblib')

        if not os.path.exists(reducer_fname):
            print("Computing PCs and projecting data", embed.shape)
            reducer = PCA(n_components=0.99, svd_solver = 'full')
            embed = reducer.fit_transform(embed)
            joblib.dump(reducer, reducer_fname)
        else:
            print("Projecting PCs", embed.shape)
            reducer = joblib.load(reducer_fname)
            embed = reducer.transform(embed)
    else:

        reducer_fname = os.path.join(out_dir, 'tsne_model.joblib')
 
        if not os.path.exists(reducer_fname):
            print("Training TSNE and projecting data", embed.shape)
            reducer = openTSNE.TSNE(n_jobs=50, verbose=True, metric="cosine", exaggeration = 4, random_state=42)
            reducer = reducer.fit(embed)
            embed = reducer.transform(embed)
            joblib.dump(reducer, reducer_fname)
        else:
            print("TSNE projecting data", embed.shape)
            reducer = joblib.load(reducer_fname)
            print(reducer_fname)
            embed = reducer.transform(embed)
 
    return embed, reducer


