import os as os
import numpy as np
import zarr

from scipy.spatial.distance import pdist, squareform


def knn_graph(w, k, symmetrize=True, metric='euclidean'):
    '''
    :param w: A weighted affinity graph of shape [N, N] or 2-d array 
    :param k: The number of neighbors to use
    :param symmetrize: Whether to symmetrize the resulting graph
    :return: An undirected, binary, KNN graph of shape [N, N]
    '''
    w_shape = w.shape
    if w_shape[0] != w_shape[1]:
        w = np.array(squareform(pdist(w, metric=metric)))

    neighborhoods = np.argsort(w, axis=1)[:, -(k+1):-1]
    A = np.zeros_like(w)
    for i, neighbors in enumerate(neighborhoods):
        for j in neighbors:
            A[i, j] = 1
            if symmetrize:
                A[j, i] = 1
    return A

def build_knn_graph(embed, out_fname):
    if embed.ndim < 3:
        k=int(np.log(embed.shape[0]))
    else:
        k=int(np.log(embed.shape[0]* embed.shape[1]))

    knn_graph_out =  knn_graph(embed, k=k, symmetrize=True, metric='euclidean')
    zarr.save(out_fname, knn_graph_out)
    print("Saved", out_fname, "KNN Graph")

