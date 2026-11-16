"""Relative (geodesic) representations, adapted from RelativeGeodesics.

Attribution
-----------
This module re-implements the representation functions, anchor selection and classifier-head
decoder of the official implementation of

    H. Yu, B. Inal, G. Arvanitidis, S. Hauberg, F. Locatello and M. Fumero,
    "Connecting Neural Models Latent Geometries with Relative Geodesic Representations,"
    Advances in Neural Information Processing Systems 38 (NeurIPS 2025).
    arXiv:2506.01599, https://openreview.net/forum?id=4JnZvkVssS

    Code: https://github.com/marc0git/RelativeGeodesics (commit 639ea8e, 2025-12-03),
    files vision_foundation_models/src/representations/{abs,cos,euc,geo_euc,geo_sphere,
    geo_ig,utils}.py, vision_foundation_models/src/anchor_utils.py and the ``train_decoder``
    head in vision_foundation_models/src/utils.py.

It builds on relative representations (L. Moschella et al., "Relative representations enable
zero-shot latent space communication," ICLR 2023). Please cite both papers when reporting
results produced with this module. The upstream repository did not carry a license file at the
referenced commit; the code here is a re-implementation for research use with attribution.

Differences from upstream
-------------------------
* One function per representation, ``score(x, anchors, decoder)`` -> ``(N, A)`` matrix,
  instead of ``get_<rep>_name_and_fn`` factories returning ``functools.partial`` objects.
* ``geo_*`` scores are chunked over both samples and anchors so memory stays bounded.
* The anchor-anchor normalisation mask is created on the tensor's device.
* ``geo_sphere`` / ``geo_ig`` energies (``NotImplementedError`` upstream) are not provided either.
* NumPy front end (``relative_representation``) for the rep_taxonomy pipeline.

Definitions (``x`` sample, ``a`` anchor, ``g`` decoder, path ``t_k = alpha_k x + (1-alpha_k) a``
with ``alpha_k = k / S`` for ``k = 0..S``):

* ``abs``        the embedding itself (scale from the anchors' std)
* ``cos``        cos(x, a)
* ``euc``        ||x - a||                                  (``euc_e``: 0.5 ||x - a||^2)
* ``geo_euc``    sum_k ||g(t_{k+1}) - g(t_k)||              (``_e``: 0.5 S sum_k ||.||^2)
* ``geo_sphere`` sum_k angle(g(t_{k+1}), g(t_k))
* ``geo_ig``     sum_k angle(sqrt p(t_{k+1}), sqrt p(t_k)),  p = softmax(g(.))  (Fisher-Rao)

i.e. lengths (or energies) of the straight latent path between ``x`` and ``a`` under the pullback
metric of the decoder. Each representation is returned with the standard deviation of its
off-diagonal anchor-to-anchor scores, which is used to normalise it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REFERENCE = ("Yu, Inal, Arvanitidis, Hauberg, Locatello, Fumero, 'Connecting Neural Models Latent "
             "Geometries with Relative Geodesic Representations', NeurIPS 2025; "
             "https://github.com/marc0git/RelativeGeodesics (639ea8e)")

REPRESENTATIONS = ("abs", "cos", "euc", "geo_euc", "geo_sphere", "geo_ig")
NEEDS_DECODER = {"geo_euc", "geo_sphere", "geo_ig"}


# ----------------------------------------------------------------------------- helpers
def spherical_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle between vectors along the last dim (upstream: utils.spherical_distance, after geomstats)."""
    cos = torch.sum(a * b, dim=-1) / (torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1))
    return torch.arccos(torch.clip(cos, -1, 1))


def _path_outputs(z1: torch.Tensor, z2: torch.Tensor, decoder: Callable, steps: int) -> torch.Tensor:
    """Decoder outputs along straight paths z2 -> z1 for all pairs: (N, M, steps+1, C)."""
    n, d = z1.shape
    m = z2.shape[0]
    alpha = torch.linspace(0, 1, steps=steps + 1, device=z1.device, dtype=z1.dtype).view(1, 1, steps + 1, 1)
    t = alpha * z1[:, None, None, :] + (1 - alpha) * z2[None, :, None, :]
    return decoder(t.reshape(-1, d)).reshape(n, m, steps + 1, -1)


def _geo_score(kind: str, z1, z2, decoder, steps: int, energy: bool) -> torch.Tensor:
    out = _path_outputs(z1, z2, decoder, steps)
    if kind == "geo_euc":                                   # upstream geo_euc.compute_geo_euc_score_matrix
        diff = torch.diff(out, dim=2)
        if energy:
            return 0.5 * torch.sum(diff.square(), dim=(2, 3)) * steps
        return torch.sum(torch.sqrt(torch.sum(diff.square(), dim=3)), dim=2)
    if kind == "geo_sphere":                                # upstream geo_sphere.compute_geo_sphere_score_matrix
        if energy:
            raise NotImplementedError("geo_sphere energy is not implemented (as upstream)")
        return torch.sum(spherical_distance(out[:, :, :-1, :], out[:, :, 1:, :]), dim=2)
    if kind == "geo_ig":                                    # upstream geo_ig.compute_geo_ig_score_matrix
        if energy:
            raise NotImplementedError("geo_ig energy is not implemented (as upstream)")
        p = torch.sqrt(torch.softmax(out, dim=-1))
        return torch.sum(spherical_distance(p[:, :, :-1, :], p[:, :, 1:, :]), dim=2)
    raise ValueError(kind)


@torch.no_grad()
def score_matrix(kind: str, x: torch.Tensor, anchors: torch.Tensor, decoder: Optional[Callable] = None,
                 steps: int = 30, energy: bool = False, chunk: int = 8, anchor_chunk: int = 256) -> torch.Tensor:
    """(N, A) relative scores of ``x`` w.r.t. ``anchors`` for representation ``kind``."""
    if kind == "abs":
        return x
    if kind == "cos":                                       # upstream cos.relative_reps
        return F.normalize(x, p=2, dim=-1) @ F.normalize(anchors, p=2, dim=-1).T
    if kind == "euc":                                       # upstream euc.compute_euc_score_matrix
        d2 = torch.cat([(z[:, None, :] - anchors[None, :, :]).square().sum(-1)
                        for z in x.split(max(1, int(chunk) * 64), dim=0)], dim=0)   # exact at d = 0
        return 0.5 * d2 if energy else d2.sqrt()
    if kind not in NEEDS_DECODER:
        raise ValueError(f"Unknown representation {kind!r}; choose from {REPRESENTATIONS}")
    if decoder is None:
        raise ValueError(f"{kind} needs a decoder")
    rows = []
    for z in x.split(int(chunk), dim=0):                    # upstream utils.geo_reps (split_steps)
        rows.append(torch.cat([_geo_score(kind, z, a, decoder, int(steps), energy)
                               for a in anchors.split(int(anchor_chunk), dim=0)], dim=1))
    return torch.cat(rows, dim=0)


@torch.no_grad()
def normalized_reps(kind: str, x: torch.Tensor, anchors: torch.Tensor, decoder: Optional[Callable] = None,
                    **kw: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(rep, std)`` as upstream ``norm_*_reps``: std of the off-diagonal anchor-anchor scores."""
    rep = score_matrix(kind, x, anchors, decoder, **kw)
    if kind == "abs":                                       # upstream abs.norm_abs_reps
        return rep, torch.std(anchors)
    aa = score_matrix(kind, anchors, anchors, decoder, **kw)
    mask = ~torch.eye(anchors.shape[0], dtype=torch.bool, device=aa.device)
    return rep, torch.std(aa[mask])


# ----------------------------------------------------------------------------- decoder
def make_decoder(input_dim: int, num_classes: int, hidden_dim: int = 500) -> nn.Module:
    """Classifier head of upstream ``utils.train_decoder`` (encoding_type='Absolute')."""
    return nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                         nn.Linear(hidden_dim, num_classes))


def train_decoder(x: np.ndarray, y: np.ndarray, *, hidden_dim: int = 500, epochs: int = 10,
                  lr: float = 5e-4, batch_size: int = 64, seed: int = 0, device=None,
                  val_fraction: float = 0.2) -> Tuple[nn.Module, Dict[str, Any]]:
    """Fit the head on frozen embeddings with Adam + cross-entropy (upstream defaults: lr 5e-4,
    batch 64, hidden 500). Returns the frozen head and its held-out accuracy."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    classes, yi = np.unique(np.asarray(y).ravel(), return_inverse=True)
    g = torch.Generator().manual_seed(int(seed))
    torch.manual_seed(int(seed))
    X = torch.as_tensor(np.asarray(x, dtype=np.float32))
    Y = torch.as_tensor(yi, dtype=torch.long)
    perm = torch.randperm(len(X), generator=g)
    n_val = int(round(val_fraction * len(X))) if len(X) >= 10 else 0
    va, tr = perm[:n_val], perm[n_val:]
    dec = make_decoder(X.shape[1], max(len(classes), 2), hidden_dim).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dec.train()
    for _ in range(int(epochs)):
        order = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(order), int(batch_size)):
            b = order[i:i + int(batch_size)]
            opt.zero_grad()
            loss_fn(dec(X[b].to(device)), Y[b].to(device)).backward()
            opt.step()
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        acc = float((dec(X[va].to(device)).argmax(1).cpu() == Y[va]).float().mean()) if n_val else float("nan")
    return dec, {"decoder_val_accuracy": acc, "decoder_classes": int(len(classes))}


# ----------------------------------------------------------------------------- anchors
def select_anchors(x: np.ndarray, n_anchors: int, scheme: str = "uniform", seed: int = 0) -> np.ndarray:
    """Anchor row indices (upstream real_models.py / anchor_utils.py): ``uniform`` random rows,
    ``fps`` farthest-point sampling and ``kmeans`` (row nearest each centroid), both on
    L2-normalised embeddings."""
    n = len(x)
    k = int(min(max(int(n_anchors), 2), n))
    rng = np.random.default_rng(int(seed))
    scheme = str(scheme).lower()
    if scheme == "uniform":
        return np.sort(rng.choice(n, size=k, replace=False))
    z = np.asarray(x, dtype=np.float64)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    if scheme == "fps":
        idx = [int(rng.integers(n))]
        d = np.linalg.norm(z - z[idx[0]], axis=1)
        for _ in range(k - 1):
            j = int(np.argmax(d))
            idx.append(j)
            d = np.minimum(d, np.linalg.norm(z - z[j], axis=1))
        return np.sort(np.unique(idx))
    if scheme == "kmeans":
        from sklearn.cluster import KMeans
        lab = KMeans(n_clusters=k, n_init=3, random_state=int(seed)).fit_predict(z)
        out = [int(m[np.argmin(np.linalg.norm(z[m] - z[m].mean(0), axis=1))])
               for m in (np.flatnonzero(lab == c) for c in range(k)) if len(m)]
        return np.sort(np.unique(out))
    raise ValueError(f"Unknown anchor scheme: {scheme}")


# ----------------------------------------------------------------------------- numpy front end
def relative_representation(kind: str, x: np.ndarray, anchor_idx: np.ndarray, decoder: Optional[nn.Module] = None,
                            *, device=None, steps: int = 30, energy: bool = False, chunk: int = 8,
                            anchor_chunk: int = 256, normalize: bool = True) -> Tuple[np.ndarray, float]:
    """Relative representation (N x A; N x d for ``abs``) of every row of ``x`` w.r.t. ``x[anchor_idx]``."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    X = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    A = X[torch.as_tensor(np.asarray(anchor_idx), device=device)]
    rep, std = normalized_reps(kind, X, A, decoder, steps=steps, energy=energy, chunk=chunk,
                               anchor_chunk=anchor_chunk)
    std = float(std.detach().cpu())
    out = rep.detach().cpu().numpy()
    if normalize and np.isfinite(std) and std > 0:
        out = out / std
    return out, std
