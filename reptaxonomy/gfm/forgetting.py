"""Catastrophic forgetting of pretrained representations during EO fine-tuning.

Fine-tuning a GFM on one task (an FTW shift task, an EPA region split, MADOS)
improves the target task; this module measures what the update *removed* from
the representation of data the model was not adapted to ("retention sets").

Retention sets (resolved by ``reptaxonomy.pipelines.gfm_experiment``)
----------------------------------------------------------------------
same_family    analysed conditions of the fine-tuning dataset family that are not the
               run's train / val / eval conditions (other FTW countries or seasons,
               other EPA ecoregions)
cross_family   analysed conditions of the other enabled families (fine-tune on FTW,
               retain EPA / MADOS); the fine-tuned backbone is embedded with the
               *retention* family's band plan and statistics
ood            (optional, ``include_ood``) the run's own OOD conditions; these are
               transfer targets, so they are reported separately

Representation forgetting (pretrained ``m`` vs fine-tuned ``m__ft__<tag>-<variant>``,
same chips, embeddings from the standard ``embed`` artifacts)
---------------------------------------------------------------------------------
probe_<kind>_pre / _ft     chip-grouped cross-validated token probe (kNN cosine, or
                           multinomial logistic regression on PCA-whitened tokens)
                           trained and tested on the retention set; balanced accuracy
forgetting_<kind>          probe_pre - probe_ft (Davari et al., CVPR 2022: forgetting is
                           measured by what a re-trained probe can still read out)
rel_forgetting_<kind>      forgetting / probe_pre
cka_pooled / cka_tokens / mutual_knn@k
                           representational drift on the retention set (1 = unchanged)
class_geometry_rho         Spearman rho of class-centroid distance matrices pre vs ft
separability_pre / _ft     cosine silhouette of labelled tokens
twonn_id_pre / _ft         global TwoNN intrinsic dimension of retention tokens

Layer-wise (live forward passes, ``layerwise.enabled``)
  layerwise_cka.csv        linear CKA(pre, ft) at evenly spaced depths on retention chips
                           (Ramasesh et al., ICLR 2021: forgetting concentrates in deeper
                           layers)

Learning-forgetting trade-off (per run x model x variant)
  tradeoff.csv             id_gain (mean ID mIoU ft - linear probe), ood_change, mean
                           forgetting over retention sets (same vs cross family), weight
                           change, CKA; correlations.csv: Spearman rho across models of
                           forgetting vs weight change / representation change / predictor
                           columns (``forgetting.predictor_csvs``, e.g. pretrained
                           separability / global ID / spectral fitness per model)

Sequential fine-tuning (``run_sequential``)
  accuracy_matrix.csv      R[j, i] = ID mIoU on task i after step j (task-i head frozen)
  sequential_summary.csv   ACC, backward transfer BWT = mean_i<T (R[T,i] - R[i,i]) and
                           forgetting F_i = max_{l<T} R[l,i] - R[T,i] (Chaudhry et al.)

Outputs: ``<out_dir>/forgetting/<family>/`` and ``<out_dir>/forgetting/sequential/<name>/``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from reptaxonomy.gfm.cross_condition import CondEmbeddings, paired_metrics
from reptaxonomy.intercompare.reference_similarity import linear_cka

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "probes": ["knn", "linear"],
    "probe_k": 10,
    "n_splits": 3,
    "test_frac": 0.25,
    "max_train_tokens": 20000,
    "max_tokens_per_chip": 64,
    "linear_pca_dim": 64,
    "min_tokens": 50,
    "silhouette_n": 3000,
    "id_max_points": 5000,
    "layerwise": {"enabled": True, "max_chips": 64, "batch_size": 8, "grid": 4, "max_depths": 8},
}


# ----------------------------------------------------------------------------- probes
def _chip_splits(chips: np.ndarray, n_splits: int, test_frac: float, seed: int) -> List[np.ndarray]:
    uniq = np.unique(chips)
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n_splits):
        k = max(1, int(round(len(uniq) * test_frac)))
        test = set(rng.choice(uniq, size=min(k, max(1, len(uniq) - 1)), replace=False).tolist())
        out.append(np.array([c in test for c in chips]))
    return out


def _standardise(Xtr: np.ndarray, Xte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def probe_scores(E: CondEmbeddings, kinds: Sequence[str] = ("knn", "linear"), k: int = 10, n_splits: int = 3,
                 test_frac: float = 0.25, max_train: int = 20000, pca_dim: int = 64, min_tokens: int = 50,
                 seed: int = 0) -> Dict[str, float]:
    """Chip-grouped CV balanced accuracy of token probes trained *and* tested on ``E``.

    The same seed gives the same chip splits for every encoder on a condition, so
    pre/ft differences are paired.
    """
    from sklearn.metrics import balanced_accuracy_score
    m = E.token_labels != E.ignore_index
    if m.sum() < min_tokens or len(np.unique(E.token_labels[m])) < 2:
        return {}
    X, y, chips = E.tokens[m].astype(np.float64), E.token_labels[m], E.token_pos[m, 0]
    rng = np.random.default_rng(seed)
    res: Dict[str, List[float]] = {kind: [] for kind in kinds}
    for te in _chip_splits(chips, n_splits, test_frac, seed):
        tr = np.where(~te)[0]
        if len(tr) > max_train:
            tr = rng.choice(tr, max_train, replace=False)
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        Xtr, Xte = _standardise(X[tr], X[te])
        for kind in kinds:
            if kind == "knn":
                from sklearn.neighbors import KNeighborsClassifier
                clf = KNeighborsClassifier(n_neighbors=min(k, len(tr) - 1), metric="cosine").fit(Xtr, y[tr])
                pred = clf.predict(Xte)
            elif kind == "linear":
                from sklearn.decomposition import PCA
                from sklearn.linear_model import LogisticRegression
                d = min(pca_dim, Xtr.shape[1], len(tr) - 1)
                pca = PCA(d, whiten=True, random_state=seed).fit(Xtr)
                clf = LogisticRegression(max_iter=1000, C=1.0).fit(pca.transform(Xtr), y[tr])
                pred = clf.predict(pca.transform(Xte))
            else:
                raise ValueError(f"unknown probe kind {kind}")
            res[kind].append(float(balanced_accuracy_score(y[te], pred)))
    return {f"probe_{kind}": float(np.mean(v)) for kind, v in res.items() if v}


def class_geometry(E: CondEmbeddings, min_tokens: int = 20) -> Tuple[List[int], Optional[np.ndarray]]:
    m = E.token_labels != E.ignore_index
    cls = [int(c) for c in np.unique(E.token_labels[m]) if (E.token_labels == c).sum() >= min_tokens]
    if len(cls) < 3:
        return cls, None
    mu = np.stack([E.tokens[E.token_labels == c].mean(0) for c in cls])
    d = np.sqrt(np.maximum((mu ** 2).sum(1)[:, None] + (mu ** 2).sum(1)[None] - 2 * mu @ mu.T, 0))
    return cls, d


def class_geometry_rho(A: CondEmbeddings, B: CondEmbeddings) -> float:
    from scipy.stats import spearmanr
    ca, da = class_geometry(A)
    cb, db = class_geometry(B)
    if da is None or db is None or ca != cb:
        return float("nan")
    iu = np.triu_indices(len(ca), 1)
    return float(spearmanr(da[iu], db[iu]).statistic)


def separability(E: CondEmbeddings, n: int = 3000, seed: int = 0) -> float:
    from sklearn.metrics import silhouette_score
    m = np.where(E.token_labels != E.ignore_index)[0]
    if len(m) < 20 or len(np.unique(E.token_labels[m])) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.choice(m, size=min(n, len(m)), replace=False)
    return float(silhouette_score(E.tokens[idx], E.token_labels[idx], metric="cosine"))


def twonn_id(X: np.ndarray, max_points: int = 5000, seed: int = 0, discard: float = 0.1) -> float:
    """TwoNN (Facco et al. 2017) MLE with the top ``discard`` fraction of ratios removed."""
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, np.float64)
    if len(X) > max_points:
        X = X[np.random.default_rng(seed).choice(len(X), max_points, replace=False)]
    X = np.unique(X, axis=0)
    if len(X) < 10:
        return float("nan")
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    mu = mu[: max(2, int(len(mu) * (1 - discard)))]
    lm = np.log(mu)
    return float(len(lm) / lm.sum()) if lm.sum() > 0 else float("nan")


def retention_metrics(A: CondEmbeddings, B: CondEmbeddings, params: Mapping[str, Any], seed: int = 0) -> Dict[str, float]:
    """All representation-forgetting metrics for one retention condition (A = pre, B = ft)."""
    p = {**DEFAULTS, **dict(params or {})}
    out: Dict[str, float] = {"n_chips": float(len(A))}
    kw = dict(kinds=p["probes"], k=p["probe_k"], n_splits=p["n_splits"], test_frac=p["test_frac"],
              max_train=p["max_train_tokens"], pca_dim=p["linear_pca_dim"], min_tokens=p["min_tokens"], seed=seed)
    pa, pb = probe_scores(A, **kw), probe_scores(B, **kw)
    for kind in p["probes"]:
        a, b = pa.get(f"probe_{kind}", np.nan), pb.get(f"probe_{kind}", np.nan)
        out[f"probe_{kind}_pre"], out[f"probe_{kind}_ft"] = a, b
        out[f"forgetting_{kind}"] = a - b
        out[f"rel_forgetting_{kind}"] = (a - b) / a if a and np.isfinite(a) and a > 0 else np.nan
    pm = paired_metrics(A, B, seed=seed)
    out["cka_pooled"] = pm.get("paired_cka_pooled", np.nan)
    out["cka_tokens"] = pm.get("paired_cka_tokens", np.nan)
    out.update({k.replace("paired_", ""): v for k, v in pm.items() if k.startswith("paired_mutual_knn")})
    out["class_geometry_rho"] = class_geometry_rho(A, B)
    out["separability_pre"], out["separability_ft"] = separability(A, p["silhouette_n"], seed), separability(B, p["silhouette_n"], seed)
    out["twonn_id_pre"] = twonn_id(A.tokens, p["id_max_points"], seed) if len(A.tokens) else np.nan
    out["twonn_id_ft"] = twonn_id(B.tokens, p["id_max_points"], seed) if len(B.tokens) else np.nan
    return out


# ----------------------------------------------------------------------------- layer-wise CKA
def depth_modules(enc, max_depths: int = 8) -> List[Tuple[str, nn.Module]]:
    """Evenly spaced depth units of a GFMEncoder backbone (input side first)."""
    bb = enc.backbone
    fam = getattr(enc, "family", "")
    cands: List[Tuple[str, nn.Module]] = []
    if fam == "resnet":
        cands = [(n, getattr(bb, n)) for n in ("conv1", "layer1", "layer2", "layer3", "layer4") if hasattr(bb, n)]
    elif fam == "swin" and hasattr(bb, "features"):
        cands = [(f"features.{i}", m) for i, m in enumerate(bb.features)]
    elif fam == "unet" and hasattr(bb, "encoder"):
        blocks = getattr(bb.encoder, "_blocks", None) or getattr(bb.encoder, "blocks", None)
        if blocks is not None:
            cands = [(f"encoder.blocks.{i}", m) for i, m in enumerate(blocks)]
    if not cands:
        best: Tuple[str, Optional[nn.ModuleList]] = ("", None)
        for name, mod in bb.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
                pref = name.endswith(("blocks", "layers"))
                score = (pref, len(mod))
                if best[1] is None or score > ((best[0].endswith(("blocks", "layers"))), len(best[1])):
                    best = (name, mod)
        if best[1] is not None:
            cands = [(f"{best[0]}.{i}", m) for i, m in enumerate(best[1])]
    if not cands:
        cands = list(bb.named_children())
    if len(cands) > max_depths:
        idx = np.unique(np.round(np.linspace(0, len(cands) - 1, max_depths)).astype(int))
        cands = [cands[i] for i in idx]
    return cands


def _pool_feature(t: Any, family: str, grid: int) -> Optional[torch.Tensor]:
    if isinstance(t, (list, tuple)):
        t = next((v for v in t if torch.is_tensor(v)), None)
    if isinstance(t, dict):
        t = next((v for v in t.values() if torch.is_tensor(v)), None)
    if not torch.is_tensor(t) or t.ndim < 2:
        return None
    t = t.detach().float()
    if t.ndim == 4:
        if family == "swin":  # torchvision Swin stages are channels-last
            t = t.permute(0, 3, 1, 2)
    elif t.ndim == 3:
        n = t.shape[1]
        s = int(round(math.sqrt(n)))
        s1 = int(round(math.sqrt(max(n - 1, 1))))
        if s * s == n:
            t = t.transpose(1, 2).reshape(t.shape[0], t.shape[2], s, s)
        elif s1 * s1 == n - 1:
            t = t[:, 1:].transpose(1, 2).reshape(t.shape[0], t.shape[2], s1, s1)
        else:
            return t.mean(1)
    elif t.ndim == 2:
        return t
    else:
        return t.reshape(t.shape[0], -1, t.shape[-1]).mean(1)
    g = F.adaptive_avg_pool2d(t, (min(grid, t.shape[-2]), min(grid, t.shape[-1])))
    return g.flatten(2).transpose(1, 2).reshape(-1, g.shape[1])


@torch.no_grad()
def collect_depth_features(enc, loader, device: torch.device, max_depths: int = 8, grid: int = 4,
                           max_batches: Optional[int] = None) -> Tuple[List[str], List[np.ndarray]]:
    mods = depth_modules(enc, max_depths)
    store: Dict[int, List[np.ndarray]] = {i: [] for i in range(len(mods))}
    hooks = []
    for i, (_, mod) in enumerate(mods):
        def hook(_m, _inp, out, i=i):
            f = _pool_feature(out, enc.family, grid)
            if f is not None:
                store[i].append(f.cpu().numpy())
        hooks.append(mod.register_forward_hook(hook))
    try:
        enc.eval()
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            enc.forward_features(batch, device=device)
    finally:
        for h in hooks:
            h.remove()
    names, feats = [], []
    for i, (n, _) in enumerate(mods):
        if store[i]:
            names.append(n); feats.append(np.concatenate(store[i]))
    return names, feats


def layerwise_cka(model: str, pre_kwargs: Dict[str, Any], ft_ckpt: str, cond_spec: Dict[str, Any],
                  params: Mapping[str, Any], device: Optional[str] = None) -> pd.DataFrame:
    from reptaxonomy.gfm.encoders import GFMEncoder
    from reptaxonomy.gfm.finetune import _loader
    lw = {**DEFAULTS["layerwise"], **dict(params or {})}
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = _loader(cond_spec["cls"], cond_spec["kwargs"], int(lw["batch_size"]), False, 0, int(lw["max_chips"]))
    rows = []
    pre = GFMEncoder(model, **{k: v for k, v in pre_kwargs.items() if k != "checkpoint"}).to(dev)
    n_pre, f_pre = collect_depth_features(pre, loader, dev, int(lw["max_depths"]), int(lw["grid"]))
    del pre
    ft = GFMEncoder(model, **dict(pre_kwargs, checkpoint=ft_ckpt)).to(dev)
    n_ft, f_ft = collect_depth_features(ft, loader, dev, int(lw["max_depths"]), int(lw["grid"]))
    del ft
    L = len(n_pre)
    for i, (na, a) in enumerate(zip(n_pre, f_pre)):
        if i >= len(f_ft) or f_ft[i].shape[0] != a.shape[0]:
            continue
        rows.append({"depth_index": i, "depth_name": na, "rel_depth": i / max(L - 1, 1),
                     "cka": linear_cka(a, f_ft[i]), "n_rows": a.shape[0], "dim": a.shape[1]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- driver-facing
def _load_metrics(finetune_root: str, family: str, tag: str, model: str, variant: str) -> Optional[Dict[str, Any]]:
    p = Path(finetune_root, family, tag, model, variant, "metrics.json")
    return json.loads(p.read_text()) if p.exists() else None


def _mean_miou(metrics: Optional[Dict[str, Any]], conds: Sequence[str]) -> float:
    if not metrics or not conds:
        return float("nan")
    v = [metrics["conditions"].get(c, {}).get("mIoU", np.nan) for c in conds]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def run_forgetting(family: str, jobs: Sequence[Dict[str, Any]], embed_dir: str, work_dir: str, out_dir: str,
                   finetune_root: str, params: Optional[Mapping[str, Any]] = None, seed: int = 0,
                   device: Optional[str] = None, predictors: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """Representation forgetting for one fine-tuning family.

    Each job: ``{"run", "model", "variant", "ft_name", "id_conditions", "ood_conditions",
    "retention": [{"condition", "family", "relation", "spec", "pre_kwargs"}], "ft_ckpt"}``.
    ``spec``/``pre_kwargs`` (condition loader + encoder kwargs of the retention family)
    are only needed for layer-wise CKA.
    """
    p = {**DEFAULTS, **dict(params or {})}
    lw = {**DEFAULTS["layerwise"], **dict(p.get("layerwise") or {})}
    out = Path(out_dir, "forgetting", family)
    out.mkdir(parents=True, exist_ok=True)
    rows, lrows = [], []
    cache: Dict[Tuple[str, str], CondEmbeddings] = {}

    def emb(cond: str, model: str) -> CondEmbeddings:
        key = (cond, model)
        if key not in cache:
            cache[key] = CondEmbeddings(embed_dir, work_dir, cond, model,
                                        max_tokens_per_chip=int(p["max_tokens_per_chip"]), seed=seed)
        return cache[key]

    for job in jobs:
        for ret in job["retention"]:
            cond = ret["condition"]
            base = {"run": job["run"], "model": job["model"], "variant": job["variant"],
                    "finetuned_model": job["ft_name"], "retention_condition": cond,
                    "retention_family": ret["family"], "relation": ret["relation"]}
            try:
                A, B = emb(cond, job["model"]), emb(cond, job["ft_name"])
                if len(A) < 3 or len(B) < 3:
                    logger.warning("[forgetting] %s / %s: missing embeddings on %s", job["model"], job["ft_name"], cond)
                    continue
                rows.append({**base, **retention_metrics(A, B, p, seed=seed)})
            except Exception as exc:
                logger.warning("[forgetting] %s on %s failed: %s", job["ft_name"], cond, exc)
            if lw.get("enabled", True) and ret.get("spec") and job.get("ft_ckpt") and ret.get("layerwise", True):
                try:
                    df = layerwise_cka(job["model"], ret["pre_kwargs"], job["ft_ckpt"], ret["spec"], lw, device)
                    for r in df.to_dict("records"):
                        lrows.append({**base, **r})
                except Exception as exc:
                    logger.warning("[forgetting] layer-wise CKA %s on %s failed: %s", job["ft_name"], cond, exc)
        # cache is per family; drop fine-tuned entries to bound memory
        for k in [k for k in cache if k[1] == job["ft_name"]]:
            cache.pop(k, None)

    ret_df = pd.DataFrame(rows)
    ret_df.to_csv(out / "retention_metrics.csv", index=False)
    lw_df = pd.DataFrame(lrows)
    lw_df.to_csv(out / "layerwise_cka.csv", index=False)
    trade = tradeoff_table(jobs, ret_df, finetune_root, family, out_dir)
    if predictors is not None and not predictors.empty and not trade.empty:
        trade = trade.merge(predictors, on="model", how="left")
    trade.to_csv(out / "tradeoff.csv", index=False)
    corr = forgetting_correlations(trade)
    corr.to_csv(out / "correlations.csv", index=False)
    if not lw_df.empty:
        depth_summary(lw_df).to_csv(out / "layerwise_summary.csv", index=False)
    try:
        plot_forgetting(trade, lw_df, out, p.get("figure_formats", ["png"]))
    except Exception as exc:
        logger.warning("[forgetting] plotting failed: %s", exc)
    return {"retention": ret_df, "layerwise": lw_df, "tradeoff": trade, "correlations": corr}


def tradeoff_table(jobs: Sequence[Dict[str, Any]], ret_df: pd.DataFrame, finetune_root: str, family: str,
                   out_dir: str) -> pd.DataFrame:
    wc = pd.DataFrame()
    pf = Path(out_dir, "finetune_compare", family, "pretrained_vs_finetuned.csv")
    if pf.exists():
        try:
            wc = pd.read_csv(pf)
        except Exception:
            wc = pd.DataFrame()
    rows = []
    for job in jobs:
        lin = _load_metrics(finetune_root, family, job["run"], job["model"], "linear")
        ftm = _load_metrics(finetune_root, family, job["run"], job["model"], job["variant"])
        rec: Dict[str, Any] = {"run": job["run"], "model": job["model"], "variant": job["variant"],
                               "finetuned_model": job["ft_name"]}
        rec["id_mIoU_linear"] = _mean_miou(lin, job.get("id_conditions", []))
        rec["id_mIoU_ft"] = _mean_miou(ftm, job.get("id_conditions", []))
        rec["id_gain"] = rec["id_mIoU_ft"] - rec["id_mIoU_linear"]
        rec["ood_mIoU_linear"] = _mean_miou(lin, job.get("ood_conditions", []))
        rec["ood_mIoU_ft"] = _mean_miou(ftm, job.get("ood_conditions", []))
        rec["ood_change"] = rec["ood_mIoU_ft"] - rec["ood_mIoU_linear"]
        if not ret_df.empty:
            g = ret_df[(ret_df.run == job["run"]) & (ret_df.finetuned_model == job["ft_name"])]
            for rel in ("same_family", "cross_family", "ood"):
                h = g[g.relation == rel]
                for col in [c for c in g.columns if c.startswith(("forgetting_", "rel_forgetting_"))] + ["cka_tokens"]:
                    if col in h and len(h):
                        rec[f"{col}__{rel}"] = float(h[col].mean())
            h = g[g.relation != "ood"]
            for col in [c for c in g.columns if c.startswith(("forgetting_", "rel_forgetting_"))] + ["cka_tokens", "class_geometry_rho"]:
                if col in h and len(h):
                    rec[f"{col}__retention"] = float(h[col].mean())
        if not wc.empty and "finetuned_model" in wc:
            w = wc[(wc.run == job["run"]) & (wc.finetuned_model == job["ft_name"])]
            for col in ("weight_rel_change_mean", "weight_rel_change_last_quartile_depth",
                        "weight_rel_change_first_quartile_depth"):
                if col in w and len(w):
                    rec[col] = float(w[col].iloc[0])
        rows.append(rec)
    return pd.DataFrame(rows)


def forgetting_correlations(trade: pd.DataFrame, min_n: int = 4) -> pd.DataFrame:
    from scipy.stats import spearmanr
    if trade.empty:
        return pd.DataFrame()
    ys = [c for c in trade.columns if c.startswith("forgetting_") and c.endswith("__retention")]
    skip = {"run", "model", "variant", "finetuned_model"}
    xs = [c for c in trade.columns if c not in skip and c not in ys and not c.startswith(("forgetting_", "rel_forgetting_"))
          and pd.api.types.is_numeric_dtype(trade[c])]
    rows = []
    for (run, var), g in trade.groupby(["run", "variant"]):
        for y in ys:
            for x in xs:
                h = g[[x, y]].dropna()
                if len(h) >= min_n and h[x].nunique() > 1 and h[y].nunique() > 1:
                    r = spearmanr(h[x], h[y])
                    rows.append({"run": run, "variant": var, "x": x, "y": y, "n_models": len(h),
                                 "spearman_rho": float(r.statistic), "p_value": float(r.pvalue)})
    # pooled over variants (learning-forgetting frontier across adaptation strategies)
    for run, g in trade.groupby("run"):
        for y in ys:
            h = g[["id_gain", y]].dropna()
            if len(h) >= min_n and h["id_gain"].nunique() > 1:
                r = spearmanr(h["id_gain"], h[y])
                rows.append({"run": run, "variant": "ALL", "x": "id_gain", "y": y, "n_models": len(h),
                             "spearman_rho": float(r.statistic), "p_value": float(r.pvalue)})
    return pd.DataFrame(rows)


def depth_summary(lw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in lw.groupby(["run", "model", "variant", "relation"]):
        g = g.sort_values("rel_depth")
        if len(g) < 2:
            continue
        from scipy.stats import spearmanr
        half = g.rel_depth >= 0.5
        rows.append(dict(zip(["run", "model", "variant", "relation"], keys),
                         cka_shallow=float(g.loc[~half, "cka"].mean()), cka_deep=float(g.loc[half, "cka"].mean()),
                         deep_minus_shallow=float(g.loc[half, "cka"].mean() - g.loc[~half, "cka"].mean()),
                         depth_rho=float(spearmanr(g.rel_depth, g.cka).statistic) if g.cka.nunique() > 1 else np.nan))
    return pd.DataFrame(rows)


def plot_forgetting(trade: pd.DataFrame, lw: pd.DataFrame, out: Path, formats: Sequence[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ycol = next((c for c in ("forgetting_linear__retention", "forgetting_knn__retention") if c in trade), None)
    if ycol and "id_gain" in trade and trade[[ycol, "id_gain"]].dropna().shape[0]:
        for run, g in trade.groupby("run"):
            fig, ax = plt.subplots(figsize=(5, 4))
            for var, h in g.groupby("variant"):
                ax.scatter(h["id_gain"], h[ycol], label=var, s=28)
                for _, r in h.iterrows():
                    ax.annotate(str(r["model"]).split("__")[0], (r["id_gain"], r[ycol]), fontsize=6, alpha=0.7)
            ax.axhline(0, lw=0.5, c="k"); ax.axvline(0, lw=0.5, c="k")
            ax.set_xlabel("ID gain (mIoU ft - linear probe)"); ax.set_ylabel(ycol.replace("__", " / "))
            ax.set_title(f"Learning vs forgetting: {run}", fontsize=9); ax.legend(fontsize=7)
            fig.tight_layout()
            for f in formats:
                fig.savefig(out / f"tradeoff_{run}.{f}", dpi=200)
            plt.close(fig)
    if not lw.empty:
        for run, g in lw.groupby("run"):
            fig, ax = plt.subplots(figsize=(5, 4))
            for (m, v), h in g.groupby(["model", "variant"]):
                h = h.groupby("rel_depth", as_index=False)["cka"].mean()
                ax.plot(h.rel_depth, h.cka, marker="o", ms=3, label=f"{m.split('__')[0]}|{v}")
            ax.set_xlabel("relative depth"); ax.set_ylabel("CKA(pre, ft) on retention sets"); ax.set_ylim(0, 1.02)
            ax.set_title(f"Layer-wise forgetting: {run}", fontsize=9); ax.legend(fontsize=6)
            fig.tight_layout()
            for f in formats:
                fig.savefig(out / f"layerwise_{run}.{f}", dpi=200)
            plt.close(fig)


# ----------------------------------------------------------------------------- sequential fine-tuning
def continual_metrics(R: np.ndarray) -> Dict[str, float]:
    """R[j, i]: performance on task i after training step j (lower triangle used)."""
    T = R.shape[0]
    out = {"ACC": float(np.nanmean(R[T - 1, :T]))}
    if T > 1:
        out["BWT"] = float(np.nanmean([R[T - 1, i] - R[i, i] for i in range(T - 1)]))
        out["forgetting"] = float(np.nanmean([np.nanmax(R[i:T - 1, i]) - R[T - 1, i] for i in range(T - 1)]))
    return out


def run_sequential(name: str, models: Sequence[str], steps: Sequence[Dict[str, Any]], out_dir: str,
                   finetune_root: str, mode: str = "full", hp: Optional[Dict[str, Any]] = None,
                   regularizer: Optional[str] = None, reg_weight: float = 0.0, fisher_batches: int = 50,
                   num_workers: int = 0, device: Optional[str] = None, seed: int = 0,
                   overwrite: bool = False) -> pd.DataFrame:
    """Fine-tune each model through ``steps`` in order; report accuracy matrix / BWT.

    Each step: ``{"label", "family", "tag", "train": {cls, kwargs}, "val": {...}|None,
    "id_conditions": {cond: {cls, kwargs}}, "num_classes", "encoder_kwargs": {model: kw}}``.
    Step ``j`` starts from the backbone of step ``j-1``; each step keeps its own head,
    which is frozen when that task is re-evaluated after later steps.
    """
    from reptaxonomy.gfm.encoders import GFMEncoder
    from reptaxonomy.gfm.finetune import _loader, estimate_fisher, evaluate_checkpoint, load_head, run_finetune
    hp = dict(hp or {})
    out = Path(out_dir, "forgetting", "sequential", name)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(finetune_root, "sequential", name)
    mat_rows, sum_rows = [], []
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    for m in models:
        T = len(steps)
        R = np.full((T, T), np.nan)
        prev_ckpt: Optional[str] = None
        fisher_acc: Optional[Dict[str, torch.Tensor]] = None
        ckpts: List[str] = []
        for j, st in enumerate(steps):
            sdir = root / m / f"step{j}_{st['label']}"
            enc_kw = dict(st["encoder_kwargs"][m])
            if prev_ckpt:
                enc_kw["checkpoint"] = prev_ckpt
            fisher_path = None
            if regularizer == "ewc" and fisher_acc is not None:
                fisher_path = str(sdir / "fisher_prev.pt"); sdir.mkdir(parents=True, exist_ok=True)
                torch.save(fisher_acc, fisher_path)
            run_finetune(m, enc_kw, st["train"], st.get("val"), st["id_conditions"], int(st["num_classes"]), sdir,
                         mode=mode, num_workers=num_workers, device=device, seed=seed, overwrite=overwrite,
                         regularizer=regularizer if j > 0 else None, reg_weight=reg_weight, fisher_path=fisher_path,
                         variant=f"seq-{mode}", **hp)
            ck = str(sdir / "model.pt"); ckpts.append(ck); prev_ckpt = ck
            if regularizer == "ewc":
                try:
                    enc = GFMEncoder(m, **dict(st["encoder_kwargs"][m], checkpoint=ck)).to(dev)
                    head = load_head(ck, dev)
                    tl = _loader(st["train"]["cls"], st["train"]["kwargs"], int(hp.get("batch_size", 8)), True,
                                 num_workers, hp.get("max_train_samples"))
                    f_new = estimate_fisher(enc, head, tl, dev, fisher_batches)
                    fisher_acc = f_new if fisher_acc is None else {k: fisher_acc.get(k, 0) + v for k, v in f_new.items()}
                    del enc
                except Exception as exc:
                    logger.warning("[sequential] Fisher estimation failed (%s step %d): %s", m, j, exc)
            # re-evaluate every task seen so far with its own (frozen) head
            for i in range(j + 1):
                si = steps[i]
                try:
                    res = evaluate_checkpoint(m, si["encoder_kwargs"][m], ck, si["id_conditions"], int(si["num_classes"]),
                                              batch_size=int(hp.get("batch_size", 8)), num_workers=num_workers,
                                              max_eval_samples=hp.get("max_eval_samples"), device=device,
                                              head_ckpt=ckpts[i])
                    v = [c["mIoU"] for c in res["conditions"].values() if np.isfinite(c["mIoU"])]
                    R[j, i] = float(np.mean(v)) if v else np.nan
                except Exception as exc:
                    logger.warning("[sequential] eval %s step %d task %d failed: %s", m, j, i, exc)
                mat_rows.append({"model": m, "after_step": j, "after_label": st["label"], "task": i,
                                 "task_label": si["label"], "mIoU": R[j, i]})
        sum_rows.append({"model": m, "sequence": "->".join(s["label"] for s in steps), "mode": mode,
                         "regularizer": regularizer or "none", **continual_metrics(R)})
    mat = pd.DataFrame(mat_rows); mat.to_csv(out / "accuracy_matrix.csv", index=False)
    summ = pd.DataFrame(sum_rows); summ.to_csv(out / "sequential_summary.csv", index=False)
    return summ
