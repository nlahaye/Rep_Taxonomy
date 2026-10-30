"""Pre-trained vs fine-tuned comparison.

For every fine-tuning run (tag), base model ``m`` and fine-tuned model
``m__ft__<tag>-<mode>`` this module reports, per evaluation condition:

Representation change (same chips, same grid cells -> fully paired)
  cka_pooled, cka_tokens        linear CKA pre vs fine-tuned (1 = unchanged geometry)
  mutual_knn@k                  chip-neighbourhood overlap pre vs fine-tuned
  mean_cosine_tokens            mean cosine between matched cells (dims must agree)

Performance (from finetune/<...>/metrics.json)
  mIoU_linear                   frozen pre-trained encoder + trained head (probe)
  mIoU_<mode>                   fine-tuned encoder + head
  delta_mIoU                    mIoU_<mode> - mIoU_linear
  id_ood_gap_linear / _<mode>   mean ID mIoU - mean OOD mIoU (FTW shift tasks, EPA held-out regions)

Weights
  weight_rel_change_{mean,median,max,first_quartile_depth,last_quartile_depth}
                                ||W_ft - W_pre||_F / ||W_pre||_F per parameter tensor

and ``summary_correlations.csv``: Spearman rho across models between
representation change (1 - cka_tokens) and delta_mIoU / OOD-gap change.

Outputs: ``<out_dir>/finetune_compare/<dataset>/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from reptaxonomy.gfm.cross_condition import CondEmbeddings, paired_metrics
from reptaxonomy.intercompare.reference_similarity import linear_cka

logger = logging.getLogger(__name__)


def ft_name(model: str, tag: str, mode: str) -> str:
    return f"{model}__ft__{tag}-{mode}"


def weight_change(pre_state: Dict[str, torch.Tensor], ft_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    rel = []
    for k, w0 in pre_state.items():
        if k not in ft_state or not torch.is_floating_point(w0) or w0.ndim < 2:
            continue
        w1 = ft_state[k]
        if w1.shape != w0.shape:
            continue
        n0 = float(w0.float().norm())
        if n0 > 0:
            rel.append(float((w1.float() - w0.float()).norm()) / n0)
    if not rel:
        return {}
    r = np.asarray(rel)
    q = max(1, len(r) // 4)
    return {"weight_rel_change_mean": float(r.mean()), "weight_rel_change_median": float(np.median(r)),
            "weight_rel_change_max": float(r.max()), "weight_rel_change_first_quartile_depth": float(r[:q].mean()),
            "weight_rel_change_last_quartile_depth": float(r[-q:].mean()), "n_weight_tensors": float(len(r))}


def _miou(metrics: Optional[Dict[str, Any]], cond: str) -> float:
    try:
        return float(metrics["conditions"][cond]["mIoU"])
    except Exception:
        return float("nan")


def run_finetune_compare(embed_dir: str, work_dir: str, out_dir: str, dataset: str,
                         runs: Sequence[Dict[str, Any]], models: Sequence[str], modes: Sequence[str],
                         finetune_root: str, encoder_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
                         compute_weight_change: bool = True, seed: int = 0) -> pd.DataFrame:
    out = Path(out_dir, "finetune_compare", dataset)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for run in runs:
        tag = run["tag"]
        for m in models:
            lin = _load_metrics(finetune_root, dataset, tag, m, "linear")
            for mode in modes:
                if mode == "linear":
                    continue
                ftm = _load_metrics(finetune_root, dataset, tag, m, mode)
                name = ft_name(m, tag, mode)
                wc: Dict[str, float] = {}
                if compute_weight_change and ftm is not None:
                    wc = _weight_change_for(m, finetune_root, dataset, tag, mode, (encoder_kwargs or {}).get(m, {}))
                for cond in run["eval_conditions"]:
                    row: Dict[str, Any] = {"run": tag, "model": m, "finetuned_model": name, "mode": mode,
                                           "condition": cond,
                                           "role": "id" if cond in run.get("id_conditions", []) else
                                                   ("ood" if cond in run.get("ood_conditions", []) else "eval"),
                                           "mIoU_linear": _miou(lin, cond), f"mIoU_ft": _miou(ftm, cond)}
                    row["delta_mIoU"] = row["mIoU_ft"] - row["mIoU_linear"]
                    try:
                        A = CondEmbeddings(embed_dir, work_dir, cond, m, seed=seed)
                        Bm = CondEmbeddings(embed_dir, work_dir, cond, name, seed=seed)
                        if len(A) >= 3 and len(Bm) >= 3:
                            pm = paired_metrics(A, Bm, seed=seed)
                            row["cka_pooled"] = pm.get("paired_cka_pooled", np.nan)
                            row["cka_tokens"] = pm.get("paired_cka_tokens", np.nan)
                            row.update({k.replace("paired_", ""): v for k, v in pm.items()
                                        if k.startswith("paired_mutual_knn")})
                            if A.tokens.shape[1] == Bm.tokens.shape[1]:
                                na = A.tokens / (np.linalg.norm(A.tokens, axis=1, keepdims=True) + 1e-8)
                                nb = Bm.tokens / (np.linalg.norm(Bm.tokens, axis=1, keepdims=True) + 1e-8)
                                if na.shape == nb.shape:
                                    row["mean_cosine_tokens"] = float((na * nb).sum(1).mean())
                    except Exception as exc:
                        logger.warning("pre/ft representation comparison failed for %s on %s: %s", name, cond, exc)
                    row.update(wc)
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pretrained_vs_finetuned.csv", index=False)
    if df.empty:
        return df

    gaps = []
    for (run, m, mode), g in df.groupby(["run", "model", "mode"]):
        idg, oodg = g[g["role"] == "id"], g[g["role"] == "ood"]
        rec = {"run": run, "model": m, "mode": mode,
               "repr_change_tokens": float(1 - g["cka_tokens"].mean()) if "cka_tokens" in g else np.nan,
               "delta_mIoU_mean": float(g["delta_mIoU"].mean())}
        if len(idg) and len(oodg):
            rec["id_ood_gap_linear"] = float(idg["mIoU_linear"].mean() - oodg["mIoU_linear"].mean())
            rec["id_ood_gap_ft"] = float(idg["mIoU_ft"].mean() - oodg["mIoU_ft"].mean())
            rec["gap_change"] = rec["id_ood_gap_ft"] - rec["id_ood_gap_linear"]
            if "cka_tokens" in g:
                rec["repr_change_id"] = float(1 - idg["cka_tokens"].mean())
                rec["repr_change_ood"] = float(1 - oodg["cka_tokens"].mean())
        gaps.append(rec)
    gdf = pd.DataFrame(gaps)
    gdf.to_csv(out / "per_model_summary.csv", index=False)

    from scipy.stats import spearmanr
    corr = []
    for (run, mode), g in gdf.groupby(["run", "mode"]):
        for y in ("delta_mIoU_mean", "gap_change"):
            if y in g and g[["repr_change_tokens", y]].dropna().shape[0] >= 4:
                h = g[["repr_change_tokens", y]].dropna()
                corr.append({"run": run, "mode": mode, "x": "repr_change_tokens", "y": y, "n_models": len(h),
                             "spearman_rho": float(spearmanr(h.iloc[:, 0], h.iloc[:, 1]).statistic)})
    pd.DataFrame(corr).to_csv(out / "summary_correlations.csv", index=False)
    return df


def _load_metrics(root: str, dataset: str, tag: str, model: str, mode: str) -> Optional[Dict[str, Any]]:
    p = Path(root, dataset, tag, model, mode, "metrics.json")
    return json.loads(p.read_text()) if p.exists() else None


def _weight_change_for(model: str, root: str, dataset: str, tag: str, mode: str,
                       enc_kwargs: Dict[str, Any]) -> Dict[str, float]:
    p = Path(root, dataset, tag, model, mode, "model.pt")
    if not p.exists():
        return {}
    try:
        from reptaxonomy.gfm.encoders import GFMEncoder
        pre = GFMEncoder(model, **{k: v for k, v in enc_kwargs.items() if k != "checkpoint"}).backbone.state_dict()
        ft = torch.load(p, map_location="cpu", weights_only=False)["backbone"]
        return weight_change(pre, ft)
    except Exception as exc:
        logger.warning("weight change failed for %s (%s): %s", model, tag, exc)
        return {}
