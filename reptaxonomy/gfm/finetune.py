"""Fine-tuning / probing of gfmtools encoders for segmentation (FTW, EPA/NLCD, MADOS).

gfmtools' ``engines`` (BaseTask/SegmentationTask), ``pipelines.ed_pipeline`` and
``models.decoders`` (SegFCN/UPerNet/LTAE) are stubs in the current release, so
the training loop lives here, on top of ``reptaxonomy.gfm.encoders.GFMEncoder``.

Modes
-----
linear   frozen pretrained encoder + trained head  (representation-quality probe;
         backbone weights are unchanged, so its representations equal the
         pretrained ones)
partial  last ``unfreeze_fraction`` of encoder parameters + head trained
full     whole encoder + head trained (encoder LR = lr * backbone_lr_mult)
lpft     LP-FT (Kumar et al., ICLR 2022): head initialised from the ``linear`` run
         (``lp_head_from``) or warmed up for ``lp_epochs`` on the frozen encoder, then
         full fine-tuning
lora     encoder frozen; rank-``lora_rank`` adapters on nn.Linear layers whose name
         matches ``lora_targets`` (ViT families); merged into the weights before saving
         so the checkpoint loads into the plain architecture

Forgetting regularisers (any mode that updates the backbone)
--------------------------------------------------------------
regularizer='l2sp'  lambda/2 * ||theta - theta_0||^2   (theta_0 = backbone at load time)
regularizer='ewc'   lambda/2 * sum_i F_i (theta_i - theta*_i)^2, diagonal Fisher F from
                    ``fisher_path`` (``estimate_fisher``; previous task in sequential runs)

``wise_interpolate`` builds WiSE-FT weight-space ensembles (Wortsman et al., CVPR 2022)
between the linear-probe run (pretrained encoder + LP head) and a fine-tuned run.

Outputs (``<out_root>/finetune/<dataset>/<run_tag>/<model>/<mode>/``)
---------------------------------------------------------------------
model.pt          {"backbone": state_dict, "head": state_dict, "meta": {...}}
metrics.json      per-condition mIoU / OA / macro-F1 / per-class IoU, train curve
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reptaxonomy.gfm.datasets import IGNORE_INDEX, LABEL_KEY, REPSET_CLASSES
from reptaxonomy.gfm.encoders import GFMEncoder
from reptaxonomy.util.experiment_init_utils import build_forward_instruments_collate_fn

logger = logging.getLogger(__name__)


class SegHead(nn.Module):
    """Light decoder on the encoder's final feature grid.

    kind='fcn'          conv3x3-BN-ReLU-conv1x1, bilinear upsample to label size
    kind='progressive'  two x2 conv-upsampling stages before the classifier
    kind='linear'       1x1 conv only (strict linear probe)
    """

    def __init__(self, in_ch: int, num_classes: int, kind: str = "fcn", hidden: int = 256) -> None:
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.body = nn.Identity()
            self.cls = nn.Conv2d(in_ch, num_classes, 1)
        elif kind == "progressive":
            self.body = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.BatchNorm2d(hidden // 2), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(hidden // 2, hidden // 2, 3, padding=1), nn.BatchNorm2d(hidden // 2), nn.ReLU(inplace=True),
            )
            self.cls = nn.Conv2d(hidden // 2, num_classes, 1)
        else:
            self.body = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1), nn.BatchNorm2d(hidden),
                                      nn.ReLU(inplace=True), nn.Dropout2d(0.1))
            self.cls = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, feats: torch.Tensor, out_hw: Sequence[int]) -> torch.Tensor:
        y = self.cls(self.body(feats))
        return F.interpolate(y, size=tuple(out_hw), mode="bilinear", align_corners=False)


def _labels(batch: Mapping[str, Any]) -> torch.Tensor:
    return torch.as_tensor(batch[LABEL_KEY][LABEL_KEY]["data"]).long()


def _loader(cls: str, kwargs: Dict[str, Any], batch_size: int, shuffle: bool, workers: int,
            max_samples: Optional[int] = None) -> DataLoader:
    kw = dict(kwargs)
    if max_samples is not None:
        kw["max_samples"] = max_samples if kw.get("max_samples") is None else min(int(kw["max_samples"]), int(max_samples))
    ds = REPSET_CLASSES[cls](**kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=build_forward_instruments_collate_fn({}), drop_last=shuffle and len(ds) > batch_size)


# ----------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen; ``merge()`` folds BA into W."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        self.rank, self.scale = int(rank), float(alpha) / max(int(rank), 1)
        self.lora_A = nn.Parameter(torch.zeros(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        self.base.weight += (self.lora_B @ self.lora_A).to(self.base.weight.dtype) * self.scale
        return self.base


def inject_lora(module: nn.Module, targets: Sequence[str], rank: int, alpha: float) -> List[nn.Parameter]:
    """Replace matching nn.Linear children in-place; return the adapter parameters."""
    params: List[nn.Parameter] = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(t in name for t in targets):
            lo = LoRALinear(child, rank, alpha)
            setattr(module, name, lo)
            params += [lo.lora_A, lo.lora_B]
        else:
            params += inject_lora(child, targets, rank, alpha)
    return params


def merge_lora(module: nn.Module) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merge()); n += 1
        else:
            n += merge_lora(child)
    return n


# ----------------------------------------------------------------------------- regularisers
def _anchor(enc: GFMEncoder) -> Dict[str, torch.Tensor]:
    return {k: p.detach().clone() for k, p in enc.backbone.named_parameters()}


def reg_penalty(enc: GFMEncoder, anchor: Dict[str, torch.Tensor], fisher: Optional[Dict[str, torch.Tensor]],
                weight: float) -> torch.Tensor:
    tot = None
    for k, p in enc.backbone.named_parameters():
        if not p.requires_grad or k not in anchor:
            continue
        d = (p - anchor[k].to(p.device)) ** 2
        if fisher is not None:
            if k not in fisher:
                continue
            d = d * fisher[k].to(p.device)
        s = d.sum()
        tot = s if tot is None else tot + s
    if tot is None:
        return torch.zeros((), device=next(enc.parameters()).device)
    return 0.5 * float(weight) * tot


def estimate_fisher(enc: GFMEncoder, head: "SegHead", loader: DataLoader, device: torch.device,
                    max_batches: int = 50) -> Dict[str, torch.Tensor]:
    """Diagonal empirical Fisher of the segmentation loss w.r.t. backbone parameters."""
    enc.eval(); head.eval()
    for p in enc.backbone.parameters():
        p.requires_grad = True
    F_: Dict[str, torch.Tensor] = {k: torch.zeros_like(p, device="cpu") for k, p in enc.backbone.named_parameters()}
    n = 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        y = _labels(batch).to(device)
        enc.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(enc.forward_features(batch, device=device), y.shape[-2:]), y,
                               ignore_index=IGNORE_INDEX)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        for k, p in enc.backbone.named_parameters():
            if p.grad is not None:
                F_[k] += p.grad.detach().cpu() ** 2
        n += 1
    return {k: v / max(n, 1) for k, v in F_.items()}


def _set_trainable(enc: GFMEncoder, mode: str, unfreeze_fraction: float) -> List[nn.Parameter]:
    params = list(enc.backbone.named_parameters())
    for _, p in params:
        p.requires_grad = mode in ("full", "lpft")
    if mode == "partial":
        k = max(1, int(round(len(params) * float(unfreeze_fraction))))
        for _, p in params[-k:]:
            p.requires_grad = True
    return [p for _, p in params if p.requires_grad]


@torch.no_grad()
def evaluate(enc: GFMEncoder, head: SegHead, loader: DataLoader, num_classes: int, device: torch.device,
             max_batches: Optional[int] = None) -> Dict[str, Any]:
    enc.eval(); head.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        y = _labels(batch).to(device)
        logits = head(enc.forward_features(batch, device=device), y.shape[-2:])
        pred = logits.argmax(1)
        m = y != IGNORE_INDEX
        idx = (y[m] * num_classes + pred[m]).cpu()
        cm += torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return confusion_metrics(cm)


def confusion_metrics(cm: torch.Tensor) -> Dict[str, Any]:
    cm = cm.double()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    support = cm.sum(1)
    iou = tp / (tp + fp + fn).clamp_min(1)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    present = support > 0
    return {
        "mIoU": float(iou[present].mean()) if present.any() else float("nan"),
        "macro_F1": float(f1[present].mean()) if present.any() else float("nan"),
        "OA": float(tp.sum() / cm.sum().clamp_min(1)),
        "per_class_IoU": [float(v) for v in iou],
        "support": [int(v) for v in support],
        "n_pixels": int(cm.sum()),
    }


def load_head(ckpt: str | Path, device: torch.device | str = "cpu") -> "SegHead":
    """Rebuild the SegHead saved in a ``model.pt`` written by ``run_finetune``."""
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = st["meta"]
    kind = meta.get("head", "fcn")
    head = SegHead(int(meta["feature_channels"]), int(meta["num_classes"]),
                   kind=("linear" if kind == "linear" else kind))
    head.load_state_dict(st["head"])
    return head.to(device).eval()


def wise_interpolate(pre_ckpt: str | Path, ft_ckpt: str | Path, alpha: float, out_ckpt: str | Path) -> Path:
    """WiSE-FT: theta_alpha = (1 - alpha) theta_LP + alpha theta_FT for backbone AND head.

    ``pre_ckpt`` is the ``linear`` run (pretrained backbone + LP head), so alpha = 0 is
    the frozen-probe model and alpha = 1 the fine-tuned model. Non-float buffers
    (BatchNorm counters) are taken from the fine-tuned model.
    """
    a = torch.load(pre_ckpt, map_location="cpu", weights_only=False)
    b = torch.load(ft_ckpt, map_location="cpu", weights_only=False)

    def mix(sa: Dict[str, torch.Tensor], sb: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, vb in sb.items():
            va = sa.get(k)
            if va is not None and torch.is_floating_point(vb) and va.shape == vb.shape:
                out[k] = (1 - alpha) * va.float() + alpha * vb.float()
                out[k] = out[k].to(vb.dtype)
            else:
                out[k] = vb
        return out

    if a["meta"].get("head") != b["meta"].get("head"):
        raise ValueError("WiSE-FT needs the same head kind in the linear and fine-tuned runs")
    meta = dict(b["meta"], mode=f"wise{alpha:.2f}", wise_alpha=float(alpha),
                wise_from=[str(pre_ckpt), str(ft_ckpt)])
    out = Path(out_ckpt); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": mix(a["backbone"], b["backbone"]), "head": mix(a["head"], b["head"]), "meta": meta}, out)
    return out


def evaluate_checkpoint(model_name: str, encoder_kwargs: Dict[str, Any], ckpt: str | Path,
                        eval_conditions: Dict[str, Dict[str, Any]], num_classes: int, batch_size: int = 8,
                        num_workers: int = 0, max_eval_samples: Optional[int] = None,
                        device: Optional[str] = None, head_ckpt: Optional[str | Path] = None) -> Dict[str, Any]:
    """mIoU etc. of (backbone from ``ckpt``, head from ``head_ckpt`` or ``ckpt``) on every condition."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = GFMEncoder(model_name, **dict(encoder_kwargs, checkpoint=str(ckpt))).to(dev).eval()
    head = load_head(head_ckpt or ckpt, dev)
    if head.cls.out_channels != num_classes:
        raise ValueError(f"head has {head.cls.out_channels} classes, condition has {num_classes}")
    res: Dict[str, Any] = {"model": model_name, "checkpoint": str(ckpt), "head_checkpoint": str(head_ckpt or ckpt),
                           "conditions": {}}
    for name, spec in eval_conditions.items():
        el = _loader(spec["cls"], spec["kwargs"], batch_size, False, num_workers, max_eval_samples)
        res["conditions"][name] = evaluate(enc, head, el, num_classes, dev)
    return res


def run_finetune(model_name: str, encoder_kwargs: Dict[str, Any], train: Dict[str, Any],
                 val: Optional[Dict[str, Any]], eval_conditions: Dict[str, Dict[str, Any]],
                 num_classes: int, out_dir: str | Path, mode: str = "full", head: str = "fcn",
                 epochs: int = 20, batch_size: int = 8, lr: float = 1e-4, backbone_lr_mult: float = 0.1,
                 weight_decay: float = 0.05, unfreeze_fraction: float = 0.25, max_train_samples: Optional[int] = None,
                 max_eval_samples: Optional[int] = None, num_workers: int = 4, device: Optional[str] = None,
                 seed: int = 0, early_stop_patience: int = 5, amp: bool = False, overwrite: bool = False,
                 regularizer: Optional[str] = None, reg_weight: float = 0.0, fisher_path: Optional[str] = None,
                 lp_head_from: Optional[str] = None, lp_epochs: int = 0, lora_rank: int = 8,
                 lora_alpha: float = 16.0, lora_targets: Sequence[str] = ("qkv", "proj", "fc1", "fc2"),
                 variant: Optional[str] = None) -> Dict[str, Any]:
    """Train one (model, mode) on ``train`` and evaluate on every eval condition.

    ``train`` / ``val`` / ``eval_conditions[name]`` are ``{"cls": ..., "kwargs": {...}}``.
    ``variant`` is the label written to metrics.json (defaults to ``mode``); see the
    module docstring for ``lpft`` / ``lora`` / ``regularizer``.
    """
    out_dir = Path(out_dir)
    metrics_path = out_dir / "metrics.json"
    ckpt_path = out_dir / "model.pt"
    if metrics_path.exists() and ckpt_path.exists() and not overwrite:
        logger.info("Fine-tune exists, skipping: %s", out_dir)
        return json.loads(metrics_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    enc = GFMEncoder(model_name, **encoder_kwargs).to(dev)
    tl = _loader(train["cls"], train["kwargs"], batch_size, True, num_workers, max_train_samples)
    vl = _loader(val["cls"], val["kwargs"], batch_size, False, num_workers, max_eval_samples) if val else None

    with torch.no_grad():
        probe = next(iter(tl))
        c = enc.forward_features(probe, device=dev).shape[1]
    seg = SegHead(c, num_classes, kind=("linear" if head == "linear" else head)).to(dev)
    if mode == "lpft" and lp_head_from and Path(lp_head_from).exists():
        seg.load_state_dict(torch.load(lp_head_from, map_location="cpu", weights_only=False)["head"])
        logger.info("[LP-FT] head initialised from %s", lp_head_from)
        lp_epochs = 0
    anchor = _anchor(enc) if regularizer in ("l2sp", "ewc") and reg_weight > 0 else None
    fisher = None
    if regularizer == "ewc" and anchor is not None:
        if fisher_path and Path(fisher_path).exists():
            fisher = torch.load(fisher_path, map_location="cpu", weights_only=False)
        else:
            logger.warning("EWC requested without a Fisher file (%s); falling back to L2-SP", fisher_path)
    if mode == "lora":
        for p in enc.backbone.parameters():
            p.requires_grad = False
        bb_params = inject_lora(enc.backbone, list(lora_targets), lora_rank, lora_alpha)
        if not bb_params:
            raise ValueError(f"{model_name}: no nn.Linear layers match lora_targets={list(lora_targets)}")
        enc.to(dev)
    else:
        bb_params = _set_trainable(enc, mode, unfreeze_fraction)
    groups = [{"params": seg.parameters(), "lr": lr}]
    if bb_params:
        groups.append({"params": bb_params, "lr": lr * backbone_lr_mult})
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    total = max(1, epochs * max(1, len(tl)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")

    best, best_state, bad, curve = -1.0, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        frozen = mode == "linear" or (mode == "lpft" and ep < lp_epochs)
        enc.train(not frozen); seg.train()
        if frozen:
            enc.eval()
        tot, n = 0.0, 0
        for batch in tl:
            y = _labels(batch).to(dev)
            with torch.autocast(dev.type, enabled=amp and dev.type == "cuda"):
                if frozen:
                    with torch.no_grad():
                        f = enc.forward_features(batch, device=dev)
                else:
                    f = enc.forward_features(batch, device=dev)
                loss = F.cross_entropy(seg(f, y.shape[-2:]), y, ignore_index=IGNORE_INDEX)
                if anchor is not None and not frozen:
                    loss = loss + reg_penalty(enc, anchor, fisher, reg_weight)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(seg.parameters()) + bb_params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss.detach()) * y.shape[0]; n += y.shape[0]
        row = {"epoch": ep, "train_loss": tot / max(n, 1)}
        if vl is not None:
            vm = evaluate(enc, seg, vl, num_classes, dev)
            row["val_mIoU"] = vm["mIoU"]
            score = vm["mIoU"]
        else:
            score = -row["train_loss"]
        curve.append(row)
        logger.info("[%s|%s] %s", model_name, mode, row)
        if score > best:
            best, bad = score, 0
            best_state = {"backbone": {k: v.detach().cpu().clone() for k, v in enc.backbone.state_dict().items()},
                          "head": {k: v.detach().cpu().clone() for k, v in seg.state_dict().items()}}
        else:
            bad += 1
            if early_stop_patience and bad >= early_stop_patience:
                break
    if best_state is not None:
        enc.backbone.load_state_dict(best_state["backbone"]); seg.load_state_dict(best_state["head"])
    if mode == "lora":
        logger.info("[LoRA] merged %d adapters", merge_lora(enc.backbone))

    results: Dict[str, Any] = {"model": model_name, "mode": mode, "variant": variant or mode, "head": head,
                               "regularizer": regularizer, "reg_weight": reg_weight, "epochs_run": len(curve),
                               "train_seconds": time.time() - t0, "curve": curve, "conditions": {}}
    for name, spec in eval_conditions.items():
        el = _loader(spec["cls"], spec["kwargs"], batch_size, False, num_workers, max_eval_samples)
        results["conditions"][name] = evaluate(enc, seg, el, num_classes, dev)
    torch.save({"backbone": enc.backbone.state_dict(), "head": seg.state_dict(),
                "meta": {"model": model_name, "mode": mode, "variant": variant or mode, "head": head,
                         "num_classes": num_classes, "feature_channels": c}}, ckpt_path)
    metrics_path.write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune one gfmtools encoder on a rep_taxonomy condition")
    ap.add_argument("--spec", required=True, help="JSON written by gfm_experiment (one fine-tune job)")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    run_finetune(**spec)


if __name__ == "__main__":
    main()
