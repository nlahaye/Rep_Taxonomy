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


def _set_trainable(enc: GFMEncoder, mode: str, unfreeze_fraction: float) -> List[nn.Parameter]:
    params = list(enc.backbone.named_parameters())
    for _, p in params:
        p.requires_grad = mode == "full"
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


def run_finetune(model_name: str, encoder_kwargs: Dict[str, Any], train: Dict[str, Any],
                 val: Optional[Dict[str, Any]], eval_conditions: Dict[str, Dict[str, Any]],
                 num_classes: int, out_dir: str | Path, mode: str = "full", head: str = "fcn",
                 epochs: int = 20, batch_size: int = 8, lr: float = 1e-4, backbone_lr_mult: float = 0.1,
                 weight_decay: float = 0.05, unfreeze_fraction: float = 0.25, max_train_samples: Optional[int] = None,
                 max_eval_samples: Optional[int] = None, num_workers: int = 4, device: Optional[str] = None,
                 seed: int = 0, early_stop_patience: int = 5, amp: bool = False, overwrite: bool = False) -> Dict[str, Any]:
    """Train one (model, mode) on ``train`` and evaluate on every eval condition.

    ``train`` / ``val`` / ``eval_conditions[name]`` are ``{"cls": ..., "kwargs": {...}}``.
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
        enc.train(mode != "linear"); seg.train()
        if mode == "linear":
            enc.eval()
        tot, n = 0.0, 0
        for batch in tl:
            y = _labels(batch).to(dev)
            with torch.autocast(dev.type, enabled=amp and dev.type == "cuda"):
                if mode == "linear":
                    with torch.no_grad():
                        f = enc.forward_features(batch, device=dev)
                else:
                    f = enc.forward_features(batch, device=dev)
                loss = F.cross_entropy(seg(f, y.shape[-2:]), y, ignore_index=IGNORE_INDEX)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(seg.parameters()) + bb_params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss) * y.shape[0]; n += y.shape[0]
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

    results: Dict[str, Any] = {"model": model_name, "mode": mode, "head": head, "epochs_run": len(curve),
                               "train_seconds": time.time() - t0, "curve": curve, "conditions": {}}
    for name, spec in eval_conditions.items():
        el = _loader(spec["cls"], spec["kwargs"], batch_size, False, num_workers, max_eval_samples)
        results["conditions"][name] = evaluate(enc, seg, el, num_classes, dev)
    torch.save({"backbone": enc.backbone.state_dict(), "head": seg.state_dict(),
                "meta": {"model": model_name, "mode": mode, "head": head, "num_classes": num_classes,
                         "feature_channels": c}}, ckpt_path)
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
