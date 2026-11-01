"""End-to-end rep_taxonomy evaluation of the gfmtools encoder suite on FTW / EPA / MADOS.

Two modes (``mode:`` in the YAML, or ``--mode``):

``pretrained``
    prepare -> embed -> pipeline -> cross_condition [-> synthesis]

    * prepare: expand every dataset's ``experiments`` into *conditions* (one
      rep_taxonomy "dataset" each, e.g. ``ftw__ftw-germany-cambodia__ood_test``,
      ``ftw__season-france-a__test``, ``epa__sierra-nevada``, ``mados__test``),
      write manifests / label caches, compute ONE band-statistics set per dataset
      family (on the reference split, so domain shift is not normalised away) and
      check band coverage for every model x dataset.
    * embed: ``reptaxonomy.pipelines.generate_embeddings.run_embedding_export``
      per condition (or the batched ``fast`` backend, same artifacts).
    * pipeline: ``reptaxonomy.pipelines.full_pipeline.run_full_pipeline`` per
      condition (all existing steps: projections, robustness, data kernel,
      reference similarity, invariance, geodesic, Demsar, synthesis).
    * cross_condition: ``reptaxonomy.gfm.cross_condition`` (MMD / C2ST / Frechet /
      class-geometry / kNN transfer / paired CKA / taxonomy stability).
    * geo_id: ``reptaxonomy.robustness.geo_intrinsic_dimension.run_geo_id`` --
      GeoINRID-style global/local intrinsic dimension of chip embeddings at the
      chips' coordinates (manifest ``lats``/``lons``), Moran's I / variogram /
      region tests of local ID, geographic-vs-embedding neighbourhood agreement,
      and (optional) location-encoder references (GeoCLIP, CSP, SatCLIP) at the
      same sites. Per condition and pooled per dataset family.
    * synthesis: ``run_cross_perspective_synthesis`` pooled over the conditions
      of each dataset family.

``finetune_compare``
    everything above, then finetune -> ft_embed -> ft_pipeline -> finetune_compare

    * finetune: ``reptaxonomy.gfm.finetune.run_finetune`` for each FT run (FTW
      shift-task train split, EPA train split, MADOS train) x model x mode
      (``linear`` probe = frozen pretrained baseline; ``full`` / ``partial``
      update the backbone).
    * ft_embed / ft_pipeline: the fine-tuned backbones become extra models
      ``<model>__ft__<tag>-<mode>`` (checkpoint override) and are run through the
      same embedding + full pipeline on the run's eval conditions, with
      ``synthesis.training_pairs = [[pretrained, finetuned], ...]``.
    * finetune_compare: ``reptaxonomy.gfm.compare_finetune`` (mIoU linear vs ft,
      ID/OOD gap change, CKA(pre, ft), weight change, rank correlations).
    * ft_geo_id: geo_id on each run's eval conditions for pretrained + fine-tuned
      models (local-ID maps before/after adaptation; outputs/geo_id/<ds>_<tag>_ft).

Catastrophic forgetting (see ``reptaxonomy.gfm.forgetting``)

    * finetune variants: ``finetune.modes`` may list ``lpft`` / ``lora`` or named
      ``finetune.variants`` (e.g. ``l2sp: {mode: full, regularizer: l2sp, reg_weight: 1e-3}``).
    * ft_wise: WiSE-FT weight-space interpolation between the linear-probe run and
      ``finetune.wise_ft.source_variant`` at ``alphas`` (variants ``wise025`` ...).
    * forgetting: representation forgetting of every fine-tuned variant on retention
      sets (same-family conditions outside the run, conditions of the other
      families), layer-wise CKA, learning-forgetting trade-off; retention-set geo_id.
    * ft_sequential: sequential fine-tuning chains (``forgetting.sequential``) with
      accuracy matrix / BWT, optionally EWC or L2-SP.

Usage::

    python -m reptaxonomy.pipelines.gfm_experiment -y reptaxonomy/config/gfm/ftw_pretrained.yaml
    python -m reptaxonomy.pipelines.gfm_experiment -y reptaxonomy/config/gfm/ftw_finetune_compare.yaml \
        --stages finetune ft_embed ft_pipeline finetune_compare

Layout (all under ``root_dir``)::

    work/manifests/<cond>.json  work/labels/<cond>.npz  work/band_stats/<ds>.json
    work/plan.json              work/model_coverage.csv
    embeddings/<cond>/<model>/test/embd_<stem>.npy        (C x h x w)
    outputs/rep_taxonomy/<cond>/...                       full_pipeline (pretrained models)
    outputs/rep_taxonomy_ft/<cond>/...                    full_pipeline (pretrained + fine-tuned)
    outputs/cross_condition/<ds>/...  outputs/finetune_compare/<ds>/...
    outputs/geo_id/<ds>/...  outputs/geo_id/<ds>_<tag>_ft/...
    finetune/<ds>/<tag>/<model>/<mode>/{model.pt,metrics.json}
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from reptaxonomy.gfm import bands as B
from reptaxonomy.gfm.catalog import get_entry, resolve_suite
from reptaxonomy.gfm.compare_finetune import ft_name, run_finetune_compare
from reptaxonomy.gfm.conditions import Condition, build_conditions
from reptaxonomy.gfm.cross_condition import run_cross_condition
from reptaxonomy.gfm.datasets import IGNORE_INDEX, LABEL_KEY, REPSET_CLASSES, load_or_compute_band_stats
from reptaxonomy.robustness.geo_intrinsic_dimension import run_geo_id

logger = logging.getLogger("reptaxonomy.gfm_experiment")

PRETRAINED_STAGES = ["prepare", "embed", "pipeline", "cross_condition", "geo_id", "synthesis"]
FINETUNE_STAGES = ["finetune", "ft_wise", "ft_embed", "ft_pipeline", "finetune_compare", "ft_geo_id", "ft_synthesis"]
# run after the per-family loop (they need every family's conditions / outputs)
POST_STAGES = ["forgetting", "ft_sequential"]
BASE_MODES = ("linear", "partial", "full", "lpft", "lora")
REPSET_INFO = {"FTWRepSet": ("s2", 10.0), "EPARepSet": ("s2", 10.0), "MADOSRepSet": ("s2", 10.0)}


# ----------------------------------------------------------------------------- config
def deep_merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    """YAML with optional ``base: <relative path>`` (or list) inheritance."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    bases = cfg.pop("base", None)
    merged: Dict[str, Any] = {}
    for b in ([bases] if isinstance(bases, str) else (bases or [])):
        merged = deep_merge(merged, load_config(path.parent / b))
    return deep_merge(merged, cfg)


def _paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    root = cfg.get("root_dir", "./gfm_runs")
    p = {
        "work_dir": cfg.get("work_dir") or os.path.join(root, "work"),
        "embed_dir": cfg.get("embed_dir") or os.path.join(root, "embeddings"),
        "out_dir": cfg.get("out_dir") or os.path.join(root, "outputs"),
        "finetune_root": cfg.get("finetune_root") or os.path.join(root, "finetune"),
    }
    p["rt_out"] = os.path.join(p["out_dir"], "rep_taxonomy")
    p["rt_out_ft"] = os.path.join(p["out_dir"], "rep_taxonomy_ft")
    for v in p.values():
        Path(v).mkdir(parents=True, exist_ok=True)
    return p


def resolve_models(cfg: Dict[str, Any]) -> List[str]:
    mc = cfg.get("models", {}) or {}
    names = resolve_suite(mc.get("suite", "core"), include_experimental=bool(mc.get("include_experimental", False)),
                          include=list(mc.get("include", []) or []), exclude=list(mc.get("exclude", []) or []))
    if not mc.get("include_scratch", True):
        names = [n for n in names if not n.endswith("__scratch")]
    return names


# ----------------------------------------------------------------------------- dataset context
class DatasetContext:
    def __init__(self, name: str, ds_cfg: Dict[str, Any]) -> None:
        self.name = name
        self.cfg = ds_cfg
        self.conditions, comps, self.runs = build_conditions(name, ds_cfg.get("data", {}), ds_cfg.get("experiments", []))
        self.by_name: Dict[str, Condition] = {c.name: c for c in self.conditions}
        for r in self.runs:  # train/val conditions of FT runs that are not analysed themselves
            for c in (r.train, r.val):
                if c is not None:
                    self.by_name.setdefault(c.name, c)
        self.comparisons = [asdict(c) for c in comps]
        self.source_bands: List[str] = []
        self.sensor, self.gsd = REPSET_INFO[self.conditions[0].cls] if self.conditions else ("s2", 10.0)
        self.raw_scale = 1.0
        self.num_classes = int(ds_cfg.get("num_classes") or 0)
        self.band_stats: Dict[str, List[float]] = {}
        self.coverage: Dict[str, float] = {}
        self.models: List[str] = []

    @property
    def analysed(self) -> List[Condition]:
        return [c for c in self.conditions if c.analyze]

    def dataset_cfg(self, cond: Condition) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """(``dataset``, ``dataset_init``) blocks for the rep_taxonomy pipelines."""
        dataset = {
            "name": cond.name, "dataset_name": cond.name, "split": "test", "task": "segmentation",
            "bands": list(self.source_bands), "target_name": LABEL_KEY,
            "num_classes": self.num_classes, "ignore_index": IGNORE_INDEX,
            "gfm_repset": cond.cls, "band_stats": self.band_stats, "gfm_family": self.name,
            "gfm_setup": cond.setup, "gfm_role": cond.role,
        }
        return dataset, dict(cond.kwargs)

    def encoder_kwargs(self, model: str, cfg: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        kw = {"source_bands": list(self.source_bands), "source_sensor": self.sensor, "band_stats": self.band_stats,
              "gsd": self.gsd, "raw_scale": self.raw_scale}
        kw.update(cfg.get("model_init", {}) or {})
        kw.update((cfg.get("per_model_overrides", {}) or {}).get(model, {}) or {})
        kw.update(extra or {})
        kw.pop("model_name", None)
        return kw


def _instantiate(cond: Condition):
    return REPSET_CLASSES[cond.cls](**cond.kwargs)


def model_band_coverage(model: str, source_bands: Sequence[str], sensor: str) -> float:
    e = get_entry(model)
    canon = B.canonical_bands(source_bands, sensor)
    if e["family"] == "dofa":  # wavelength-conditioned: any optical band works
        return 1.0 if any(B.CANONICAL_WAVELENGTH_UM.get(c, -1) > 0 for c in canon) else 0.0
    return B.plan_bands(canon, B.canonical_bands(e["bands"], e["sensor"])).coverage


def _nearest(lab: np.ndarray, size: int) -> np.ndarray:
    H, W = lab.shape
    if max(H, W) <= size:
        return lab
    ys = (np.arange(size) * H / size + H / (2 * size)).astype(int).clip(0, H - 1)
    xs = (np.arange(size) * W / size + W / (2 * size)).astype(int).clip(0, W - 1)
    return lab[np.ix_(ys, xs)]


def write_condition_cache(ds, cond: Condition, work_dir: str, label_grid: int, overwrite: bool) -> Dict[str, Any]:
    man_p = Path(work_dir, "manifests", f"{cond.name}.json")
    lab_p = Path(work_dir, "labels", f"{cond.name}.npz")
    man_p.parent.mkdir(parents=True, exist_ok=True)
    lab_p.parent.mkdir(parents=True, exist_ok=True)
    if man_p.exists() and lab_p.exists() and not overwrite:
        return json.loads(man_p.read_text())
    stems = [Path(f).stem for f in ds.img_files]
    man = {"condition": cond.name, "setup": cond.setup, "role": cond.role, "paired_with": cond.paired_with,
           "tags": cond.tags, "kwargs": cond.kwargs, "stems": stems, "sample_keys": list(ds.sample_keys),
           "pair_keys": list(ds.pair_keys), "describe": ds.describe()}
    labels: Dict[str, np.ndarray] = {}
    hist = np.zeros(max(int(ds.num_classes), 1) + 1, dtype=np.int64)
    lats, lons, times = [float("nan")] * len(stems), [float("nan")] * len(stems), [None] * len(stems)
    for i, stem in enumerate(stems):
        try:
            sample = ds[i]
            y = sample[LABEL_KEY][LABEL_KEY]["data"].numpy().astype(np.int64)
        except Exception as exc:
            logger.warning("label read failed %s[%d]: %s", cond.name, i, exc)
            continue
        meta = sample.get("rt_meta", {}) or {}
        lats[i], lons[i], times[i] = meta.get("lat", float("nan")), meta.get("lon", float("nan")), meta.get("time")
        labels[stem] = _nearest(y, label_grid).astype(np.uint8 if y.max() < 256 else np.int32)
        v = y[y != IGNORE_INDEX]
        hist += np.bincount(v.clip(0, len(hist) - 1), minlength=len(hist))[: len(hist)]
    man["class_pixel_counts"] = hist.tolist()
    # geolocation for the geo_id stage (chip-centre lat/lon in EPSG:4326, acquisition time)
    man["lats"], man["lons"], man["times"] = lats, lons, times
    rows = getattr(ds, "meta_rows", None)
    if rows and len(rows) == len(stems):
        man["point_tags"] = {k: [r.get(k) for r in rows] for k in rows[0] if k != "aoi_id"}
    np.savez_compressed(lab_p, **labels)
    man_p.write_text(json.dumps(man, indent=1, default=str))
    return man


def prepare(ctxs: Dict[str, DatasetContext], cfg: Dict[str, Any], P: Dict[str, str], models: List[str]) -> None:
    overwrite = bool(cfg.get("overwrite_cache", False))
    label_grid = int(cfg.get("label_grid", 64))
    min_cov = float(cfg.get("min_band_coverage", 0.25))
    cov_rows = []
    for name, ctx in ctxs.items():
        bs_cfg = ctx.cfg.get("band_stats", {}) or {}
        ref_name = bs_cfg.get("reference")
        if not ref_name:
            trains = [r.train.name for r in ctx.runs]
            ref_name = trains[0] if trains else ctx.conditions[0].name
        ref = ctx.by_name[ref_name]
        ref_ds = _instantiate(ref)
        ctx.source_bands = list(ref_ds.band_names)
        ctx.sensor, ctx.gsd = getattr(ref_ds, "sensor", ctx.sensor), float(getattr(ref_ds, "gsd", ctx.gsd))
        ctx.raw_scale = float(getattr(ref_ds, "raw_scale", 1.0))
        ctx.num_classes = ctx.num_classes or int(ref_ds.num_classes)
        ctx.band_stats = bs_cfg.get("values") or load_or_compute_band_stats(
            ref_ds, Path(P["work_dir"], "band_stats", f"{name}.json"), n=int(bs_cfg.get("n_samples", 256)),
            seed=int(cfg.get("seed", 0)), prefer_default=bs_cfg.get("source", "default") == "default")
        logger.info("[%s] reference=%s bands=%s classes=%d", name, ref_name, ctx.source_bands, ctx.num_classes)

        needed = {c.name for c in ctx.analysed}
        for r in ctx.runs:
            needed |= set(r.eval_conditions)
        for cname in sorted(needed):
            cond = ctx.by_name[cname]
            try:
                ds = ref_ds if cname == ref_name else _instantiate(cond)
                man = write_condition_cache(ds, cond, P["work_dir"], label_grid, overwrite)
                logger.info("[%s] %s: %d chips", name, cname, len(man["stems"]))
            except Exception as exc:
                logger.error("[%s] condition %s failed to build: %s", name, cname, exc)
                cond.analyze = False

        for m in models:
            try:
                cov = model_band_coverage(m, ctx.source_bands, ctx.sensor)
            except Exception as exc:
                cov = 0.0
                logger.warning("coverage(%s) failed: %s", m, exc)
            ctx.coverage[m] = cov
            ok = cov >= min_cov
            if ok:
                ctx.models.append(m)
            cov_rows.append({"dataset": name, "model": m, "band_coverage": round(cov, 3), "used": ok,
                             "reason": "" if ok else f"coverage<{min_cov}"})
        Path(P["work_dir"], "band_stats").mkdir(parents=True, exist_ok=True)
        Path(P["work_dir"], "band_stats", f"{name}.json").write_text(json.dumps(ctx.band_stats, indent=1))

    with open(Path(P["work_dir"], "model_coverage.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "model", "band_coverage", "used", "reason"])
        w.writeheader(); w.writerows(cov_rows)
    write_plan(ctxs, cfg, P)


def write_plan(ctxs: Dict[str, DatasetContext], cfg: Dict[str, Any], P: Dict[str, str]) -> Dict[str, Any]:
    plan = {"embed_dir": P["embed_dir"], "work_dir": P["work_dir"], "out_dir": P["out_dir"],
            "finetune_root": P["finetune_root"], "datasets": {}}
    for name, ctx in ctxs.items():
        plan["datasets"][name] = {
            "source_bands": ctx.source_bands, "sensor": ctx.sensor, "gsd": ctx.gsd, "raw_scale": ctx.raw_scale,
            "num_classes": ctx.num_classes, "band_stats": ctx.band_stats, "models": ctx.models,
            "coverage": ctx.coverage,
            "conditions": [c.to_dict() for c in ctx.conditions],
            "analysed": [c.name for c in ctx.analysed],
            "comparisons": [c for c in ctx.comparisons
                            if ctx.by_name[c["a"]].analyze and ctx.by_name[c["b"]].analyze],
            "finetune_runs": [{"tag": r.tag, "train": r.train.to_dict(), "val": r.val.to_dict() if r.val else None,
                               "eval_conditions": r.eval_conditions, "id_conditions": r.id_conditions,
                               "ood_conditions": r.ood_conditions} for r in ctx.runs],
        }
    Path(P["work_dir"], "plan.json").write_text(json.dumps(plan, indent=1, default=str))
    return plan


def load_plan_into(ctxs: Dict[str, DatasetContext], P: Dict[str, str]) -> bool:
    p = Path(P["work_dir"], "plan.json")
    if not p.exists():
        return False
    plan = json.loads(p.read_text())
    for name, ctx in ctxs.items():
        d = plan["datasets"].get(name)
        if not d:
            return False
        ctx.source_bands, ctx.sensor, ctx.gsd = d["source_bands"], d["sensor"], d["gsd"]
        ctx.raw_scale, ctx.num_classes, ctx.band_stats = d["raw_scale"], d["num_classes"], d["band_stats"]
        ctx.models, ctx.coverage = d["models"], d["coverage"]
        analysed = set(d["analysed"])
        for c in ctx.conditions:
            c.analyze = c.name in analysed
    return True


# ----------------------------------------------------------------------------- embeddings
def _embed_done(P: Dict[str, str], cond: str, model: str) -> bool:
    man = Path(P["work_dir"], "manifests", f"{cond}.json")
    if not man.exists():
        return False
    stems = json.loads(man.read_text())["stems"]
    d = Path(P["embed_dir"], cond, model, "test")
    return bool(stems) and all((d / f"embd_{s}.npy").exists() for s in stems)


def embed_condition(ctx: DatasetContext, cond: Condition, models: List[str], cfg: Dict[str, Any],
                    P: Dict[str, str], overrides: Dict[str, Dict[str, Any]]) -> None:
    todo = [m for m in models if cfg.get("overwrite_embeddings") or not _embed_done(P, cond.name, m)]
    if not todo:
        return
    backend = str(cfg.get("embedding_backend", "reptaxonomy"))
    if backend == "fast":
        return _embed_fast(ctx, cond, todo, cfg, P, overrides)
    from reptaxonomy.pipelines.generate_embeddings import run_embedding_export
    dataset, dataset_init = ctx.dataset_cfg(cond)
    ecfg = {
        "dataset": dataset, "dataset_init": dataset_init, "model_names": todo,
        "model_init": dict(cfg.get("model_init", {}) or {}),
        "per_model_overrides": {m: dict(overrides.get(m, {})) for m in todo},
        "embed_dir": P["embed_dir"], "work_dir": P["work_dir"], "split": "test", "batch_size": 1,
        "num_workers": int(cfg.get("num_workers", 0)), "overwrite": bool(cfg.get("overwrite_embeddings", False)),
        "fail_fast": bool(cfg.get("fail_fast", False)), "seed": int(cfg.get("seed", 42)),
        "device": cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"),
        "test_num_workers": int(cfg.get("num_workers", 0)),
    }
    run_embedding_export(ecfg)


def _embed_fast(ctx: DatasetContext, cond: Condition, models: List[str], cfg: Dict[str, Any],
                P: Dict[str, str], overrides: Dict[str, Dict[str, Any]]) -> None:
    """Batched GFMEncoder export writing the same artifacts as generate_embeddings."""
    from torch.utils.data import DataLoader
    from reptaxonomy.gfm.encoders import GFMEncoder
    from reptaxonomy.util.experiment_init_utils import build_forward_instruments_collate_fn
    dev = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = _instantiate(cond)
    stems = [Path(f).stem for f in ds.img_files]
    loader = DataLoader(ds, batch_size=int(cfg.get("embed_batch_size", 8)), shuffle=False,
                        num_workers=int(cfg.get("num_workers", 0)), collate_fn=build_forward_instruments_collate_fn({}))
    for m in models:
        out = Path(P["embed_dir"], cond.name, m, "test"); out.mkdir(parents=True, exist_ok=True)
        try:
            enc = GFMEncoder(m, **ctx.encoder_kwargs(m, cfg, overrides.get(m))).to(dev).eval()
            k = 0
            with torch.no_grad():
                for batch in loader:
                    g = enc.forward_features(batch, device=dev).float().cpu().numpy()
                    for j in range(g.shape[0]):
                        np.save(out / f"embd_{stems[k]}.npy", g[j]); k += 1
            logger.info("[EMBED-fast] %s %s: %d", cond.name, m, k)
            del enc
        except Exception:
            logger.error("[EMBED-fast] %s %s failed\n%s", cond.name, m, traceback.format_exc())
            if cfg.get("fail_fast"):
                raise
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- full pipeline
def pipeline_condition(ctx: DatasetContext, cond: Condition, models: List[str], cfg: Dict[str, Any],
                       P: Dict[str, str], out_root: str, overrides: Dict[str, Dict[str, Any]],
                       training_pairs: Sequence[Sequence[str]] = ()) -> None:
    from reptaxonomy.pipelines.full_pipeline import run_full_pipeline
    present = [m for m in models if _embed_done(P, cond.name, m)]
    missing = sorted(set(models) - set(present))
    if missing:
        logger.warning("[PIPELINE] %s: skipping models without complete embeddings: %s", cond.name, missing)
    if len(present) < 3:
        # data_kernel_analysis embeds the model x model distance matrix with 2-D classical MDS
        logger.warning("[PIPELINE] %s: full_pipeline needs >= 3 models with embeddings (have %d), skipped",
                       cond.name, len(present))
        return
    dataset, dataset_init = ctx.dataset_cfg(cond)
    pcfg = copy.deepcopy(cfg.get("pipeline", {}) or {})
    pcfg.update({
        "dataset": dataset, "dataset_init": dataset_init, "model_names": present,
        "model_init": dict(cfg.get("model_init", {}) or {}),
        "per_model_overrides": {m: dict(overrides.get(m, {})) for m in present},
        "embed_dir": P["embed_dir"], "work_dir": os.path.join(P["work_dir"], "rt", cond.name), "out_dir": out_root,
        "seed": int(cfg.get("seed", 42)),
    })
    pcfg.setdefault("test_num_workers", int(cfg.get("num_workers", 0)))
    if cfg.get("device"):
        pcfg["model_init"].setdefault("device", cfg["device"])
    pairs = [list(p) for p in training_pairs if p[0] in present and p[1] in present]
    steps = pcfg.setdefault("steps", {})
    syn_step = steps.setdefault("cross_perspective_synthesis", {"enabled": True})
    syn = syn_step.setdefault("config_overrides", {}).setdefault("synthesis", {})
    syn["datasets"] = [cond.name]
    syn["training_pairs"] = pairs + [p for p in syn.get("training_pairs", []) or [] if p[0] in present and p[1] in present]
    inv = steps.get("invariance_checks", {}).get("config_overrides", {})
    if inv is not None and not inv.get("invariance_pairs") and pairs:
        steps.setdefault("invariance_checks", {}).setdefault("config_overrides", {})["invariance_pairs"] = pairs[:2]
    Path(pcfg["work_dir"]).mkdir(parents=True, exist_ok=True)
    try:
        run_full_pipeline(pcfg)
    except Exception:
        logger.error("[PIPELINE] %s failed\n%s", cond.name, traceback.format_exc())
        if cfg.get("fail_fast"):
            raise


def family_synthesis(ctx: DatasetContext, cond_names: List[str], models: List[str], cfg: Dict[str, Any],
                     P: Dict[str, str], out_root: str, tag: str, training_pairs: Sequence[Sequence[str]] = ()) -> None:
    from reptaxonomy.summarize.cross_perspective_synthesis import run_cross_perspective_synthesis
    have = [c for c in cond_names if Path(out_root, c).exists()]
    if not have:
        return
    pipe = cfg.get("pipeline", {}) or {}
    methods = [str(m).lower() for m in pipe.get("upsample_methods", ["pixelshuffle"])]
    base_syn = (pipe.get("steps", {}).get("cross_perspective_synthesis", {}) or {}).get("config_overrides", {}).get("synthesis", {})
    syn = dict(base_syn or {})
    syn.update({"datasets": have, "tag": tag, "out_subdir": "synthesis_pooled",
                "training_pairs": [list(p) for p in training_pairs],
                "variants": {m: (out_root if i == 0 else os.path.join(out_root, f"upsample_{m}"))
                             for i, m in enumerate(methods)},
                "primary_variant": methods[0]})
    syn.update(cfg.get("synthesis", {}) or {})
    try:
        run_cross_perspective_synthesis({"out_dir": out_root, "embed_dir": P["embed_dir"], "model_names": models,
                                         "projections": pipe.get("projections"), "synthesis": syn})
    except Exception:
        logger.error("[SYNTHESIS] %s failed\n%s", tag, traceback.format_exc())


# ----------------------------------------------------------------------------- fine-tuning
def wise_names(cfg: Dict[str, Any]) -> List[str]:
    w = ((cfg.get("finetune", {}) or {}).get("wise_ft", {}) or {})
    if not w.get("enabled", False):
        return []
    return [f"wise{int(round(float(a) * 100)):03d}" for a in w.get("alphas", [0.5]) if 0 < float(a) < 1]


def ft_modes(cfg: Dict[str, Any], include_wise: bool = True) -> List[str]:
    """Variant labels: base modes, named ``finetune.variants`` and (optionally) WiSE-FT."""
    modes = list((cfg.get("finetune", {}) or {}).get("modes", ["linear", "full"]))
    if "linear" not in modes:
        modes = ["linear"] + modes  # linear probe on the frozen backbone = pretrained reference
    if include_wise:
        modes += [w for w in wise_names(cfg) if w not in modes]
    return modes


def variant_spec(cfg: Dict[str, Any], name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """(base mode, extra hyper-parameters) of a variant label; None for derived (WiSE) variants."""
    variants = ((cfg.get("finetune", {}) or {}).get("variants", {}) or {})
    if name in variants:
        v = dict(variants[name] or {})
        mode = v.pop("mode", "full")
        if mode not in BASE_MODES:
            raise ValueError(f"finetune.variants.{name}.mode={mode!r} not in {BASE_MODES}")
        return mode, v
    if name in BASE_MODES:
        return name, {}
    if name.startswith("wise"):
        return None
    raise ValueError(f"Unknown fine-tuning mode/variant '{name}' (base modes {BASE_MODES} or finetune.variants)")


def ft_models(ctx: DatasetContext, cfg: Dict[str, Any]) -> List[str]:
    sel = (cfg.get("finetune", {}) or {}).get("models")
    return [m for m in ctx.models if (not sel or m in sel)]


def run_finetunes(ctx: DatasetContext, cfg: Dict[str, Any], P: Dict[str, str]) -> None:
    from reptaxonomy.gfm.finetune import run_finetune
    fc = dict(cfg.get("finetune", {}) or {})
    hp = {k: fc[k] for k in ("head", "epochs", "batch_size", "lr", "backbone_lr_mult", "weight_decay",
                             "unfreeze_fraction", "max_train_samples", "max_eval_samples", "early_stop_patience",
                             "amp") if k in fc}
    per_family = (fc.get("per_family", {}) or {})
    for r in ctx.runs:
        if fc.get("runs") and r.tag not in fc["runs"]:
            continue
        evals = {c: {"cls": ctx.by_name[c].cls, "kwargs": ctx.by_name[c].kwargs} for c in r.eval_conditions
                 if c in ctx.by_name}
        for m in ft_models(ctx, cfg):
            fam = get_entry(m)["family"]
            # linear first: LP-FT initialises its head from the linear run
            for name in sorted(ft_modes(cfg, include_wise=False), key=lambda v: v != "linear"):
                spec = variant_spec(cfg, name)
                if spec is None:
                    continue
                mode, extra = spec
                h = deep_merge(hp, (per_family.get(fam, {}) or {}).get(mode, per_family.get(fam, {}) or {}))
                h = {k: v for k, v in h.items() if not isinstance(v, dict)}
                h.update({k: v for k, v in extra.items() if not isinstance(v, dict)})
                if mode == "lpft":
                    h.setdefault("lp_head_from", str(Path(P["finetune_root"], ctx.name, r.tag, m, "linear", "model.pt")))
                out = Path(P["finetune_root"], ctx.name, r.tag, m, name)
                logger.info("[FT] %s %s %s (%s) -> %s", r.tag, m, name, mode, out)
                try:
                    run_finetune(m, ctx.encoder_kwargs(m, cfg), {"cls": r.train.cls, "kwargs": r.train.kwargs},
                                 {"cls": r.val.cls, "kwargs": r.val.kwargs} if r.val else None, evals,
                                 ctx.num_classes, out, mode=mode, num_workers=int(cfg.get("num_workers", 0)),
                                 device=cfg.get("device"), seed=int(cfg.get("seed", 0)),
                                 overwrite=bool(fc.get("overwrite", False)), variant=name, **h)
                except Exception:
                    logger.error("[FT] %s %s %s failed\n%s", r.tag, m, name, traceback.format_exc())
                    if cfg.get("fail_fast"):
                        raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def run_wise(ctx: DatasetContext, cfg: Dict[str, Any], P: Dict[str, str]) -> None:
    """WiSE-FT checkpoints + metrics for every (run, model): linear run <-> source variant."""
    from reptaxonomy.gfm.finetune import evaluate_checkpoint, wise_interpolate
    fc = dict(cfg.get("finetune", {}) or {})
    w = dict(fc.get("wise_ft", {}) or {})
    src = str(w.get("source_variant", "full"))
    for r in ctx.runs:
        if fc.get("runs") and r.tag not in fc["runs"]:
            continue
        evals = {c: {"cls": ctx.by_name[c].cls, "kwargs": ctx.by_name[c].kwargs} for c in r.eval_conditions
                 if c in ctx.by_name}
        for m in ft_models(ctx, cfg):
            base = Path(P["finetune_root"], ctx.name, r.tag, m)
            lin, ft = base / "linear" / "model.pt", base / src / "model.pt"
            if not (lin.exists() and ft.exists()):
                logger.warning("[WISE] %s %s: need linear and %s checkpoints", r.tag, m, src)
                continue
            for a in w.get("alphas", [0.5]):
                a = float(a)
                if not 0 < a < 1:
                    continue
                name = f"wise{int(round(a * 100)):03d}"
                out = base / name
                if (out / "metrics.json").exists() and not fc.get("overwrite", False):
                    continue
                try:
                    ck = wise_interpolate(lin, ft, a, out / "model.pt")
                    res = evaluate_checkpoint(m, ctx.encoder_kwargs(m, cfg), ck, evals, ctx.num_classes,
                                              batch_size=int(fc.get("batch_size", 8)),
                                              num_workers=int(cfg.get("num_workers", 0)),
                                              max_eval_samples=fc.get("max_eval_samples"), device=cfg.get("device"))
                    res.update(mode=name, variant=name, wise_alpha=a, wise_source=src, head=fc.get("head", "fcn"))
                    (out / "metrics.json").write_text(json.dumps(res, indent=2))
                    logger.info("[WISE] %s %s alpha=%.2f done", r.tag, m, a)
                except Exception:
                    logger.error("[WISE] %s %s %s failed\n%s", r.tag, m, name, traceback.format_exc())


def ft_variants(ctx: DatasetContext, cfg: Dict[str, Any], P: Dict[str, str], run) -> Tuple[List[str], Dict[str, Dict[str, Any]], List[List[str]]]:
    """Fine-tuned model names, checkpoint overrides and (pre, ft) pairs for one FT run."""
    names, ov, pairs = [], {}, []
    for m in ft_models(ctx, cfg):
        for mode in ft_modes(cfg):
            if mode == "linear":
                continue
            ck = Path(P["finetune_root"], ctx.name, run.tag, m, mode, "model.pt")
            if not ck.exists():
                continue
            n = ft_name(m, run.tag, mode)
            names.append(n)
            ov[n] = dict((cfg.get("per_model_overrides", {}) or {}).get(m, {}) or {}, checkpoint=str(ck))
            pairs.append([m, n])
    return names, ov, pairs


# ----------------------------------------------------------------------------- helpers
def _spec(c: Condition) -> Dict[str, Any]:
    return {"cls": c.cls, "kwargs": dict(c.kwargs)}


# ----------------------------------------------------------------------------- forgetting
_FTW_COUNTRIES = {"austria", "belgium", "brazil", "cambodia", "corsica", "croatia", "denmark", "estonia", "finland",
                  "france", "germany", "india", "kenya", "latvia", "lithuania", "luxembourg", "netherlands",
                  "portugal", "rwanda", "slovakia", "slovenia", "south_africa", "spain", "sweden", "vietnam"}


def _ftw_task_countries(cond: Condition) -> Tuple[Optional[set], set]:
    """(train countries | None = all, ood countries) of an FTW shift-task condition."""
    spec = cond.kwargs.get("shift_spec") or {}
    if spec:
        tr = set(spec.get("id_countries") or spec.get("countries") or [])
        return (tr or None), set(spec.get("ood_countries") or [])
    task = str(cond.kwargs.get("shift_task") or cond.tags.get("task") or "")
    toks = task.replace("ftw-", "", 1).replace("south-africa", "south_africa").split("-")
    if "all" in toks:
        return None, set()
    found = [t for t in toks if t in _FTW_COUNTRIES]
    if not found:
        return None, set()
    return {found[0]}, set(found[1:])


def retention_conditions(ctxs: Dict[str, DatasetContext], ctx: DatasetContext, run, m: str,
                         fc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Retention sets for (family, run, base model): same-family, cross-family, optional OOD."""
    rc = dict(fc.get("retention", {}) or {})
    explicit = (rc.get("explicit", {}) or {}).get(run.tag)
    kmax = int(rc.get("max_per_family", 3))
    used = set(run.eval_conditions) | {run.train.name} | ({run.val.name} if run.val else set())
    out: List[Dict[str, Any]] = []

    def add(g: DatasetContext, cname: str, rel: str) -> None:
        if cname in g.by_name and g.by_name[cname].analyze and m in g.models:
            out.append({"condition": cname, "family": g.name, "relation": rel})

    if explicit:
        for e in explicit:
            fam, cname = e.split(":", 1) if ":" in e else (ctx.name, e)
            if fam in ctxs:
                add(ctxs[fam], cname, "same_family" if fam == ctx.name else "cross_family")
        return out
    if rc.get("same_family", True):
        tr_c, ood_c = (_ftw_task_countries(run.train) if ctx.name == "ftw" else (set(), set()))
        cands = []
        for c in ctx.analysed:
            if c.name in used or c.tags.get("in_train_split") or c.role in ("train", "val"):
                continue
            if ctx.name == "ftw":
                if tr_c is None:
                    continue  # run trains on every country -> nothing is held out
                ctry = c.tags.get("country")
                if ctry is None or ctry in tr_c or (ctry in ood_c and not rc.get("include_ood", False)):
                    continue
                if c.cls == "FTWRepSet" and c.kwargs.get("shift_task"):
                    continue  # prefer official-test country conditions over shift-task splits
            cands.append(c.name)
        for cname in sorted(cands)[:kmax]:
            add(ctx, cname, "same_family")
    if rc.get("cross_family", True):
        pref = list(rc.get("prefer_roles", ["test", "country", "region", "id_region", "id_test", "season"]))
        for g in ctxs.values():
            if g.name == ctx.name or (rc.get("families") and g.name not in rc["families"]):
                continue
            cands = sorted(g.analysed, key=lambda c: (pref.index(c.role) if c.role in pref else len(pref), c.name))
            cands = [c for c in cands if c.role not in ("train", "val")] or cands
            for c in cands[:kmax]:
                add(g, c.name, "cross_family")
    if rc.get("include_ood", False):
        for cname in run.ood_conditions:
            add(ctx, cname, "ood")
    return out


def _predictors(cfg: Dict[str, Any], P: Dict[str, str], family: str) -> Optional["pd.DataFrame"]:
    import pandas as pd
    frames = []
    for extra in (cfg.get("forgetting", {}) or {}).get("predictor_csvs", []) or []:
        if Path(extra).exists():
            frames.append(pd.read_csv(extra))
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="model", how="outer")
    return out


def forgetting_stage(ctxs: Dict[str, DatasetContext], cfg: Dict[str, Any], P: Dict[str, str],
                     base_ov: Dict[str, Dict[str, Any]]) -> None:
    from reptaxonomy.gfm.forgetting import run_forgetting
    fc = dict(cfg.get("forgetting", {}) or {})
    if not fc.get("enabled", True):
        return
    lw_max = int((fc.get("layerwise", {}) or {}).get("max_retention", 2))
    gcfg = dict(cfg.get("geo_id", {}) or {})
    for name, ctx in ctxs.items():
        if fc.get("families") and name not in fc["families"]:
            continue
        jobs = []
        for r in ctx.runs:
            if (cfg.get("finetune", {}) or {}).get("runs") and r.tag not in cfg["finetune"]["runs"]:
                continue
            ft_names, ft_ov, pairs = ft_variants(ctx, cfg, P, r)
            for pre, ftn in pairs:
                variant = ftn.rsplit(f"{r.tag}-", 1)[1]
                ret = retention_conditions(ctxs, ctx, r, pre, fc)
                if not ret:
                    continue
                for k, e in enumerate(ret):
                    g = ctxs[e["family"]]
                    cond = g.by_name[e["condition"]]
                    ov = dict(base_ov, **{ftn: dict(ft_ov[ftn])})
                    embed_condition(g, cond, [pre, ftn], cfg, P, ov)  # pre is a no-op when cached
                    e.update(spec=_spec(cond), pre_kwargs=g.encoder_kwargs(pre, cfg), layerwise=k < lw_max)
                jobs.append({"run": r.tag, "model": pre, "variant": variant, "ft_name": ftn,
                             "id_conditions": r.id_conditions or r.eval_conditions, "ood_conditions": r.ood_conditions,
                             "ft_ckpt": ft_ov[ftn]["checkpoint"], "retention": ret})
        if not jobs:
            logger.info("[FORGET] %s: no fine-tuned variants with retention sets", name)
            continue
        params = {k: v for k, v in fc.items() if k not in ("enabled", "retention", "sequential", "families",
                                                           "predictor_csvs", "geo_id")}
        logger.info("[FORGET] %s: %d jobs", name, len(jobs))
        run_forgetting(name, jobs, P["embed_dir"], P["work_dir"], P["out_dir"], P["finetune_root"], params,
                       seed=int(cfg.get("seed", 0)), device=cfg.get("device"), predictors=_predictors(cfg, P, name))
        if fc.get("geo_id", False) and gcfg.get("enabled", True):
            by_fam: Dict[str, Dict[str, set]] = {}
            for j in jobs:
                for e in j["retention"]:
                    d = by_fam.setdefault(e["family"], {"conds": set(), "models": set()})
                    d["conds"].add(e["condition"]); d["models"] |= {j["model"], j["ft_name"]}
            for fam, d in by_fam.items():
                geo_id_stage(fam, sorted(d["conds"]), sorted(d["models"]), gcfg, P, f"{name}_retention_{fam}")


def sequential_stage(ctxs: Dict[str, DatasetContext], cfg: Dict[str, Any], P: Dict[str, str]) -> None:
    """``forgetting.sequential: [{name, steps: ["ftw:<tag>", "mados:mados-train"], models, mode, ...}]``."""
    from reptaxonomy.gfm.forgetting import run_sequential
    fc = dict(cfg.get("finetune", {}) or {})
    hp = {k: fc[k] for k in ("head", "epochs", "batch_size", "lr", "backbone_lr_mult", "weight_decay",
                             "unfreeze_fraction", "max_train_samples", "max_eval_samples", "early_stop_patience",
                             "amp") if k in fc}
    for chain in (cfg.get("forgetting", {}) or {}).get("sequential", []) or []:
        chain = dict(chain)
        steps, fams = [], []
        try:
            for st in chain["steps"]:
                fam, tag = st.split(":", 1)
                ctx = ctxs[fam]
                r = next(x for x in ctx.runs if x.tag == tag)
                ids = r.id_conditions or r.eval_conditions
                steps.append({"label": f"{fam}-{tag}", "family": fam, "tag": tag, "train": _spec(r.train),
                              "val": _spec(r.val) if r.val else None,
                              "id_conditions": {c: _spec(ctx.by_name[c]) for c in ids if c in ctx.by_name},
                              "num_classes": ctx.num_classes, "encoder_kwargs": {}})
                fams.append(ctx)
        except (KeyError, StopIteration, ValueError) as exc:
            logger.error("[SEQ] chain %s: bad step spec (%s)", chain.get("name"), exc)
            continue
        models = chain.get("models") or sorted(set.intersection(*[set(ft_models(c, cfg)) for c in fams]))
        models = [m for m in models if all(m in c.models for c in fams)]
        for st, ctx in zip(steps, fams):
            st["encoder_kwargs"] = {m: ctx.encoder_kwargs(m, cfg) for m in models}
        h = dict(hp, **(chain.get("hp", {}) or {}))
        logger.info("[SEQ] %s: %s x %s", chain.get("name"), models, [s["label"] for s in steps])
        try:
            run_sequential(str(chain.get("name", "chain")), models, steps, P["out_dir"], P["finetune_root"],
                           mode=chain.get("mode", "full"), hp=h, regularizer=chain.get("regularizer"),
                           reg_weight=float(chain.get("reg_weight", 0.0)),
                           fisher_batches=int(chain.get("fisher_batches", 50)),
                           num_workers=int(cfg.get("num_workers", 0)), device=cfg.get("device"),
                           seed=int(cfg.get("seed", 0)), overwrite=bool(chain.get("overwrite", False)))
        except Exception:
            logger.error("[SEQ] %s failed\n%s", chain.get("name"), traceback.format_exc())


# ----------------------------------------------------------------------------- driver
def run(cfg: Dict[str, Any], stages: Optional[Sequence[str]] = None, datasets: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=getattr(logging, str(cfg.get("log_level", "INFO")).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mode = cfg.get("mode", "pretrained")
    if not (stages or cfg.get("stages")):
        stages = PRETRAINED_STAGES + (FINETUNE_STAGES + POST_STAGES if mode == "finetune_compare" else [])
    stages = list(stages or cfg.get("stages"))
    unknown = set(stages) - set(PRETRAINED_STAGES + FINETUNE_STAGES + POST_STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}")
    P = _paths(cfg)
    torch.manual_seed(int(cfg.get("seed", 42)))
    models = resolve_models(cfg)
    ds_cfgs = {k: v for k, v in (cfg.get("datasets", {}) or {}).items()
               if (v or {}).get("enabled", True) and (not datasets or k in datasets)}
    ctxs = {k: DatasetContext(k, v) for k, v in ds_cfgs.items()}
    logger.info("mode=%s stages=%s datasets=%s models=%d", mode, stages, list(ctxs), len(models))
    Path(P["work_dir"], "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if "prepare" in stages or not load_plan_into(ctxs, P):
        prepare(ctxs, cfg, P, models)
    base_ov = {m: dict(v or {}) for m, v in (cfg.get("per_model_overrides", {}) or {}).items()}

    for name, ctx in ctxs.items():
        conds = ctx.analysed
        if "embed" in stages:
            for c in conds:
                embed_condition(ctx, c, ctx.models, cfg, P, base_ov)
        if "pipeline" in stages:
            for c in conds:
                pipeline_condition(ctx, c, ctx.models, cfg, P, P["rt_out"], base_ov,
                                   training_pairs=_scratch_pairs(ctx.models))
        if "cross_condition" in stages and (cfg.get("cross_condition", {}) or {}).get("enabled", True):
            cc = dict(cfg.get("cross_condition", {}) or {}); cc.pop("enabled", None)
            plan = write_plan(ctxs, cfg, P)["datasets"][name]
            run_cross_condition(P["embed_dir"], P["work_dir"], P["out_dir"], name, plan["comparisons"],
                                ctx.models, [c.name for c in conds], **cc)
        gcfg = dict(cfg.get("geo_id", {}) or {})
        geo_on = bool(gcfg.get("enabled", True))
        if "geo_id" in stages and geo_on:
            geo_id_stage(name, [c.name for c in conds], ctx.models, gcfg, P, name)
        if "synthesis" in stages:
            family_synthesis(ctx, [c.name for c in conds], ctx.models, cfg, P, P["rt_out"], f"{name}_pretrained",
                             _scratch_pairs(ctx.models))

        if mode != "finetune_compare":
            continue
        if "finetune" in stages:
            run_finetunes(ctx, cfg, P)
        if "ft_wise" in stages and wise_names(cfg):
            run_wise(ctx, cfg, P)
        all_ft, all_pairs = [], []
        for r in ctx.runs:
            if (cfg.get("finetune", {}) or {}).get("runs") and r.tag not in cfg["finetune"]["runs"]:
                continue
            ft_names, ft_ov, pairs = ft_variants(ctx, cfg, P, r)
            all_ft += ft_names; all_pairs += pairs
            ov = dict(base_ov, **ft_ov)
            evals = [ctx.by_name[c] for c in r.eval_conditions if c in ctx.by_name and ctx.by_name[c].analyze]
            if "ft_embed" in stages:
                for c in evals:
                    embed_condition(ctx, c, ft_names, cfg, P, ov)
            if "ft_pipeline" in stages:
                pre = [p[0] for p in pairs]
                for c in evals:
                    pipeline_condition(ctx, c, list(dict.fromkeys(pre + ft_names)), cfg, P,
                                       os.path.join(P["rt_out_ft"], r.tag), ov, training_pairs=pairs)
            if "ft_geo_id" in stages and geo_on and gcfg.get("finetuned", True) and pairs:
                geo_id_stage(name, [c.name for c in evals], list(dict.fromkeys([p[0] for p in pairs] + ft_names)),
                             gcfg, P, f"{name}_{r.tag}_ft")
            if "ft_synthesis" in stages and pairs:
                family_synthesis(ctx, [c.name for c in evals], list(dict.fromkeys([p[0] for p in pairs] + ft_names)),
                                 cfg, P, os.path.join(P["rt_out_ft"], r.tag), f"{name}_{r.tag}_ft", pairs)
        if "finetune_compare" in stages:
            runs = [{"tag": r.tag, "eval_conditions": [c for c in r.eval_conditions if c in ctx.by_name],
                     "id_conditions": r.id_conditions, "ood_conditions": r.ood_conditions} for r in ctx.runs]
            run_finetune_compare(P["embed_dir"], P["work_dir"], P["out_dir"], name, runs, ft_models(ctx, cfg),
                                 ft_modes(cfg), P["finetune_root"],
                                 encoder_kwargs={m: ctx.encoder_kwargs(m, cfg) for m in ft_models(ctx, cfg)},
                                 compute_weight_change=bool((cfg.get("finetune", {}) or {}).get("weight_change", True)),
                                 seed=int(cfg.get("seed", 0)))

    # ---- post-family stages (need every family's conditions / fine-tuning outputs)
    if mode == "finetune_compare":
        if "forgetting" in stages:
            forgetting_stage(ctxs, cfg, P, base_ov)
        if "ft_sequential" in stages:
            sequential_stage(ctxs, cfg, P)
    logger.info("done. outputs under %s", P["out_dir"])


def geo_id_stage(name: str, conditions: Sequence[str], models: Sequence[str], gcfg: Dict[str, Any],
                 P: Dict[str, str], tag: str) -> None:
    """Geolocated ID for one dataset family; failures are logged, never fatal."""
    gcfg = {k: v for k, v in gcfg.items() if k not in ("enabled", "finetuned")}
    per_ds = (gcfg.pop("datasets", {}) or {}).get(name, {}) or {}
    params = {**gcfg, **per_ds}
    if params.pop("skip", False):
        return
    coords_csv = params.pop("coords_csv", None)
    try:
        run_geo_id(P["embed_dir"], P["work_dir"], P["out_dir"], tag, conditions, models, params,
                   coords_csv=coords_csv)
    except Exception:
        logger.error("[%s] geo_id failed:\n%s", tag, traceback.format_exc())


def _scratch_pairs(models: Sequence[str]) -> List[List[str]]:
    """(pretrained, scratch) pairs of the same architecture -> synthesis training deltas."""
    scratch = {m.split("__")[0]: m for m in models if m.endswith("__scratch")}
    return [[m, scratch[m.split("__")[0]]] for m in models
            if not m.endswith("__scratch") and m.split("__")[0] in scratch]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--mode", choices=["pretrained", "finetune_compare"])
    ap.add_argument("--stages", nargs="*", help=f"subset of {PRETRAINED_STAGES + FINETUNE_STAGES + POST_STAGES}")
    ap.add_argument("--datasets", nargs="*", help="subset of the config's datasets (ftw epa mados)")
    ap.add_argument("--models", nargs="*", help="override the model list (catalog keys)")
    ap.add_argument("--root-dir")
    ap.add_argument("--device")
    ap.add_argument("--max-samples", type=int, help="cap chips per condition (smoke runs)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides, YAML values")
    a = ap.parse_args()
    cfg = load_config(a.yaml)
    if a.mode:
        cfg["mode"] = a.mode
    if a.root_dir:
        cfg["root_dir"] = a.root_dir
    if a.device:
        cfg["device"] = a.device
    if a.models:
        cfg.setdefault("models", {})
        cfg["models"].update({"suite": a.models, "include": [], "exclude": []})
    if a.max_samples:
        for d in (cfg.get("datasets", {}) or {}).values():
            d.setdefault("data", {})["max_samples"] = a.max_samples
    for kv in a.set:
        k, v = kv.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(v)
    run(cfg, a.stages, a.datasets)


if __name__ == "__main__":
    main()
