"""Audit gfm_experiment runs: which units are complete and which still have to run.

Read-only. For each config it rebuilds the units that ``gfm_experiment`` would run (same config
loading, conditions, model coverage from ``work/plan.json``, fine-tune runs and variants, upsample
variants and pipeline steps) and checks each one on disk.

Status of a unit
    done         complete
    stale        complete, but upstream outputs are newer (or, with markers, the configuration
                 changed); the next run recomputes it
    failed       ``_FAILED.<key>.json`` present (the error is shown)
    running      ``_RUNNING.<key>.json`` from a live process on this host
    interrupted  ``_RUNNING.<key>.json`` from a dead process (or another host, unverifiable)
    partial      some outputs present (e.g. 812/1000 chips; a resumable fine-tune checkpoint)
    todo         nothing on disk yet
    blocked      waiting for an upstream unit (e.g. embeddings, fine-tuned checkpoint)
    unverified   no output found, but this step may legitimately produce none (no reference
                 models, no invariance pairs, unsupported capability)

Completion markers (``_DONE.*``, restart support) are used when present. Runs from versions
without markers are judged from their output files; note that such versions recompute the
pipeline stages on every invocation, so restrict reruns with ``--stages`` / ``--datasets``.

Usage::

    python -m reptaxonomy.pipelines.audit_experiments -y cfgA.yaml cfgB.yaml [--root-dir ...]
        [--mode finetune_compare] [--stages ...] [--datasets ...] [--models ...] [--set k=v ...]
        [--show 25] [--all] [--validate] [--csv audit.csv] [--json audit.json]

Prints a per-stage summary, the outstanding units and the commands that complete the run.
Exit status: 0 = nothing outstanding, 1 = work remains, 2 = a config could not be audited.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as _dt
import json
import os
import socket
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from reptaxonomy.pipelines import gfm_experiment as G

try:  # restart support (newer versions)
    from reptaxonomy.util import run_state as RS
except ImportError:  # pragma: no cover - older checkouts
    RS = None

DONE, FAILED, RUNNING = "_DONE.", "_FAILED.", "_RUNNING."
STATUSES = ["done", "stale", "failed", "running", "interrupted", "partial", "todo", "blocked", "unverified"]
OUTSTANDING = {"stale", "failed", "interrupted", "partial", "todo", "blocked"}
PER_MODEL_STEPS = ["compute_resource_reqs", "weight_matrix_analysis", "run_projections", "embedding_robustness"]
GLOBAL_STEPS = ["data_kernel_analysis", "reference_similarity", "invariance_checks", "geodesic_compare",
                "demsar_wrapper"]
FINAL_STEPS = ["cross_perspective_synthesis"]
UPSAMPLE_INDEPENDENT = {"compute_resource_reqs", "weight_matrix_analysis"}
PRIMARY_ONLY_GLOBAL = {"invariance_checks"}
OPTIONAL_OUTPUT = {"compute_resource_reqs", "weight_matrix_analysis", "reference_similarity", "invariance_checks",
                   "geodesic_compare"}
HOST = socket.gethostname()


# ----------------------------------------------------------------------------- small helpers
def _safe(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_.=+" else "_" for ch in str(key))


def marker(d: os.PathLike, key: str, prefix: str = DONE) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(d, f"{prefix}{_safe(key)}.json").read_text())
    except Exception:
        return None


def _ts(m: Optional[Dict[str, Any]], k: str = "completed_at") -> Optional[float]:
    try:
        return _dt.datetime.fromisoformat(str(m[k])).timestamp()
    except Exception:
        return None


def mtime(p: os.PathLike) -> Optional[float]:
    try:
        return Path(p).stat().st_mtime
    except OSError:
        return None


def newest(paths: Iterable[os.PathLike]) -> Optional[float]:
    ts = [t for t in (mtime(p) for p in paths) if t is not None]
    return max(ts) if ts else None


def pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def _err(m: Dict[str, Any]) -> str:
    e = str(m.get("error") or m.get("traceback") or "")
    lines = [ln.strip() for ln in e.strip().splitlines() if ln.strip()]
    return (lines[-1] if lines else "failed")[:160]


def npy_ok(p: Path) -> bool:
    if RS is not None and hasattr(RS, "npy_ok"):
        return RS.npy_ok(p)
    try:
        import numpy as np
        np.load(p, mmap_mode="r")
        return True
    except Exception:
        return False


class Unit(dict):
    """One row of the audit."""


MARKERS_REQUIRED = False  # set per config: True when the audited version writes completion markers


def unit_status(d: os.PathLike, key: Optional[str], legacy: Callable[[], Tuple[str, str]],
                upstream_ts: Optional[float] = None, out_ts: Callable[[], Optional[float]] = lambda: None,
                fp_now: Optional[str] = None, check_config: bool = True) -> Tuple[str, str]:
    """Status from markers if present, else from ``legacy()`` -> (status, detail)."""
    if key is not None:
        done = marker(d, key)
        f = marker(d, key, FAILED)
        if done is not None and f is not None and (_ts(f, "failed_at") or 0) >= (_ts(done) or 0):
            done = None
        if done is not None:
            if check_config and fp_now and done.get("fingerprint") not in (None, fp_now):
                return "stale", "configuration changed since it ran"
            t = _ts(done)
            if upstream_ts and t and upstream_ts > t + 1:
                return "stale", "upstream outputs are newer"
            return "done", ("adopted pre-marker outputs" if done.get("adopted") else "")
        f = marker(d, key, FAILED)
        if f is not None:
            return "failed", _err(f)
        r = marker(d, key, RUNNING)
        if r is not None:
            if r.get("host") == HOST and pid_alive(r.get("pid")):
                return "running", f"pid {r.get('pid')} since {r.get('started_at')}"
            if r.get("host") not in (None, HOST):
                return "interrupted", f"started on {r.get('host')} {r.get('started_at')} (cannot verify from here)"
            return "interrupted", f"killed run from {r.get('started_at')}"
    st, det = legacy()
    if st == "done" and key is not None and MARKERS_REQUIRED and det.startswith("no marker"):
        return "todo", "outputs present but no completion marker (the driver will recompute)"
    if st == "unverified" and key is not None and MARKERS_REQUIRED:
        return "todo", "no completion marker"
    if st == "done" and upstream_ts:
        t = out_ts()
        if t and upstream_ts > t + 1:
            return "stale", "upstream outputs are newer"
    return st, det


# ----------------------------------------------------------------------------- config / plan
def paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    """``gfm_experiment._paths`` without creating directories."""
    root = cfg.get("root_dir", "./gfm_runs")
    p = {"work_dir": cfg.get("work_dir") or os.path.join(root, "work"),
         "embed_dir": cfg.get("embed_dir") or os.path.join(root, "embeddings"),
         "out_dir": cfg.get("out_dir") or os.path.join(root, "outputs"),
         "finetune_root": cfg.get("finetune_root") or os.path.join(root, "finetune")}
    p["rt_out"] = os.path.join(p["out_dir"], "rep_taxonomy")
    p["rt_out_ft"] = os.path.join(p["out_dir"], "rep_taxonomy_ft")
    return p


def apply_overrides(cfg: Dict[str, Any], a: argparse.Namespace) -> Dict[str, Any]:
    """Same command-line overrides as gfm_experiment.main."""
    if a.mode:
        cfg["mode"] = a.mode
    if a.root_dir:
        cfg["root_dir"] = a.root_dir
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
    return cfg


def stage_list(cfg: Dict[str, Any], stages: Optional[Sequence[str]]) -> List[str]:
    mode = cfg.get("mode", "pretrained")
    post = list(getattr(G, "POST_STAGES", []))
    if stages or cfg.get("stages"):
        return list(stages or cfg.get("stages"))
    return list(G.PRETRAINED_STAGES) + (list(G.FINETUNE_STAGES) + post if mode == "finetune_compare" else [])


def _ft_modes(cfg: Dict[str, Any], include_wise: bool) -> List[str]:
    try:
        return G.ft_modes(cfg, include_wise=include_wise)
    except TypeError:  # older signature
        return G.ft_modes(cfg)


def _trainable(cfg: Dict[str, Any], name: str) -> bool:
    spec = getattr(G, "variant_spec", None)
    if spec is None:
        return True
    try:
        return spec(cfg, name) is not None
    except ValueError:
        return False


def _wise(cfg: Dict[str, Any]) -> List[str]:
    f = getattr(G, "wise_names", None)
    return f(cfg) if f else []


# ----------------------------------------------------------------------------- auditor
class Auditor:
    def __init__(self, cfg_path: str, cfg: Dict[str, Any], stages: List[str], datasets: Optional[Sequence[str]],
                 validate: bool = False) -> None:
        self.cfg_path, self.cfg, self.stages, self.validate = cfg_path, cfg, stages, validate
        self.P = paths(cfg)
        self.mode = cfg.get("mode", "pretrained")
        self.markers = RS is not None
        global MARKERS_REQUIRED
        MARKERS_REQUIRED = self.markers
        self.check_config = bool((cfg.get("resume", {}) or {}).get("check_config", True)) if isinstance(
            cfg.get("resume", {}), dict) else True
        if RS is not None and hasattr(RS, "set_root"):
            RS.set_root(cfg.get("root_dir", "./gfm_runs"))
        self.units: List[Unit] = []
        self.notes: List[str] = []
        self.models = G.resolve_models(cfg)
        ds_cfgs = {k: v for k, v in (cfg.get("datasets", {}) or {}).items()
                   if (v or {}).get("enabled", True) and (not datasets or k in datasets)}
        self.ctxs = {}
        for k, v in ds_cfgs.items():
            try:
                self.ctxs[k] = G.DatasetContext(k, v)
            except Exception as exc:  # e.g. EPA region listing needs the data root
                ctx = ctx_from_plan(self.P, k, v)
                if ctx is None:
                    raise
                self.ctxs[k] = ctx
                self.notes.append(f"{k}: conditions taken from plan.json (config expansion failed: {exc})")
        self.plan_ok = self._load_plan()
        self._stems: Dict[str, List[str]] = {}

    # -- plan
    def _load_plan(self) -> bool:
        try:
            plan = json.loads(Path(self.P["work_dir"], "plan.json").read_text())
        except Exception:
            plan = None
        ok = True
        for name, ctx in self.ctxs.items():
            d = (plan or {}).get("datasets", {}).get(name)
            if not d:
                ok = False
                ctx.models = list(self.models)
                self.notes.append(f"{name}: no plan.json entry (prepare not finished); model list is the configured "
                                  "suite before band-coverage filtering")
                continue
            ctx.source_bands, ctx.sensor, ctx.gsd = d["source_bands"], d["sensor"], d["gsd"]
            ctx.raw_scale, ctx.num_classes, ctx.band_stats = d["raw_scale"], d["num_classes"], d["band_stats"]
            ctx.models, ctx.coverage = list(d["models"]), d.get("coverage", {})
            analysed = set(d["analysed"])
            for c in ctx.conditions:
                c.analyze = c.name in analysed
            dropped = sorted(set(self.models) - set(ctx.models))
            if dropped:
                self.notes.append(f"{name}: {len(dropped)} models excluded by band coverage: {', '.join(dropped)}")
        return ok

    def add(self, ds: str, stage: str, status: str, detail: str = "", path: os.PathLike = "", **kw: Any) -> str:
        u = Unit(config=self.cfg_path, dataset=ds, stage=stage, status=status, detail=detail, path=str(path))
        u.update({k: kw.get(k, "") for k in ("condition", "model", "variant", "step", "run")})
        self.units.append(u)
        return status

    def on(self, stage: str) -> bool:
        return stage in self.stages

    # -- per-unit checks
    def stems(self, cond: str) -> List[str]:
        if cond not in self._stems:
            try:
                self._stems[cond] = json.loads(Path(self.P["work_dir"], "manifests", f"{cond}.json").read_text())["stems"]
            except Exception:
                self._stems[cond] = []
        return self._stems[cond]

    def embed_dir(self, cond: str, model: str) -> Path:
        return Path(self.P["embed_dir"], cond, model, "test")

    def embed_status(self, ctx, cond, model: str, overrides=None) -> Tuple[str, str]:
        d = self.embed_dir(cond.name, model)
        stems = self.stems(cond.name)
        if not stems:
            return "blocked", "condition cache (prepare) missing"

        def legacy() -> Tuple[str, str]:
            ok = npy_ok if self.validate else (lambda p: p.exists())
            n = sum(ok(d / f"embd_{s}.npy") for s in stems)
            if n == len(stems):
                return "done", "no marker; all chips present"
            return ("partial", f"{n}/{len(stems)} chips") if n else ("todo", "")

        fp = None
        if self.markers and self.plan_ok and hasattr(G, "embed_fp") and self.check_config:
            try:
                fp = G.embed_fp(ctx, cond, model, self.cfg, overrides)
            except Exception:
                fp = None
        st, det = unit_status(d, "embed", legacy, fp_now=fp, check_config=self.check_config)
        if st in ("interrupted", "failed"):
            n = sum((d / f"embd_{s}.npy").exists() for s in stems)
            det = f"{det}; {n}/{len(stems)} chips on disk (kept on resume)"
        return st, det

    def embed_ts(self, cond: str, model: str) -> Optional[float]:
        m = marker(self.embed_dir(cond, model), "embed")
        if m is not None:
            return _ts(m)
        stems = self.stems(cond)
        return newest(self.embed_dir(cond, model) / f"embd_{s}.npy" for s in stems[-3:]) if stems else None

    # -- stages
    def audit(self) -> List[Unit]:
        for name, ctx in self.ctxs.items():
            self.audit_prepare(name, ctx)
            conds = ctx.analysed
            if self.on("embed"):
                for c in conds:
                    for m in ctx.models:
                        st, det = self.embed_status(ctx, c, m)
                        self.add(name, "embed", st, det, self.embed_dir(c.name, m), condition=c.name, model=m)
            if self.on("pipeline"):
                for c in conds:
                    self.audit_pipeline(name, ctx, c, ctx.models, self.P["rt_out"], "pipeline")
            if self.on("cross_condition") and (self.cfg.get("cross_condition", {}) or {}).get("enabled", True):
                self.audit_cross_condition(name, ctx, conds)
            gcfg = dict(self.cfg.get("geo_id", {}) or {})
            geo_on = bool(gcfg.get("enabled", True)) and not (((gcfg.get("datasets", {}) or {}).get(name, {}) or {})
                                                              .get("skip", False))
            if self.on("geo_id") and geo_on:
                self.audit_geo(name, name, [c.name for c in conds], ctx.models, "geo_id")
            if self.on("synthesis"):
                self.audit_family_synthesis(name, [c.name for c in conds], ctx.models, self.P["rt_out"],
                                            f"{name}_pretrained", "synthesis")
            if self.mode == "finetune_compare":
                self.audit_finetune_family(name, ctx, gcfg, geo_on)
        if self.mode == "finetune_compare":
            if self.on("forgetting"):
                self.audit_forgetting()
            if self.on("ft_sequential"):
                self.audit_sequential()
        return self.units

    def audit_prepare(self, name: str, ctx) -> None:
        if not self.on("prepare") and self.plan_ok:
            return
        wd = Path(self.P["work_dir"])
        needed = {c.name for c in ctx.analysed}
        for r in ctx.runs:
            needed |= set(r.eval_conditions)
        label_grid = int(self.cfg.get("label_grid", 64))
        for cname in sorted(needed):
            cond = ctx.by_name.get(cname)
            man_p, lab_p = wd / "manifests" / f"{cname}.json", wd / "labels" / f"{cname}.npz"
            if man_p.exists() and lab_p.exists():
                st, det = "done", ""
                try:
                    man = json.loads(man_p.read_text())
                    if self.markers and cond is not None and hasattr(G, "_cache_fp") and self.check_config \
                            and man.get("fingerprint") not in (None, G._cache_fp(cond, label_grid)):
                        st, det = "stale", "condition definition changed since the cache was built"
                    else:
                        det = f"{len(man.get('stems', []))} chips"
                except Exception:
                    st, det = "partial", "manifest unreadable"
            elif lab_p.exists() or man_p.exists():
                st, det = "partial", "manifest or labels missing"
            else:
                st, det = "todo", ""
            self.add(name, "prepare", st, det, man_p, condition=cname)
        st = "done" if self.plan_ok else "todo"
        self.add(name, "prepare", st, "" if self.plan_ok else "plan.json missing or lacks this dataset",
                 wd / "plan.json", step="plan")

    # full_pipeline steps of one condition (pretrained or fine-tuned models)
    def audit_pipeline(self, name: str, ctx, cond, models: List[str], out_root: str, stage: str,
                       blocked_models: Sequence[str] = ()) -> None:
        present = [m for m in models if m not in blocked_models and self.embed_status(ctx, cond, m)[0] in ("done", "stale")]
        missing = [m for m in models if m not in present]
        pipe = self.cfg.get("pipeline", {}) or {}
        steps_cfg = pipe.get("steps", {}) or {}
        methods = [str(m).lower() for m in pipe.get("upsample_methods", [pipe.get("upsample_method", "pixelshuffle")])]
        primary = methods[0]
        enabled = lambda s: bool((steps_cfg.get(s, {}) or {}).get("enabled", True))
        if len(present) < 3:
            for s in [x for x in PER_MODEL_STEPS + GLOBAL_STEPS + FINAL_STEPS if enabled(x)]:
                self.add(name, stage, "blocked", f"needs >= 3 models with complete embeddings (have {len(present)})",
                         Path(out_root, cond.name), condition=cond.name, step=s)
            return
        if missing:
            self.notes.append(f"{stage} {cond.name}: runs without {len(missing)} model(s) lacking embeddings: "
                              f"{', '.join(missing)}")
        state_dir = Path(out_root, "_state", cond.name)
        c = cond.name
        emb_ts = {m: self.embed_ts(c, m) for m in present}
        all_ts = max([t for t in emb_ts.values() if t] or [0]) or None
        any_per_ts: List[float] = []
        for method in methods:
            vout = Path(out_root) if method == primary else Path(out_root, f"upsample_{method}")
            for s in PER_MODEL_STEPS:
                if not enabled(s) or (method != primary and s in UPSAMPLE_INDEPENDENT):
                    continue
                for m in present:
                    files = self._step_files(s, vout, c, m)
                    st, det = unit_status(state_dir, f"{method}.{s}.{m}", lambda: self._legacy(s, files),
                                          upstream_ts=emb_ts[m], out_ts=lambda: newest(files))
                    t = _ts(marker(state_dir, f"{method}.{s}.{m}")) or newest(files)
                    if t:
                        any_per_ts.append(t)
                    self.add(name, stage, st, det, files[0] if files else vout, condition=c, model=m,
                             variant=method, step=s)
            per_ts = max(any_per_ts) if any_per_ts else all_ts
            for s in GLOBAL_STEPS:
                if not enabled(s) or (method != primary and s in PRIMARY_ONLY_GLOBAL):
                    continue
                files = self._step_files(s, vout, c, None)
                st, det = unit_status(state_dir, f"{method}.{s}.global", lambda: self._legacy(s, files),
                                      upstream_ts=per_ts, out_ts=lambda: newest(files))
                if s == "demsar_wrapper" and st == "done" and not self.markers:
                    det = "inferred: demsar_inputs is shared by all conditions of this output root"
                self.add(name, stage, st, det, files[0] if files else vout, condition=c, variant=method, step=s)
        for s in FINAL_STEPS:
            if not enabled(s):
                continue
            d = Path(out_root, "synthesis", c)
            files = sorted(d.glob("*.csv")) if d.exists() else []
            st, det = unit_status(state_dir, f"final.{s}.global",
                                  lambda: ("done", "no marker; outputs present") if files else ("todo", ""))
            self.add(name, stage, st, det, d, condition=c, step=s)

    def _step_files(self, s: str, vout: Path, c: str, m: Optional[str]) -> List[Path]:
        E = Path(self.P["embed_dir"], c, m or "", "test")
        if s == "compute_resource_reqs":
            return [E / "calflops_stats" / f"{m}.flops_summary.csv"]
        if s == "weight_matrix_analysis":
            return sorted((E / "ww_stats").glob("*.summary.csv")) or [E / "ww_stats" / f"{m}.summary.csv"]
        if s == "run_projections":
            return [vout / c / m / f"silhouette_stats.{m}.{c}.csv"]
        if s == "embedding_robustness":
            return [vout / c / m / f"silhouette_stats_recomputed.{m}.{c}.csv"]
        if s == "data_kernel_analysis":
            return [vout / c / "data_kernel_manifest.json"]
        if s == "reference_similarity":
            return [vout / c / "pairwise_reference_summary.csv"]
        if s == "invariance_checks":
            return [vout / c / "invariance_summary.csv"]
        if s == "geodesic_compare":
            g = vout / c / "geodesic_compare"
            return sorted(g.glob("*_pairwise_comparison.summary.csv")) or [g]
        if s == "demsar_wrapper":
            return [vout / "demsar_inputs" / "all_metrics_mean_std_summary.csv"]
        return []

    def _legacy(self, s: str, files: List[Path]) -> Tuple[str, str]:
        if files and all(f.exists() and f.is_file() for f in files):
            return "done", "no marker; outputs present"
        if s in OPTIONAL_OUTPUT:
            return "unverified", "no output found (this step may produce none for this configuration)"
        return "todo", ""

    def audit_cross_condition(self, name: str, ctx, conds) -> None:
        unit = Path(self.P["out_dir"], "cross_condition", name)
        cn = [c.name for c in conds]
        ups = [self.embed_ts(c, m) for c in cn for m in ctx.models]
        incomplete = sum(self.embed_status(ctx, c, m)[0] not in ("done", "stale") for c in conds for m in ctx.models)
        f = unit / "condition_summary.csv"
        st, det = unit_status(unit, "cross_condition",
                              lambda: ("done", "no marker; outputs present") if f.exists() else ("todo", ""),
                              upstream_ts=max([t for t in ups if t] or [0]) or None, out_ts=lambda: mtime(f))
        if st == "todo" and incomplete:
            st, det = "blocked", f"{incomplete} (condition, model) embedding sets incomplete"
        elif st == "done" and incomplete:
            det = (det + "; " if det else "") + f"ran with {incomplete} embedding sets missing"
        self.add(name, "cross_condition", st, det, unit)

    def audit_geo(self, name: str, tag: str, conds: List[str], models: List[str], stage: str) -> None:
        unit = Path(self.P["out_dir"], "geo_id", tag)
        f = unit / "params.json"
        ups = [self.embed_ts(c, m) for c in conds for m in models]
        st, det = unit_status(unit, "geo_id", lambda: ("done", "no marker; outputs present") if f.exists() else ("todo", ""),
                              upstream_ts=max([t for t in ups if t] or [0]) or None, out_ts=lambda: mtime(f))
        if st == "todo" and not any(ups):
            st, det = "blocked", "no embeddings yet"
        self.add(name, stage, st, det, unit, step=tag)

    def audit_family_synthesis(self, name: str, conds: List[str], models: List[str], out_root: str, tag: str,
                               stage: str) -> None:
        have = [c for c in conds if Path(out_root, c).exists()]
        unit = Path(out_root, "_state", f"synthesis_{tag}")
        d = Path(out_root, "synthesis_pooled", tag)
        files = sorted(d.glob("*.csv")) if d.exists() else []
        ups = [newest(Path(out_root, "_state", c).glob(DONE + "*")) for c in have] if self.markers else []
        st, det = unit_status(unit, "synthesis", lambda: ("done", "no marker; outputs present") if files else ("todo", ""),
                              upstream_ts=max([t for t in ups if t] or [0]) or None)
        if st == "todo" and not have:
            st, det = "blocked", "no pipeline outputs yet"
        elif st == "done" and len(have) < len(conds):
            det = (det + "; " if det else "") + f"pooled {len(have)}/{len(conds)} conditions"
            if not self.markers:
                st = "stale"
        self.add(name, stage, st, det, d, step=tag)

    # -- fine-tuning
    def ft_dir(self, ds: str, tag: str, m: str, v: str) -> Path:
        return Path(self.P["finetune_root"], ds, tag, m, v)

    def ft_status(self, d: Path) -> Tuple[str, str]:
        if marker(d, "finetune") is not None and not (d / "metrics.json").exists():
            return "todo", "marker without metrics.json (outputs removed)"
        def legacy() -> Tuple[str, str]:
            if (d / "metrics.json").exists():
                return "done", "no marker; metrics.json present"
            if (d / "train_state.pt").exists():
                return "partial", "resumable checkpoint (train_state.pt)"
            if (d / "model.pt").exists():
                return "partial", "model.pt without metrics.json (killed during evaluation)"
            return "todo", ""
        if (d / "metrics.json").exists():
            if marker(d, "finetune", FAILED) is None and marker(d, "finetune", RUNNING) is None:
                return "done", ""
        st, det = unit_status(d, "finetune" if marker(d, "finetune") is None else None, legacy)
        if st == "interrupted" and (d / "train_state.pt").exists():
            det += "; resumes from train_state.pt"
        return st, det

    def audit_finetune_family(self, name: str, ctx, gcfg: Dict[str, Any], geo_on: bool) -> None:
        fc = self.cfg.get("finetune", {}) or {}
        runs = [r for r in ctx.runs if not (fc.get("runs") and r.tag not in fc["runs"])]
        ftm = G.ft_models(ctx, self.cfg)
        train_modes = [v for v in _ft_modes(self.cfg, False) if _trainable(self.cfg, v)]
        wise = _wise(self.cfg)
        src = str(((fc.get("wise_ft", {}) or {}).get("source_variant", "full")))
        for r in runs:
            for m in ftm:
                for v in train_modes:
                    d = self.ft_dir(name, r.tag, m, v)
                    if self.on("finetune"):
                        st, det = self.ft_status(d)
                        if st == "todo" and v == "lpft" and not (self.ft_dir(name, r.tag, m, "linear") / "metrics.json").exists():
                            st, det = "blocked", "needs the linear run (LP-FT head)"
                        self.add(name, "finetune", st, det, d, run=r.tag, model=m, variant=v)
                if self.on("ft_wise"):
                    lin, srcd = self.ft_dir(name, r.tag, m, "linear"), self.ft_dir(name, r.tag, m, src)
                    for w in wise:
                        d = self.ft_dir(name, r.tag, m, w)
                        if not ((lin / "metrics.json").exists() and (srcd / "metrics.json").exists()):
                            self.add(name, "ft_wise", "blocked", f"needs linear and {src} fine-tunes", d,
                                     run=r.tag, model=m, variant=w)
                            continue
                        f = d / "metrics.json"
                        st, det = unit_status(d, "wise", lambda: ("done", "") if f.exists() else ("todo", ""),
                                              upstream_ts=newest([lin / "model.pt", srcd / "model.pt"]),
                                              out_ts=lambda: mtime(f))
                        self.add(name, "ft_wise", st, det, d, run=r.tag, model=m, variant=w)
            # fine-tuned model names the downstream stages expect
            expected, ready, pairs = [], [], []
            for m in ftm:
                for v in _ft_modes(self.cfg, True):
                    if v == "linear":
                        continue
                    n = G.ft_name(m, r.tag, v)
                    d = self.ft_dir(name, r.tag, m, v)
                    expected.append(n)
                    pairs.append((m, n))
                    if (d / "model.pt").exists() and (d / "metrics.json").exists():
                        ready.append(n)
            not_ready = [n for n in expected if n not in ready]
            ov = {n: dict((self.cfg.get("per_model_overrides", {}) or {}).get(n.split("__ft__")[0], {}) or {},
                          checkpoint=str(self.ft_dir(name, r.tag, *_split_ft(n, r.tag), ) / "model.pt"))
                  for n in ready}
            evals = [ctx.by_name[c] for c in r.eval_conditions if c in ctx.by_name and ctx.by_name[c].analyze]
            if self.on("ft_embed"):
                for c in evals:
                    for n in expected:
                        if n in not_ready:
                            self.add(name, "ft_embed", "blocked", "fine-tuned checkpoint not complete",
                                     self.embed_dir(c.name, n), run=r.tag, condition=c.name, model=n)
                            continue
                        st, det = self.embed_status(ctx, c, n, ov)
                        ck_ts = mtime(ov[n]["checkpoint"])
                        if st == "done" and ck_ts and (self.embed_ts(c.name, n) or 0) + 1 < ck_ts:
                            st, det = "stale", "checkpoint is newer than the embeddings"
                        self.add(name, "ft_embed", st, det, self.embed_dir(c.name, n), run=r.tag, condition=c.name,
                                 model=n)
            pre = list(dict.fromkeys(p[0] for p in pairs if p[1] in ready))
            if self.on("ft_pipeline"):
                for c in evals:
                    self.audit_pipeline(name, ctx, c, list(dict.fromkeys(pre + expected)),
                                        os.path.join(self.P["rt_out_ft"], r.tag), "ft_pipeline",
                                        blocked_models=not_ready)
            if self.on("ft_geo_id") and geo_on and gcfg.get("finetuned", True) and expected:
                self.audit_geo(name, f"{name}_{r.tag}_ft", [c.name for c in evals], pre + ready, "ft_geo_id")
            if self.on("ft_synthesis") and expected:
                self.audit_family_synthesis(name, [c.name for c in evals], pre + ready,
                                            os.path.join(self.P["rt_out_ft"], r.tag), f"{name}_{r.tag}_ft",
                                            "ft_synthesis")
        if self.on("finetune_compare"):
            f = Path(self.P["out_dir"], "finetune_compare", name, "pretrained_vs_finetuned.csv")
            ft_done = [u for u in self.units if u["dataset"] == name and u["stage"] == "finetune"]
            newest_ft = newest(Path(u["path"], "metrics.json") for u in ft_done)
            if f.exists():
                st = "stale" if newest_ft and mtime(f) + 1 < newest_ft else "done"
                det = "fine-tunes finished after it ran" if st == "stale" else "rebuilt on every run"
            else:
                st, det = ("todo", "") if newest_ft else ("blocked", "no fine-tunes yet")
            self.add(name, "finetune_compare", st, det, f)

    def audit_forgetting(self) -> None:
        fc = dict(self.cfg.get("forgetting", {}) or {})
        rc = getattr(G, "retention_conditions", None)
        if not fc.get("enabled", True) or rc is None:
            return
        for name, ctx in self.ctxs.items():
            if fc.get("families") and name not in fc["families"]:
                continue
            units = Path(self.P["out_dir"], "forgetting", name, "_units")
            for r in ctx.runs:
                if (self.cfg.get("finetune", {}) or {}).get("runs") and r.tag not in self.cfg["finetune"]["runs"]:
                    continue
                for m in G.ft_models(ctx, self.cfg):
                    for v in _ft_modes(self.cfg, True):
                        if v == "linear":
                            continue
                        n = G.ft_name(m, r.tag, v)
                        ready = (self.ft_dir(name, r.tag, m, v) / "metrics.json").exists()
                        for e in rc(self.ctxs, ctx, r, m, fc):
                            key = f"{n}__{e['condition']}"
                            if not ready:
                                self.add(name, "forgetting", "blocked", "fine-tuned checkpoint not complete", units,
                                         run=r.tag, model=n, condition=e["condition"])
                                continue
                            f = units / f"{_safe(key)}.json"
                            st, det = unit_status(units, key, lambda: ("done", "") if f.exists() else ("todo", ""))
                            self.add(name, "forgetting", st, (det + f" ({e['relation']})").strip(), f, run=r.tag,
                                     model=n, condition=e["condition"])

    def audit_sequential(self) -> None:
        for chain in (self.cfg.get("forgetting", {}) or {}).get("sequential", []) or []:
            cname = str(chain.get("name", "chain"))
            try:
                steps = []
                for st in chain["steps"]:
                    fam, tag = st.split(":", 1)
                    steps.append((self.ctxs[fam], fam, tag))
            except (KeyError, ValueError):
                self.add("-", "ft_sequential", "blocked", "step references a dataset not in this config", "",
                         step=cname)
                continue
            fams = [s[0] for s in steps]
            models = chain.get("models") or sorted(set.intersection(*[set(G.ft_models(c, self.cfg)) for c in fams]))
            models = [m for m in models if all(m in c.models for c in fams)]
            root = Path(self.P["finetune_root"], "sequential", cname)
            for m in models:
                prev_ok = True
                for j, (ctx, fam, tag) in enumerate(steps):
                    d = root / m / f"step{j}_{fam}-{tag}"
                    st, det = self.ft_status(d)
                    if st == "todo" and not prev_ok:
                        st, det = "blocked", "previous step not complete"
                    evals = [d / f"eval_task{i}.json" for i in range(j + 1)]
                    n_ev = sum(e.exists() for e in evals)
                    if st == "done" and n_ev < len(evals):
                        st, det = "partial", f"trained; {n_ev}/{len(evals)} task evaluations"
                    self.add(fam, "ft_sequential", st, det, d, model=m, step=f"{cname}:{j}", run=tag)
                    prev_ok = st in ("done", "stale")


def ctx_from_plan(P: Dict[str, str], name: str, ds_cfg: Dict[str, Any]):
    """DatasetContext rebuilt from work/plan.json (used when the config cannot be expanded here)."""
    from reptaxonomy.gfm.conditions import Condition, FinetuneRun
    try:
        d = json.loads(Path(P["work_dir"], "plan.json").read_text())["datasets"][name]
    except Exception:
        return None
    mk = lambda x: Condition(**x) if x else None
    ctx = G.DatasetContext.__new__(G.DatasetContext)
    ctx.name, ctx.cfg = name, ds_cfg
    ctx.conditions = [mk(c) for c in d["conditions"]]
    ctx.runs = [FinetuneRun(tag=r["tag"], dataset=name, train=mk(r["train"]), val=mk(r.get("val")),
                            eval_conditions=r["eval_conditions"], id_conditions=r.get("id_conditions", []),
                            ood_conditions=r.get("ood_conditions", [])) for r in d.get("finetune_runs", [])]
    ctx.by_name = {c.name: c for c in ctx.conditions}
    for r in ctx.runs:
        for c in (r.train, r.val):
            if c is not None:
                ctx.by_name.setdefault(c.name, c)
    ctx.comparisons = d.get("comparisons", [])
    ctx.source_bands, ctx.sensor, ctx.gsd, ctx.raw_scale = d["source_bands"], d["sensor"], d["gsd"], d["raw_scale"]
    ctx.num_classes, ctx.band_stats = d["num_classes"], d["band_stats"]
    ctx.coverage, ctx.models = d.get("coverage", {}), list(d["models"])
    return ctx


def _split_ft(n: str, tag: str) -> Tuple[str, str]:
    m, rest = n.split("__ft__", 1)
    return m, rest[len(tag) + 1:]


# ----------------------------------------------------------------------------- reporting
def live_runs() -> List[Tuple[int, str]]:
    out = []
    for p in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore").strip()
        except Exception:
            continue
        if "gfm_experiment" in cmd and "audit_experiments" not in cmd and int(p.name) != os.getpid():
            out.append((int(p.name), cmd))
    return out


def summarize(units: List[Unit]) -> "OrderedDict[Tuple[str, str], Counter]":
    s: "OrderedDict[Tuple[str, str], Counter]" = OrderedDict()
    for u in units:
        s.setdefault((u["dataset"], u["stage"]), Counter())[u["status"]] += 1
    return s


def _label(u: Unit) -> str:
    parts = [u[k] for k in ("run", "condition", "model", "variant", "step") if u.get(k)]
    return " | ".join(parts) or u["path"]


def driver_args(a: argparse.Namespace) -> str:
    """Overrides given to the audit, repeated so the suggested driver command sees the same config."""
    out = []
    if a.mode:
        out.append(f"--mode {a.mode}")
    if a.root_dir:
        out.append(f"--root-dir {a.root_dir}")
    if a.models:
        out.append("--models " + " ".join(a.models))
    if a.max_samples:
        out.append(f"--max-samples {a.max_samples}")
    if a.set:
        out.append("--set " + " ".join(a.set))
    return (" " + " ".join(out)) if out else ""


def rerun_commands(cfg_path: str, units: List[Unit], stage_order: List[str], markers: bool,
                   extra: str = "") -> List[str]:
    todo = [u for u in units if u["status"] in OUTSTANDING]
    if not todo:
        return []
    stages = [s for s in stage_order if any(u["stage"] == s for u in todo)]
    dss = sorted({u["dataset"] for u in todo if u["dataset"] != "-"})
    # downstream summaries rebuild from what exists; include them after the stages with work left
    follow = {"embed": ["pipeline", "cross_condition", "geo_id", "synthesis"],
              "pipeline": ["cross_condition", "synthesis", "finetune_compare"], "finetune": ["ft_wise", "ft_embed", "ft_pipeline", "finetune_compare",
                                                       "ft_geo_id", "ft_synthesis", "forgetting"],
              "ft_embed": ["ft_pipeline", "finetune_compare", "ft_geo_id", "ft_synthesis", "forgetting"],
              "ft_pipeline": ["finetune_compare", "ft_synthesis"]}
    want = list(stages)
    for s in stages:
        want += [x for x in follow.get(s, []) if x in stage_order]
    want = [s for s in stage_order if s in set(want)]
    cmd = f"python -m reptaxonomy.pipelines.gfm_experiment -y {cfg_path}{extra} --stages {' '.join(want)}"
    if dss:
        cmd += f" --datasets {' '.join(dss)}"
    out = [cmd]
    gpu = [s for s in want if s in ("embed", "finetune", "ft_wise", "ft_embed", "ft_sequential")]
    if gpu and len(gpu) < len(want):
        out.append(f"  (GPU part: --stages {' '.join(gpu)}; CPU part: --stages "
                   f"{' '.join(s for s in want if s not in gpu)})")
    if not markers:
        out.append("  (this version has no completion markers: pipeline stages recompute every condition they "
                   "are given; embeddings and fine-tunes with complete outputs are skipped)")
    return out


def print_report(aud: Auditor, show: int, show_all: bool, stage_order: List[str], fh=sys.stdout) -> None:
    w = fh.write
    w(f"\n=== {aud.cfg_path}\n")
    w(f"root_dir={aud.cfg.get('root_dir', './gfm_runs')}  mode={aud.mode}  "
      f"datasets={','.join(aud.ctxs) or '-'}  markers={'yes' if aud.markers else 'no (pre-restart version)'}\n")
    for n in aud.notes:
        w(f"  note: {n}\n")
    s = summarize(aud.units)
    cols = [c for c in STATUSES if any(cnt[c] for cnt in s.values())]
    head = f"  {'dataset':<10} {'stage':<17} {'units':>6} " + " ".join(f"{c:>11}" for c in cols)
    w(head + "\n  " + "-" * (len(head) - 2) + "\n")
    for (ds, st), cnt in sorted(s.items(), key=lambda kv: (kv[0][0], stage_order.index(kv[0][1])
                                                            if kv[0][1] in stage_order else 99)):
        w(f"  {ds:<10} {st:<17} {sum(cnt.values()):>6} " + " ".join(f"{cnt[c] or '':>11}" for c in cols) + "\n")
    tot = Counter(u["status"] for u in aud.units)
    w(f"  total: {len(aud.units)} units; " + ", ".join(f"{tot[c]} {c}" for c in STATUSES if tot[c]) + "\n")
    pending = [u for u in aud.units if show_all or u["status"] not in ("done", "unverified")]
    if pending:
        w("\n  outstanding units:\n")
        by: "OrderedDict[Tuple[str, str], List[Unit]]" = OrderedDict()
        for u in pending:
            by.setdefault((u["dataset"], u["stage"]), []).append(u)
        for (ds, st), us in by.items():
            w(f"  [{ds} / {st}]\n")
            for u in us[:show]:
                w(f"    {u['status']:<11} {_label(u)}" + (f"  -- {u['detail']}" if u["detail"] else "") + "\n")
            if len(us) > show:
                w(f"    ... {len(us) - show} more (use --show or --csv)\n")
    unv = [u for u in aud.units if u["status"] == "unverified"] if not show_all else []
    if unv:
        w(f"\n  unverified (no output; may be legitimate for this config -- check, or rerun the stage):\n")
        for u in unv[:show]:
            w(f"    {_label(u)}\n")
        if len(unv) > show:
            w(f"    ... {len(unv) - show} more\n")
    cmds = rerun_commands(aud.cfg_path, aud.units, stage_order, aud.markers, getattr(aud, "extra", ""))
    if cmds:
        w("\n  to finish:\n")
        for c in cmds:
            w(f"    {c}\n")
    else:
        w("\n  nothing outstanding.\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-y", "--yaml", nargs="+", required=True, help="one or more gfm_experiment configs")
    ap.add_argument("--mode", choices=["pretrained", "finetune_compare"])
    ap.add_argument("--stages", nargs="*", help="audit only these stages (default: the stages the config runs)")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--root-dir")
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--validate", action="store_true", help="check .npy headers of embeddings without markers (slower)")
    ap.add_argument("--show", type=int, default=25, help="outstanding units listed per (dataset, stage)")
    ap.add_argument("--all", action="store_true", help="list completed units too")
    ap.add_argument("--csv", help="write every unit to this CSV")
    ap.add_argument("--json", help="write every unit (and per-config summaries) to this JSON")
    a = ap.parse_args(argv)

    all_units: List[Unit] = []
    summaries: Dict[str, Any] = {}
    rc = 0
    order = list(G.PRETRAINED_STAGES) + list(G.FINETUNE_STAGES) + list(getattr(G, "POST_STAGES", []))
    procs = live_runs()
    if procs:
        print(f"active gfm_experiment processes on {HOST}:")
        for pid, cmd in procs:
            print(f"  pid {pid}: {cmd[:200]}")
    for y in a.yaml:
        try:
            cfg = apply_overrides(G.load_config(y), a)
            stages = stage_list(cfg, a.stages)
            aud = Auditor(y, cfg, stages, a.datasets, validate=a.validate)
            aud.extra = driver_args(a)
            aud.audit()
        except Exception as exc:  # keep auditing the other configs
            print(f"\n=== {y}\n  could not audit: {type(exc).__name__}: {exc}")
            rc = 2
            continue
        print_report(aud, a.show, a.all, order)
        all_units += aud.units
        summaries[y] = {f"{ds}/{st}": dict(c) for (ds, st), c in summarize(aud.units).items()}
        if rc == 0 and any(u["status"] in OUTSTANDING for u in aud.units):
            rc = 1
    fields = ["config", "dataset", "stage", "run", "condition", "model", "variant", "step", "status", "detail", "path"]
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows([{k: u.get(k, "") for k in fields} for u in all_units])
        print(f"\nwrote {len(all_units)} units to {a.csv}")
    if a.json:
        Path(a.json).write_text(json.dumps({"host": HOST, "generated": _dt.datetime.now().isoformat(timespec="seconds"),
                                            "summaries": summaries, "units": all_units}, indent=1, default=str))
        print(f"wrote {a.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
