"""Gather and compress the outputs of gfm_experiment runs for manuscript updates.

The run commands (or the same -y / --set / --root-dir arguments the driver took) decide what is
collected: the config chain is resolved with ``gfm_experiment.load_config`` and the driver's
command-line overrides, the run root comes from the resolved config, the conditions from
``work/plan.json`` and the upsampling variants from ``pipeline.upsample_methods`` (union over all
commands, plus any ``upsample_<method>`` folder found on disk).

Collected (paths inside the archive mirror the run root):
  configs/                        every YAML of each config chain, resolved configs, run commands
  work/                           plan.json, resolved_config.yaml, model_coverage.csv, band_stats,
                                  manifests (chip / class counts), parallel-worker logs
  outputs/rep_taxonomy/<cond>/    primary variant; upsample_<m>/<cond>/ for the other variants,
                                  demsar_inputs, synthesis, synthesis_pooled/<ds>_*, _state markers
  outputs/cross_condition/<ds>/   outputs/geo_id/<ds>*/   outputs/rep_taxonomy_ft/ (if present)
  embeddings/<cond>/<model>/      small files only (ww_stats, calflops_stats, logs); never embd_*.npy
  finetune/<ds>/                  metrics / configs only, never checkpoints
Written alongside: MANIFEST.csv (every file seen, included or skipped with the reason),
dataset_summary.csv (chips and class pixel counts per condition), embeddings_index.csv,
CHECKLIST.md / checklist.csv (manuscript items -> files, with missing items flagged) and the
audit_experiments report.

Examples
    python -m reptaxonomy.pipelines.gather_outputs -y reptaxonomy/config/gfm/mados_pretrained_fast.yaml \\
        --set "pipeline.upsample_methods=[pixelshuffle,patch,nearest,bilinear]" -o mados_outputs.tar.gz
    python -m reptaxonomy.pipelines.gather_outputs --commands mados_commands.txt -o mados_outputs.tar.gz
    python -m reptaxonomy.pipelines.gather_outputs --commands mados_commands.txt --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fnmatch
import io
import json
import os
import shlex
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

CKPT_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx"}
ARRAY_EXT = {".npy", ".npz", ".h5", ".hdf5", ".zarr", ".nc", ".tif", ".tiff"}
LOG_EXT = {".log", ".out", ".err"}
RT_SPECIAL = {"_state", "synthesis_pooled", "synthesis"}


# ----------------------------------------------------------------------------- run specs
def driver_parser() -> argparse.ArgumentParser:
    """The gfm_experiment arguments that change what a run writes."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("-y", "--yaml")
    ap.add_argument("--mode")
    ap.add_argument("--stages", nargs="*")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--root-dir")
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--conditions", nargs="*")
    return ap


@dataclass
class RunSpec:
    yaml: str
    sets: List[str] = field(default_factory=list)
    root_dir: Optional[str] = None
    stages: Optional[List[str]] = None
    datasets: Optional[List[str]] = None
    conditions: Optional[List[str]] = None
    models: Optional[List[str]] = None
    mode: Optional[str] = None
    max_samples: Optional[int] = None
    command: str = ""


def _cut_shell(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        if t in {"|", "&", "&&", ";", "||"} or t.startswith((">", "2>", "1>", "&>")):
            break
        out.append(t)
    return out


def parse_commands(text: str) -> List[RunSpec]:
    """gfm_experiment invocations from a shell history / script (continuations, comments,
    env assignments, nohup, redirections and pipes are handled)."""
    lines, buf = [], ""
    for raw in text.splitlines():
        s = raw.rstrip()
        if s.endswith("\\"):
            buf += s[:-1] + " "
            continue
        lines.append(buf + s)
        buf = ""
    if buf:
        lines.append(buf)
    specs = []
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#") or "gfm_experiment" not in ln:
            continue
        try:
            toks = shlex.split(ln, comments=True)
        except ValueError:
            continue
        idx = next((i for i, t in enumerate(toks) if "gfm_experiment" in t), None)
        if idx is None:
            continue
        args, _ = driver_parser().parse_known_args(_cut_shell(toks[idx + 1:]))
        if not args.yaml:
            continue
        specs.append(RunSpec(args.yaml, list(args.set or []), args.root_dir, args.stages, args.datasets,
                             args.conditions, args.models, args.mode, args.max_samples, command=ln))
    return specs


def config_chain(path: Path) -> List[Path]:
    """The YAML file and every ``base:`` file it inherits from (depth first, no duplicates)."""
    seen: List[Path] = []

    def walk(p: Path) -> None:
        p = p.resolve()
        if p in seen or not p.exists():
            return
        seen.append(p)
        bases = (yaml.safe_load(p.read_text()) or {}).get("base")
        for b in ([bases] if isinstance(bases, str) else (bases or [])):
            walk(p.parent / b)

    walk(path)
    return seen


def resolve_spec(spec: RunSpec, cwd: Path) -> Dict[str, Any]:
    from reptaxonomy.pipelines import audit_experiments as AE
    from reptaxonomy.pipelines import gfm_experiment as G
    y = Path(spec.yaml) if Path(spec.yaml).is_absolute() else cwd / spec.yaml
    cfg = G.load_config(y)
    ns = argparse.Namespace(mode=spec.mode, root_dir=spec.root_dir, models=spec.models,
                            max_samples=spec.max_samples, set=spec.sets)
    cfg = AE.apply_overrides(cfg, ns)
    for k in ("root_dir", "work_dir", "embed_dir", "out_dir", "finetune_root"):
        if cfg.get(k) and not Path(cfg[k]).is_absolute():
            cfg[k] = str(cwd / cfg[k])
    cfg.setdefault("root_dir", str(cwd / "gfm_runs"))
    return {"cfg": cfg, "P": AE.paths(cfg), "yaml": y, "chain": config_chain(y)}


def methods_of(cfg: Dict[str, Any]) -> List[str]:
    from reptaxonomy.util.general_utils import as_method_list
    return as_method_list((cfg.get("pipeline", {}) or {}).get("upsample_methods"))


# ----------------------------------------------------------------------------- collection
@dataclass
class Entry:
    src: Path
    arc: str
    size: int
    included: bool
    reason: str
    truncate_to: Optional[int] = None


class Collector:
    def __init__(self, root: Path, max_file_mb: float, max_array_mb: float, with_embeddings: bool,
                 include_labels: bool):
        self.root = root
        self.max_file = int(max_file_mb * 2 ** 20)
        self.max_array = int(max_array_mb * 2 ** 20)
        self.with_embeddings = with_embeddings
        self.include_labels = include_labels
        self.entries: Dict[str, Entry] = {}

    def arc(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return ("external/" + p.name)

    def add_file(self, p: Path, small_only: bool = False) -> None:
        if not p.is_file():
            return
        a = self.arc(p)
        if a in self.entries:
            return
        size = p.stat().st_size
        ext = p.suffix.lower()
        inc, why, trunc = True, "", None
        if ext in CKPT_EXT:
            inc, why = False, "checkpoint"
        elif p.name.startswith("embd_") and ext == ".npy":
            inc, why = (True, "embedding (--with-embeddings)") if self.with_embeddings else (False, "embedding array")
        elif ext in ARRAY_EXT:
            if small_only:
                inc, why = False, "array in embeddings tree"
            elif size > self.max_array:
                inc, why = False, f"array > {self.max_array / 2 ** 20:g} MB"
        elif ext in LOG_EXT and size > self.max_file:
            trunc, why = self.max_file, "log truncated to its tail"
        elif size > self.max_file:
            inc, why = False, f"file > {self.max_file / 2 ** 20:g} MB"
        self.entries[a] = Entry(p, a, size, inc, why, trunc)

    def add_tree(self, d: Path, small_only: bool = False, skip_dirs: Iterable[str] = ()) -> None:
        if not d.is_dir():
            return
        skip = set(skip_dirs)
        for dp, dns, fns in os.walk(d):
            dns[:] = sorted(x for x in dns if x not in skip and not x.endswith(".zarr"))
            for f in sorted(fns):
                self.add_file(Path(dp, f), small_only)

    def add_glob(self, d: Path, pattern: str) -> None:
        if d.is_dir():
            for p in sorted(d.glob(pattern)):
                self.add_file(p)

    def included(self) -> List[Entry]:
        return [e for e in self.entries.values() if e.included]


def gather_root(col: Collector, P: Dict[str, str], dss: Dict[str, Dict[str, Any]],
                all_conds: Sequence[str]) -> None:
    sel = {c for d in dss.values() for c in d["conditions"]}
    other = set(all_conds) - sel
    W, E, O = Path(P["work_dir"]), Path(P["embed_dir"]), Path(P["out_dir"])
    for f in ("plan.json", "resolved_config.yaml", "model_coverage.csv"):
        col.add_file(W / f)
    col.add_tree(W / "band_stats")
    col.add_tree(W / "_parallel" / "logs")
    for c in sel:
        col.add_file(W / "manifests" / f"{c}.json")
        if col.include_labels:
            col.add_file(W / "labels" / f"{c}.npz")
        rt = W / "rt" / c
        if rt.is_dir():
            for p in rt.rglob("*"):
                if p.suffix.lower() in {".yaml", ".yml", ".json", ".log"}:
                    col.add_file(p)
    for c in sel:
        col.add_tree(E / c, small_only=not col.with_embeddings)
    col.add_tree(E / "_shared", small_only=True)

    def out_root(vout: Path) -> None:
        if not vout.is_dir():
            return
        for ch in sorted(vout.iterdir()):
            n = ch.name
            if ch.is_file():
                col.add_file(ch)
            elif n in other:
                continue
            elif n.startswith("upsample_"):
                out_root(ch)
            elif n == "_state":
                for st in ch.iterdir():
                    if st.name in sel or (st.name.startswith("synthesis_") and
                                          any(st.name[10:].startswith(ds) for ds in dss)) or st.is_file():
                        col.add_tree(st) if st.is_dir() else col.add_file(st)
            elif n == "synthesis_pooled":
                for t in ch.iterdir():
                    if any(t.name.startswith(ds) for ds in dss):
                        col.add_tree(t)
            elif n == "synthesis":
                for t in ch.iterdir():
                    if t.name in sel:
                        col.add_tree(t)
            else:
                col.add_tree(ch, skip_dirs=other)

    for top in ("rep_taxonomy", "rep_taxonomy_ft"):
        T = O / top
        if top == "rep_taxonomy_ft" and T.is_dir():
            for tag in sorted(T.iterdir()):
                out_root(tag) if tag.is_dir() else col.add_file(tag)
        else:
            out_root(T)
    for ds in dss:
        col.add_tree(O / "cross_condition" / ds)
        if (O / "geo_id").is_dir():
            for t in sorted((O / "geo_id").iterdir()):
                if t.is_dir() and t.name.startswith(ds):
                    col.add_tree(t)
        F = Path(P["finetune_root"], ds)
        if F.is_dir():
            for p in F.rglob("*"):
                if p.suffix.lower() in {".json", ".csv", ".yaml", ".yml", ".log", ".txt"}:
                    col.add_file(p)
    known = {"rep_taxonomy", "rep_taxonomy_ft", "cross_condition", "geo_id"}
    if O.is_dir():   # forgetting, finetune_compare, ... : dataset-scoped folders, conditions filtered
        for top in sorted(O.iterdir()):
            if top.is_dir() and top.name not in known:
                col.add_tree(top, skip_dirs=other)


# ----------------------------------------------------------------------------- summaries
def dataset_summary(W: Path, dss: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for ds, d in dss.items():
        for c in d["conditions"]:
            p = W / "manifests" / f"{c}.json"
            if not p.exists():
                rows.append({"dataset": ds, "condition": c, "status": "manifest missing"})
                continue
            m = json.loads(p.read_text())
            hist = m.get("class_pixel_counts") or []
            lats = [x for x in (m.get("lats") or []) if x == x and x is not None]
            r = {"dataset": ds, "condition": c, "status": "ok", "setup": m.get("setup"), "role": m.get("role"),
                 "n_chips": len(m.get("stems") or []), "n_georeferenced": len(lats),
                 "n_labelled_pixels": int(sum(hist[:-1])) if hist else 0,
                 "n_classes_present": int(sum(1 for h in hist[:-1] if h > 0)) if hist else 0}
            r.update({f"class_{i}_pixels": int(h) for i, h in enumerate(hist[:-1])})
            rows.append(r)
    return rows


def embeddings_index(E: Path, dss: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    import numpy as np
    rows = []
    for ds, d in dss.items():
        for c in d["conditions"]:
            for m in d["models"]:
                td = E / c / m / "test"
                fs = sorted(td.glob("embd_*.npy")) if td.is_dir() else []
                shape = ""
                if fs:
                    try:
                        with open(fs[0], "rb") as fh:
                            v = np.lib.format.read_magic(fh)
                            hdr = (np.lib.format.read_array_header_1_0 if v == (1, 0)
                                   else np.lib.format.read_array_header_2_0)(fh)
                            shape = "x".join(map(str, hdr[0])) + f" {hdr[2]}"
                    except Exception:
                        shape = "unreadable"
                rows.append({"dataset": ds, "condition": c, "model": m, "n_files": len(fs),
                             "total_mb": round(sum(f.stat().st_size for f in fs) / 2 ** 20, 2),
                             "first_shape_dtype": shape})
    return rows


# ----------------------------------------------------------------------------- checklist
# (manuscript section, item, scope, pattern(s), per_model)
#  scopes: work | variant (each upsampling variant, per condition) | primary (primary variant, per
#  condition) | patch (patch variant, per condition) | vroot (each variant root) | embed (per condition)
#  | synthesis | cross | geo        {V}=variant root, {c}=condition, {ds}=dataset
CHECKS: List[Tuple[str, str, str, Tuple[str, ...], bool]] = [
    ("Data (chips, classes, bands)", "condition manifests (chips, class pixel counts)", "work", ("work/manifests/{c}.json",), False),
    ("Data (chips, classes, bands)", "model band coverage (Table V)", "work1", ("work/model_coverage.csv",), False),
    ("Data (chips, classes, bands)", "plan / resolved config", "work1", ("work/plan.json", "work/resolved_config.yaml"), False),
    ("Data (chips, classes, bands)", "band statistics", "work1", ("work/band_stats/*",), False),
    ("P1 class separability", "silhouette per model (native + projections)", "variant", ("{V}/{c}/*/silhouette_stats.*.csv",), True),
    ("P1 class separability", "silhouette under perturbation", "variant", ("{V}/{c}/*/silhouette_stats_recomputed.*.csv",), True),
    ("P1 class separability", "intrinsic dimension", "variant", ("{V}/{c}/*intrinsic_dimension*",), False),
    ("P2 similarity", "data-kernel residual / omnibus summaries", "variant", ("{V}/{c}/data_kernel_manifest.json", "{V}/{c}/*pairwise_residual_summary*.csv"), False),
    ("P2 similarity", "reference-set similarity", "variant", ("{V}/{c}/pairwise_reference_summary.csv",), False),
    ("P2 similarity", "relative representations (geodesic)", "variant", ("{V}/{c}/geodesic_compare/*_pairwise_comparison*.csv",), False),
    ("P2 similarity", "invariance checks (primary only)", "primary", ("{V}/{c}/invariance_summary.csv",), False),
    ("Patch-level variant", "patch class retention (cell purity filter)", "patch", ("{V}/{c}/patch_class_retention.csv",), False),
    ("P3 weights / resources", "WeightWatcher summaries", "embed", ("embeddings/{c}/*/test/ww_stats/*.summary.*", "embeddings/_shared/*ww_stats/*.summary.*"), True),
    ("P3 weights / resources", "FLOPs / parameters", "embed", ("embeddings/{c}/*/test/calflops_stats/*.flops*",), True),
    ("P3 weights / resources", "compression / inference energy / latency", "embed", ("embeddings/{c}/*/test/calflops_stats/*.inference_energy.csv", "embeddings/{c}/*/test/calflops_stats/*.latencies.csv"), True),
    ("P3 weights / resources", "hardware environment", "embed", ("embeddings/{c}/*/test/calflops_stats/*.environment.json",), True),
    ("Demsar ranking", "metric summaries / avg ranks / Nemenyi", "vroot", ("{V}/demsar_inputs/all_metrics_mean_std_summary.csv", "{V}/*avg_ranks*.csv"), False),
    ("Synthesis", "ranking table", "synthesis", ("{S}/ranking_table.csv",), False),
    ("Synthesis", "variance decomposition", "synthesis", ("{S}/variance_decomposition*.csv",), False),
    ("Synthesis", "lens sensitivity", "synthesis", ("{S}/lens_sensitivity_silhouette.csv", "{S}/lens_sensitivity_similarity.csv"), False),
    ("Synthesis", "data-kernel centrality / reference Mantel", "synthesis", ("{S}/centrality.csv", "{S}/reference_vs_data_kernel_mantel.csv"), False),
    ("Synthesis", "invariance summary", "synthesis", ("{S}/invariance_summary.csv",), False),
    ("Synthesis", "resource / environment tables", "synthesis", ("{S}/resource_table.csv", "{S}/environment_table.csv"), False),
    ("Synthesis", "upsampling sensitivity", "synthesis_multi", ("{S}/upsampling_sensitivity_silhouette.csv", "{S}/upsampling_sensitivity_similarity.csv"), False),
    ("Synthesis", "patch class retention (pooled)", "synthesis_patch", ("{S}/patch_class_retention.csv",), False),
    ("Cross-condition", "condition summary / pairwise metrics / stability", "cross", ("outputs/cross_condition/{ds}/condition_summary.csv", "outputs/cross_condition/{ds}/*pairwise*.csv", "outputs/cross_condition/{ds}/*stability*.csv"), False),
    ("Geolocated ID", "global / local ID, spatial statistics, variogram", "geo", ("outputs/geo_id/{ds}/*global_id*.csv", "outputs/geo_id/{ds}/*local_id*.csv", "outputs/geo_id/{ds}/*spatial*.csv", "outputs/geo_id/{ds}/*variogram*.csv"), False),
    ("Geolocated ID", "maps", "geo", ("outputs/geo_id/{ds}/*.png",), False),
]


def checklist(paths: Sequence[str], dss: Dict[str, Dict[str, Any]], methods: List[str],
              syn_tags: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    primary = methods[0]
    vroot = {m: "outputs/rep_taxonomy" if m == primary else f"outputs/rep_taxonomy/upsample_{m}" for m in methods}
    rows = []

    def count(pats: Sequence[str]) -> Tuple[int, List[int]]:
        per = [len(fnmatch.filter(paths, p)) for p in pats]
        return sum(per), per

    for sec, item, scope, pats, per_model in CHECKS:
        for ds, d in dss.items():
            nm = len(d["models"])
            jobs: List[Tuple[str, Dict[str, str]]] = []
            if scope == "work":
                jobs = [(c, {"c": c}) for c in d["conditions"]]
            elif scope == "work1":
                jobs = [("", {})]
            elif scope in ("variant", "primary", "patch"):
                ms = {"variant": methods, "primary": [primary],
                      "patch": [m for m in methods if m == "patch"]}[scope]
                jobs = [(f"{m}:{c}", {"V": vroot[m], "c": c}) for m in ms for c in d["analysed"]]
            elif scope == "vroot":
                jobs = [(m, {"V": vroot[m]}) for m in methods]
            elif scope == "embed":
                jobs = [(c, {"c": c}) for c in d["conditions"]]
            elif scope.startswith("synthesis"):
                if scope == "synthesis_multi" and len(methods) < 2:
                    continue
                if scope == "synthesis_patch" and "patch" not in methods:
                    continue
                tags = syn_tags.get(ds) or [f"{ds}_pretrained"]
                jobs = [(t, {"S": f"outputs/rep_taxonomy/synthesis_pooled/{t}"}) for t in tags]
            elif scope in ("cross", "geo"):
                jobs = [("", {})]
            for unit, sub in jobs:
                sub = {**sub, "ds": ds}
                ex = [p.format(**sub) for p in pats]
                n, per = count(ex)
                if per_model:
                    status = "ok" if n >= nm else ("partial" if n else "MISSING")
                    found = f"{n}/{nm} models"
                else:
                    status = "ok" if all(per) else ("partial" if n else "MISSING")
                    found = str(n)
                rows.append({"section": sec, "item": item, "dataset": ds, "unit": unit,
                             "found": found, "status": status, "patterns": "; ".join(ex)})
    return rows


def synthesis_variants(root: Path, tag: str) -> Optional[List[str]]:
    f = root / "outputs" / "rep_taxonomy" / "synthesis_pooled" / tag / "upsampling_sensitivity_silhouette.csv"
    if not f.exists():
        return None
    with open(f) as fh:
        return sorted({r["variant"] for r in csv.DictReader(fh) if r.get("variant")})


# ----------------------------------------------------------------------------- audit
def run_audit(spec: RunSpec, res: Dict[str, Any], methods: List[str], datasets: Optional[List[str]],
              tmp: Path) -> Dict[str, bytes]:
    from reptaxonomy.pipelines import audit_experiments as AE
    sets = [s for s in spec.sets if not s.startswith("pipeline.upsample_methods=")]
    sets.append(f"pipeline.upsample_methods=[{','.join(methods)}]")
    argv = ["-y", str(res["yaml"]), "--root-dir", str(res["cfg"]["root_dir"]), "--set", *sets, "--csv", str(tmp / "audit.csv"), "--json", str(tmp / "audit.json"),
            "--timings"]
    if spec.mode:
        argv += ["--mode", spec.mode]
    for flag, v in (("--datasets", datasets), ("--conditions", spec.conditions), ("--models", spec.models)):
        if v:
            argv += [flag, *v]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = AE.main(argv)
        buf.write(f"\n[audit exit code {rc}]\n")
    except BaseException as exc:   # noqa: BLE001 - the archive is still useful without the audit
        buf.write(f"\n[audit failed: {exc!r}]\n")
    out = {"audit/audit_report.txt": ("audit_experiments " + shlex.join(argv) + "\n\n" + buf.getvalue()).encode()}
    for n in ("audit.csv", "audit.json"):
        if (tmp / n).exists():
            out[f"audit/{n}"] = (tmp / n).read_bytes()
    return out


# ----------------------------------------------------------------------------- writing
def _csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


def checklist_md(rows: List[Dict[str, Any]], notes: List[str], header: str) -> bytes:
    lines = [f"# Output checklist\n\n{header}\n"]
    if notes:
        lines.append("## Action items\n")
        lines += [f"- {n}" for n in notes]
        lines.append("")
    sec = None
    for r in rows:
        if r["section"] != sec:
            sec = r["section"]
            lines += ["", f"## {sec}", "", "| dataset | item | unit | found | status |", "|---|---|---|---|---|"]
        lines.append(f"| {r['dataset']} | {r['item']} | {r['unit'] or '-'} | {r['found']} | {r['status']} |")
    return ("\n".join(lines) + "\n").encode()


def write_archive(out: Path, top: str, files: List[Entry], extra: Dict[str, bytes]) -> None:
    tmp = out.with_name(out.name + ".partial")
    if out.suffix == ".zip":
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            for n, b in extra.items():
                z.writestr(f"{top}/{n}", b)
            for e in files:
                if e.truncate_to:
                    z.writestr(f"{top}/{e.arc}", _tail(e.src, e.truncate_to))
                else:
                    z.write(e.src, f"{top}/{e.arc}")
    else:
        mode = "w:gz" if out.name.endswith((".tar.gz", ".tgz")) else ("w:xz" if out.name.endswith(".xz") else "w")
        with tarfile.open(tmp, mode) as t:
            def addb(name: str, b: bytes) -> None:
                ti = tarfile.TarInfo(f"{top}/{name}")
                ti.size, ti.mtime = len(b), int(time.time())
                t.addfile(ti, io.BytesIO(b))
            for n, b in extra.items():
                addb(n, b)
            for e in files:
                if e.truncate_to:
                    addb(e.arc, _tail(e.src, e.truncate_to))
                else:
                    t.add(e.src, f"{top}/{e.arc}", recursive=False)
    os.replace(tmp, out)


def _tail(p: Path, n: int) -> bytes:
    with open(p, "rb") as fh:
        fh.seek(max(0, p.stat().st_size - n))
        return b"[... truncated: head removed ...]\n" + fh.read()


def _mb(n: int) -> str:
    return f"{n / 2 ** 20:,.1f} MB"


# ----------------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], parents=[driver_parser()])
    ap.add_argument("--commands", help="file with the gfm_experiment commands that were run (shell history / script)")
    ap.add_argument("--variants", nargs="*", help="upsampling variants to check (default: union of the commands' "
                                                  "pipeline.upsample_methods and upsample_* folders on disk)")
    ap.add_argument("-o", "--out", help="archive (.tar.gz, .tar.xz, .tar or .zip); default <dataset>_outputs.tar.gz")
    ap.add_argument("--cwd", default=".", help="directory the runs were started from (relative paths)")
    ap.add_argument("--max-file-mb", type=float, default=100.0, help="skip non-array files larger than this")
    ap.add_argument("--max-array-mb", type=float, default=5.0,
                    help="skip .npy/.npz/... analysis arrays larger than this (projections, kNN graphs)")
    ap.add_argument("--with-embeddings", action="store_true", help="also pack embd_*.npy (very large)")
    ap.add_argument("--include-labels", action="store_true", help="also pack work/labels/<cond>.npz")
    ap.add_argument("--no-audit", action="store_true", help="skip the audit_experiments report")
    ap.add_argument("--dry-run", action="store_true", help="report what would be packed; write nothing")
    a = ap.parse_args(argv)
    cwd = Path(a.cwd).resolve()

    specs: List[RunSpec] = []
    cmd_text = ""
    if a.commands:
        cmd_text = Path(a.commands).read_text()
        specs = parse_commands(cmd_text)
        if not specs:
            ap.error(f"no gfm_experiment commands found in {a.commands}")
    if a.yaml:
        specs.append(RunSpec(a.yaml, list(a.set or []), a.root_dir, a.stages, a.datasets, a.conditions,
                             a.models, a.mode, a.max_samples, command="(command line)"))
    if not specs:
        ap.error("give -y CONFIG (with the driver's --set / --root-dir ...) or --commands FILE")

    resolved = [resolve_spec(s, cwd) for s in specs]
    roots: Dict[str, List[int]] = {}
    for i, r in enumerate(resolved):
        roots.setdefault(str(Path(r["cfg"]["root_dir"]).resolve()), []).append(i)
    if len(roots) > 1:
        print(f"note: {len(roots)} run roots; each is packed under its own folder")

    extra: Dict[str, bytes] = {}
    all_files: List[Entry] = []
    manifest_rows: List[Dict[str, Any]] = []
    summary_lines: List[str] = []
    multi = len(roots) > 1
    first_ds = None
    for root_s, idxs in roots.items():
        root = Path(root_s)
        pre = f"{root.name}/" if multi else ""
        cfgs = [resolved[i] for i in idxs]
        P = cfgs[0]["P"]
        plan_p = Path(P["work_dir"], "plan.json")
        plan = json.loads(plan_p.read_text()) if plan_p.exists() else {"datasets": {}}
        want_ds: List[str] = []
        for i in idxs:
            names = list((resolved[i]["cfg"].get("datasets", {}) or {}).keys())
            sel = specs[i].datasets or a.datasets
            want_ds += [n for n in names if (not sel or n in sel) and n not in want_ds]
        globs = [g for i in idxs for g in (specs[i].conditions or [])] or (a.conditions or [])
        dss: Dict[str, Dict[str, Any]] = {}
        all_conds: List[str] = []
        for ds, d in (plan.get("datasets") or {}).items():
            names = [c["name"] for c in d.get("conditions", [])]
            all_conds += names
            if ds not in want_ds:
                continue
            keep = [c for c in names if not globs or any(fnmatch.fnmatch(c, g) for g in globs)]
            dss[ds] = {"conditions": keep, "analysed": [c for c in d.get("analysed", names) if c in keep],
                       "models": list(d.get("models", []))}
        missing_ds = [d for d in want_ds if d not in dss]
        first_ds = first_ds or (list(dss) or want_ds or ["gfm"])[0]

        methods: List[str] = []
        for c in cfgs:
            methods += [m for m in methods_of(c["cfg"]) if m not in methods]
        rt = Path(P["out_dir"], "rep_taxonomy")
        on_disk = sorted(p.name[9:] for p in rt.glob("upsample_*") if p.is_dir()) if rt.is_dir() else []
        if a.variants:
            methods = list(dict.fromkeys(a.variants))
        else:
            methods += [m for m in on_disk if m not in methods]

        col = Collector(root, a.max_file_mb, a.max_array_mb, a.with_embeddings, a.include_labels)
        gather_root(col, P, dss, all_conds)
        inc = col.included()
        for e in col.entries.values():
            manifest_rows.append({"path": pre + e.arc, "bytes": e.size, "included": int(e.included),
                                  "note": e.reason})
        all_files += [Entry(e.src, pre + e.arc, e.size, True, e.reason, e.truncate_to) for e in inc]

        # configs and commands
        for k, c in enumerate(cfgs):
            for y in c["chain"]:
                extra.setdefault(f"{pre}configs/{y.name}", y.read_bytes())
            extra[f"{pre}configs/resolved_{k:02d}_{c['yaml'].stem}.yaml"] = yaml.safe_dump(
                c["cfg"], sort_keys=False).encode()
        extra[f"{pre}configs/commands.txt"] = ("\n".join(specs[i].command for i in idxs) + "\n").encode()
        if cmd_text:
            extra[f"{pre}configs/commands_source.txt"] = cmd_text.encode()

        W, E = Path(P["work_dir"]), Path(P["embed_dir"])
        extra[f"{pre}dataset_summary.csv"] = _csv_bytes(dataset_summary(W, dss))
        extra[f"{pre}embeddings_index.csv"] = _csv_bytes(embeddings_index(E, dss))

        syn_tags = {}
        sp = rt / "synthesis_pooled"
        for ds in dss:
            syn_tags[ds] = sorted(t.name for t in sp.glob(f"{ds}*") if t.is_dir()) if sp.is_dir() else []
        paths = [e.arc for e in inc]
        rows = checklist(paths, dss, methods, syn_tags)
        notes: List[str] = []
        if missing_ds:
            notes.append(f"datasets not in work/plan.json (prepare never ran?): {', '.join(missing_ds)}")
        undeclared = [m for m in on_disk if m not in methods_of(cfgs[0]["cfg"]) and not a.variants]
        if undeclared and len(cfgs) == 1:
            notes.append(f"variant folders on disk not in this config's upsample_methods: {', '.join(undeclared)} "
                         "(collected; pass every run command with --commands to record how they were made)")
        nonprim = methods[1:]
        for ds in dss:
            for tag in syn_tags.get(ds) or [f"{ds}_pretrained"]:
                have = synthesis_variants(root, tag)
                lack = [m for m in nonprim if have is None or m not in have]
                if nonprim and lack:
                    y = specs[idxs[0]].yaml
                    notes.append(
                        f"synthesis_pooled/{tag} lacks variant(s) {', '.join(lack)} (the last synthesis run listed "
                        f"other upsample_methods). Re-run only the synthesis with all variants:\n"
                        f"  `python -m reptaxonomy.pipelines.gfm_experiment -y {y} --stages synthesis --set "
                        f"\"pipeline.upsample_methods=[{','.join(methods)}]\"`")
        miss = [r for r in rows if r["status"] != "ok"]
        if miss:
            notes.append(f"{len(miss)} checklist entries partial or missing (see tables below / checklist.csv)")
        header = (f"Run root: `{root}`  \nDatasets: {', '.join(dss) or '-'}  \nConditions: "
                  f"{', '.join(c for d in dss.values() for c in d['conditions']) or '-'}  \n"
                  f"Variants: {', '.join(methods)} (primary: {methods[0]})  \n"
                  f"Models: {', '.join(sorted({m for d in dss.values() for m in d['models']})) or '-'}  \n"
                  f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        extra[f"{pre}CHECKLIST.md"] = checklist_md(rows, notes, header)
        extra[f"{pre}checklist.csv"] = _csv_bytes(rows)

        if not a.no_audit and not a.dry_run:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                for n, b in run_audit(specs[idxs[0]], cfgs[0], methods, list(dss) or None, Path(td)).items():
                    extra[pre + n] = b

        skipped = [e for e in col.entries.values() if not e.included]
        summary_lines += [f"[{root}] datasets={','.join(dss) or '-'} variants={','.join(methods)}",
                          f"  include {len(inc)} files ({_mb(sum(e.size for e in inc))}); "
                          f"skip {len(skipped)} ({_mb(sum(e.size for e in skipped))})",
                          f"  checklist: {sum(r['status'] == 'ok' for r in rows)} ok, "
                          f"{sum(r['status'] == 'partial' for r in rows)} partial, "
                          f"{sum(r['status'] == 'MISSING' for r in rows)} missing"]
        summary_lines += ["  - " + n.replace("\n", "\n    ") for n in notes]

    extra["MANIFEST.csv"] = _csv_bytes(manifest_rows)
    print("\n".join(summary_lines))
    if a.dry_run:
        return 0
    out = Path(a.out or f"{first_ds}_outputs.tar.gz").resolve()
    top = out.name.split(".")[0]
    write_archive(out, top, all_files, extra)
    print(f"wrote {out} ({_mb(out.stat().st_size)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
