"""Restart / retry support: atomic writes, completion markers, overwrite policy.

A long run is split into *units* (one condition cache, one (condition, model)
embedding set, one full_pipeline step, one fine-tune, ...). A unit is complete
when its completion marker ``_DONE.<key>.json`` exists in the unit's directory.
The marker is written last, only after the unit's outputs are on disk, so
killing a run leaves no marker and the next run redoes that unit. Failures are
logged to ``_FAILED.<key>.json`` (cleared on success), so a plain re-run retries
exactly the failed and the unfinished units.

Every marker stores a *fingerprint*, a hash of the configuration and the
upstream markers the unit depends on. With ``resume.check_config: true``
(default) a unit whose fingerprint no longer matches is treated as stale and
recomputed. Because upstream completion stamps are part of downstream
fingerprints, recomputing embeddings automatically invalidates the analyses
built on them.

Config (top level of a gfm_experiment YAML; ``overwrite`` is also read by
full_pipeline and generate_embeddings)::

    overwrite: false           # false | true | [stage, "stage:step", ...]
    resume:
      check_config: true       # recompute units whose config/upstream changed
      validate_files: true     # verify .npy headers when adopting pre-marker outputs

The legacy keys ``overwrite_cache``, ``overwrite_embeddings``,
``finetune.overwrite`` and ``forgetting.sequential[].overwrite`` are still honoured.

Overwrite clears a unit's marker before recomputing it. If an overwrite run is
killed, restart it WITHOUT ``--overwrite``: cleared units carry no marker and
are redone, while units that the overwrite run already finished are kept.

CLI: ``python -m reptaxonomy.util.run_state <root_dir> [--failed]`` prints the
completion / failure markers under a run root.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import os
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, os.PathLike]
DONE_PREFIX, FAILED_PREFIX, RUNNING_PREFIX = "_DONE.", "_FAILED.", "_RUNNING."

# legacy per-stage keys -> stages they force
_LEGACY = {
    "overwrite_cache": ("prepare",),
    "overwrite_embeddings": ("embed", "ft_embed"),
}


# ----------------------------------------------------------------------------- atomic writes
@contextlib.contextmanager
def atomic_path(path: PathLike, suffix: str = "") -> Iterator[Path]:
    """Yield a temp path next to ``path``; move it into place only if the block succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=f".tmp{suffix}", dir=path.parent)
    os.close(fd)
    tmp_p = Path(tmp)
    try:
        yield tmp_p
        os.replace(tmp_p, path)
    except BaseException:
        if tmp_p.is_dir():
            shutil.rmtree(tmp_p, ignore_errors=True)
        else:
            tmp_p.unlink(missing_ok=True)
        raise


def save_npy(path: PathLike, arr: Any, allow_pickle: bool = False) -> Path:
    with atomic_path(path, ".npy") as tmp:
        with open(tmp, "wb") as fh:
            np.save(fh, arr, allow_pickle=allow_pickle)
    return Path(path)


def torch_save(obj: Any, path: PathLike) -> Path:
    import torch
    with atomic_path(path) as tmp:
        torch.save(obj, tmp)
    return Path(path)


def write_text(path: PathLike, text: str) -> Path:
    with atomic_path(path) as tmp:
        tmp.write_text(text)
    return Path(path)


def write_json(path: PathLike, obj: Any, indent: int = 1) -> Path:
    return write_text(path, json.dumps(obj, indent=indent, default=str))


def to_csv(df: Any, path: PathLike, **kw: Any) -> Path:
    with atomic_path(path, ".csv") as tmp:
        df.to_csv(tmp, **kw)
    return Path(path)


def savez_compressed(path: PathLike, **arrays: Any) -> Path:
    with atomic_path(path, ".npz") as tmp:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
    return Path(path)


def joblib_dump(obj: Any, path: PathLike) -> Path:
    import joblib
    with atomic_path(path) as tmp:
        joblib.dump(obj, tmp)
    return Path(path)


def zarr_save(path: PathLike, arr: np.ndarray) -> Path:
    """zarr directory stores cannot be written into a temp *file*; write a sibling dir, then rename."""
    import zarr
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        zarr.save(str(tmp), arr)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return path


# ----------------------------------------------------------------------------- validation
def npy_ok(path: PathLike) -> bool:
    """True if ``path`` is a complete .npy file (header parses and the data size matches)."""
    p = Path(path)
    try:
        size = p.stat().st_size
        with open(p, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            reader = {(1, 0): np.lib.format.read_array_header_1_0,
                      (2, 0): np.lib.format.read_array_header_2_0}.get(version)
            if reader is None:
                return True
            shape, _fortran, dtype = reader(fh)
            if dtype.hasobject:
                return size > fh.tell()
            return size == fh.tell() + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    except Exception:
        return False


# ----------------------------------------------------------------------------- fingerprints
_ROOTS: list = []


def set_root(root: PathLike) -> None:
    """Fingerprints treat paths under ``root`` as root-relative (moved/symlinked roots stay valid)."""
    forms = {str(root), os.path.abspath(str(root)), os.path.realpath(str(root))}
    _ROOTS[:] = sorted((f.rstrip(os.sep) for f in forms if f), key=len, reverse=True)


def _rel(obj: Any) -> Any:
    if isinstance(obj, (str, Path)):
        s = str(obj)
        for r in _ROOTS:
            if s == r or s.startswith(r + os.sep):
                return "<root>" + s[len(r):]
        return obj if isinstance(obj, str) else s
    if isinstance(obj, Mapping):
        return {str(k): _rel(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_rel(v) for v in (sorted(obj, key=str) if isinstance(obj, set) else obj)]
    return obj


def fingerprint(*parts: Any) -> str:
    blob = json.dumps(_rel(list(parts)), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def file_stamp(path: PathLike) -> Optional[str]:
    """Cheap identity of an upstream file (size + mtime); None if missing."""
    try:
        st = Path(path).stat()
        return f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return None


# ----------------------------------------------------------------------------- markers
def _safe(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_.=+" else "_" for ch in str(key))


def marker_path(unit_dir: PathLike, key: str = "unit", prefix: str = DONE_PREFIX) -> Path:
    return Path(unit_dir, f"{prefix}{_safe(key)}.json")


def read_marker(unit_dir: PathLike, key: str = "unit", prefix: str = DONE_PREFIX) -> Optional[Dict[str, Any]]:
    p = marker_path(unit_dir, key, prefix)
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def stamp(unit_dir: PathLike, key: str = "unit") -> Optional[str]:
    """Completion identity of a unit (used in downstream fingerprints); None if incomplete."""
    m = read_marker(unit_dir, key)
    return None if m is None else f"{m.get('fingerprint')}@{m.get('completed_at')}"


def mark_done(unit_dir: PathLike, key: str = "unit", fp: Optional[str] = None, **info: Any) -> Path:
    marker_path(unit_dir, key, FAILED_PREFIX).unlink(missing_ok=True)
    marker_path(unit_dir, key, RUNNING_PREFIX).unlink(missing_ok=True)
    rec = {"key": key, "fingerprint": fp, "completed_at": _dt.datetime.now().isoformat(timespec="microseconds"),
           "host": socket.gethostname(), "pid": os.getpid(), **info}
    return write_json(marker_path(unit_dir, key), rec)


def mark_failed(unit_dir: PathLike, key: str = "unit", error: str = "", **info: Any) -> Path:
    rec = {"key": key, "failed_at": _dt.datetime.now().isoformat(timespec="seconds"),
           "host": socket.gethostname(), "error": str(error)[-4000:], **info}
    return write_json(marker_path(unit_dir, key, FAILED_PREFIX), rec)


def mark_running(unit_dir: PathLike, key: str = "unit", fp: Optional[str] = None) -> Path:
    return write_json(marker_path(unit_dir, key, RUNNING_PREFIX),
                      {"key": key, "fingerprint": fp, "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
                       "host": socket.gethostname(), "pid": os.getpid()})


def clear(unit_dir: PathLike, key: str = "unit") -> None:
    for pre in (DONE_PREFIX, FAILED_PREFIX, RUNNING_PREFIX):
        marker_path(unit_dir, key, pre).unlink(missing_ok=True)


# ----------------------------------------------------------------------------- policy
class RunPolicy:
    """Decides whether a unit must (re)run; built once per invocation from the experiment config.

    ``should_run`` is the single entry point for a unit. A unit finished during the
    current invocation (``done``) is never redone in the same invocation, even under
    ``overwrite``, so shared units (e.g. a pretrained embedding reused by several
    stages) are forced once, not once per caller.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        cfg = dict(cfg or {})
        ow = cfg.get("overwrite", False)
        self.all = ow is True
        self.items = set() if isinstance(ow, bool) or ow is None else {str(x) for x in (
            [ow] if isinstance(ow, str) else ow)}
        for k, stages in _LEGACY.items():
            if cfg.get(k):
                self.items.update(stages)
        if (cfg.get("finetune", {}) or {}).get("overwrite"):
            self.items.update({"finetune", "ft_wise"})
        rs = cfg.get("resume", {})
        rs = {} if rs is True or rs is None else ({"enabled": False} if rs is False else dict(rs))
        self.enabled = bool(rs.get("enabled", True))
        self.check_config = bool(rs.get("check_config", True))
        self.validate_files = bool(rs.get("validate_files", True))
        self._session: set = set()

    # -- overwrite requests
    def overwrite(self, stage: str, sub: Optional[str] = None) -> bool:
        """True if the user asked to recompute ``stage`` (or ``stage:sub`` / bare ``sub``)."""
        if self.all or not self.enabled:
            return True
        return stage in self.items or (sub is not None and (f"{stage}:{sub}" in self.items or sub in self.items))

    # -- unit decisions
    def _id(self, unit_dir: PathLike, key: str) -> str:
        return str(marker_path(unit_dir, key).resolve())

    def done_now(self, unit_dir: PathLike, key: str = "unit") -> bool:
        return self._id(unit_dir, key) in self._session

    def is_done(self, unit_dir: PathLike, key: str = "unit", fp: Optional[str] = None) -> bool:
        """Marker present and (when check_config) fingerprint unchanged; ignores overwrite."""
        if self.done_now(unit_dir, key):
            return True
        m = read_marker(unit_dir, key)
        if m is None:
            return False
        return not (self.check_config and fp is not None and m.get("fingerprint") not in (None, fp))

    def should_run(self, unit_dir: PathLike, key: str = "unit", fp: Optional[str] = None,
                   stage: Optional[str] = None, sub: Optional[str] = None) -> bool:
        if self.done_now(unit_dir, key):
            return False
        if stage is not None and self.overwrite(stage, sub):
            return True
        return not self.is_done(unit_dir, key, fp)

    def is_stale(self, unit_dir: PathLike, key: str = "unit", fp: Optional[str] = None,
                 stage: Optional[str] = None, sub: Optional[str] = None) -> bool:
        """Existing partial/complete outputs must be discarded (overwrite or config change)."""
        if self.done_now(unit_dir, key):
            return False
        if stage is not None and self.overwrite(stage, sub):
            return True
        if not self.check_config or fp is None:
            return False
        for pre in (DONE_PREFIX, RUNNING_PREFIX):
            m = read_marker(unit_dir, key, pre)
            if m is not None and m.get("fingerprint") not in (None, fp):
                return True
        return False

    def begin(self, unit_dir: PathLike, key: str = "unit", fp: Optional[str] = None) -> None:
        marker_path(unit_dir, key).unlink(missing_ok=True)
        mark_running(unit_dir, key, fp)

    def done(self, unit_dir: PathLike, key: str = "unit", fp: Optional[str] = None, **info: Any) -> None:
        mark_done(unit_dir, key, fp, **info)
        self._session.add(self._id(unit_dir, key))

    def failed(self, unit_dir: PathLike, key: str = "unit", error: str = "", **info: Any) -> None:
        mark_failed(unit_dir, key, error, **info)

    def to_dict(self) -> Dict[str, Any]:
        return {"overwrite": True if self.all else sorted(self.items),
                "resume": {"enabled": self.enabled, "check_config": self.check_config,
                           "validate_files": self.validate_files}}


def policy_from(cfg: Optional[Mapping[str, Any]]) -> RunPolicy:
    pol = (cfg or {}).get("_run_policy")
    return pol if isinstance(pol, RunPolicy) else RunPolicy(cfg)


# ----------------------------------------------------------------------------- status CLI
def scan(root: PathLike, prefixes: Sequence[str] = (DONE_PREFIX, FAILED_PREFIX, RUNNING_PREFIX)) -> Iterable[Dict[str, Any]]:
    for p in sorted(Path(root).rglob("_*.json")):
        for pre in prefixes:
            if p.name.startswith(pre):
                try:
                    rec = json.loads(p.read_text())
                except Exception:
                    rec = {}
                yield {"status": pre.strip("_.").lower(), "dir": str(p.parent), **rec}


def main() -> None:
    ap = argparse.ArgumentParser(description="List completion / failure markers under a run root.")
    ap.add_argument("root")
    ap.add_argument("--failed", action="store_true", help="only failed and interrupted (running) units")
    a = ap.parse_args()
    pre = (FAILED_PREFIX, RUNNING_PREFIX) if a.failed else (DONE_PREFIX, FAILED_PREFIX, RUNNING_PREFIX)
    n = {}
    for r in scan(a.root, pre):
        n[r["status"]] = n.get(r["status"], 0) + 1
        when = r.get("completed_at") or r.get("failed_at") or r.get("started_at") or ""
        err = (r.get("error") or "").strip().splitlines()[-1:] or [""]
        print(f"{r['status']:8s} {when:26s} {r['dir']}  [{r.get('key')}] {err[0][:120]}")
    print("summary:", n or "no markers")


if __name__ == "__main__":
    main()
