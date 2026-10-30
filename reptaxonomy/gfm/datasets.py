"""rep_taxonomy "representation sets" built from gfmtools datasets.

Each wrapper turns one gfmtools dataset (or one split / country / window /
ecoregion of it) into a dataset that

* emits the gfmtools sample-dict layout rep_taxonomy already consumes
  (``instruments`` -> instrument -> band -> ``{"data": (H, W)}``),
* exposes a single instrument (so ``build_dataset_spec`` infers one image size),
* remaps labels to contiguous ``0..K-1`` with ``ignore_index = 255`` under the
  uniform target key ``"label"``,
* adds ``rt_meta`` (time string, lat/lon, ``sample_key``, ``pair_key``) used by
  time/location-aware encoders and by paired cross-split comparisons,
* exposes ``img_files`` (stable, filename-safe ids) so embeddings land in
  ``embed_dir/<dataset_name>/<model>/test/embd_<stem>.npy``.

Wrappers are constructible from plain kwargs, which is what
``experiment_init_utils.build_test_loader`` requires (it re-instantiates
``DATASET_REGISTRY[dataset_cls_name](**dataset_kwargs)``).
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IGNORE_INDEX = 255
LABEL_KEY = "label"

# NLCD 2019 CONUS legend (codes present in the EPA dataset), level-2 order.
NLCD_CODES = [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
NLCD_NAMES = {
    11: "open_water", 12: "perennial_ice_snow", 21: "developed_open", 22: "developed_low",
    23: "developed_medium", 24: "developed_high", 31: "barren", 41: "deciduous_forest",
    42: "evergreen_forest", 43: "mixed_forest", 52: "shrub_scrub", 71: "grassland_herbaceous",
    81: "pasture_hay", 82: "cultivated_crops", 90: "woody_wetlands", 95: "emergent_herbaceous_wetlands",
}
NLCD_L1 = {1: "water", 2: "developed", 3: "barren", 4: "forest", 5: "shrubland", 7: "herbaceous",
           8: "planted_cultivated", 9: "wetlands"}

FTW_CLASS_NAMES = {0: "background", 1: "field", 2: "boundary"}
# EarthShift FTW statistics (reflectance units, reflectance_scale = 1e-4).
FTW_EARTHSHIFT_MEAN = {"B4": 0.1552, "B3": 0.1355, "B2": 0.1105, "B8": 0.2743}
FTW_EARTHSHIFT_STD = {"B4": 0.1888, "B3": 0.1757, "B2": 0.1809, "B8": 0.1742}


def slug(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(s)).strip("-")


def stable_subsample(keys: Sequence[str], n: Optional[int], seed: int = 0) -> List[int]:
    """Deterministic subset that depends only on the *keys* (not their order).

    Two conditions sharing pair keys (e.g. FTW window A / window B of the same
    AOIs) therefore select the same chips, which keeps paired comparisons valid.
    """
    idx = list(range(len(keys)))
    if n is None or n >= len(keys):
        return idx
    h = [hashlib.sha1(f"{seed}:{k}".encode()).hexdigest() for k in keys]
    return sorted(sorted(idx, key=lambda i: h[i])[: int(n)])


def _latlon_from_grid(grid: Optional[Mapping[str, Any]]) -> tuple[float, float]:
    try:
        from pyproj import Transformer
        t = grid["transform"]
        w, h = grid["width"], grid["height"]
        x = t[2] + t[0] * w / 2 + t[1] * h / 2
        y = t[5] + t[3] * w / 2 + t[4] * h / 2
        lon, lat = Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True).transform(x, y)
        return float(lat), float(lon)
    except Exception:
        return float("nan"), float("nan")


def _time_string(entry: Mapping[str, Any], index: int = -1) -> Optional[str]:
    tw = entry.get("time_window")
    if isinstance(tw, Mapping) and tw.get("start") is not None:
        return f"{tw['start']}/{tw.get('end', tw['start'])}"
    times = entry.get("time") or []
    if times:
        t = times[index] if -len(times) <= index < len(times) else times[-1]
        return None if t is None else str(t)
    return None


class _RepSet(Dataset):
    """Shared plumbing for the gfmtools wrappers."""

    instrument: str = "Sentinel-2"
    sensor: str = "s2"
    gsd: float = 10.0
    raw_scale: float = 1.0  # multiply pixel values by this to get S2 L2A DN (OlmoEarth)
    class_names: Dict[int, str] = {}
    num_classes: int = 0
    ignore_index: int = IGNORE_INDEX

    def __init__(self) -> None:
        self.indices: List[int] = []
        self.sample_keys: List[str] = []
        self.pair_keys: List[str] = []
        self.band_names: List[str] = []

    # subclasses implement
    def _raw(self, base_index: int) -> Dict[str, Any]:
        raise NotImplementedError

    def _convert(self, s: Mapping[str, Any], i: int) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def img_files(self) -> List[str]:
        return [f"{k}.tif" for k in self.sample_keys]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self._convert(self._raw(self.indices[i]), i)

    def _pack(self, bands: Dict[str, torch.Tensor], label: torch.Tensor, i: int,
              time: Optional[str], grid: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        lat, lon = _latlon_from_grid(grid)
        inst = {"bands": list(bands)}
        for b, d in bands.items():
            inst[b] = {"data": d.float()}
        return {
            "instruments": [self.instrument],
            "targets": [LABEL_KEY],
            self.instrument: inst,
            LABEL_KEY: {"bands": [LABEL_KEY], LABEL_KEY: {"data": label.long()}},
            "rt_meta": {"time": time, "lat": lat, "lon": lon,
                        "sample_key": self.sample_keys[i], "pair_key": self.pair_keys[i]},
        }

    def describe(self) -> Dict[str, Any]:
        return {"cls": type(self).__name__, "n": len(self), "instrument": self.instrument,
                "bands": self.band_names, "num_classes": self.num_classes,
                "class_names": self.class_names, "ignore_index": self.ignore_index}


# ----------------------------------------------------------------------------- FTW
def register_ftw_task(spec: Mapping[str, Any]) -> str:
    """Build a custom ``ftw_shifts`` task and register it by name.

    ``spec = {kind: geographic, id_countries: [...], ood_countries: [...], windows: [a, b], name: ...}``
    or ``{kind: seasonal, countries: [...] | all, id_window: a, ood_window: b, name: ...}``.
    ``build_shift_datasets`` resolves names through ``EARTHSHIFT_FTW_TASKS``.
    """
    from gfmtools.data.datasets import ftw_shifts as S
    kw = dict(spec)
    kind = kw.pop("kind", "geographic")
    for k in ("id_countries", "ood_countries", "windows"):
        if isinstance(kw.get(k), list):
            kw[k] = tuple(kw[k])
    if isinstance(kw.get("countries"), list):
        kw["countries"] = tuple(kw["countries"])
    task = S.geographic_shift(**kw) if kind == "geographic" else S.seasonal_shift(**kw)
    S.EARTHSHIFT_FTW_TASKS[task.name] = task
    return task.name


@functools.lru_cache(maxsize=16)
def _ftw_shift(task: str, data_root: str, reflectance_scale: float, group_holdout_by_aoi: Optional[bool]):
    from gfmtools.data.datasets.ftw_shifts import build_shift_datasets
    return build_shift_datasets(task, data_root=data_root, reflectance_scale=reflectance_scale,
                                group_holdout_by_aoi=group_holdout_by_aoi)


class FTWRepSet(_RepSet):
    """One FTW representation set.

    * ``shift_task`` given: ``split`` in {train, val, id_test, ood_test} of
      ``gfmtools.data.datasets.ftw_shifts.build_shift_datasets(shift_task)``.
    * otherwise: ``FieldsOfTheWorld(split=split, countries=..., windows=...)``.

    ``country_filter`` / ``window_filter`` further restrict the samples (e.g. the
    per-country breakdown of ``ftw-all-window`` OOD test).
    """

    instrument = "Sentinel-2"
    sensor = "s2"
    gsd = 10.0
    class_names = dict(FTW_CLASS_NAMES)
    num_classes = 3

    def __init__(self, split: str = "test", data_root: str = "/mnt/data/ftw",
                 shift_task: Optional[str] = None, countries: Any = "all",
                 windows: Sequence[str] = ("a", "b"), split_strategy: str = "official",
                 reflectance_scale: float = 1.0, group_holdout_by_aoi: Optional[bool] = None,
                 country_filter: Optional[Sequence[str]] = None,
                 window_filter: Optional[Sequence[str]] = None,
                 max_samples: Optional[int] = None, sample_seed: int = 0,
                 include_presence_only: bool = True, shift_spec: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        from gfmtools.data.datasets.ftw import FieldsOfTheWorld
        if shift_spec:  # custom geographic / seasonal task (re-registered in every worker)
            shift_task = register_ftw_task(shift_spec)
        self.reflectance_scale = float(reflectance_scale)
        self.raw_scale = 1.0 / self.reflectance_scale if self.reflectance_scale else 1.0
        if shift_task:
            sd = _ftw_shift(shift_task, str(data_root), self.reflectance_scale, group_holdout_by_aoi)
            self.base = sd.as_dict()[split]
        else:
            self.base = FieldsOfTheWorld(data_root=data_root, split=split, countries=countries,
                                         windows=tuple(windows) if not isinstance(windows, str) else windows,
                                         split_strategy=split_strategy, reflectance_scale=self.reflectance_scale)
        samples = list(self.base.samples)
        keep = list(range(len(samples)))
        if country_filter:
            cf = {c.lower() for c in country_filter}
            keep = [i for i in keep if samples[i].country.lower() in cf]
        if window_filter:
            wf = set(window_filter)
            keep = [i for i in keep if samples[i].window in wf]
        if not include_presence_only:
            from gfmtools.data.datasets.ftw import PRESENCE_ONLY_COUNTRIES  # type: ignore
            keep = [i for i in keep if samples[i].country not in PRESENCE_ONLY_COUNTRIES]
        pair = [f"{samples[i].country}__{samples[i].aoi_id}" for i in keep]
        sub = stable_subsample(pair, max_samples, sample_seed)
        self.indices = [keep[j] for j in sub]
        self.pair_keys = [pair[j] for j in sub]
        self.sample_keys = [slug(f"{samples[i].country}__{samples[i].aoi_id}__{samples[i].window}")
                            for i in self.indices]
        self.band_names = ["B4", "B3", "B2", "B8"]
        self.meta_rows = [{"country": samples[i].country, "aoi_id": samples[i].aoi_id,
                           "window": samples[i].window} for i in self.indices]

    def _raw(self, base_index: int) -> Dict[str, Any]:
        return self.base[base_index]

    def _convert(self, s: Mapping[str, Any], i: int) -> Dict[str, Any]:
        inst = s[s["instruments"][0]]
        tname = s["targets"][0]
        lab = torch.as_tensor(s[tname][tname]["data"]).long()
        ignore = s[tname].get("ignore_value", 3)
        out = lab.clone()
        out[(lab == ignore) | (lab < 0) | (lab >= self.num_classes)] = IGNORE_INDEX
        bands = {b: torch.as_tensor(inst[b]["data"]) for b in self.band_names}
        return self._pack(bands, out, i, _time_string(inst), inst.get("grid"))

    def default_band_stats(self) -> Dict[str, List[float]]:
        f = 1.0 if self.reflectance_scale != 1.0 else 1e4  # DN when reflectance_scale == 1
        return {b: [FTW_EARTHSHIFT_MEAN[b] * f, FTW_EARTHSHIFT_STD[b] * f] for b in self.band_names}


# ----------------------------------------------------------------------------- EPA
class EPARepSet(_RepSet):
    """EPA / NLCD representation set, optionally restricted to one EPA L3 ecoregion.

    ``region`` (one ``US_L3NAME``) -> ``EPALabels(split=None, regions=[region])``.
    ``split`` + ``split_file`` -> the region list of that split.
    ``instrument``: ``S2L2A`` (10 m, 384 px; labels nearest-upsampled) or ``L8``
    (30 m, 128 px, label-native).
    """

    def __init__(self, split: Optional[str] = None, region: Optional[str] = None,
                 regions: Optional[Sequence[str]] = None, data_root: str = "/mnt/data/mtruong/",
                 split_file: Optional[str] = None, instrument: str = "S2L2A",
                 included_timepoints: Any = (2, 3), timepoint: int = -1, img_size: int = 128,
                 gsd: int = 30, sample_limit: int = 5000, seed: int = 1917,
                 nlcd_level: int = 2, max_samples: Optional[int] = None, sample_seed: int = 0) -> None:
        super().__init__()
        from gfmtools.data.datasets.epa import EPALabels
        kw: Dict[str, Any] = dict(data_root=data_root, img_size=img_size, gsd=gsd,
                                  included_timepoints=tuple(included_timepoints)
                                  if isinstance(included_timepoints, (list, tuple)) else included_timepoints,
                                  seed=seed, sample_limit=sample_limit)
        if split_file:
            kw["split_file"] = split_file
        if region is not None or regions:
            kw.update(split=None, regions=[region] if region is not None else list(regions))
        else:
            kw.update(split=split)
        self.base = EPALabels(**kw)
        self.region = region
        self.instrument = instrument
        self.sensor = "s2" if instrument.upper().startswith("S2") else "oli"
        self.gsd = 10.0 if self.sensor == "s2" else float(gsd)
        self.timepoint = int(timepoint)
        self.nlcd_level = int(nlcd_level)
        if self.nlcd_level == 1:
            l1 = sorted({c // 10 for c in NLCD_CODES})
            self._lut = {c: l1.index(c // 10) for c in NLCD_CODES}
            self.class_names = {i: NLCD_L1[k] for i, k in enumerate(l1)}
        else:
            self._lut = {c: i for i, c in enumerate(NLCD_CODES)}
            self.class_names = {i: NLCD_NAMES[c] for i, c in enumerate(NLCD_CODES)}
        self.num_classes = len(self.class_names)
        lut = np.full(256, IGNORE_INDEX, dtype=np.int64)
        for c, k in self._lut.items():
            lut[c] = k
        self._lut_t = torch.as_tensor(lut)
        n = len(self.base)
        tag = slug(region) if region else slug(split or "all")
        keys = [f"{tag}__{i:06d}" for i in range(n)]
        self.indices = stable_subsample(keys, max_samples, sample_seed)
        self.sample_keys = [keys[i] for i in self.indices]
        self.pair_keys = list(self.sample_keys)
        self.band_names = (["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
                           if self.sensor == "s2" else ["B2", "B3", "B4", "B5", "B6", "B7"])
        self.raw_scale = 1.0

    def _raw(self, base_index: int) -> Dict[str, Any]:
        return self.base[base_index]

    def _select_time(self, d: torch.Tensor) -> torch.Tensor:
        d = torch.as_tensor(d)
        if d.ndim == 3:
            d = d[self.timepoint]
        if d.ndim != 2:
            raise ValueError(f"EPA band array has unexpected shape {tuple(d.shape)} "
                             "(see GFMTOOLS_NOTES.md: EPALabels band indexing with several timepoints)")
        return d

    def _convert(self, s: Mapping[str, Any], i: int) -> Dict[str, Any]:
        inst = s[self.instrument]
        avail = [b for b in self.band_names if b in inst]
        bands = {b: self._select_time(inst[b]["data"]) for b in avail}
        tname = s["targets"][0]
        lab = torch.as_tensor(s[tname][s[tname]["bands"][0]]["data"]).long().clamp(0, 255)
        lab = self._lut_t[lab]
        H, W = next(iter(bands.values())).shape
        if lab.shape != (H, W):
            lab = F.interpolate(lab[None, None].float(), size=(H, W), mode="nearest")[0, 0].long()
        return self._pack(bands, lab, i, _time_string(inst, self.timepoint), inst.get("grid"))


def list_epa_regions(data_root: str, split: Optional[str] = None, split_file: Optional[str] = None) -> List[str]:
    """US_L3NAME ecoregions available in the EPA dataset (or in one split of split_file)."""
    if split is not None and split_file:
        import yaml
        with open(split_file) as fh:
            return list(yaml.safe_load(fh)[split])
    import geopandas as gpd
    path = Path(data_root) / "epa_shapes" / "all_epa3_subsets.geojson"
    if path.exists():
        return sorted(set(gpd.read_file(path)["US_L3NAME"]))
    raise RuntimeError("Provide epa.regions explicitly (could not enumerate ecoregions)")


# ----------------------------------------------------------------------------- MADOS
class MADOSRepSet(_RepSet):
    instrument = "Sentinel-2"
    sensor = "s2"
    gsd = 10.0
    raw_scale = 1e4  # MADOS stores surface reflectance (0-1)

    def __init__(self, split: str = "test", data_root: str = "/mnt/data/rdemilt/mados/MADOS",
                 max_samples: Optional[int] = None, sample_seed: int = 0) -> None:
        super().__init__()
        from gfmtools.data.datasets.mados import MADOS
        self.base = MADOS(data_root=data_root, split=split)
        names = dict(getattr(self.base, "CLASS_NAMES", {}))
        self.class_names = {k - 1: v for k, v in sorted(names.items()) if k >= 1} or {i: str(i + 1) for i in range(15)}
        self.num_classes = len(self.class_names)
        crops = list(getattr(self.base, "crops", range(len(self.base))))
        keys = [slug(f"{split}__{getattr(c, 'patch_id', i)}") for i, c in enumerate(crops)]
        self.indices = stable_subsample(keys, max_samples, sample_seed)
        self.sample_keys = [keys[i] for i in self.indices]
        self.pair_keys = list(self.sample_keys)
        self.band_names = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    def _raw(self, base_index: int) -> Dict[str, Any]:
        return self.base[base_index]

    def _convert(self, s: Mapping[str, Any], i: int) -> Dict[str, Any]:
        inst = s["Sentinel-2"]
        bands = {b: torch.as_tensor(inst[b]["data"]) for b in self.band_names}
        lab = torch.as_tensor(s["mados_class"]["mados_class"]["data"]).long()
        out = lab - 1
        out[(lab <= 0) | (lab > self.num_classes)] = IGNORE_INDEX
        return self._pack(bands, out, i, _time_string(inst), inst.get("grid"))


REPSET_CLASSES = {"FTWRepSet": FTWRepSet, "EPARepSet": EPARepSet, "MADOSRepSet": MADOSRepSet}


# ----------------------------------------------------------------------------- band stats
def compute_band_stats(ds: _RepSet, n: int = 256, seed: int = 0) -> Dict[str, List[float]]:
    """Per-band mean/std over (up to) n samples; ignores NaNs."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False) if len(ds) else []
    s1: Dict[str, float] = {}
    s2: Dict[str, float] = {}
    cnt: Dict[str, float] = {}
    for i in idx:
        s = ds[int(i)]
        inst = s[s["instruments"][0]]
        for b in ds.band_names:
            if b not in inst:
                continue
            v = torch.as_tensor(inst[b]["data"]).double()
            v = v[torch.isfinite(v)]
            s1[b] = s1.get(b, 0.0) + float(v.sum())
            s2[b] = s2.get(b, 0.0) + float((v * v).sum())
            cnt[b] = cnt.get(b, 0.0) + float(v.numel())
    out = {}
    for b in s1:
        m = s1[b] / max(cnt[b], 1)
        var = max(s2[b] / max(cnt[b], 1) - m * m, 1e-12)
        out[b] = [m, math.sqrt(var)]
    return out


def load_or_compute_band_stats(ds: _RepSet, cache_path: str | Path, n: int = 256, seed: int = 0,
                               prefer_default: bool = True) -> Dict[str, List[float]]:
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    if prefer_default and hasattr(ds, "default_band_stats"):
        stats = ds.default_band_stats()
    else:
        stats = compute_band_stats(ds, n=n, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(stats, indent=2))
    return stats
