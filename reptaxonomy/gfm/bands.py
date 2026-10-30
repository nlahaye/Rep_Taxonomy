"""Sensor-aware band harmonisation for gfmtools datasets and encoders.

Every dataset band and every encoder input band is mapped to a *canonical
spectral name* (``blue``, ``nir``, ``swir1`` ...). Encoder inputs are then
assembled by canonical name, so a Sentinel-2 dataset can feed a Landsat- or
HLS-pretrained encoder (and vice versa) without hand-written band lists.

Bands the dataset cannot provide are filled (see ``assemble_bands``) and the
fraction of encoder bands that were actually observed is reported as
``band_coverage``; the experiment driver can skip model/dataset pairs whose
coverage falls below ``min_band_coverage``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Centre wavelengths (micrometres) of each canonical band; used by DOFA.
CANONICAL_WAVELENGTH_UM: Dict[str, float] = {
    "coastal": 0.443,
    "blue": 0.490,
    "green": 0.560,
    "red": 0.665,
    "re1": 0.705,
    "re2": 0.740,
    "re3": 0.783,
    "nir": 0.842,
    "nir_narrow": 0.865,
    "water_vapour": 0.945,
    "cirrus": 1.375,
    "swir1": 1.610,
    "swir2": 2.190,
    "pan": 0.590,
    "tir1": 10.90,
    "tir2": 12.00,
    "tir": 11.45,
    "vv": -1.0,     # SAR polarisations have no optical wavelength
    "vh": -2.0,
    "ndvi": -3.0,   # derived index
}

# If a canonical band is missing, these substitutes are tried in order
# (spectrally adjacent bands; e.g. HLS/Landsat B5 ~= Sentinel-2 B8A ~= B8).
CANONICAL_SUBSTITUTES: Dict[str, Tuple[str, ...]] = {
    "nir_narrow": ("nir",),
    "nir": ("nir_narrow",),
    "tir": ("tir1", "tir2"),
    "tir1": ("tir",),
    "tir2": ("tir",),
}

_S2 = {
    "B1": "coastal", "B2": "blue", "B3": "green", "B4": "red", "B5": "re1",
    "B6": "re2", "B7": "re3", "B8": "nir", "B8A": "nir_narrow", "B9": "water_vapour",
    "B10": "cirrus", "B11": "swir1", "B12": "swir2", "NDVI": "ndvi",
}
_OLI = {  # Landsat 8/9 OLI(-TIRS) and HLS (HLS uses Landsat numbering)
    "B1": "coastal", "B2": "blue", "B3": "green", "B4": "red", "B5": "nir_narrow",
    "B6": "swir1", "B7": "swir2", "B8": "pan", "B9": "cirrus", "B10": "tir1", "B11": "tir2",
}
_TM = {  # Landsat 4/5 TM and Landsat 7 ETM+
    "B1": "blue", "B2": "green", "B3": "red", "B4": "nir", "B5": "swir1",
    "B6": "tir", "B6_VCID_1": "tir1", "B6_VCID_2": "tir2", "B7": "swir2", "B8": "pan",
}
_RGBN = {"R": "red", "G": "green", "B": "blue", "N": "nir", "NIR": "nir"}
_SAR = {"VV": "vv", "VH": "vh"}

# instrument / sensor key -> table
SENSOR_TABLES: Dict[str, Dict[str, str]] = {
    "s2": _S2, "oli": _OLI, "hls": _OLI, "tm": _TM, "etm": _TM, "rgbn": _RGBN, "sar": _SAR,
}

_INSTRUMENT_TO_SENSOR = {
    "sentinel-2": "s2", "sentinel2": "s2", "s2": "s2", "s2l2a": "s2", "s2l1c": "s2",
    "l8": "oli", "l9": "oli", "landsat8": "oli", "landsat9": "oli", "oli": "oli",
    "hls": "hls", "l5": "tm", "l4": "tm", "tm": "tm", "l7": "etm", "etm": "etm",
    "naip": "rgbn", "oam": "rgbn", "rgb": "rgbn", "s1grd": "sar", "sentinel-1": "sar", "s1": "sar",
}


def sensor_of(instrument: Optional[str], default: str = "s2") -> str:
    if instrument is None:
        return default
    return _INSTRUMENT_TO_SENSOR.get(str(instrument).lower().replace("_", ""), default)


def normalise_band_name(band: str) -> str:
    """'B01' -> 'B1', 'B8a' -> 'B8A', 'SR_B4' -> 'B4'. Non-numbered names are upper-cased."""
    b = str(band).strip().upper()
    if b.startswith("SR_"):
        b = b[3:]
    if b.startswith("B") and len(b) > 1:
        head, digits, tail = "B", "", ""
        for i, ch in enumerate(b[1:]):
            if ch.isdigit():
                digits += ch
            else:
                tail = b[1 + i:]
                break
        if digits:
            return f"{head}{int(digits)}{tail}"
    return b


def canonical_name(band: str, sensor: str) -> str:
    table = SENSOR_TABLES.get(sensor, _S2)
    nb = normalise_band_name(band)
    if nb in table:
        return table[nb]
    for other in ("s2", "rgbn", "sar"):
        if nb in SENSOR_TABLES[other]:
            return SENSOR_TABLES[other][nb]
    return nb.lower()


def canonical_bands(bands: Sequence[str], sensor: str) -> List[str]:
    return [canonical_name(b, sensor) for b in bands]


def wavelengths_um(canon: Sequence[str]) -> List[float]:
    out = []
    for c in canon:
        if c not in CANONICAL_WAVELENGTH_UM or CANONICAL_WAVELENGTH_UM[c] <= 0:
            raise ValueError(f"Canonical band '{c}' has no optical wavelength (DOFA needs one).")
        out.append(CANONICAL_WAVELENGTH_UM[c])
    return out


@dataclass
class BandPlan:
    """How encoder input channels are assembled from dataset channels."""

    target_canonical: List[str]
    source_index: List[Optional[int]]  # index into dataset channels, None = filled
    derived: Dict[int, str]            # target channel -> derived index name (e.g. ndvi)

    @property
    def coverage(self) -> float:
        n = len(self.target_canonical)
        hit = sum(1 for i, s in enumerate(self.source_index) if s is not None or i in self.derived)
        return hit / n if n else 0.0

    @property
    def missing(self) -> List[str]:
        return [c for c, s, i in zip(self.target_canonical, self.source_index, range(len(self.source_index)))
                if s is None and i not in self.derived]


def plan_bands(source_canonical: Sequence[str], target_canonical: Sequence[str],
               allow_substitutes: bool = True) -> BandPlan:
    src = list(source_canonical)
    idx: List[Optional[int]] = []
    derived: Dict[int, str] = {}
    for t_i, t in enumerate(target_canonical):
        if t in src:
            idx.append(src.index(t))
            continue
        found = None
        if allow_substitutes:
            for alt in CANONICAL_SUBSTITUTES.get(t, ()):
                if alt in src:
                    found = src.index(alt)
                    break
        if found is None and t == "ndvi" and "red" in src and ("nir" in src or "nir_narrow" in src):
            derived[t_i] = "ndvi"
        idx.append(found)
    return BandPlan(list(target_canonical), idx, derived)


def assemble_bands(x: torch.Tensor, source_canonical: Sequence[str], plan: BandPlan,
                   fill: str = "zero") -> torch.Tensor:
    """x: (B, C_src, H, W) already standardised -> (B, C_tgt, H, W).

    fill='zero'  : missing channels set to 0 (= dataset mean after z-scoring).
    fill='mean'  : missing channels set to the mean of the observed channels.
    """
    B, _, H, W = x.shape
    out = x.new_zeros((B, len(plan.target_canonical), H, W))
    src = list(source_canonical)
    for t_i, s_i in enumerate(plan.source_index):
        if s_i is not None:
            out[:, t_i] = x[:, s_i]
    for t_i, name in plan.derived.items():
        if name == "ndvi":
            r = x[:, src.index("red")]
            n = x[:, src.index("nir")] if "nir" in src else x[:, src.index("nir_narrow")]
            out[:, t_i] = (n - r) / (n.abs() + r.abs() + 1e-6)
    if fill == "mean":
        miss = [i for i, s in enumerate(plan.source_index) if s is None and i not in plan.derived]
        if miss:
            out[:, miss] = x.mean(dim=1, keepdim=True)
    return out
