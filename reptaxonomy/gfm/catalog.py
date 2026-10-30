"""Catalog of every gfmtools encoder / pretrained-weight combination.

The catalog is generated from the gfmtools weight enums at import time, so new
weights added to gfmtools show up automatically. Each entry is a plain dict
(JSON-serialisable, safe to store in ModelBundle kwargs):

    key          unique model name used throughout rep_taxonomy outputs
    family       resnet | swin | dofa | croma | scalemae | unet | prithvi | olmoearth
    arch         builder / architecture id inside the family
    module       gfmtools.models.<module>
    enum         weights enum class name (None for OlmoEarth / scratch)
    weights      weights enum member name (None => random init)
    sensor       canonical sensor table used for the encoder's input bands
    bands        encoder input band names (sensor-native naming)
    input_size   spatial size fed to the encoder (pixels)
    experimental True when the gfmtools wrapper/weights are known to be fragile

Model names
-----------
* ``<arch>__<weights_lower>``        e.g. ``resnet50__sentinel2_all_moco``
* ``<arch>__scratch``                random-init negative control (S2-native bands)
* ``<key>__ft__<tag>``                  fine-tuned variant; checkpoint supplied via
                                      ``per_model_overrides[<name>].checkpoint``

Suites
------
* ``core``      one representative per family (+ scratch controls)
* ``optical``   every entry whose bands are optical (drops Sentinel-1-only weights)
* ``full``      every entry in the catalog (incompatible pairs are skipped at
                run time by ``min_band_coverage``)
"""

from __future__ import annotations

import copy
import importlib
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

S2_12 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
S2_13 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
S2_RGB = ["B4", "B3", "B2"]
S2_SECO_ECO_12 = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
S2_NDVI_9 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "NDVI"]  # verify against weight card
S2_SATLAS_MS = ["B4", "B3", "B2", "B5", "B6", "B7", "B8", "B11", "B12"]
HLS6 = ["B2", "B3", "B4", "B5", "B6", "B7"]
OLMOEARTH_S2 = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12", "B1", "B9"]

# family -> (module, [(arch, builder, enum_cls_name, kind, default_input_size)])
_FAMILIES: Dict[str, Any] = {
    "resnet": ("resnet", [
        ("resnet18", "resnet18", "ResNet18_Weights", None),
        ("resnet50", "resnet50", "ResNet50_Weights", None),
        ("resnet152", "resnet152", "ResNet152_Weights", None),
    ]),
    "swin": ("swin", [
        ("swin_v2_t", "swin_v2_t", "Swin_V2_T_Weights", None),
        ("swin_v2_b", "swin_v2_b", "Swin_V2_B_Weights", None),
        ("swin_t", "swin_t", "Swin_T_Weights", None),
        ("swin_s", "swin_s", "Swin_S_Weights", None),
        ("swin_b", "swin_b", "Swin_B_Weights", None),
    ]),
    "dofa": ("dofa", [
        ("dofa_base", "dofa_base_patch16_224", "DOFABase16_Weights", 224),
        ("dofa_large", "dofa_large_patch16_224", "DOFALarge16_Weights", 224),
    ]),
    "croma": ("croma", [
        ("croma_base", "croma_base", "CROMABase_Weights", 120),
        ("croma_large", "croma_large", "CROMALarge_Weights", 120),
    ]),
    "scalemae": ("scale_mae", [
        ("scalemae_large", "scalemae_large_patch16", "ScaleMAELarge16_Weights", 224),
    ]),
    "unet": ("unet", [
        ("unet", "unet", "Unet_Weights", None),
    ]),
    "prithvi": ("prithvi", [
        ("prithvi_eo1", None, "PrithviEO1_Weights", 224),
        ("prithvi_eo2", None, "PrithviEO2_Weights", 224),
    ]),
}

_OLMOEARTH_ARTIFACTS = [
    "OLMOEARTH_V1_NANO", "OLMOEARTH_V1_BASE", "OLMOEARTH_V1_1_NANO",
    "OLMOEARTH_V1_1_BASE", "OLMOEARTH_V1_2_NANO", "OLMOEARTH_V1_2_BASE",
]

# Known-fragile weights (wrapper/checkpoint mismatches noted while integrating).
_EXPERIMENTAL = {
    ("swin_t", "CITYSCAPES_SEMSEG"), ("swin_s", "CITYSCAPES_SEMSEG"), ("swin_b", "CITYSCAPES_SEMSEG"),
    ("resnet152", "SENTINEL2_MI_MS_SATLAS"), ("resnet152", "SENTINEL2_MI_RGB_SATLAS"),
    ("resnet152", "SENTINEL2_SI_MS_SATLAS"), ("resnet152", "SENTINEL2_SI_RGB_SATLAS"),
    ("resnet50", "LANDSAT_ETM_SR_MOCO"), ("resnet50", "LANDSAT_ETM_SR_SIMCLR"),
    ("resnet50", "SENTINEL2_ALL_NDVI_SECO_ECO"),
    ("prithvi_eo2", "EO_2_0_600M"), ("prithvi_eo2", "EO_2_0_600M_TL"),
}


def _infer_sensor_and_bands(arch: str, wname: str, meta: Dict[str, Any]) -> tuple[str, List[str]]:
    n = wname.upper()
    bands = list(meta.get("bands") or [])
    inst = str(meta.get("instrument") or "").upper()
    if "SENTINEL1" in n or inst == "S1GRD" and "SENTINEL2" not in n:
        return "sar", bands or ["VV", "VH"]
    if "LANDSAT" in n:
        if "_TM_" in n:
            return "tm", bands or ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        if "_ETM_" in n:
            return "etm", bands
        return "oli", bands
    if "NAIP" in n or "OAM" in n or inst in ("NAIP", "OAM"):
        return "rgbn", bands or ["R", "G", "B"]
    if "FMOW" in n or "CITYSCAPES" in n:
        return "rgbn", ["R", "G", "B"]
    if arch.startswith("prithvi"):
        return "hls", bands or HLS6
    if "SENTINEL2" in n or inst.startswith("S2") or arch.startswith(("croma", "unet")):
        if bands:
            return "s2", bands
        in_chans = meta.get("in_chans")
        if "NDVI" in n:
            return "s2", S2_NDVI_9
        if "SECO_ECO" in n:
            return "s2", S2_SECO_ECO_12
        if "RGB" in n or in_chans == 3:
            return "s2", S2_RGB
        if "SATLAS" in n and in_chans == 9:
            return "s2", S2_SATLAS_MS
        if in_chans == 13:
            return "s2", S2_13
        if in_chans == 12:
            return "s2", S2_12
        return "s2", S2_13
    return "s2", bands or S2_12


def _build_catalog() -> Dict[str, Dict[str, Any]]:
    cat: Dict[str, Dict[str, Any]] = {}
    for family, (module, archs) in _FAMILIES.items():
        try:
            mod = importlib.import_module(f"gfmtools.models.{module}")
        except Exception as exc:  # optional deps missing (e.g. segmentation_models_pytorch)
            logger.warning("gfmtools.models.%s unavailable (%s); skipping family %s", module, exc, family)
            continue
        for arch, builder, enum_name, size in archs:
            enum_cls = getattr(mod, enum_name, None)
            if enum_cls is None:
                continue
            for w in enum_cls:
                meta = dict(w.meta)
                if family == "croma":
                    sensor, bands = "s2", list(S2_12)
                elif family == "dofa":
                    sensor, bands = "s2", list(S2_12)  # wavelength-agnostic; uses dataset bands
                elif family == "scalemae":
                    sensor, bands = "rgbn", ["R", "G", "B"]
                else:
                    sensor, bands = _infer_sensor_and_bands(arch, w.name, meta)
                key = f"{arch}__{w.name.lower()}"
                entry = {
                    "key": key, "family": family, "arch": arch, "module": module,
                    "builder": builder, "enum": enum_name, "weights": w.name,
                    "sensor": sensor, "bands": list(bands), "input_size": size,
                    "experimental": (arch, w.name) in _EXPERIMENTAL,
                }
                if family == "prithvi":
                    entry["prithvi_cfg"] = {
                        k: (list(v) if isinstance(v, tuple) else v) for k, v in meta.items()
                        if k in ("image_size", "patch_size", "num_frames", "embed_dim", "depth",
                                 "num_heads", "mlp_ratio", "coords_encoding", "coords_scale_learn",
                                 "decoder_embed_dim", "decoder_depth", "decoder_num_heads", "in_chans")
                    }
                    entry["input_size"] = int(meta.get("image_size", 224))
                if family == "unet":
                    entry["unet_encoder"] = meta.get("encoder")
                    entry["unet_classes"] = meta.get("num_classes")
                    entry["bitemporal"] = len(bands) == 8
                cat[key] = entry
            # random-init control per architecture (S2-native band set)
            scratch_bands = {"croma": S2_12, "dofa": S2_12, "scalemae": ["R", "G", "B"],
                             "prithvi": HLS6}.get(family, S2_12)
            scratch_sensor = {"scalemae": "rgbn", "prithvi": "hls"}.get(family, "s2")
            first = next(iter(enum_cls))
            entry = {
                "key": f"{arch}__scratch", "family": family, "arch": arch, "module": module,
                "builder": builder, "enum": enum_name, "weights": None, "template_weights": first.name,
                "sensor": scratch_sensor, "bands": list(scratch_bands), "input_size": size,
                "experimental": False,
            }
            if family == "prithvi":
                entry.update({k: v for k, v in cat[f"{arch}__{first.name.lower()}"].items()
                              if k in ("prithvi_cfg", "input_size")})
                entry["prithvi_cfg"] = dict(entry["prithvi_cfg"], coords_encoding=[])
            if family == "unet":
                entry["unet_encoder"] = first.meta.get("encoder")
                entry["unet_classes"] = first.meta.get("num_classes")
                entry["bitemporal"] = False
            cat[entry["key"]] = entry

    for art in _OLMOEARTH_ARTIFACTS:
        key = art.lower().replace("olmoearth_", "olmoearth_")
        cat[key] = {
            "key": key, "family": "olmoearth", "arch": "olmoearth", "module": "olmoearth",
            "builder": "OlmoEarthModel", "enum": None, "weights": art, "sensor": "s2",
            "bands": list(OLMOEARTH_S2), "input_size": None, "experimental": False,
        }
    return cat


_CATALOG: Optional[Dict[str, Dict[str, Any]]] = None


def catalog() -> Dict[str, Dict[str, Any]]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _build_catalog()
    return _CATALOG


CORE_SUITE = [
    "croma_base__croma_vit", "croma_large__croma_vit",
    "dofa_base__dofa_mae", "dofa_large__dofa_mae",
    "resnet18__sentinel2_all_moco", "resnet50__sentinel2_all_moco", "resnet50__sentinel2_all_dino",
    "resnet50__sentinel2_si_ms_satlas", "resnet50__landsat_oli_sr_moco",
    "swin_v2_t__sentinel2_si_ms_satlas", "swin_v2_b__sentinel2_si_ms_satlas",
    "scalemae_large__fmow_rgb",
    "prithvi_eo1__eo_1_0_100m", "prithvi_eo2__eo_2_0_300m", "prithvi_eo2__eo_2_0_300m_tl",
    "unet__sentinel2_3class_ftw",
    "olmoearth_v1_nano", "olmoearth_v1_base",
    # random-init negative controls
    "resnet50__scratch", "dofa_base__scratch", "croma_base__scratch",
]


def split_model_name(name: str) -> tuple[str, Optional[str]]:
    """'dofa_base__dofa_mae__ft__ftw-germany-cambodia_full' -> (base_key, tag)."""
    if "__ft__" in name:
        base, tag = name.split("__ft__", 1)
        return base, tag
    return name, None


def get_entry(name: str) -> Dict[str, Any]:
    base, tag = split_model_name(name)
    cat = catalog()
    if base not in cat:
        raise KeyError(f"Unknown gfmtools model '{base}'. Known: {sorted(cat)[:10]} ... ({len(cat)} total)")
    entry = copy.deepcopy(cat[base])
    entry["finetune_tag"] = tag
    entry["name"] = name
    return entry


def is_gfm_model(name: str) -> bool:
    try:
        base, _ = split_model_name(name)
        return base in catalog()
    except Exception:
        return False


def resolve_suite(suite: str | List[str], include_experimental: bool = False,
                  include: Optional[List[str]] = None, exclude: Optional[List[str]] = None) -> List[str]:
    cat = catalog()
    if isinstance(suite, (list, tuple)):
        names = list(suite)
    elif suite == "core":
        names = [n for n in CORE_SUITE if n in cat]
    elif suite == "optical":
        names = [k for k, e in cat.items() if e["sensor"] != "sar"]
    elif suite == "full":
        names = list(cat)
    else:
        raise ValueError(f"Unknown suite '{suite}' (core | optical | full | explicit list)")
    if not include_experimental:
        names = [n for n in names if not cat.get(split_model_name(n)[0], {}).get("experimental", False)]
    names += [n for n in (include or []) if n not in names]
    excl = set(exclude or [])
    return [n for n in names if n not in excl]
