"""Ecological structure of the EPA / NLCD ecoregions used by the GFM experiments.

The dataset covers ten EPA Level III ecoregions (``US_L3NAME``), each with Landsat
(``2019LS_*``), Sentinel-2 (``2019S2_*``) and NLCD label (``2019NLCD_*``) stores.

* ``ECO_GROUPS``: five ecological groups that follow the EPA Level I regions (western forested
  mountains, dry West = desert + semiarid prairie, northern forests, eastern temperate forests,
  Gulf coastal prairie). The Gulf coastal prairie is kept on its own: it is flat grassland /
  cropland and its Level I membership is reported inconsistently.
* ``SIDE``: humid East vs dry / montane West (the main land-cover gradient of the set).
* ``SPLIT_SCHEMES``: region-disjoint train / val / test splits
    within_group  every test region has a same-group region in train (new region, familiar biome)
    east_west     train on the humid East, test on the dry and mountain West (new biome)
* ``eco_distance``: 0 same group, 1 different group on the same side, 2 opposite sides.
* ``region_hierarchy``: Level I / II labels from the dataset's ecoregion GeoJSON when it carries
  them (``NA_L1NAME`` / ``NA_L2NAME`` or ``L1_KEY`` / ``L2_KEY``), used as a second distance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

ECO_GROUPS: Dict[str, str] = {
    "Blue Mountains": "western_mountains",
    "Southern Rockies": "western_mountains",
    "Central Basin and Range": "dry_west",
    "Northwestern Great Plains": "dry_west",
    "Northern Lakes and Forests": "northern_forests",
    "Northeastern Highlands": "northern_forests",
    "Central Appalachians": "eastern_temperate",
    "Piedmont": "eastern_temperate",
    "Eastern Corn Belt Plains": "eastern_temperate",
    "Western Gulf Coastal Plain": "gulf_prairie",
}

SIDE: Dict[str, str] = {
    "western_mountains": "west", "dry_west": "west",
    "northern_forests": "east", "eastern_temperate": "east", "gulf_prairie": "east",
}

SPLIT_SCHEMES: Dict[str, Dict[str, list]] = {
    "within_group": {
        "train": ["Southern Rockies", "Northwestern Great Plains", "Northern Lakes and Forests",
                  "Piedmont", "Eastern Corn Belt Plains"],
        "val": ["Central Appalachians"],
        "test": ["Blue Mountains", "Central Basin and Range", "Northeastern Highlands",
                 "Western Gulf Coastal Plain"],
    },
    "east_west": {
        "train": ["Central Appalachians", "Piedmont", "Eastern Corn Belt Plains",
                  "Northern Lakes and Forests", "Northeastern Highlands"],
        "val": ["Western Gulf Coastal Plain"],
        "test": ["Blue Mountains", "Southern Rockies", "Central Basin and Range",
                 "Northwestern Great Plains"],
    },
}

DISTANCE_LABELS = {0: "same_group", 1: "same_side", 2: "opposite_side"}


def eco_group(region: str) -> Optional[str]:
    return ECO_GROUPS.get(region)


def eco_distance(a: str, b: str) -> Optional[int]:
    ga, gb = ECO_GROUPS.get(a), ECO_GROUPS.get(b)
    if ga is None or gb is None:
        return None
    if ga == gb:
        return 0
    return 1 if SIDE[ga] == SIDE[gb] else 2


def resolve_scheme(spec, name: Optional[str] = None) -> Dict[str, list]:
    """``within_group`` / ``east_west`` or an explicit ``{train: [...], val: [...], test: [...]}``."""
    if isinstance(spec, str):
        if spec not in SPLIT_SCHEMES:
            raise ValueError(f"unknown EPA split scheme '{spec}' (known: {sorted(SPLIT_SCHEMES)})")
        return {k: list(v) for k, v in SPLIT_SCHEMES[spec].items()}
    if isinstance(spec, Mapping) and "train" in spec:
        return {k: list(v) for k, v in spec.items()}
    raise ValueError(f"EPA split scheme {name or spec!r} needs train / val / test region lists")


def check_scheme(name: str, scheme: Mapping[str, Sequence[str]]) -> None:
    seen: Dict[str, str] = {}
    for sp, regions in scheme.items():
        for r in regions:
            if r in seen:
                raise ValueError(f"EPA split scheme '{name}': {r} is in both {seen[r]} and {sp}")
            seen[r] = sp


def region_hierarchy(data_root: Optional[str]) -> Dict[str, Dict[str, str]]:
    """``{US_L3NAME: {"l1": ..., "l2": ...}}`` from the dataset GeoJSON ({} when unavailable)."""
    try:
        import geopandas as gpd
        if not data_root:
            from gfmtools.utils.paths import default_data_root
            data_root = default_data_root("epa", "/mnt/data/mtruong/")
        path = Path(data_root) / "epa_shapes" / "all_epa3_subsets.geojson"
        if not path.exists():
            return {}
        df = gpd.read_file(path, ignore_geometry=True)
    except Exception:
        return {}
    cols = {c.upper(): c for c in df.columns}
    l1 = next((cols[c] for c in ("NA_L1NAME", "L1_KEY", "NA_L1CODE") if c in cols), None)
    l2 = next((cols[c] for c in ("NA_L2NAME", "L2_KEY", "NA_L2CODE") if c in cols), None)
    if "US_L3NAME" not in df.columns or not (l1 or l2):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.drop_duplicates("US_L3NAME").iterrows():
        out[str(row["US_L3NAME"])] = {k: str(row[c]) for k, c in (("l1", l1), ("l2", l2)) if c}
    return out


def hierarchy_distance(a: str, b: str, hier: Mapping[str, Mapping[str, str]]) -> Optional[int]:
    """0 same Level II, 1 same Level I, 2 different Level I (None without labels)."""
    ha, hb = hier.get(a), hier.get(b)
    if not ha or not hb:
        return None
    if ha.get("l2") and ha.get("l2") == hb.get("l2"):
        return 0
    if ha.get("l1") and hb.get("l1"):
        return 1 if ha["l1"] == hb["l1"] else 2
    return None
