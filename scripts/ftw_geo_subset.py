#!/usr/bin/env python
"""Fetch a hash-stable Fields-of-the-World subset (or only its chip footprints) for geo_id runs.

Writes the layout ``gfmtools.data.datasets.ftw.FieldsOfTheWorld`` expects::

    <root>/<country>/chips_<country>.parquet
    <root>/<country>/data_config_<country>.json
    <root>/<country>/s2_images/window_{a,b}/<aoi>.tif
    <root>/<country>/label_masks/semantic_3class/<aoi>.tif

and ``<root>/ftw_chip_sites.csv`` (stem, country, aoi_id, split, lat, lon) with chip centroids
for *all* chips of the requested countries (used by the location-only geo_id run).

Example::

    python scripts/ftw_geo_subset.py --root /data/ftw_geo --countries kenya france \
        --split test --n-per-country 60 --windows a
    python scripts/ftw_geo_subset.py --root /data/ftw_geo --countries all --footprints-only
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE = "https://data.source.coop/kerner-lab/fields-of-the-world"
ALL = ["austria", "belgium", "brazil", "cambodia", "corsica", "croatia", "denmark", "estonia", "finland",
       "france", "germany", "india", "kenya", "latvia", "lithuania", "luxembourg", "netherlands", "portugal",
       "rwanda", "slovakia", "slovenia", "south_africa", "spain", "sweden", "vietnam"]


def _get(url: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rep_taxonomy-geo_id/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        dst.write_bytes(data)
        return True
    except Exception as exc:
        print(f"  miss {url}: {exc}", file=sys.stderr)
        return False


def centroids(parquet: Path) -> pd.DataFrame:
    import shapely
    df = pd.read_parquet(parquet)
    geom = shapely.from_wkb(df["geometry"].values)
    c = shapely.centroid(geom)
    return pd.DataFrame({"aoi_id": df["aoi_id"].astype(str), "split": df["split"].astype(str),
                         "lon": shapely.get_x(c), "lat": shapely.get_y(c)})


def stable_pick(ids, n, seed=0):
    ids = sorted(ids, key=lambda k: hashlib.sha1(f"{seed}:{k}".encode()).hexdigest())
    return ids if n is None else ids[: int(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--countries", nargs="+", default=["all"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-per-country", type=int, default=60)
    ap.add_argument("--windows", nargs="+", default=["a"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--footprints-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.root)
    countries = ALL if a.countries == ["all"] else a.countries
    sites = []
    for c in countries:
        cdir = root / c
        _get(f"{BASE}/{c}/chips_{c}.parquet", cdir / f"chips_{c}.parquet")
        _get(f"{BASE}/{c}/data_config_{c}.json", cdir / f"data_config_{c}.json")
        cen = centroids(cdir / f"chips_{c}.parquet").assign(country=c)
        sites.append(cen)
        if a.footprints_only:
            continue
        pick = stable_pick(cen.loc[cen["split"] == a.split, "aoi_id"].tolist(), a.n_per_country, a.seed)
        jobs = [(f"{BASE}/{c}/label_masks/semantic_3class/{i}.tif", cdir / "label_masks" / "semantic_3class" / f"{i}.tif")
                for i in pick]
        jobs += [(f"{BASE}/{c}/s2_images/window_{w}/{i}.tif", cdir / "s2_images" / f"window_{w}" / f"{i}.tif")
                 for w in a.windows for i in pick]
        with ThreadPoolExecutor(a.workers) as ex:
            ok = sum(ex.map(lambda j: _get(*j), jobs))
        print(f"{c}: {len(pick)} chips, {ok}/{len(jobs)} files")
    s = pd.concat(sites, ignore_index=True)
    s["stem"] = s["country"] + "__" + s["aoi_id"]
    s.to_csv(root / "ftw_chip_sites.csv", index=False)
    print(f"wrote {root / 'ftw_chip_sites.csv'} ({len(s)} chips)")


if __name__ == "__main__":
    main()
