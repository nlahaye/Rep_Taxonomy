"""Location-only geolocated ID: location-encoder ID at a benchmark's sample sites.

Needs no imagery, only a site table (``stem, lat, lon`` + grouping columns such as
``country`` / ``split``), so it can cover a full benchmark footprint (e.g. all 70k FTW
chips) and the exact ID / OOD site sets of each shift task. It answers: how complex is
the geography the benchmark samples, as seen by pre-trained location encoders
(GeoCLIP, CSP, SatCLIP), and do ID and OOD site sets differ in that complexity?

Scopes are built from the config::

    sites_csv: /data/ftw_geo/ftw_chip_sites.csv     # scripts/ftw_geo_subset.py --footprints-only
    scopes:
      all: {}                                       # every site
      by: [split, country]                          # one scope per value of each column
      sets:                                         # named filters  {column: [values]}
        ftw-germany-cambodia__id:  {country: [germany], split: [train]}
        ftw-germany-cambodia__ood: {country: [cambodia], split: [test]}
      ftw_shift_tasks: [table1, ftw-europe-africa]  # expands gfmtools.ftw_shifts tasks -> __id / __ood
    geo_id: {...}                                   # same block as config/gfm/base.yaml

Usage::

    python -m reptaxonomy.pipelines.geo_id_locations -y reptaxonomy/config/gfm/ftw_geo_id_locations.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

import numpy as np

from reptaxonomy.robustness.geo_intrinsic_dimension import (DEFAULTS, clean_for_id, location_encoder_embeddings,
                                                            run_location_only)

logger = logging.getLogger("reptaxonomy.geo_id_locations")


def _ftw_task_sets(names: List[Any]) -> Dict[str, Dict[str, list]]:
    """gfmtools ftw_shifts geographic tasks -> {task__id: filter, task__ood: filter}.

    ID sites = ID countries on the official train split; OOD sites = OOD countries on the
    official test split (the sites a model is trained on vs. evaluated on). Seasonal tasks
    share sites between ID and OOD and are reported once (``task__sites``)."""
    from gfmtools.data.datasets import ftw_shifts as S
    out: Dict[str, Dict[str, list]] = {}
    tasks = []
    for n in names:
        if n == "table1":
            tasks += [S.get_task(t) for t in S.EARTHSHIFT_TABLE1_TASKS]
        elif n == "earthshift":
            tasks += list(S.EARTHSHIFT_FTW_TASKS.values())
        elif isinstance(n, dict):
            kind = n.get("kind", "geographic")
            tasks.append(S.geographic_shift(n["id_countries"], n["ood_countries"], name=n.get("name")) if kind == "geographic"
                         else S.seasonal_shift(n["countries"], name=n.get("name")))
        else:
            tasks.append(S.get_task(n))
    for t in tasks:
        idc, oodc = t.in_domain.countries, t.out_domain.countries
        if t.shift_type == "geographic":
            out[f"{t.name}__id"] = {"country": list(idc), "split": ["train"]}
            out[f"{t.name}__ood"] = {"country": list(oodc), "split": ["test"]}
        else:
            out[f"{t.name}__sites"] = {} if idc == "all" else {"country": list(idc)}
    return out


def build_scoped_sites(sites: pd.DataFrame, scopes: Dict[str, Any]) -> pd.DataFrame:
    parts = []
    if "all" in scopes or not scopes:
        parts.append(sites.assign(scope="all"))
    for col in scopes.get("by", []) or []:
        for v, g in sites.groupby(col):
            parts.append(g.assign(scope=f"{col}={v}"))
    sets = dict(scopes.get("sets", {}) or {})
    if scopes.get("ftw_shift_tasks"):
        sets.update(_ftw_task_sets(scopes["ftw_shift_tasks"]))
    for name, flt in sets.items():
        m = pd.Series(True, index=sites.index)
        for col, vals in (flt or {}).items():
            m &= sites[col].astype(str).isin([str(v) for v in vals])
        parts.append(sites[m].assign(scope=name))
    # identical site sets (e.g. a seasonal task's sites == a country scope) are analysed once
    seen, keep, aliases = {}, [], {}
    for df in parts:
        key = hash(tuple(sorted(df["stem"].astype(str))))
        if key in seen:
            logger.info("scope %s duplicates %s; skipped", df["scope"].iat[0] if len(df) else "?", seen[key])
            if len(df):
                aliases[str(df["scope"].iat[0])] = seen[key]
            continue
        seen[key] = df["scope"].iat[0] if len(df) else "?"
        keep.append(df)
    out = pd.concat(keep, ignore_index=True)
    out.attrs["scope_aliases"] = aliases
    return out


def run(cfg: Dict[str, Any], size_matched_only: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sites = pd.read_csv(cfg["sites_csv"])
    sites["stem"] = sites["stem"].astype(str)
    scoped = build_scoped_sites(sites, cfg.get("scopes", {}) or {})
    logger.info("scopes: %s", scoped.groupby("scope").size().to_dict())
    out_dir = cfg.get("out_dir", str(Path(cfg.get("root_dir", ".")) / "outputs"))
    params = dict(cfg.get("geo_id", {}) or {})
    params.setdefault("group_keys", ["country", "split"])
    name = cfg.get("name", "locations")
    out = Path(out_dir, "geo_id", name); out.mkdir(parents=True, exist_ok=True)
    sm = dict(cfg.get("size_matched", {}) or {})
    sm_enabled = bool(sm.pop("enabled", True))
    if size_matched_only:
        size_matched_shift(scoped, params, out, **sm)
        return
    run_location_only(scoped, out_dir, name, params, scope_col="scope")
    if sm_enabled:
        size_matched_shift(scoped, params, out, **sm)
    # skipped duplicate scopes -> the scope that was analysed (used by summarize.geo_id_report)
    Path(out_dir, "geo_id", name, "scope_aliases.json").write_text(
        json.dumps(scoped.attrs.get("scope_aliases", {}), indent=1))


def size_matched_shift(scoped: pd.DataFrame, params: Dict[str, Any], out: Path, n_rep: int = 50,
                       methods=("MLE", "TwoNN"), k: int = 10, seed: int = 0) -> pd.DataFrame:
    """Global ID of each OOD site set vs ``n_rep`` random ID subsets of the same size.

    ID estimates depend on sample size, and EarthShift OOD site sets are 1-2 orders of magnitude smaller
    than their ID sets, so the raw ID/OOD contrast is confounded. Reports the OOD estimate, the matched-ID
    mean / sd and the fraction of matched-ID draws below the OOD value (empirical quantile).
    """
    import skdim
    aliases = scoped.attrs.get("scope_aliases", {})
    by = {k_: g.drop_duplicates("stem") for k_, g in scoped.groupby("scope")}
    uniq = scoped.drop_duplicates("stem").reset_index(drop=True)
    p = {**DEFAULTS, **params}
    embs = location_encoder_embeddings(uniq[["lon", "lat"]].to_numpy(float), {"location_encoders": p["location_encoders"]})
    row_of = {s_: i for i, s_ in enumerate(uniq["stem"])}
    rng = np.random.default_rng(seed)

    def est(X: np.ndarray, m: str) -> float:
        X = clean_for_id(X, seed)
        try:
            if m == "MLE":
                e = skdim.id.MLE(); e.fit(X, n_neighbors=min(k, len(X) - 1))
            else:
                e = getattr(skdim.id, m)(); e.fit(X)
            return float(e.dimension_)
        except Exception:
            return float("nan")

    rows = []
    for ood in [s_ for s_ in by if str(s_).endswith("__ood")]:
        task = str(ood)[:-5]
        idn = f"{task}__id"
        idn = idn if idn in by else aliases.get(idn)
        if idn not in by:
            continue
        io, ii = by[ood]["stem"].map(row_of).to_numpy(), by[idn]["stem"].map(row_of).to_numpy()
        n = len(io)
        if n < max(k + 2, 20) or len(ii) <= n:
            continue
        draws = [rng.choice(ii, n, replace=False) for _ in range(n_rep)]
        for L, E in embs.items():
            for m in methods:
                v_ood = est(E[io], m)
                v_id = np.array([est(E[d], m) for d in draws])
                rows.append({"task": task, "model": L, "method": m, "k": k if m == "MLE" else np.nan, "n": n,
                             "n_id_pool": len(ii), "id_ood": v_ood, "id_matched_mean": np.nanmean(v_id),
                             "id_matched_sd": np.nanstd(v_id), "frac_matched_below_ood": np.nanmean(v_id < v_ood),
                             "ratio_ood_matched": v_ood / np.nanmean(v_id)})
            logger.info("size-matched %s / %s (n=%d)", task, L, n)
    df = pd.DataFrame(rows)
    df.to_csv(out / "shift_size_matched.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-y", "--config", required=True)
    ap.add_argument("--size-matched-only", action="store_true",
                    help="only the size-matched ID/OOD comparison (shift_size_matched.csv)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if cfg.get("base"):
        from reptaxonomy.pipelines.gfm_experiment import load_config
        cfg = load_config(a.config)
    run(cfg, size_matched_only=a.size_matched_only)


if __name__ == "__main__":
    main()
