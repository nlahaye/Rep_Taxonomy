"""Experimental setups -> "conditions" (representation sets) and fine-tuning runs.

A *condition* is one dataset slice that rep_taxonomy treats as its own dataset
(``dataset_name``): embeddings, projections, robustness, etc. are computed per
condition, and ``cross_condition`` then compares conditions for every model.

Supported setups (``experiments`` list in the YAML):

FTW (``dataset: ftw``)
  shift_task        EarthShift tasks from ``gfmtools...ftw_shifts`` (``tasks: table1 | all | [..]``);
                    one condition per split (train / val / id_test / ood_test).
                    Comparisons: train<->id_test (in-distribution reference),
                    train<->ood_test and id_test<->ood_test (shift).
  shift_by_country  OOD (or any) split of a multi-country task broken out by country.
  per_country       official split, one condition per country (country x country matrix).
  seasonal_paired   same AOIs, window A vs window B, per country -> *paired* chips.
EPA (``dataset: epa``; another dataset key with ``family: epa`` prefixes its conditions with that key)
  ecoregions        one condition per US_L3NAME ecoregion (``regions: all | [..] | split:<name>``).
                    Pairwise comparisons carry ecological distance labels (``epa_regions``):
                    eco_distance 0 same group / 1 same side / 2 East vs West, plus the Level I / II
                    hierarchy distance when the ecoregion GeoJSON provides it.
  region_splits     named region-disjoint schemes (``schemes: [within_group, east_west]`` or explicit
                    ``{name: {train, val, test}}``); one pooled condition per scheme split
                    (``<ds>__<scheme>__<split>``), balanced per region, and a fine-tuning run
                    ``<ds>-<scheme>`` (train -> val, evaluated on test and on the per-region test
                    conditions). ``finetune_max_samples_per_region`` sizes separate training sets.
  timepoint_pair    the same chips of one scheme split at another composite (``timepoint``, e.g. 0 =
                    leaf-off) -> paired comparison with the default (leaf-on) condition.
  splits            split_file splits (train / val / test region groups).
MADOS (``dataset: mados``)
  splits            train / val / test.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from reptaxonomy.gfm import epa_regions as ER
from reptaxonomy.gfm.datasets import list_epa_regions, slug

EARTHSHIFT_TABLE1 = ("ftw-south_africa-window", "ftw-germany-year", "ftw-all-window",
                     "ftw-germany-cambodia", "ftw-germany-denmark")


@dataclass
class Condition:
    name: str
    dataset: str                       # ftw | epa | mados
    cls: str                           # FTWRepSet | EPARepSet | MADOSRepSet
    kwargs: Dict[str, Any]
    setup: str                         # setup id (groups conditions for comparisons)
    role: str = "eval"                 # train | val | id_test | ood_test | country | season | region | split
    paired_with: Optional[str] = None  # condition sharing pair_keys (seasonal)
    analyze: bool = True               # False: only used to train/validate fine-tuning
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Comparison:
    setup: str
    a: str
    b: str
    kind: str              # reference | shift | paired | pairwise
    paired: bool = False
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinetuneRun:
    tag: str
    dataset: str
    train: Condition
    val: Optional[Condition]
    eval_conditions: List[str]
    id_conditions: List[str] = field(default_factory=list)
    ood_conditions: List[str] = field(default_factory=list)


def _ftw_tasks(spec: Any) -> List[str]:
    if spec in (None, "table1"):
        return list(EARTHSHIFT_TABLE1)
    if spec == "all":
        from gfmtools.data.datasets.ftw_shifts import EARTHSHIFT_FTW_TASKS
        return list(EARTHSHIFT_FTW_TASKS)
    return list(spec)


def _epa_scheme_condition(px: str, base: Dict[str, Any], scheme: str, split: str, regions: Sequence[str],
                          cap: Any = "same", suffix: str = "", analyze: bool = True) -> Condition:
    """Pooled condition of one scheme split, drawn per region (keys shared with per-region conditions)."""
    kw = {k: v for k, v in base.items() if k not in ("split", "region", "split_file")}
    kw["regions"] = list(regions)
    if cap != "same":
        kw["max_samples_per_region"] = cap
    kw.setdefault("max_samples_per_region", None)
    if kw["max_samples_per_region"] is not None:
        kw["max_samples"] = None   # the per-region cap sizes pooled conditions
    groups = sorted({g for g in (ER.eco_group(r) for r in regions) if g})
    return Condition(f"{px}__{slug(scheme)}__{split}{suffix}", "epa", "EPARepSet", kw, f"region_split:{scheme}",
                     role=split, analyze=analyze and not suffix,
                     tags={"scheme": scheme, "regions": list(regions), "eco_groups": groups})


def build_conditions(dataset: str, data_cfg: Dict[str, Any], experiments: Sequence[Dict[str, Any]],
                     prefix: Optional[str] = None):
    """Return (conditions, comparisons, finetune_runs). ``prefix`` (default: ``dataset``) names the
    EPA conditions, so a second EPA family (e.g. Landsat) has its own condition names."""
    px = prefix or dataset
    conds: Dict[str, Condition] = {}
    comps: List[Comparison] = []
    runs: List[FinetuneRun] = []
    common = {k: v for k, v in data_cfg.items() if k not in ("experiments",)}

    def add(c: Condition) -> Condition:
        conds.setdefault(c.name, c)
        return conds[c.name]

    for exp in experiments:
        t = exp["type"]
        ms = exp.get("max_samples", common.get("max_samples"))
        base = {k: v for k, v in common.items() if k not in ("max_samples",)}
        base["max_samples"] = ms

        if dataset == "ftw" and t == "shift_task":
            splits = exp.get("splits", ["train", "id_test", "ood_test"])
            for task in _ftw_tasks(exp.get("tasks")):
                spec = None
                if isinstance(task, dict):  # custom ftw_shifts.geographic_shift / seasonal_shift
                    from reptaxonomy.gfm.datasets import register_ftw_task
                    spec, task = dict(task), register_ftw_task(task)
                setup = f"shift:{task}"
                made = {}
                for sp in splits:
                    kw = dict(base, shift_task=task, split=sp)
                    if spec:
                        kw["shift_spec"] = spec
                    made[sp] = add(Condition(f"ftw__{slug(task)}__{sp}", "ftw", "FTWRepSet", kw, setup, role=sp,
                                             tags={"task": task}))
                if "train" in made and "id_test" in made:
                    comps.append(Comparison(setup, made["train"].name, made["id_test"].name, "reference"))
                for a in ("train", "id_test"):
                    if a in made and "ood_test" in made:
                        comps.append(Comparison(setup, made[a].name, made["ood_test"].name, "shift"))
                tr = conds.get(f"ftw__{slug(task)}__train") or add(Condition(
                    f"ftw__{slug(task)}__train", "ftw", "FTWRepSet",
                    dict(base, shift_task=task, split="train", **({"shift_spec": spec} if spec else {})),
                    setup, role="train", tags={"task": task}, analyze=False))
                va = add(Condition(f"ftw__{slug(task)}__val", "ftw", "FTWRepSet",
                                   dict(base, shift_task=task, split="val", **({"shift_spec": spec} if spec else {})), setup, role="val",
                                   tags={"task": task}, analyze="val" in splits)) if exp.get("finetune_val", True) else None
                ev = [c.name for sp, c in made.items() if sp in ("id_test", "ood_test")]
                runs.append(FinetuneRun(tag=slug(task), dataset="ftw", train=tr, val=va, eval_conditions=ev,
                                        id_conditions=[c.name for sp, c in made.items() if sp == "id_test"],
                                        ood_conditions=[c.name for sp, c in made.items() if sp == "ood_test"]))

        elif dataset == "ftw" and t == "shift_by_country":
            task = exp["task"]
            sp = exp.get("split", "ood_test")
            names = []
            for c in exp["countries"]:
                kw = dict(base, shift_task=task, split=sp, country_filter=[c])
                names.append(add(Condition(f"ftw__{slug(task)}__{sp}__{slug(c)}", "ftw", "FTWRepSet", kw,
                                           f"shift_by_country:{task}", role="country", tags={"country": c})).name)
            comps += [Comparison(f"shift_by_country:{task}", a, b, "pairwise") for a, b in itertools.combinations(names, 2)]

        elif dataset == "ftw" and t == "per_country":
            sp = exp.get("split", "test")
            names = []
            for c in exp["countries"]:
                kw = dict(base, split=sp, countries=[c], windows=exp.get("windows", ["a", "b"]))
                names.append(add(Condition(f"ftw__country-{slug(c)}__{sp}", "ftw", "FTWRepSet", kw, "per_country",
                                           role="country", tags={"country": c})).name)
            comps += [Comparison("per_country", a, b, "pairwise") for a, b in itertools.combinations(names, 2)]

        elif dataset == "ftw" and t == "seasonal_paired":
            sp = exp.get("split", "test")
            for c in exp["countries"]:
                pair = []
                for w in ("a", "b"):
                    kw = dict(base, split=sp, countries=[c], windows=[w])
                    pair.append(add(Condition(f"ftw__season-{slug(c)}-{w}__{sp}", "ftw", "FTWRepSet", kw,
                                              f"seasonal:{c}", role="season", tags={"country": c, "window": w})))
                pair[0].paired_with, pair[1].paired_with = pair[1].name, pair[0].name
                comps.append(Comparison(f"seasonal:{c}", pair[0].name, pair[1].name, "paired", paired=True))

        elif dataset == "epa" and t == "ecoregions":
            regions = exp.get("regions", "all")
            if regions == "all":
                regions = list_epa_regions(common.get("data_root"))
            elif isinstance(regions, str) and regions.startswith("split:"):
                regions = list_epa_regions(common.get("data_root"), regions.split(":", 1)[1], common.get("split_file"))
            train_regions = set()
            if common.get("split_file") and exp.get("train_split", "train"):
                try:
                    train_regions = set(list_epa_regions(common.get("data_root"), exp.get("train_split", "train"),
                                                         common.get("split_file")))
                except Exception:
                    train_regions = set()
            names = []
            for r in regions:
                kw = dict(base, region=r)
                kw.pop("split", None)
                kw.pop("max_samples_per_region", None)   # only used by pooled scheme conditions
                role = "id_region" if r in train_regions else "region"
                names.append(add(Condition(f"{px}__{slug(r)}", "epa", "EPARepSet", kw, "ecoregions", role=role,
                                           tags={"region": r, "in_train_split": r in train_regions,
                                                 "eco_group": ER.eco_group(r)})).name)
            hier = ER.region_hierarchy(common.get("data_root")) if exp.get("hierarchy", True) else {}
            for a, b in itertools.combinations(names, 2):
                ra, rb = conds[a].tags["region"], conds[b].tags["region"]
                d = ER.eco_distance(ra, rb)
                tags = {"region_a": ra, "region_b": rb, "group_a": ER.eco_group(ra), "group_b": ER.eco_group(rb),
                        "eco_distance": d, "eco_distance_label": ER.DISTANCE_LABELS.get(d)}
                hd = ER.hierarchy_distance(ra, rb, hier)
                if hd is not None:
                    tags["hier_distance"] = hd
                comps.append(Comparison("ecoregions", a, b, "pairwise", tags=tags))
            if exp.get("finetune", True) and common.get("split_file"):
                tr = Condition(f"{px}__split-train", "epa", "EPARepSet", dict(base, split="train"), "epa_splits",
                               role="train", analyze=False)
                va = Condition(f"{px}__split-val", "epa", "EPARepSet", dict(base, split="val"), "epa_splits",
                               role="val", analyze=False)
                runs.append(FinetuneRun(tag=f"{px}-train", dataset="epa", train=tr, val=va, eval_conditions=names,
                                        id_conditions=[n for n in names if conds[n].tags.get("in_train_split")],
                                        ood_conditions=[n for n in names if not conds[n].tags.get("in_train_split")]))

        elif dataset == "epa" and t == "splits":
            names = []
            for sp in exp.get("splits", ["train", "val", "test"]):
                names.append(add(Condition(f"{px}__split-{sp}", "epa", "EPARepSet", {k: v for k, v in dict(base, split=sp).items()
                                                                            if k != "max_samples_per_region"},
                                           "epa_splits", role=sp)).name)
            comps += [Comparison("epa_splits", a, b, "pairwise") for a, b in itertools.combinations(names, 2)]
            if f"{px}__split-train" in conds and f"{px}__split-test" in conds:
                runs.append(FinetuneRun(tag=f"{px}-train", dataset="epa", train=conds[f"{px}__split-train"],
                                        val=conds.get(f"{px}__split-val"), eval_conditions=[f"{px}__split-test"],
                                        ood_conditions=[f"{px}__split-test"]))

        elif dataset == "epa" and t == "region_splits":
            schemes = exp.get("schemes", ["within_group", "east_west"])
            items = list(schemes.items()) if isinstance(schemes, dict) else [(n, n) for n in schemes]
            splits = list(exp.get("splits", ["train", "val", "test"]))
            for sname, spec in items:
                scheme = ER.resolve_scheme(spec, sname)
                ER.check_scheme(sname, scheme)
                made = {sp: add(_epa_scheme_condition(px, base, sname, sp, scheme[sp]))
                        for sp in splits if scheme.get(sp)}
                names = [c.name for c in made.values()]
                for a, b in itertools.combinations(names, 2):
                    kind = "shift" if conds[b].role == "test" or conds[a].role == "test" else "reference"
                    comps.append(Comparison(f"region_split:{sname}", a, b, kind, tags={"scheme": sname}))
                if not exp.get("finetune", True) or not scheme.get("train"):
                    continue
                ft_cap = exp.get("finetune_max_samples_per_region", "same")
                tr, va = made.get("train"), made.get("val")
                if ft_cap != "same":   # separate (not analysed) training / validation sets
                    fb = dict(base, max_samples=None)
                    tr = add(_epa_scheme_condition(px, fb, sname, "train", scheme["train"], ft_cap, "-ft"))
                    va = add(_epa_scheme_condition(px, fb, sname, "val", scheme["val"], ft_cap, "-ft")) \
                        if scheme.get("val") else None
                if tr is None:
                    tr = add(_epa_scheme_condition(px, base, sname, "train", scheme["train"], analyze=False))
                test = [made["test"].name] if "test" in made else []
                per_region = [f"{px}__{slug(r)}" for r in scheme.get("test", []) if f"{px}__{slug(r)}" in conds]
                runs.append(FinetuneRun(tag=f"{px}-{slug(sname)}", dataset="epa", train=tr, val=va,
                                        eval_conditions=test + per_region, ood_conditions=test + per_region))

        elif dataset == "epa" and t == "timepoint_pair":
            sname = exp.get("scheme", "within_group")
            sp = exp.get("split", "test")
            scheme = ER.resolve_scheme(exp.get("regions_from", sname), sname)
            ref = add(_epa_scheme_condition(px, base, sname, sp, scheme[sp]))
            tp = int(exp.get("timepoint", 0))
            label = slug(exp.get("label") or f"t{tp}")
            alt = _epa_scheme_condition(px, dict(base, timepoint=tp), sname, sp, scheme[sp])
            alt.name, alt.setup, alt.role = f"{ref.name}__{label}", f"timepoint:{sname}-{sp}", "season"
            alt.tags = dict(alt.tags, timepoint=tp, timepoint_label=label)
            alt = add(alt)
            ref.paired_with, alt.paired_with = alt.name, ref.name
            comps.append(Comparison(alt.setup, ref.name, alt.name, "paired", paired=True,
                                    tags={"scheme": sname, "split": sp, "timepoint_b": tp, "label_b": label}))

        elif dataset == "mados" and t == "splits":
            names = []
            for sp in exp.get("splits", ["train", "val", "test"]):
                names.append(add(Condition(f"mados__{sp}", "mados", "MADOSRepSet", dict(base, split=sp),
                                           "mados_splits", role=sp)).name)
            comps += [Comparison("mados_splits", a, b, "pairwise") for a, b in itertools.combinations(names, 2)]
            if "mados__train" in conds:
                runs.append(FinetuneRun(tag="mados-train", dataset="mados", train=conds["mados__train"],
                                        val=conds.get("mados__val"),
                                        eval_conditions=[n for n in names if n != "mados__train"],
                                        id_conditions=[n for n in names if n != "mados__train"]))
        else:
            raise ValueError(f"Unknown experiment type '{t}' for dataset '{dataset}'")

    # de-duplicate fine-tune runs by tag (eval lists merged)
    merged: Dict[str, FinetuneRun] = {}
    for r in runs:
        if r.tag in merged:
            m = merged[r.tag]
            m.eval_conditions = sorted(set(m.eval_conditions) | set(r.eval_conditions))
            m.id_conditions = sorted(set(m.id_conditions) | set(r.id_conditions))
            m.ood_conditions = sorted(set(m.ood_conditions) | set(r.ood_conditions))
        else:
            merged[r.tag] = r
    return list(conds.values()), comps, list(merged.values())
