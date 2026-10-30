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
EPA (``dataset: epa``)
  ecoregions        one condition per US_L3NAME ecoregion (``regions: all | [..] | split:<name>``).
  splits            split_file splits (train / val / test region groups).
MADOS (``dataset: mados``)
  splits            train / val / test.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

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


def build_conditions(dataset: str, data_cfg: Dict[str, Any], experiments: Sequence[Dict[str, Any]]):
    """Return (conditions, comparisons, finetune_runs)."""
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
                regions = list_epa_regions(common.get("data_root", "/mnt/data/mtruong/"))
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
                role = "id_region" if r in train_regions else "region"
                names.append(add(Condition(f"epa__{slug(r)}", "epa", "EPARepSet", kw, "ecoregions", role=role,
                                           tags={"region": r, "in_train_split": r in train_regions})).name)
            comps += [Comparison("ecoregions", a, b, "pairwise") for a, b in itertools.combinations(names, 2)]
            if exp.get("finetune", True) and common.get("split_file"):
                tr = Condition("epa__split-train", "epa", "EPARepSet", dict(base, split="train"), "epa_splits", role="train",
                               analyze=False)
                va = Condition("epa__split-val", "epa", "EPARepSet", dict(base, split="val"), "epa_splits", role="val",
                               analyze=False)
                runs.append(FinetuneRun(tag="epa-train", dataset="epa", train=tr, val=va, eval_conditions=names,
                                        id_conditions=[n for n in names if conds[n].tags.get("in_train_split")],
                                        ood_conditions=[n for n in names if not conds[n].tags.get("in_train_split")]))

        elif dataset == "epa" and t == "splits":
            names = []
            for sp in exp.get("splits", ["train", "val", "test"]):
                names.append(add(Condition(f"epa__split-{sp}", "epa", "EPARepSet", dict(base, split=sp),
                                           "epa_splits", role=sp)).name)
            comps += [Comparison("epa_splits", a, b, "pairwise") for a, b in itertools.combinations(names, 2)]
            if "epa__split-train" in conds and "epa__split-test" in conds:
                runs.append(FinetuneRun(tag="epa-train", dataset="epa", train=conds["epa__split-train"],
                                        val=conds.get("epa__split-val"), eval_conditions=["epa__split-test"],
                                        ood_conditions=["epa__split-test"]))

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
