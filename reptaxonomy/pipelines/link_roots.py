"""Build a joint run root from per-family run roots with symbolic links (no copies, no re-embedding).

Cross-family stages (forgetting with cross-family retention sets, ft_sequential) need every
family in one root_dir, while the per-family runs (e.g. ./gfm_runs/ftw, ./gfm_runs/mados) keep
their own roots. This links, for every family in each source's work/plan.json:

  embeddings/<cond>/            -> <src>/embeddings/<cond>/      (pretrained + fine-tuned embeddings, markers)
  finetune/<family>/            -> <src>/finetune/<family>/      (checkpoints, metrics)
  work/manifests/<cond>.json, work/labels/<cond>.npz, work/band_stats/<family>.json
and writes a merged work/plan.json, so the joint config needs no prepare / embed / finetune.

Completion markers and fingerprints are root-relative, so linked units count as done. New
embeddings written by the joint run (e.g. retention sets) land in the source roots through the
links. Do not run --overwrite on embed / ft_embed / finetune in the joint root: it would delete
the sources' files.

    python -m reptaxonomy.pipelines.link_roots --dest ./gfm_runs/ftw_mados_joint \\
        --src ./gfm_runs/ftw ./gfm_runs/mados
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _link(src: Path, dst: Path, log: List[str], dry: bool) -> None:
    if not src.exists():
        return
    if dst.is_symlink():
        if Path(os.readlink(dst)) == src:
            return
        raise FileExistsError(f"{dst} links to {os.readlink(dst)}, not {src}")
    if dst.exists():
        raise FileExistsError(f"{dst} exists and is not a link (refusing to shadow real data)")
    log.append(f"{dst} -> {src}")
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src, target_is_directory=src.is_dir())


def link_roots(dest: Path, sources: Sequence[Path], families: Optional[Sequence[str]] = None,
               dry: bool = False) -> Dict[str, object]:
    dest = dest.absolute()
    plan_out: Dict[str, object] = {"datasets": {}}
    owner: Dict[str, Path] = {}
    log: List[str] = []
    for src in sources:
        src = src.resolve()
        plan_p = src / "work" / "plan.json"
        if not plan_p.exists():
            raise FileNotFoundError(f"{plan_p} missing (run the prepare stage in {src} first)")
        plan = json.loads(plan_p.read_text())
        for fam, d in (plan.get("datasets") or {}).items():
            if families and fam not in families:
                continue
            conds = [c["name"] for c in d.get("conditions", [])]
            if not any((src / "embeddings" / c).exists() for c in conds):
                continue   # family only mentioned in this root's plan, its data lives elsewhere
            if fam in owner:
                raise ValueError(f"family {fam} has embeddings in both {owner[fam]} and {src}; "
                                 f"choose one with --families")
            owner[fam] = src
            plan_out["datasets"][fam] = d
            for c in conds:
                _link(src / "embeddings" / c, dest / "embeddings" / c, log, dry)
                _link(src / "work" / "manifests" / f"{c}.json", dest / "work" / "manifests" / f"{c}.json", log, dry)
                _link(src / "work" / "labels" / f"{c}.npz", dest / "work" / "labels" / f"{c}.npz", log, dry)
            _link(src / "work" / "band_stats" / f"{fam}.json", dest / "work" / "band_stats" / f"{fam}.json", log, dry)
            _link(src / "finetune" / fam, dest / "finetune" / fam, log, dry)
    if not owner:
        raise ValueError("no family with embeddings found in the sources")
    plan_out.update(embed_dir=str(dest / "embeddings"), work_dir=str(dest / "work"),
                    out_dir=str(dest / "outputs"), finetune_root=str(dest / "finetune"))
    if not dry:
        p = dest / "work" / "plan.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        old = {}
        if p.exists():
            old = (json.loads(p.read_text()).get("datasets") or {})
        plan_out["datasets"] = {**{k: v for k, v in old.items() if k not in owner}, **plan_out["datasets"]}
        p.write_text(json.dumps(plan_out, indent=1))
    return {"families": {k: str(v) for k, v in owner.items()}, "links": log}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True, help="joint root_dir to create / update")
    ap.add_argument("--src", nargs="+", required=True, help="per-family root_dirs")
    ap.add_argument("--families", nargs="*", help="only these families")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    res = link_roots(Path(a.dest), [Path(s) for s in a.src], a.families, a.dry_run)
    for fam, src in res["families"].items():
        print(f"{fam}: {src}")
    print(f"{len(res['links'])} new link(s){' (dry run)' if a.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
