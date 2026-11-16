from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from dataclasses import asdict, dataclass, field, is_dataclass

from reptaxonomy.util.general_utils import read_yaml
from reptaxonomy.util import run_state as RS

from reptaxonomy.projections.run_projections import run_knn_gen
from reptaxonomy.robustness.embedding_robustness import analyze_robustness
from reptaxonomy.intercompare.data_kernel_analysis import run_data_kernel_analysis
from reptaxonomy.intercompare.geodesic_compare import run_geodesic_compare
from reptaxonomy.resources.compute_resource_reqs import compute_resource_reqs
from reptaxonomy.robustness.weight_matrix_analysis import run_weight_mtx_analysis
from reptaxonomy.summarize.demsar_wrapper import demsar_wrapper
from reptaxonomy.summarize.cross_perspective_synthesis import run_cross_perspective_synthesis
from reptaxonomy.intercompare.reference_similarity import run_reference_similarity
from reptaxonomy.robustness.invariance_checks import run_invariance_checks

from reptaxonomy.util.experiment_init_utils import (
    build_dataset_bundle,
    build_dataset_spec,
    build_model_bundle,
    resolve_model_spec,
    serialize_dataset_bundle,
    serialize_model_bundle,
)

logger = logging.getLogger(__name__)


def ensure_dict(cfg: Any, name: str = "config") -> Dict[str, Any]:
    if cfg is None:
        raise ValueError(f"{name} is empty or unreadable")
    if not isinstance(cfg, dict):
        raise ValueError(f"{name} must be a dictionary")
    return cfg


def load_pipeline_config(path: str | Path) -> Dict[str, Any]:
    cfg = read_yaml(str(path))
    return ensure_dict(cfg, "pipeline config")


def load_model_names(model_names_cfg: Any) -> List[str]:
    if isinstance(model_names_cfg, list):
        model_names = model_names_cfg
    elif isinstance(model_names_cfg, (str, Path)):
        with open(model_names_cfg, "r") as f:
            model_names = json.load(f)
    else:
        raise ValueError("model_names must be a list or path to a JSON file")

    if not isinstance(model_names, list) or not model_names:
        raise ValueError("model_names must resolve to a non-empty list")

    return [str(x) for x in model_names]


def merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def should_run(step_cfg: Dict[str, Any]) -> bool:
    return bool(step_cfg.get("enabled", True))


def validate_pipeline_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = ensure_dict(cfg)

    if "dataset" not in cfg:
        raise ValueError("Missing required top-level config key: 'dataset'")
    if "model_names" not in cfg:
        raise ValueError("Missing required top-level config key: 'model_names'")

    cfg["dataset"] = ensure_dict(cfg["dataset"], "dataset")
    if "dataset_name" not in cfg["dataset"]:
        if "name" in cfg["dataset"]:
            cfg["dataset"]["dataset_name"] = cfg["dataset"]["name"]
        else:
            raise ValueError("dataset must define either 'dataset_name' or 'name'")

    cfg["model_names"] = load_model_names(cfg["model_names"])
    cfg["steps"] = ensure_dict(cfg.get("steps", {}), "steps")
    cfg["per_model_overrides"] = ensure_dict(cfg.get("per_model_overrides", {}), "per_model_overrides")

    cfg.setdefault("work_dir", "./work")
    cfg.setdefault("embed_dir", "./embeddings")
    cfg.setdefault("out_dir", "./outputs")
    cfg.setdefault("n_runs", 3)
    cfg.setdefault("seed", 42)
    cfg.setdefault("projections", ["native", "umap", "tsne", "pca"])
    from reptaxonomy.util.general_utils import as_method_list
    cfg["upsample_methods"] = as_method_list(cfg.get("upsample_methods"), cfg.get("upsample_method", "pixelshuffle"))
    cfg.setdefault("dataset_init", {})
    cfg.setdefault("model_init", {})
    cfg.setdefault("analysis", {})

    return cfg


def attach_bundles_to_cfg(
    cfg: Dict[str, Any],
    dataset_bundle: DatasetBundle,
    model_bundle: Optional[ModelBundle] = None,
) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)

    dataset_spec_dict = asdict(dataset_bundle.dataset_spec)
    dataset_cls_name = dataset_bundle.dataset_cls.__name__
    dataset_kwargs = copy.deepcopy(dataset_bundle.dataset_kwargs)
    sample_schema = copy.deepcopy(dataset_bundle.sample_schema)
    dataset_metadata = copy.deepcopy(dataset_bundle.metadata)

    collate_fn_name = None
    if getattr(dataset_bundle, "collate_fn", None) is not None:
        collate_fn_name = getattr(dataset_bundle.collate_fn, "__name__", None)

    out["dataset"] = copy.deepcopy(dataset_spec_dict)
    out["dataset_bundle"] = {
        "dataset_spec": dataset_spec_dict,
        "dataset_cls_name": dataset_cls_name,
        "dataset_kwargs": dataset_kwargs,
        "sample_schema": sample_schema,
        "metadata": dataset_metadata,
        "collate_fn_name": collate_fn_name,
    }

    if model_bundle is not None:
        model_spec_dict = asdict(model_bundle.model_spec)
        model_dataset_spec_dict = asdict(model_bundle.dataset_spec)
        model_cls_name = model_bundle.model_cls.__name__
        model_kwargs = copy.deepcopy(model_bundle.model_kwargs)
        model_metadata = copy.deepcopy(model_bundle.metadata)

        out["model_bundle"] = {
            "model_name": model_bundle.model_name,
            "model_spec": model_spec_dict,
            "dataset_spec": model_dataset_spec_dict,
            "model_cls_name": model_cls_name,
            "model_kwargs": model_kwargs,
            "metadata": model_metadata,
        }

        out["encoder"] = model_bundle.model_name

    return out


def call_step(step_name: str, fn, cfg: Dict[str, Any]) -> None:
    dataset_name = cfg["dataset_bundle"]["dataset_spec"]["dataset_name"]
    model_name = cfg.get("model_bundle", {}).get("model_name", "global")
    print(f"[PIPELINE] Running step={step_name} dataset={dataset_name} model={model_name}")
    fn(cfg)


# Steps whose outputs do not depend on how token grids are upsampled to pixels.
UPSAMPLE_INDEPENDENT_STEPS = {"compute_resource_reqs", "weight_matrix_analysis"}

# Steps whose outputs depend on the model weights only (not on the condition's data). With
# ``share_across_conditions: [<step>, ...]`` they run once per model and dataset family; other
# conditions receive a copy of the outputs (sub-directories of embed_dir/<cond>/<model>/test).
SHAREABLE_STEP_OUTPUTS = {"weight_matrix_analysis": ["ww_stats", "ww_plots"]}

# Global steps that are evaluated once on the primary variant only.
PRIMARY_ONLY_GLOBAL_STEPS = {"invariance_checks"}


def variant_out_dir(base_out_dir: str, method: str, primary: str) -> str:
    """Primary upsampling variant writes to out_dir; others to out_dir/upsample_<method>."""
    return base_out_dir if method == primary else os.path.join(base_out_dir, f"upsample_{method}")


@dataclass
class StepState:
    """Step-level resume: one marker per (variant, step, model) under ``state_dir``.

    A step is skipped when its marker exists and its fingerprint (step config, model,
    variant, upstream embedding stamps) is unchanged. ``force`` reruns every step in
    the block (set when an upstream block recomputed something this invocation).
    """
    policy: Any
    stage: str
    state_dir: str
    method: str = ""
    upstream: Dict[str, Any] = field(default_factory=dict)
    force: bool = False
    share_root: str | None = None          # cache of condition-independent step outputs
    shared_steps: frozenset = frozenset()


def _share_paths(step_name: str, model: str, step_cfg: Dict[str, Any], cfg: Dict[str, Any],
                 state: "StepState") -> tuple[Path, Path]:
    sfp = RS.fingerprint(step_name, model, step_cfg, cfg.get("model_init"),
                         (cfg.get("per_model_overrides") or {}).get(model))
    dest = Path(cfg["embed_dir"], str(cfg["dataset"]["dataset_name"]), model, "test")
    return Path(state.share_root, step_name, model, sfp), dest


def _copy_outputs(step_name: str, src: Path, dst: Path) -> bool:
    subs = [d for d in SHAREABLE_STEP_OUTPUTS[step_name] if (src / d).is_dir()]
    if not subs:
        return False
    for d in subs:
        shutil.copytree(src / d, dst / d, dirs_exist_ok=True)
    return True


def _publish_shared(step_name: str, cache: Path, dest: Path) -> None:
    """Store a condition's outputs as the shared copy (first writer wins)."""
    if RS.read_marker(cache, "shared") is not None:
        return
    tmp = cache.with_name(cache.name + f".tmp{os.getpid()}")
    try:
        if _copy_outputs(step_name, dest, tmp):
            RS.mark_done(tmp, "shared", source=str(dest))
            tmp.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(tmp, cache)
            except OSError:   # another worker published first
                pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_step_block(
    steps: List[Any],
    cfg: Dict[str, Any],
    base_cfg: Dict[str, Any],
    model_bundle: Any = None,
    skip: set | None = None,
    state: "StepState | None" = None,
    errors: "Dict[str, str] | None" = None,
    tolerate: "set | bool" = False,
) -> bool:
    """Run a block of steps; returns True if any step actually ran.

    A failing step writes its FAILED marker and re-raises, which stops the block. When ``errors``
    is a dict and the step is tolerated (``tolerate`` is True or contains the step name), the
    failure is recorded in ``errors`` instead and the remaining steps still run, so one broken
    step (e.g. geodesic_compare) no longer silently prevents later steps, the other upsampling
    variants and the synthesis from ever running.
    """
    ran = False
    for item in steps:
        step_name, fn = item[0], item[1]
        capability = item[2] if len(item) > 2 else None
        if skip and step_name in skip:
            continue
        step_cfg = ensure_dict(cfg["steps"].get(step_name, {}), f"steps.{step_name}")
        if not should_run(step_cfg):
            continue
        if capability is not None and model_bundle is not None and not model_bundle.metadata.get(capability, True):
            print(
                f"[PIPELINE] Skipping step={step_name} "
                f"model={base_cfg.get('encoder')}; capability {capability}=False"
            )
            continue
        final_cfg = merge_dicts(base_cfg, step_cfg.get("config_overrides", {}))
        if state is None:
            call_step(step_name, fn, final_cfg)
            ran = True
            continue
        model = base_cfg.get("encoder") if model_bundle is not None else None
        key = f"{state.method or 'final'}.{step_name}.{model or 'global'}"
        up = state.upstream.get(model) if model else sorted(state.upstream.items())
        # per-model steps depend on their own model only; global steps on the whole model set
        parts = [step_name, model, state.method, step_cfg, None if model else cfg.get("model_names"), up]
        if str(cfg.get("pixelshuffle_factor", "auto")).lower() != "auto" and state.method == "pixelshuffle":
            parts.append(("pixelshuffle_factor", cfg.get("pixelshuffle_factor")))
        if state.method == "patch":
            parts.append(("patch", float(cfg.get("patch_cell_px", 32)), float(cfg.get("patch_min_purity", 0.75))))
        fp = RS.fingerprint(*parts)
        pol = state.policy
        shared = bool(model and state.share_root and step_name in state.shared_steps
                      and step_name in SHAREABLE_STEP_OUTPUTS)
        cache = dest = None
        if shared:
            cache, dest = _share_paths(step_name, model, step_cfg, cfg, state)
        if not state.force and not pol.should_run(state.state_dir, key, fp, state.stage, step_name):
            print(f"[PIPELINE] Skipping step={step_name} model={model or 'global'} variant={state.method}: complete")
            if shared:
                _publish_shared(step_name, cache, dest)
            continue
        if shared and not pol.overwrite(state.stage, step_name) and RS.read_marker(cache, "shared") is not None \
                and _copy_outputs(step_name, cache, dest):
            print(f"[PIPELINE] step={step_name} model={model}: copied condition-independent outputs from {cache}")
            pol.done(state.state_dir, key, fp, shared_from=str(cache))
            ran = True
            continue
        if state.force or pol.is_stale(state.state_dir, key, fp, state.stage, step_name) \
                or RS.read_marker(state.state_dir, key) is not None:
            final_cfg["refit_projections"] = True   # never reuse reducers fitted on old embeddings
        pol.begin(state.state_dir, key, fp)
        try:
            call_step(step_name, fn, final_cfg)
        except BaseException as exc:
            import traceback
            pol.failed(state.state_dir, key, traceback.format_exc() if isinstance(exc, Exception) else repr(exc))
            ok_to_continue = (errors is not None and isinstance(exc, Exception)
                              and (tolerate is True or (tolerate and step_name in tolerate)))
            if not ok_to_continue:
                raise
            errors[key] = f"{type(exc).__name__}: {exc}"
            logger.error("[PIPELINE] step=%s model=%s variant=%s failed (%s); continuing with the next step",
                         step_name, model or "global", state.method, errors[key])
            continue
        pol.done(state.state_dir, key, fp)
        if shared:
            _publish_shared(step_name, cache, dest)
        ran = True
    return ran


def _embed_dir_stamp(embed_dir: str, dataset: str, model: str, split: str = "test") -> Any:
    """Upstream identity of a model's embeddings when no driver-provided stamp exists."""
    d = Path(embed_dir, dataset, model, split)
    m = RS.stamp(d, "embed")
    if m is not None:
        return m
    files = list(d.glob("embd_*.npy"))
    return [len(files), max((f.stat().st_mtime_ns for f in files), default=0)]


def run_full_pipeline(cfg: Dict[str, Any]) -> None:
    """Run every enabled step. Re-running resumes: completed steps (``_state/_DONE.*``
    markers) are skipped, failed/interrupted ones are redone. ``overwrite: true |
    [step, ...]`` forces steps; ``resume: false`` disables step markers."""
    cfg = dict(cfg)
    policy = cfg.pop("_run_policy", None)
    stage = cfg.pop("_stage", "pipeline")
    state_dir = cfg.pop("_state_dir", None)
    upstream = cfg.pop("_upstream", None)
    if policy is None:
        policy = RS.RunPolicy({"overwrite": cfg.get("overwrite", False), "resume": cfg.get("resume", {})})
    cfg = validate_pipeline_cfg(cfg)

    dataset_spec = build_dataset_spec(
        dataset_cfg=cfg["dataset"],
        dataset_init=ensure_dict(cfg.get("dataset_init", {}), "dataset_init"),
    )
    dataset_bundle = build_dataset_bundle(dataset_spec)

    common_cfg = copy.deepcopy(cfg)
    common_cfg["dataset"] = vars(dataset_spec)
    common_cfg["model_names"] = cfg["model_names"]
    common_cfg["projections"] = cfg["projections"]
    common_cfg = attach_bundles_to_cfg(common_cfg, dataset_bundle)

    per_model_steps = [
        ("compute_resource_reqs", compute_resource_reqs, "supports_resource_requirements"),
        ("weight_matrix_analysis", run_weight_mtx_analysis, "supports_weight_matrix_analysis"),
        ("run_projections", run_knn_gen, "supports_projection_analysis"),
        ("embedding_robustness", analyze_robustness, "supports_embedding_robustness"),
    ]

    global_steps = [
        ("data_kernel_analysis", run_data_kernel_analysis),
        ("reference_similarity", run_reference_similarity),
        ("invariance_checks", run_invariance_checks),
        ("geodesic_compare", run_geodesic_compare),
        ("demsar_wrapper", demsar_wrapper),
    ]

    # Final, cross-variant synthesis (runs once after every variant is complete).
    final_steps = [("cross_perspective_synthesis", run_cross_perspective_synthesis)]

    shared_model_init = ensure_dict(cfg.get("model_init", {}), "model_init")
    base_out_dir = str(cfg["out_dir"])
    from reptaxonomy.util.general_utils import as_method_list
    methods = as_method_list(cfg["upsample_methods"])
    primary = methods[0]
    variants: Dict[str, str] = {}

    model_bundles = {}
    for model_name in cfg["model_names"]:
        per_model_cfg = ensure_dict(
            cfg.get("per_model_overrides", {}).get(model_name, {}),
            f"per_model_overrides.{model_name}",
        )
        model_spec = resolve_model_spec(model_name, shared_model_init, per_model_cfg)
        model_bundles[model_name] = build_model_bundle(model_spec, dataset_bundle)

    dataset_name = dataset_spec.dataset_name if hasattr(dataset_spec, "dataset_name") else cfg["dataset"]["dataset_name"]
    if upstream is None:
        upstream = {m: _embed_dir_stamp(cfg["embed_dir"], dataset_name, m, str(cfg["dataset"].get("split", "test")))
                    for m in cfg["model_names"]}
    state_dir = state_dir or os.path.join(base_out_dir, "_state", str(dataset_name))
    share_root = cfg.get("_share_root")
    shared_steps = frozenset(cfg.get("share_across_conditions") or [])
    fail_fast = bool(cfg.get("fail_fast", False))
    failed: Dict[str, str] = {}    # model -> error; the other models (and global steps) still run
    step_errors: Dict[str, str] | None = None if fail_fast else {}   # tolerated step failures
    any_ran = False

    for method in methods:
        out_dir = variant_out_dir(base_out_dir, method, primary)
        variants[method] = out_dir
        variant_cfg = copy.deepcopy(common_cfg)
        variant_cfg["out_dir"] = out_dir
        variant_cfg["upsample_method"] = method
        # Every pixel variant samples the same pixels as the primary variant; the patch-level
        # variant samples cells of its own grid (shared by all models), cached in its out_dir.
        variant_cfg["index_cache_root"] = base_out_dir
        if method == "patch":   # cell samples depend on the cell size and purity threshold
            variant_cfg["index_cache_root"] = os.path.join(
                out_dir, f"_cells_{float(cfg.get('patch_cell_px', 32)):g}px_p{float(cfg.get('patch_min_purity', 0.75)):g}")
        print(f"[PIPELINE] Upsampling variant={method} out_dir={out_dir}")

        per_ran = False
        for model_name, model_bundle in model_bundles.items():
            if model_name in failed:
                continue
            model_cfg = attach_bundles_to_cfg(variant_cfg, dataset_bundle, model_bundle)
            model_cfg["encoder"] = model_name
            try:
                per_ran |= run_step_block(
                    per_model_steps, cfg, model_cfg, model_bundle=model_bundle,
                    skip=None if method == primary else UPSAMPLE_INDEPENDENT_STEPS,
                    state=StepState(policy, stage, state_dir, method, upstream,
                                    share_root=share_root, shared_steps=shared_steps),
                    errors=step_errors, tolerate=UPSAMPLE_INDEPENDENT_STEPS,
                )
            except Exception as exc:
                if fail_fast:
                    raise
                failed[model_name] = f"{type(exc).__name__}: {exc}"
                logger.error("[PIPELINE] %s / %s failed (%s); continuing with the other models",
                             dataset_name, model_name, failed[model_name])

        ok = [m for m in cfg["model_names"] if m not in failed]
        if len(ok) < 3:
            logger.error("[PIPELINE] %s: %d model(s) left after failures; global steps skipped", dataset_name, len(ok))
            break
        gcfg, gvar = cfg, copy.deepcopy(variant_cfg)
        if failed:   # global steps on the models that completed (their fingerprint records the set)
            gcfg = dict(cfg, model_names=ok)
            gvar["model_names"] = ok
        # global steps read per-model outputs: rerun them whenever a per-model step reran
        glob_ran = run_step_block(
            global_steps, gcfg, gvar,
            skip=None if method == primary else PRIMARY_ONLY_GLOBAL_STEPS,
            state=StepState(policy, stage, state_dir, method, upstream, force=per_ran),
            errors=step_errors, tolerate=True,
        )
        any_ran |= per_ran or glob_ran

    final_cfg = copy.deepcopy(common_cfg)
    final_cfg["out_dir"] = base_out_dir
    syn = dict(final_cfg.get("synthesis", {}) or {})
    syn.setdefault("variants", variants)
    syn.setdefault("primary_variant", primary)
    final_cfg["synthesis"] = syn
    ok = [m for m in cfg["model_names"] if m not in failed]
    if len(ok) >= 3:
        fcfg = cfg
        if failed:
            fcfg = dict(cfg, model_names=ok)
            final_cfg["model_names"] = ok
        run_step_block(final_steps, fcfg, final_cfg,
                       state=StepState(policy, stage, state_dir, "", upstream, force=any_ran),
                       errors=step_errors, tolerate=True)
    msgs = [f"{m}: {e}" for m, e in failed.items()] + [f"{k}: {e}" for k, e in (step_errors or {}).items()]
    if msgs:
        raise RuntimeError(f"{dataset_name}: {len(msgs)} step(s)/model(s) failed (rerun retries them): "
                           + "; ".join(msgs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline from a YAML config.")
    parser.add_argument("yaml", help="Pipeline YAML config file.")
    args = parser.parse_args()

    cfg = load_pipeline_config(args.yaml)
    run_full_pipeline(cfg)


if __name__ == "__main__":
    main()
