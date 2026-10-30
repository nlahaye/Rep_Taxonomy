"""Hooks that plug the gfmtools representation sets / encoders into
``reptaxonomy.util.experiment_init_utils`` (dataset + model bundles).

* Dataset: a pipeline cfg whose ``dataset`` block carries ``gfm_repset:
  <FTWRepSet|EPARepSet|MADOSRepSet>`` is routed to ``build_gfm_dataset_bundle``;
  ``dataset_init`` holds the wrapper kwargs.
* Model: any name in ``reptaxonomy.gfm.catalog`` (``<arch>__<weights>``,
  ``<arch>__scratch``, ``<key>__ft__<tag>``) resolves to family ``gfm`` and is
  built as ``GFMEncoder``.
"""

from __future__ import annotations

from typing import Any, Dict

from reptaxonomy.gfm.catalog import is_gfm_model
from reptaxonomy.gfm.datasets import IGNORE_INDEX, LABEL_KEY, REPSET_CLASSES

_REPSET_INFO = {
    "FTWRepSet": {"instrument": "Sentinel-2", "sensor": "s2", "gsd": 10.0},
    "EPARepSet": {"instrument": "S2L2A", "sensor": "s2", "gsd": 10.0},
    "MADOSRepSet": {"instrument": "Sentinel-2", "sensor": "s2", "gsd": 10.0, "raw_scale": 1e4},
}


def is_gfm_dataset_spec(spec: Any) -> bool:
    return bool((getattr(spec, "extra", None) or {}).get("gfm_repset"))


def build_gfm_dataset_bundle(spec: Any):
    from reptaxonomy.util.experiment_init_utils import DatasetBundle, default_loader_schema

    extra = dict(spec.extra or {})
    cls_name = extra["gfm_repset"]
    cls = REPSET_CLASSES[cls_name]
    kwargs = dict(spec.init or {})
    info = dict(_REPSET_INFO[cls_name])
    if cls_name == "EPARepSet":
        inst = kwargs.get("instrument", "S2L2A")
        info["instrument"] = inst
        if not str(inst).upper().startswith("S2"):
            info.update(sensor="oli", gsd=30.0)
    if cls_name == "FTWRepSet":
        rs = float(kwargs.get("reflectance_scale", 1.0))
        info["raw_scale"] = 1.0 / rs if rs else 1.0
    bands = extra.get("source_bands") or spec.bands
    sample_schema = {
        "task": spec.task or "segmentation",
        "instrument": info["instrument"],
        "bands": list(bands or []),
        "target_name": LABEL_KEY,
        "modalities": [],
        "multi_image": False,
        **default_loader_schema(split_arg_name="split", test_split_value=kwargs.get("split")),
        "passthrough_keys": [],
    }
    metadata = {
        "dataset_name": spec.dataset_name or spec.name,
        "iterable_style": "dict_sample",
        "gfm_repset": cls_name,
        "source_bands": list(bands or []),
        "source_sensor": info["sensor"],
        "gsd": info["gsd"],
        "raw_scale": info.get("raw_scale", 1.0),
        "band_stats": extra.get("band_stats") or {},
        "num_classes": spec.num_classes,
        "ignore_index": spec.ignore_index if spec.ignore_index is not None else IGNORE_INDEX,
    }
    spec.instrument = info["instrument"]
    spec.target_name = LABEL_KEY
    if spec.ignore_index is None:
        spec.ignore_index = IGNORE_INDEX
    return DatasetBundle(dataset_spec=spec, dataset_cls=cls, dataset_kwargs=kwargs,
                         sample_schema=sample_schema, metadata=metadata)


def gfm_model_alias(name: str) -> Dict[str, Any] | None:
    return {"family": "gfm", "variant": name} if is_gfm_model(name) else None


def build_gfm_model_bundle(model_spec: Any, dataset_bundle: Any):
    from reptaxonomy.gfm.encoders import GFMEncoder
    from reptaxonomy.util.experiment_init_utils import ModelBundle

    md = dataset_bundle.metadata or {}
    kwargs: Dict[str, Any] = {
        "model_name": model_spec.variant or model_spec.alias,
        "source_bands": md.get("source_bands") or None,
        "source_sensor": md.get("source_sensor"),
        "band_stats": md.get("band_stats") or {},
        "gsd": md.get("gsd", 10.0),
        "raw_scale": md.get("raw_scale", 1.0),
        "pretrained": bool(model_spec.pretrained),
    }
    kwargs.update({k: v for k, v in (model_spec.extra or {}).items() if v is not None})
    if model_spec.device:
        kwargs["device"] = model_spec.device
    return ModelBundle(model_name=model_spec.alias, model_spec=model_spec,
                       dataset_spec=dataset_bundle.dataset_spec, model_cls=GFMEncoder,
                       model_kwargs=kwargs, metadata={"family": "gfm"})
