from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Dict, Optional, Type

from torchvision.models._api import WeightsEnum

from gfmtools.data.datasets.epa import EPALabels
from gfmtools.data.datasets.hlsburnscars import HLSBurnScars
from gfmtools.data.datasets.landfire import LandfireDataset

from gfmtools.models.resnet import ResNetEncoder
from gfmtools.models.swin import SwinEncoder
from gfmtools.models.dofa import DofaEncoder
from gfmtools.models.unet import unet


from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from gfmtools.data.datasets.epa import EPALabels
from gfmtools.data.datasets.hlsburnscars import HLSBurnScars
from gfmtools.data.datasets.landfire import LandfireDataset



DATASET_REGISTRY = {
    "EPALabels": EPALabels,
    "HLSBurnScars": HLSBurnScars,
    "LandfireDataset": LandfireDataset,
}


def _build_collate_fn(sample_schema: Dict[str, Any]):
    image_key = str(sample_schema.get("image_key", "image"))
    target_key = str(sample_schema.get("target_key", "target"))
    metadata_key = str(sample_schema.get("metadata_key", "metadata"))
    filename_key = str(sample_schema.get("filename_key", "filename"))

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}

        out: Dict[str, Any] = {}

        if image_key in batch[0]:
            first_image = batch[0][image_key]
            if isinstance(first_image, Mapping):
                image_dict: Dict[str, Any] = {}
                for k in first_image.keys():
                    image_dict[k] = default_collate([sample[image_key][k] for sample in batch])
                out["image"] = image_dict
            else:
                out["image"] = default_collate([sample[image_key] for sample in batch])

        if target_key in batch[0]:
            out["target"] = default_collate([sample[target_key] for sample in batch])

        if filename_key in batch[0]:
            out["filename"] = [sample[filename_key] for sample in batch]

        if metadata_key in batch[0]:
            metadata_batch = [sample[metadata_key] for sample in batch]
            if metadata_batch and isinstance(metadata_batch[0], Mapping):
                merged_meta: Dict[str, Any] = {}
                for k in metadata_batch[0].keys():
                    values = [m[k] for m in metadata_batch]
                    try:
                        merged_meta[k] = default_collate(values)
                    except Exception:
                        merged_meta[k] = values
                out["metadata"] = merged_meta
            else:
                out["metadata"] = metadata_batch

        for key in batch[0].keys():
            if key in {image_key, target_key, metadata_key, filename_key}:
                continue
            try:
                out[key] = default_collate([sample[key] for sample in batch])
            except Exception:
                out[key] = [sample[key] for sample in batch]

        return out

    return collate_fn


def build_test_loader(cfg: Dict[str, Any]) -> DataLoader:
    cfg = ensure_dict(cfg, "run_projections cfg")
    bundle_cfg = cfg.get("dataset_bundle")
    if not isinstance(bundle_cfg, dict):
        raise ValueError("cfg['dataset_bundle'] is required to build the test DataLoader")

    dataset_cls_name = bundle_cfg.get("dataset_cls_name")
    dataset_kwargs = dict(bundle_cfg.get("dataset_kwargs", {}))
    sample_schema = dict(bundle_cfg.get("sample_schema", {}))
    dataset_spec = dict(bundle_cfg.get("dataset_spec", {}))

    if not dataset_cls_name:
        raise ValueError("cfg['dataset_bundle']['dataset_cls_name'] is required")

    dataset_cls = DATASET_REGISTRY.get(str(dataset_cls_name))
    if dataset_cls is None:
        raise ValueError(
            f"Unsupported dataset_cls_name '{dataset_cls_name}'. "
            f"Register it in DATASET_REGISTRY before using build_test_loader()."
        )

    dataset_init_kwargs = dict(dataset_kwargs)

    split_key = sample_schema.get("split_arg_name", "split")
    dataset_init_kwargs[split_key] = sample_schema.get("test_split_value", "test")

    if "dataset_name" in dataset_spec and "dataset_name" not in dataset_init_kwargs:
        dataset_init_kwargs["dataset_name"] = dataset_spec["dataset_name"]

    if "root" in cfg and "root" not in dataset_init_kwargs:
        dataset_init_kwargs["root"] = cfg["root"]
    if "data_root" in cfg and "data_root" not in dataset_init_kwargs:
        dataset_init_kwargs["data_root"] = cfg["data_root"]

    dataset = dataset_cls(**dataset_init_kwargs)

    batch_size = int(cfg.get("test_batch_size", 1))
    num_workers = int(cfg.get("test_num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", True))
    persistent_workers = bool(cfg.get("persistent_workers", num_workers > 0))
    drop_last = bool(cfg.get("drop_last", False))

    collate_fn = _build_collate_fn(sample_schema)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class DatasetSpec:
    name: str
    dataset_name: Optional[str] = None
    split: Optional[str] = None
    task: Optional[str] = None
    instrument: Optional[str] = None
    bands: Optional[List[str]] = None
    target_name: Optional[str] = None
    num_classes: Optional[int] = None
    ignore_index: Optional[int] = None
    bandset: Optional[str] = None
    multi_image: Optional[bool] = None
    modalities: Optional[List[str]] = None
    resolution: Optional[float] = None
    data_root: Optional[str] = None
    init: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    dataset_spec: DatasetSpec
    dataset_cls: Type[Any]
    dataset_kwargs: Dict[str, Any]
    sample_schema: Dict[str, Any]
    collate_fn: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInitSpec:
    alias: str
    family: str
    variant: Optional[str] = None
    weights: Optional[str | WeightsEnum] = None
    pretrained: bool = True
    features_only: bool = True
    device: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelBundle:
    model_name: str
    model_spec: ModelInitSpec
    dataset_spec: DatasetSpec
    model_cls: Type[Any]
    model_kwargs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentBundle:
    dataset_bundle: DatasetBundle
    model_bundle: ModelBundle


DATASET_PRESETS: Dict[str, DatasetSpec] = {
    "epa": DatasetSpec(
        name="epa",
        dataset_name="epa",
        split="train",
        task="segmentation",
        instrument="L9",
        bands=["B2", "B3", "B4", "B5", "B6", "B7"],
        target_name="NLCD",
        bandset="landsat-optical-6",
        modalities=["optical"],
        resolution=30,
    ),
    "hlsburnscars": DatasetSpec(
        name="hlsburnscars",
        dataset_name="hlsburnscars",
        split="train",
        task="segmentation",
        instrument="HLS",
        bands=["B2", "B3", "B4", "B5", "B6", "B7"],
        target_name="burnscar",
        bandset="hls-optical-6",
        modalities=["optical"],
        resolution=30,
    ),
    "landfire": DatasetSpec(
        name="landfire",
        dataset_name="landfire",
        split="train",
        task="segmentation",
        instrument="HLS",
        target_name="fbfm40",
        modalities=["optical"],
        resolution=30,
    ),
}




DATASET_CLASS_REGISTRY: dict[str, Type[Any]] = {
    "EPALabels": EPALabels,
    "HLSBurnScars": HLSBurnScars,
    "LandfireDataset": LandfireDataset,
}

# --- model aliases ---
MODEL_ALIASES: dict[str, dict[str, Any]] = {
    "resnet18": {"family": "resnet", "variant": "resnet18"},
    "resnet50": {"family": "resnet", "variant": "resnet50"},
    "resnet152": {"family": "resnet", "variant": "resnet152"},
    "swinv2t": {"family": "swin", "variant": "t"},
    "swinv2b": {"family": "swin", "variant": "b"},
    "dofa_base": {"family": "dofa", "variant": "base"},
    "dofa_large": {"family": "dofa", "variant": "large"},
    "croma_base": {"family": "croma", "variant": "base"},
    "croma_large": {"family": "croma", "variant": "large"},
    "unet": {"family": "unet", "variant": None},
}


# --- model registry ---
MODEL_CLASS_REGISTRY: dict[str, Type[Any]] = {
    "ResNetEncoder": ResNetEncoder,
    "SwinEncoder": SwinEncoder,
    "DofaEncoder": DofaEncoder,
    "CROMAEncoder": CROMAEncoder,
    "unet": unet,
}



def resolve_dataset_spec(dataset: str, overrides: Any) -> DatasetSpec:
    if dataset not in DATASET_PRESETS:
        raise ValueError(f"Unknown dataset preset: {dataset}")

    spec = DatasetSpec(**asdict(DATASET_PRESETS[dataset]))
    for k, v in (overrides or {}).items():
        if hasattr(spec, k) and v is not None:
            setattr(spec, k, v)
        elif v is not None:
            spec.extra[k] = v
    return spec


def build_dataset_spec(dataset_cfg: Dict[str, Any], dataset_init: Dict[str, Any]) -> DatasetSpec:
    merged = merge_dicts(dataset_cfg, {"init": dataset_init})

    return DatasetSpec(
        name=merged.get("name", merged.get("dataset_name")),
        dataset_name=merged.get("dataset_name", merged.get("name")),
        split=merged.get("split", "test"),
        task=merged.get("task", "segmentation"),
        instrument=merged.get("instrument"),
        bands=merged.get("bands"),
        target_name=merged.get("target_name", merged.get("targetname")),
        num_classes=merged.get("num_classes", merged.get("numclasses")),
        ignore_index=merged.get("ignore_index", merged.get("ignoreindex")),
        bandset=merged.get("bandset"),
        multi_image=merged.get("multi_image", merged.get("multiimage")),
        modalities=merged.get("modalities"),
        resolution=merged.get("resolution"),
        data_root=merged.get("data_root", merged.get("dataroot")),
        init=merged.get("init", {}),
        extra={k: v for k, v in merged.items() if k not in {
            "name", "dataset_name", "split", "task", "instrument", "bands", "target_name",
            "targetname", "num_classes", "numclasses", "ignore_index", "ignoreindex",
            "bandset", "multi_image", "multiimage", "modalities", "resolution",
            "data_root", "dataroot", "init"
        }},
    )


def default_loader_schema(
    *,
    image_key: str = "image",
    target_key: str = "target",
    metadata_key: str = "metadata",
    filename_key: str = "filename",
    split_arg_name: str = "split",
    test_split_value: str = "test",
) -> Dict[str, Any]:
    return {
        "image_key": image_key,
        "target_key": target_key,
        "metadata_key": metadata_key,
        "filename_key": filename_key,
        "split_arg_name": split_arg_name,
        "test_split_value": test_split_value,
    }


def build_dataset_bundle(spec: DatasetSpec) -> DatasetBundle:
    dataset_name = spec.dataset_name or spec.name
    init = spec.init or {}

    common_sample_schema = {
        "task": spec.task,
        "instrument": spec.instrument,
        "bands": spec.bands or [],
        "target_name": spec.target_name,
        "modalities": spec.modalities or [],
        "multi_image": bool(spec.multi_image) if spec.multi_image is not None else False,
        **default_loader_schema(),
    }

    if dataset_name == "hlsburnscars":
        dataset_kwargs = {
            "dataroot": spec.data_root or "/mnt/data/rdemilth/hlsburnscars",
            "label_name": init.get("label_name", "burnscar"),
            "split": init.get("split", spec.split or "test"),
            "crs": init.get("crs", "EPSG:5070"),
            "additional_instruments": init.get("additional_instruments", None),
        }
        sample_schema = {
            **common_sample_schema,
            "filename_key": init.get("filename_key", "filename"),
            "metadata_filename_key": init.get("metadata_filename_key", "image_filename"),
        }
        metadata = {
            "dataset_name": dataset_name,
            "iterable_style": "dict_sample",
        }
        return DatasetBundle(
            dataset_spec=spec,
            dataset_cls=HLSBurnScars,
            dataset_kwargs=dataset_kwargs,
            sample_schema=sample_schema,
            metadata=metadata,
        )

    if dataset_name == "epa":
        dataset_kwargs = {
            "dataroot": spec.data_root or "/mnt/data/mtruong",
            "split": init.get("split", spec.split or "test"),
            "split_file": init.get("split_file", "/home/rdemilt/sparkbenchmark/epa_splits.yml"),
            "regions": init.get("regions", None),
            "img_size": init.get("img_size", 128),
            "gsd": init.get("gsd", 30),
            "interpolation": init.get("interpolation", "bilinear"),
            "included_time_points": init.get("included_time_points", 0),
            "transform": init.get("transform", None),
            "sampling_algorithm": init.get("sampling_algorithm", None),
            "seed": init.get("seed", 1917),
            "sample_limit": init.get("sample_limit", 5000),
        }
        sample_schema = {
            **common_sample_schema,
            "filename_key": init.get("filename_key", "filename"),
            "metadata_filename_key": init.get("metadata_filename_key", "image_filename"),
        }
        metadata = {
            "dataset_name": dataset_name,
            "iterable_style": "dict_sample",
        }
        return DatasetBundle(
            dataset_spec=spec,
            dataset_cls=EPALabels,
            dataset_kwargs=dataset_kwargs,
            sample_schema=sample_schema,
            metadata=metadata,
        )

    if dataset_name == "landfire":
        dataset_kwargs = {
            "dataroot": spec.data_root or "/mnt/data/rdemilt/fbfm40",
            "targets": init.get("targets", "fbfm40"),
            "crs": init.get("crs", "EPSG:5070"),
            "instruments": init.get("instruments", "HLS"),
            "split": init.get("split", spec.split or "test"),
        }
        sample_schema = {
            **common_sample_schema,
            "instrument": init.get("instruments", "HLS"),
            "target_name": init.get("targets", "fbfm40"),
            "filename_key": init.get("filename_key", "filename"),
            "metadata_filename_key": init.get("metadata_filename_key", "image_filename"),
        }
        metadata = {
            "dataset_name": dataset_name,
            "warning": "LandfireDataset is a stub and not fully iterable.",
            "iterable_style": "dict_sample",
        }
        return DatasetBundle(
            dataset_spec=spec,
            dataset_cls=LandfireDataset,
            dataset_kwargs=dataset_kwargs,
            sample_schema=sample_schema,
            metadata=metadata,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")



def resolve_model_spec(
    model_name: str,
    shared_model_init: Optional[Dict[str, Any]] = None,
    per_model_cfg: Optional[Dict[str, Any]] = None,
) -> ModelInitSpec:
    if model_name not in MODEL_ALIASES:
        raise ValueError(f"Unknown model alias: {model_name}")
    shared_model_init = shared_model_init or {}
    per_model_cfg = per_model_cfg or {}
    merged = merge_dicts(shared_model_init, per_model_cfg)
    alias_info = MODEL_ALIASES[model_name]
    return ModelInitSpec(
        alias=model_name,
        family=alias_info["family"],
        variant=alias_info["variant"],
        weights=merged.get("weights"),
        pretrained=merged.get("pretrained", True),
        features_only=merged.get("features_only", True),
        device=merged.get("device"),
        extra={k: v for k, v in merged.items() if k not in {"weights", "pretrained", "features_only", "device"}},
    )


def infer_dofa_wavelengths(dataset_spec: DatasetSpec) -> Optional[Dict[str, Dict[str, float]]]:
    if dataset_spec.instrument == "HLS":
        return {
            "S2": {
                "B2": 0.49,
                "B3": 0.56,
                "B4": 0.665,
                "B5": 0.705,
                "B6": 0.74,
                "B7": 0.783,
            }
        }
    if dataset_spec.instrument in {"L9", "LANDSAT"}:
        return {
            "L9": {
                "B2": 0.49,
                "B3": 0.56,
                "B4": 0.665,
                "B5": 0.864,
                "B6": 1.61,
                "B7": 2.20,
            }
        }
    return None


def build_model_bundle(model_spec: ModelInitSpec, dataset_bundle: DatasetBundle) -> ModelBundle:
    ds = dataset_bundle.dataset_spec

    if model_spec.family == "resnet":
        model_cls = ResNetEncoder
        model_kwargs = {
            "weights": model_spec.weights,
            "pretrained": model_spec.pretrained,
            "backbone": model_spec.variant,
            "gap_features": model_spec.features_only,
            **model_spec.extra,
        }
        metadata = {"expected_instrument": ds.instrument, "expected_bands": ds.bands}

    elif model_spec.family == "swin":
        band_set = ds.bandset or ("sentinel2" if ds.instrument == "HLS" else "rgb")
        model_cls = SwinEncoder
        model_kwargs = {
            "weights": model_spec.weights,
            "version": "v2",
            "size": model_spec.variant.upper() if model_spec.variant else "B",
            "band_set": band_set,
            "multi_image": bool(model_spec.extra.get("multi_image", False)),
            "pretrained": model_spec.pretrained,
            "gap_features": model_spec.features_only,
        }
        metadata = {
            "expected_instrument": ds.instrument,
            "expected_bands": ds.bands,
            "band_set": band_set,
        }

    elif model_spec.family == "dofa":
        wavelengths = model_spec.extra.get("wavelengths", infer_dofa_wavelengths(ds))
        model_cls = DofaEncoder
        model_kwargs = {
            "weights": model_spec.weights or "\.\./checkpoints/dofa/",
            "configuration": model_spec.variant or "base",
            "wavelengths": wavelengths,
        }
        metadata = {
            "expected_instrument": ds.instrument,
            "expected_bands": ds.bands,
            "wavelengths": wavelengths,
        }

    elif model_spec.family == "unet":
        model_cls = unet
        model_kwargs = {
            "weights": model_spec.weights,
            "classes": model_spec.extra.get("nclasses", ds.num_classes or 1),
        }
        metadata = {
            "expected_instrument": ds.instrument,
            "expected_bands": ds.bands,
            "num_classes": model_kwargs["classes"],
        }

    elif model_spec.family == "croma":
        model_cls = CROMAEncoder
        modalities = model_spec.extra.get("modalities", ["optical"])

        model_kwargs = {
            "weights": model_spec.weights,
            "configuration": model_spec.variant or "base",
            "modalities": modalities,
            "encoder_dim": model_spec.extra.get("encoder_dim", 768 if (model_spec.variant or "base") == "base" else 1024),
            "encoder_depth": model_spec.extra.get("encoder_depth", 12 if (model_spec.variant or "base") == "base" else 24),
            "num_heads": model_spec.extra.get("num_heads", 16),
            "patch_size": model_spec.extra.get("patch_size", 8),
            "image_size": model_spec.extra.get("image_size", 120),
        }

        metadata = {
            "expected_instrument": ds.instrument,
            "expected_bands": ds.bands,
            "modalities": modalities,
            "configuration": model_kwargs["configuration"],
        }

    else:
        raise ValueError(f"Unsupported model family: {model_spec.family}")

    return ModelBundle(
        model_name=model_spec.alias,
        model_spec=model_spec,
        dataset_spec=ds,
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        metadata=metadata,
    )


def instantiate_dataset(dataset_bundle: DatasetBundle) -> Any:
    return dataset_bundle.dataset_cls(**dataset_bundle.dataset_kwargs)


def instantiate_model(model_bundle: ModelBundle) -> Any:
    return model_bundle.model_cls(**model_bundle.model_kwargs)


def _to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_plain_dict(v) for v in value)
    if isinstance(value, type):
        return value.__name__
    if callable(value):
        return getattr(value, "__name__", str(value))
    if isinstance(value, WeightsEnum):
        return str(value)
    return value


def serialize_dataset_bundle(dataset_bundle: DatasetBundle) -> Dict[str, Any]:
    return {
        "dataset_spec": _to_plain_dict(dataset_bundle.dataset_spec),
        "dataset_cls_name": dataset_bundle.dataset_cls.__name__,
        "dataset_kwargs": _to_plain_dict(dataset_bundle.dataset_kwargs),
        "sample_schema": _to_plain_dict(dataset_bundle.sample_schema),
        "collate_fn_name": getattr(dataset_bundle.collate_fn, "__name__", None),
        "metadata": _to_plain_dict(dataset_bundle.metadata),
    }


def serialize_model_bundle(model_bundle: ModelBundle) -> Dict[str, Any]:
    return {
        "model_name": model_bundle.model_name,
        "model_spec": _to_plain_dict(model_bundle.model_spec),
        "dataset_spec": _to_plain_dict(model_bundle.dataset_spec),
        "model_cls_name": model_bundle.model_cls.__name__,
        "model_kwargs": _to_plain_dict(model_bundle.model_kwargs),
        "metadata": _to_plain_dict(model_bundle.metadata),
    }


def dataset_bundle_from_serialized(bundle_cfg: Dict[str, Any]) -> DatasetBundle:
    spec = DatasetSpec(**bundle_cfg["dataset_spec"])
    cls_name = bundle_cfg["dataset_cls_name"]
    if cls_name not in DATASET_CLASS_REGISTRY:
        raise ValueError(f"Unknown dataset class name in bundle: {cls_name}")
    return DatasetBundle(
        dataset_spec=spec,
        dataset_cls=DATASET_CLASS_REGISTRY[cls_name],
        dataset_kwargs=copy.deepcopy(bundle_cfg["dataset_kwargs"]),
        sample_schema=copy.deepcopy(bundle_cfg["sample_schema"]),
        collate_fn=None,
        metadata=copy.deepcopy(bundle_cfg.get("metadata", {})),
    )


def model_bundle_from_serialized(bundle_cfg: Dict[str, Any]) -> ModelBundle:
    ds = DatasetSpec(**bundle_cfg["dataset_spec"])
    model_spec = ModelInitSpec(**bundle_cfg["model_spec"])
    cls_name = bundle_cfg["model_cls_name"]
    if cls_name not in MODEL_CLASS_REGISTRY:
        raise ValueError(f"Unknown model class name in bundle: {cls_name}")
    return ModelBundle(
        model_name=bundle_cfg["model_name"],
        model_spec=model_spec,
        dataset_spec=ds,
        model_cls=MODEL_CLASS_REGISTRY[cls_name],
        model_kwargs=copy.deepcopy(bundle_cfg["model_kwargs"]),
        metadata=copy.deepcopy(bundle_cfg.get("metadata", {})),
    )


def initialize_experiment(
    model_name: str,
    dataset_cfg: Dict[str, Any],
    dataset_init: Optional[Dict[str, Any]] = None,
    shared_model_init: Optional[Dict[str, Any]] = None,
    per_model_cfg: Optional[Dict[str, Any]] = None,
) -> ExperimentBundle:
    dataset_spec = build_dataset_spec(dataset_cfg, dataset_init)
    dataset_bundle = build_dataset_bundle(dataset_spec)
    model_spec = resolve_model_spec(model_name, shared_model_init, per_model_cfg)
    model_bundle = build_model_bundle(model_spec, dataset_bundle)
    return ExperimentBundle(dataset_bundle=dataset_bundle, model_bundle=model_bundle)
