
import yaml

from typing import Any


def read_yaml(fpath_yaml):
    yml_conf = None
    with open(fpath_yaml) as f_yaml:
        yml_conf = yaml.load(f_yaml, Loader=yaml.FullLoader)
    return yml_conf


def resolve_dataset_name(cfg: dict[str, Any]) -> str:
    dataset_bundle = cfg.get("dataset_bundle")
    if isinstance(dataset_bundle, dict):
        dataset_spec = dataset_bundle.get("dataset_spec", {})
        if isinstance(dataset_spec, dict):
            name = dataset_spec.get("dataset_name") or dataset_spec.get("name")
            if name:
                return str(name)

    dataset_cfg = cfg.get("dataset")
    if isinstance(dataset_cfg, dict):
        name = dataset_cfg.get("dataset_name") or dataset_cfg.get("name")
        if name:
            return str(name)

    if isinstance(dataset_cfg, str):
        return dataset_cfg

    raise ValueError("Could not resolve dataset name from cfg['dataset_bundle'] or cfg['dataset'].")


def resolve_model_names(cfg: dict[str, Any]) -> list[str]:
    model_names_cfg = cfg.get("model_names")
    if isinstance(model_names_cfg, (str, Path)):
        model_names = load_json(Path(model_names_cfg))
    elif isinstance(model_names_cfg, list):
        model_names = model_names_cfg
    else:
        raise ValueError("model_names must be either a list or a path to a JSON file")

    if not isinstance(model_names, list) or not model_names:
        raise ValueError("model_names must resolve to a non-empty list")

    return [str(x) for x in model_names]

