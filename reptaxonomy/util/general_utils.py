
import yaml



def read_yaml(fpath_yaml):
    yml_conf = None
    with open(fpath_yaml) as f_yaml:
        yml_conf = yaml.load(f_yaml, Loader=yaml.FullLoader)
    return yml_conf

def resolve_model_names(cfg: DictConfig) -> List[str]:
    model_names = OmegaConf.to_container(getattr(cfg, "model_names", None), resolve=True)
    if not model_names or not isinstance(model_names, list):
        raise ValueError("cfg['model_names'] must be provided as a non-empty list")
    model_names = [str(x) for x in model_names]
    if len(model_names) < 2:
        raise ValueError("cfg['model_names'] must contain at least two models")
    return model_names


def resolve_projections(cfg: DictConfig) -> List[str]:
    projections = OmegaConf.to_container(getattr(cfg, "projections", ["umap", "tsne", "pca"]), resolve=True)
    if not projections or not isinstance(projections, list):
        raise ValueError("cfg['projections'] must be a non-empty list")
    return [str(x).lower() for x in projections]
