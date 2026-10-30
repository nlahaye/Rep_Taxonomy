"""Geographic location encoders (implicit neural representations) for the geolocated
intrinsic-dimension analysis.

Location encoders map (lon, lat) to an embedding without imagery. Their ID at the
dataset's own sample locations is a *location-only reference*: it measures how much
geographic structure a pre-trained Earth representation encodes at the places the
benchmark samples, which the image-encoder local ID can then be compared to.

Loaders follow GeoINRID (Rao et al., ICLR 2026, https://github.com/arjunarao619/GeoINRID):

* ``geoclip``             GeoCLIP location encoder (weights ship with GeoINRID / the ``geoclip`` pip package)
* ``csp_fmow``/``csp_inat`` CSP grid-cell encoders (weights ship with GeoINRID)
* ``satclip_<tag>``       SatCLIP via the microsoft/satclip code base + a checkpoint path

All encoders return float64 arrays (N, D); input is (N, 2) in **(lon, lat)** degrees.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

CSP_WEIGHTS = {
    "csp_fmow": "csp/weights/model_dir/model_fmow/model_fmow_gridcell_0.0010_32_0.1000000_1_512_gelu_UNSUPER-contsoftmax_0.000050_1.000_1_0.100_TMP1.0000_1.0000_1.0000.pth.tar",
    "csp_inat": "csp/weights/model_dir/model_inat_2018/model_inat_2018_gridcell_0.0010_32_0.1000000_1_512_leakyrelu_UNSUPER-contsoftmax_0.000500_1.000_1_1.000_TMP20.0000_1.0000_1.0000.pth.tar",
}


@contextlib.contextmanager
def _on_path(p: Path, purge: tuple = ()):
    """Temporarily prepend ``p`` to sys.path; drop clashing top-level modules
    (GeoINRID's sub-projects reuse names such as ``model``/``models``/``utils``)."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] in purge}
    sys.path.insert(0, str(p))
    try:
        yield
    finally:
        sys.path.remove(str(p))
        for k in [k for k in sys.modules if k.split(".")[0] in purge]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)


class LocationEncoder:
    def __init__(self, name: str, fn: Callable[[torch.Tensor], torch.Tensor], device: torch.device,
                 order: str = "lonlat", batch: int = 8192) -> None:
        self.name, self.fn, self.device, self.order, self.batch = name, fn, device, order, batch

    @torch.no_grad()
    def __call__(self, lonlat: np.ndarray) -> np.ndarray:
        xy = np.asarray(lonlat, dtype=np.float32)
        if self.order == "latlon":
            xy = xy[:, [1, 0]]
        out = []
        for i in range(0, len(xy), self.batch):
            out.append(self.fn(torch.from_numpy(xy[i:i + self.batch]).to(self.device)).float().cpu().numpy())
        return np.concatenate(out, 0).astype(np.float64)


def _geoclip(root: Optional[Path], device):
    if root is not None and (root / "geoclip" / "model" / "location_encoder.py").exists():
        with _on_path(root / "geoclip", purge=("model",)):
            mod = importlib.import_module("model.location_encoder")
            enc = mod.LocationEncoder(sigma=[2 ** 0, 2 ** 4, 2 ** 8], from_pretrained=True)
    else:  # pip install geoclip
        from geoclip import LocationEncoder as _LE
        enc = _LE()
    enc = enc.to(device).eval()
    return LocationEncoder("geoclip", enc, device, order="latlon")  # GeoCLIP expects (lat, lon)


def _csp(name: str, root: Path, device, checkpoint: Optional[str] = None):
    ck = Path(checkpoint) if checkpoint else root / CSP_WEIGHTS[name]
    with _on_path(root / "csp" / "main", purge=("models", "module", "SpatialRelationEncoder", "utils")):
        models = importlib.import_module("models")
        sre = importlib.import_module("SpatialRelationEncoder")
        module = importlib.import_module("module")
        state = torch.load(ck, map_location=device, weights_only=False)
        p = state["params"]
        p["coord_dim"] = 2
        ffn = module.MultiLayerFeedForwardNN(
            input_dim=p["coord_dim"] * p["frequency_num"] * 2, output_dim=p["num_filts"],
            num_hidden_layers=p["num_hidden_layer"], dropout_rate=p["dropout"], hidden_dim=p["hidden_dim"],
            activation=p["spa_f_act"], use_layernormalize=p["use_layn"], skip_connection=p["skip_connection"],
            context_str="spa_enc")
        spa = sre.GridCellSpatialRelationEncoder(
            spa_embed_dim=p["num_filts"], coord_dim=p["coord_dim"], frequency_num=p["frequency_num"],
            max_radius=p["max_radius"], min_radius=p["min_radius"], freq_init=p["freq_init"], ffn=ffn,
            device=str(device))
        loc = models.LocationEncoder(spa_enc=spa, num_inputs=p["coord_dim"], num_classes=p["num_classes"],
                                     num_filts=p["num_filts"], num_users=p.get("num_users", 1))
        wrap = models.LocationImageEncoder(loc_enc=loc, train_loss=p["train_loss"], unsuper_loss=p["unsuper_loss"],
                                           cnn_feat_dim=p.get("cnn_feat_dim", 2048), spa_enc_type=p["spa_enc_type"])
        wrap.load_state_dict(state["state_dict"])
        enc = wrap.loc_enc.to(device).eval()
    return LocationEncoder(name, lambda x: enc(x, return_feats=True), device, order="lonlat")


def _satclip(name: str, satclip_repo: Path, checkpoint: str, device):
    with _on_path(satclip_repo / "satclip", purge=("load", "main", "model", "location_encoder", "positional_encoding")):
        load = importlib.import_module("load")
        enc = load.get_satclip(checkpoint, device=str(device)).to(device).eval()
    return LocationEncoder(name, lambda x: enc(x.double()), device, order="lonlat")


def load_location_encoder(name: str, geoinrid_root: Optional[str] = None, device: Optional[str] = None,
                          checkpoint: Optional[str] = None, satclip_repo: Optional[str] = None) -> LocationEncoder:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(geoinrid_root) / "src" / "intrinsic_dimension" if geoinrid_root else None
    if root is not None and not root.exists():
        root = Path(geoinrid_root)  # already pointing at src/intrinsic_dimension
    if name == "geoclip":
        return _geoclip(root, dev)
    if name in CSP_WEIGHTS:
        if root is None:
            raise ValueError("CSP needs geo_id.location_encoders.geoinrid_root")
        return _csp(name, root, dev, checkpoint)
    if name.startswith("satclip"):
        if not (satclip_repo and checkpoint):
            raise ValueError("SatCLIP needs satclip_repo (microsoft/satclip clone) and checkpoint")
        return _satclip(name, Path(satclip_repo), checkpoint, dev)
    raise KeyError(f"unknown location encoder {name!r}")


def available_location_encoders(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``geo_id.location_encoders`` -> {name: kwargs}; accepts a list of names or a mapping."""
    le = cfg.get("location_encoders") or {}
    if isinstance(le, dict) and "models" in le:
        common = {k: v for k, v in le.items() if k != "models"}
        models = le["models"]
    else:
        common, models = {}, le
    if isinstance(models, (list, tuple)):
        models = {m: {} for m in models}
    return {m: {**common, **(kw or {})} for m, kw in (models or {}).items()}
