"""Uniform rep_taxonomy adapter around every gfmtools encoder.

``GFMEncoder`` is registered in ``MODEL_CLASS_REGISTRY`` and exposes the
interface the existing rep_taxonomy pipelines already rely on:

* ``forward_instruments(batch, device=None, **_)`` -> ``(B, N, C)`` patch tokens
  with ``N = h*w`` square and no class token, so
  ``generate_embeddings.select_encoder_output`` reshapes them to ``C x h x w``
  exactly as it does for the existing encoders.
* ``forward_features(batch)`` -> ``(B, C, h, w)`` (used by fine-tuning).
* ``.backbone`` holds the underlying gfmtools / timm / torchvision module, so
  weight-matrix and resource analyses see the real parameters.

The gfmtools ``*Encoder.tensorize`` helpers are intentionally bypassed: they
assume instrument keys (``S2L2A``/``HLS``/``LANDSAT``) and full band sets that
FTW (``Sentinel-2``, 4 bands) and MADOS (``Sentinel-2``, 11 bands) do not
provide, and several have constructor bugs (see GFMTOOLS_NOTES.md). Instead the
adapter calls the gfmtools *builders* + weight enums directly and assembles the
input tensor by canonical band name (``reptaxonomy.gfm.bands``).
"""

from __future__ import annotations

import datetime as dt
import importlib
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from reptaxonomy.gfm import bands as B
from reptaxonomy.gfm.catalog import get_entry

logger = logging.getLogger(__name__)

_META_KEYS = {"bands", "grid", "time", "band_wavelength_nm", "platform"}


# ----------------------------------------------------------------------------- helpers
def _first_str(x: Any) -> Any:
    while isinstance(x, (list, tuple)) and x:
        x = x[0]
    return x


def batch_instrument(batch: Mapping[str, Any]) -> str:
    inst = _first_str(batch.get("instruments"))
    if not isinstance(inst, str):
        raise ValueError("Batch has no 'instruments' entry")
    return inst


def batch_band_names(batch: Mapping[str, Any], instrument: str) -> List[str]:
    entry = batch[instrument]
    return [k for k, v in entry.items() if k not in _META_KEYS and isinstance(v, Mapping) and "data" in v]


def stack_source(batch: Mapping[str, Any], instrument: str, band_names: Sequence[str],
                 time_index: int = -1) -> torch.Tensor:
    chans = []
    for b in band_names:
        d = batch[instrument][b]["data"]
        d = torch.as_tensor(d)
        if d.ndim == 2:      # unbatched (H, W)
            d = d.unsqueeze(0)
        if d.ndim == 4:      # (B, T, H, W) -> select time
            d = d[:, time_index]
        chans.append(d.float())
    x = torch.stack(chans, dim=1)
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _round_to(n: int, m: int) -> int:
    return max(m, (int(n) // m) * m)


def _get_enum_member(module: str, enum_name: str, weights: Optional[str]):
    if weights is None:
        return None
    mod = importlib.import_module(f"gfmtools.models.{module}")
    return getattr(getattr(mod, enum_name), weights)


def _parse_time(s: Any) -> Optional[dt.datetime]:
    if s is None:
        return None
    if isinstance(s, Mapping):
        a, b = _parse_time(s.get("start")), _parse_time(s.get("end"))
        if a and b:
            return a + (b - a) / 2
        return a or b
    s = str(s)
    if "/" in s:  # "start/end" time window (FTW) -> midpoint
        a, _, b = s.partition("/")
        return _parse_time({"start": a, "end": b or a})
    for cand in (s, s[:19], s[:10], s[:7] + "-15", s[:4] + "-07-01"):
        try:
            return dt.datetime.fromisoformat(cand)
        except ValueError:
            continue
    try:  # gfmtools time expressions ("2019/2020", "2019-04/2019-06" ...)
        from gfmtools.utils.time import TimeExpression
        start, end = TimeExpression.from_string(s).components[0].bounds()
        return start + (end - start) / 2
    except Exception:
        return None


# ----------------------------------------------------------------------------- adapter
class GFMEncoder(nn.Module):
    """rep_taxonomy-facing wrapper for a gfmtools encoder + weights."""

    def __init__(
        self,
        model_name: str,
        source_bands: Optional[Sequence[str]] = None,
        source_sensor: Optional[str] = None,
        band_stats: Optional[Dict[str, Sequence[float]]] = None,
        normalization: str = "dataset_zscore",
        fill: str = "zero",
        input_size: Optional[int] = None,
        time_index: int = -1,
        gsd: float = 10.0,
        raw_scale: float = 1.0,
        default_time: str = "2020-07-01",
        default_location: Sequence[float] = (0.0, 0.0),
        checkpoint: Optional[str] = None,
        strict_checkpoint: bool = False,
        pretrained: bool = True,
        feature_layer: int = -1,
        unet_feature: str = "encoder",
        olmoearth_cache_dir: Optional[str] = None,
        olmoearth_patch_size: int = 8,
        device: Optional[str] = None,
        **unused: Any,
    ) -> None:
        super().__init__()
        self.entry = get_entry(model_name)
        self.model_name = model_name
        self.family = self.entry["family"]
        self.source_bands = list(source_bands) if source_bands else None
        self.source_sensor = source_sensor
        self.band_stats = {str(k): [float(v[0]), float(v[1])] for k, v in (band_stats or {}).items()}
        self.normalization = normalization
        self.fill = fill
        self.input_size = input_size if input_size is not None else self.entry.get("input_size")
        self.time_index = int(time_index)
        self.gsd = float(gsd)
        self.raw_scale = float(raw_scale)
        self.default_time = default_time
        self.default_location = [float(v) for v in default_location]
        self.feature_layer = int(feature_layer)
        self.unet_feature = unet_feature
        self.olmoearth_patch_size = int(olmoearth_patch_size)
        self.pretrained = bool(pretrained) and self.entry.get("weights") is not None
        self.target_canonical = B.canonical_bands(self.entry["bands"], self.entry["sensor"])
        self._plan: Optional[B.BandPlan] = None
        self._warned_stats = False
        if unused:
            logger.debug("GFMEncoder(%s) ignoring kwargs %s", model_name, sorted(unused))

        self.backbone = self._build_backbone(olmoearth_cache_dir)
        self.embed_dim: Optional[int] = None
        if checkpoint:
            self.load_finetuned(checkpoint, strict=strict_checkpoint)
        self.eval()

    # ------------------------------------------------------------------ build
    def _build_backbone(self, olmoearth_cache_dir: Optional[str]) -> nn.Module:
        e = self.entry
        fam = self.family
        w = _get_enum_member(e["module"], e["enum"], e["weights"]) if self.pretrained and fam != "olmoearth" else None
        mod = importlib.import_module(f"gfmtools.models.{e['module']}")
        n_in = len(e["bands"])

        if fam == "resnet":
            if w is not None:
                return getattr(mod, e["builder"])(w)
            import timm
            return timm.create_model(e["arch"], in_chans=n_in, num_classes=0, pretrained=False)

        if fam == "swin":
            if w is not None:
                return getattr(mod, e["builder"])(w)
            import torchvision
            m = getattr(torchvision.models, e["arch"])(weights=None)
            conv = m.features[0][0]
            if conv.in_channels != n_in:
                m.features[0][0] = nn.Conv2d(n_in, conv.out_channels, conv.kernel_size, conv.stride)
            return m

        if fam == "dofa":
            # The released DOFA checkpoints carry no final norm (neither ``norm`` nor
            # ``fc_norm``); gfmtools' builder only tolerates a missing fc_norm, so build
            # with global_pool=True and switch to token output with an identity-initialised
            # LayerNorm (plain per-token standardisation, no learned affine).
            m = getattr(mod, e["builder"])(w, global_pool=True)
            m.global_pool = False
            m.norm = nn.LayerNorm(m.embed_dim, eps=1e-6)
            if hasattr(m, "fc_norm"):
                del m.fc_norm
            return m

        if fam == "croma":
            return getattr(mod, e["builder"])(w, modalities=["optical"], image_size=int(self.input_size))

        if fam == "scalemae":
            m = getattr(mod, e["builder"])(w)
            m.res = self.gsd
            return m

        if fam == "unet":
            if w is not None:
                return mod.unet(w)
            import segmentation_models_pytorch as smp
            return smp.Unet(encoder_name=e.get("unet_encoder") or "efficientnet-b3", encoder_weights=None,
                            in_channels=n_in, classes=int(e.get("unet_classes") or 3))

        if fam == "prithvi":
            cfg = dict(e["prithvi_cfg"])
            ps = cfg.get("patch_size", (1, 16, 16))
            return mod.PrithviEncoder(
                weights=w,
                image_size=int(cfg.get("image_size", 224)),
                patch_size=tuple(ps) if isinstance(ps, (list, tuple)) else ps,
                num_frames=int(cfg.get("num_frames", 1)),
                in_chans=int(cfg.get("in_chans", 6)),
                embed_dim=int(cfg.get("embed_dim", 768)),
                depth=int(cfg.get("depth", 12)),
                num_heads=int(cfg.get("num_heads", 12)),
                decoder_embed_dim=int(cfg.get("decoder_embed_dim", 512)),
                decoder_depth=int(cfg.get("decoder_depth", 8)),
                decoder_num_heads=int(cfg.get("decoder_num_heads", 16)),
                mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
                coords_encoding=list(cfg.get("coords_encoding") or []),
                coords_scale_learn=bool(cfg.get("coords_scale_learn", False)),
                encoder_only=True,
            )

        if fam == "olmoearth":
            artifact = getattr(mod, e["weights"])
            return mod.OlmoEarthModel(artifact=artifact, patch_size=self.olmoearth_patch_size,
                                      input_res=self.gsd, normalize=True,
                                      cache_dir=olmoearth_cache_dir, load_weights=bool(pretrained_flag(self)))
        raise ValueError(f"Unsupported family {fam}")

    def load_finetuned(self, path: str, strict: bool = False) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, Mapping) and "backbone" in state:
            state = state["backbone"]
        missing, unexpected = self.backbone.load_state_dict(state, strict=strict)
        logger.info("Loaded fine-tuned backbone %s (missing=%d unexpected=%d)", path, len(missing), len(unexpected))

    # ------------------------------------------------------------------ inputs
    def _source(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, List[str], List[str]]:
        inst = batch_instrument(batch)
        names = self.source_bands or batch_band_names(batch, inst)
        sensor = self.source_sensor or B.sensor_of(inst)
        x = stack_source(batch, inst, names, self.time_index)
        return x, names, B.canonical_bands(names, sensor)

    def _standardise(self, x: torch.Tensor, names: Sequence[str]) -> torch.Tensor:
        if self.normalization == "none":
            return x
        if self.normalization == "dataset_zscore" and all(n in self.band_stats for n in names):
            mean = torch.tensor([self.band_stats[n][0] for n in names], dtype=x.dtype, device=x.device)
            std = torch.tensor([max(self.band_stats[n][1], 1e-6) for n in names], dtype=x.dtype, device=x.device)
            return (x - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        if not self._warned_stats:
            logger.warning("%s: band_stats missing for %s; using per-image z-scoring (removes radiometric "
                           "shift signal - run gfm_experiment so stats are computed)", self.model_name, list(names))
            self._warned_stats = True
        mu = x.mean(dim=(2, 3), keepdim=True)
        sd = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd

    def band_plan(self, source_canonical: Sequence[str]) -> B.BandPlan:
        if self.family == "dofa":
            optical = [c for c in source_canonical if B.CANONICAL_WAVELENGTH_UM.get(c, -1) > 0]
            return B.plan_bands(source_canonical, optical, allow_substitutes=False)
        return B.plan_bands(source_canonical, self.target_canonical)

    def band_coverage(self, source_bands: Sequence[str], source_sensor: str) -> float:
        canon = B.canonical_bands(source_bands, source_sensor)
        if self.family == "dofa":
            return 1.0 if any(B.CANONICAL_WAVELENGTH_UM.get(c, -1) > 0 for c in canon) else 0.0
        return B.plan_bands(canon, self.target_canonical).coverage

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        if self.input_size is None:
            mult = 32 if self.family in ("resnet", "swin", "unet") else 1
            th, tw = _round_to(H, mult), _round_to(W, mult)
        else:
            th = tw = int(self.input_size)
        if (th, tw) != (H, W):
            x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        return x

    def prepare(self, batch: Mapping[str, Any], device: Optional[torch.device] = None):
        x, names, canon = self._source(batch)
        if device is not None:
            x = x.to(device)
        if self.family == "olmoearth":
            return x * self.raw_scale, names, canon
        x = self._standardise(x, names)
        plan = self.band_plan(canon)
        self._plan = plan
        x = B.assemble_bands(x, canon, plan, fill=self.fill)
        return self._resize(x), names, [plan.target_canonical[i] for i in range(len(plan.target_canonical))]

    # ------------------------------------------------------------------ forward
    def _tokens_to_grid(self, t: torch.Tensor, drop_cls: bool) -> torch.Tensor:
        if drop_cls:
            t = t[:, 1:, :]
        n = t.shape[1]
        s = int(round(math.sqrt(n)))
        if s * s != n:
            raise ValueError(f"{self.model_name}: {n} tokens are not a square grid")
        return t.transpose(1, 2).reshape(t.shape[0], t.shape[2], s, s)

    def _prithvi_coords(self, batch: Mapping[str, Any], bsz: int, device) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # follow what the built backbone actually expects (cfg may enable time/location)
        vit = getattr(self.backbone, "encoder", None)
        vit = getattr(vit, "encoder", vit)
        enc = []
        if getattr(vit, "temporal_encoding", False):
            enc.append("time")
        if getattr(vit, "location_encoding", False):
            enc.append("location")
        meta = batch.get("rt_meta", {}) or {}
        tc = lc = None
        if "time" in enc:
            times = meta.get("time") or [None] * bsz
            rows = []
            for t in list(times)[:bsz]:
                d = _parse_time(t) or _parse_time(self.default_time)
                rows.append([[float(d.year), float(d.timetuple().tm_yday)]])
            tc = torch.tensor(rows, dtype=torch.float32, device=device)
        if "location" in enc:
            lat = meta.get("lat")
            lon = meta.get("lon")
            vals = []
            for i in range(bsz):
                try:
                    a, o = float(lat[i]), float(lon[i])
                    if math.isnan(a) or math.isnan(o):
                        raise ValueError
                except Exception:
                    a, o = self.default_location
                vals.append([a, o])
            lc = torch.tensor(vals, dtype=torch.float32, device=device)
        return tc, lc

    def _olmoearth_features(self, x_raw: torch.Tensor, canon: Sequence[str], batch: Mapping[str, Any]) -> torch.Tensor:
        from olmoearth_pretrain_minimal import Normalizer
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample

        model = self.backbone
        plan = B.plan_bands(canon, self.target_canonical)
        self._plan = plan
        x = B.assemble_bands(x_raw, canon, plan, fill="zero")  # raw units, missing = 0
        H, W = x.shape[-2:]
        p = self.olmoearth_patch_size
        th, tw = _round_to(H, p), _round_to(W, p)
        if (th, tw) != (H, W):
            x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
        bsz = x.shape[0]
        x = x.permute(0, 2, 3, 1).unsqueeze(3)  # B,H,W,T=1,C
        normed = Normalizer(std_multiplier=2.0).normalize(Modality.SENTINEL2_L2A, x.detach().cpu().numpy())
        x = torch.as_tensor(normed, dtype=torch.float32, device=x_raw.device)
        missing = [i for i, s in enumerate(plan.source_index) if s is None]
        if missing:
            x[..., missing] = 0.0  # missing bands sit at the normaliser mean
        meta = batch.get("rt_meta", {}) or {}
        times = list(meta.get("time") or [None] * bsz)[:bsz]
        ts = []
        for t in times:
            d = _parse_time(t) or _parse_time(self.default_time)
            ts.append([[d.day, d.month - 1, d.year]])
        timestamps = torch.tensor(ts, dtype=torch.long, device=x.device)
        n_bs = model._num_bandsets(Modality.SENTINEL2_L2A)
        mask = torch.zeros((*x.shape[:4], n_bs), dtype=torch.long, device=x.device)
        sample = MaskedOlmoEarthSample(sentinel2_l2a=x.contiguous(), sentinel2_l2a_mask=mask, timestamps=timestamps)
        out = model.encoder(sample, patch_size=model.patch_size, input_res=model.input_res, fast_pass=model.fast_pass)
        tam = out.get("tokens_and_masks", out) if isinstance(out, Mapping) else out
        tok = getattr(tam, "sentinel2_l2a", None)
        if tok is None and isinstance(tam, Mapping):
            tok = tam.get("sentinel2_l2a")
        if tok is None:
            raise RuntimeError("Could not locate sentinel2_l2a tokens in OlmoEarth output")
        # (B, h, w, T, bandsets, D) -> (B, D, h, w)
        while tok.ndim > 4:
            tok = tok.mean(dim=3)
        return tok.permute(0, 3, 1, 2).contiguous()

    def forward_features(self, batch: Mapping[str, Any], device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        x, _, canon = self.prepare(batch, device=device)
        fam = self.family
        m = self.backbone
        if fam == "olmoearth":
            return self._olmoearth_features(x, canon, batch)
        if fam == "resnet":
            return m.forward_features(x)
        if fam == "swin":
            y = m.norm(m.features(x))
            return y.permute(0, 3, 1, 2).contiguous()
        if fam == "dofa":
            wl = B.wavelengths_um(canon)
            return self._tokens_to_grid(m.forward_features(x, wl), drop_cls=True)
        if fam == "croma":
            return self._tokens_to_grid(m(x_optical=x)["optical_encodings"], drop_cls=False)
        if fam == "scalemae":
            return self._tokens_to_grid(m.forward_features(x), drop_cls=True)
        if fam == "unet":
            feats = m.encoder(x)
            if self.unet_feature == "decoder":
                try:
                    return m.decoder(feats)
                except TypeError:
                    return m.decoder(*feats)
            return feats[-1]
        if fam == "prithvi":
            tc, lc = self._prithvi_coords(batch, x.shape[0], x.device)
            x5 = x.unsqueeze(2)  # (B, C, T=1, H, W)
            vit = m.encoder.encoder if hasattr(m.encoder, "encoder") else m.encoder
            # gfmtools' PrithviViT reads patch_embed.num_frames (never set by
            # PatchEmbed) when temporal encoding is on; single-frame input here.
            vit.patch_embed.num_frames = x5.shape[2]
            if tc is not None:
                tc = tc[:, : x5.shape[2]]
            feats = m.forward_preprocessed(x5, tc, lc)
            t = feats[self.feature_layer] if isinstance(feats, (list, tuple)) else feats
            return self._tokens_to_grid(t, drop_cls=True)
        raise ValueError(fam)

    def forward_instruments(self, batch: Mapping[str, Any], device: Optional[torch.device] = None, **_: Any) -> torch.Tensor:
        g = self.forward_features(batch, device=device)
        self.embed_dim = int(g.shape[1])
        return g.flatten(2).transpose(1, 2)  # (B, h*w, C)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:  # noqa: D401
        return self.forward_features(batch)


def pretrained_flag(enc: GFMEncoder) -> bool:
    return enc.pretrained
