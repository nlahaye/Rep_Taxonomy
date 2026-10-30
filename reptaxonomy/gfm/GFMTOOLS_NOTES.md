# gfmtools issues found while integrating (snapshot: gfmtools-main as supplied)

`reptaxonomy.gfm` does **not** depend on the gfmtools `*Encoder` wrapper classes or the
engines/decoders. It calls the gfmtools model *builders* and weight enums directly and works
around the issues below, so no gfmtools change is required to run the pipelines. The list is
here so the fixes can be made upstream; each item notes how rep_taxonomy handles it.

## Model wrappers

| Location | Issue | rep_taxonomy handling |
|---|---|---|
| `models/scalemae.py` `ScaleMAEEncoder`, `models/unet.py` `UNetEncoder` | `nn.Module` subclass without `super().__init__()` (fails on first attribute assignment) | builders used directly |
| `models/unet.py` `UNetEncoder` | `meta.bands` accessed as an attribute (meta is a dict) | uses `weights.meta["bands"]` |
| `EncoderDecoderPipeline`, engines, decoders | no `super().__init__()`; engines are stubs; `SegFCN` is not an `nn.Module` | own heads/training in `gfm/finetune.py` |
| `models/resnet.py` `ResNetEncoder` | string weights skip the build; `next(filter(...))` arguments reversed for Landsat | timm builders + weight enums directly |
| `models/swin.py` | `model_mapper` arity mismatch; Swin T/S/B Cityscapes weights have no `bands` meta | builders directly; Cityscapes treated as RGB |
| `models/dofa.py` | builder asserts that only `fc_norm`/`head` are missing, but released checkpoints have **no final norm at all**, so `global_pool=False` (token output) fails the assert; duplicate `reshape_input` definitions; default checkpoint is a local string path | build with `global_pool=True`, then switch to token output with an identity-initialised LayerNorm |
| `models/prithvi.py` | `reshape_input()` calls `initialize_weights()` (wipes pretrained weights when image/patch size changes); only 3 configs vs 6 enum members (600M meta says patch 16 - verify); `PrithviViT` reads `patch_embed.num_frames`, which `PatchEmbed` never sets (AttributeError with temporal encoding) | native 224 input only; `patch_embed.num_frames` set per call; `forward_preprocessed` |
| `models/croma.py` | `reshape_input` references `self.img_size` (undefined) | builder with `image_size` argument |

## Weight metadata

- `ResNet152` Satlas meta says model `resnet50`, no instrument.
- `ResNet50 LANDSAT_ETM_SR_*` meta says `resnet18`.
- `Swin_V2_B SENTINEL2_MI_MS` meta says instrument `S1GRD`.
- Several weights have no `instrument` / `bands`; `catalog._infer_sensor_and_bands` infers them
  from the weight name and `in_chans`.
- FTW UNet weights take 8 channels (window A + window B). The single-window rep sets feed the
  same window twice, so treat these as an in-domain reference, not a native bi-temporal run.

## Datasets

- `data/datasets/epa.py`: with several `included_timepoints`, `image_tensor[band_index]`
  indexes the time axis, not the band axis. `EPARepSet` selects one timepoint (`timepoint:`)
  before band indexing.
- `EPALabels` no longer accepts `transform`, and the key is `included_timepoints` (the old
  rep_taxonomy preset passed `included_time_points`/`transform` -> `TypeError`; fixed in
  `experiment_init_utils.py`). Default `split_file` path is `/home/rdemilt/spark/benchmark/epa_splits.yml`.
- `gfmtools.data.datasets.__init__` imports every dataset eagerly, so importing `ftw_shifts`
  needs `xarray`, `tifffile`, `geojson`, `rasterio`, `geopandas` even for FTW-only runs.
- `hlsburnscars_prealigned` and `models/precomputed_embeddings` are not in this release; the
  rep_taxonomy registry now imports them optionally.

## Environment notes (not gfmtools)

- Python 3.13+/networkx 3: `graspologic 0.3.x` references `nx.OrderedGraph` and rejects
  SciPy `csr_array`. Use a graspologic release that supports your Python, or put `tests/compat` on
  `PYTHONPATH` (its `sitecustomize.py` aliases `nx.Ordered*` and converts `sparray -> csr_matrix`).
- OlmoEarth needs `olmoearth_pretrain_minimal` (pip `olmoearth-pretrain-minimal`).

## Geolocation (geo_id)

- `FTWRepSet.meta_rows` carries chip-centre lat/lon (from the chip footprint bounds, reprojected to
  EPSG:4326); the prepare stage stores them per condition. EPA and MADOS rows carry coordinates only
  if their gfmtools loaders expose bounds/CRS; otherwise set `geo_id.datasets.<family>.coords_csv`.
- source.coop returns HTTP 403 without a `User-Agent` header; `scripts/ftw_geo_subset.py` sets one.
