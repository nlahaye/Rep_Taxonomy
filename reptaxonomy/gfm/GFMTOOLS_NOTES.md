# gfmtools requirements

`reptaxonomy.gfm` requires **gfmtools >= 0.2.0**. `reptaxonomy.gfm.require_gfmtools()` checks
`gfmtools.__features__` when the catalog or the experiment registry is first imported and fails
with an explicit message on an older install (older releases produced silently wrong EPA inputs).

`reptaxonomy.gfm` still calls the gfmtools model *builders* and weight enums directly rather than
the `*Encoder.tensorize` helpers, because those expect fixed instrument keys and full band sets
that the FTW (4 bands) and MADOS (11 bands) rep sets do not provide.

## Fixed in gfmtools 0.2.0 (rep_taxonomy workaround removed)

| gfmtools issue | Previous rep_taxonomy workaround | Now |
|---|---|---|
| DOFA checkpoints have no final norm; token output (`global_pool=False`) failed | build pooled, swap in an identity LayerNorm | `builder(w, global_pool=False, final_norm="identity")` (bit-identical output) |
| Prithvi `PatchEmbed.num_frames` never set | set `patch_embed.num_frames` before every forward | removed (bit-identical output) |
| Prithvi 600M meta patch size 16 (HF config: 14); `reshape_input` wiped weights | 600M marked experimental; native size only | meta fixed; `reshape_input` keeps weights |
| Weight meta: wrong `model`, missing `instrument`/`bands` | `_infer_sensor_and_bands` guessed from weight names + `in_chans`; hard-coded CROMA/DOFA/Scale-MAE bands | `catalog._sensor_and_bands` reads `instrument`/`bands`/`sensor` from the meta (sensor and band lists identical for all 110 entries) |
| Known-fragile weights (ResNet152 Satlas, ResNet50 ETM SR, Swin Cityscapes, Prithvi 600M) | hard-coded `_EXPERIMENTAL` set | only weights with `derived_bands` (NDVI SECO) stay experimental |
| EPA: with several timepoints the band index hit the time axis | `_select_time` assumed a leading time axis | arrays are band-first by name and carry `time_axis`; `_select_time` uses it |
| `hlsburnscars_prealigned`, `models/precomputed_embeddings` missing | `_optional_import` placeholders | imported directly; placeholders kept only for classes that need extras |
| Hard-coded `/mnt/data/...` defaults | rep_taxonomy passed paths | `data_root=None` resolves via `GFMTOOLS_DATA_ROOT` (legacy path when unset) |
| Eager `gfmtools.data.datasets` imports | (import cost / extra deps) | lazy |

Also fixed upstream, not used by rep_taxonomy: `ScaleMAEEncoder` / `UNetEncoder` constructors,
`ResNetEncoder` / `SwinEncoder` string weights and `model_mapper`, CROMA `reshape_input`,
`SegFCN` as an `nn.Module`, `EncoderDecoderPipeline` constructor.

## Still relevant

- FTW UNet weights take 8 channels (window A + window B, meta `num_windows: 2`). The single-window
  rep sets feed the same window twice, so treat them as an in-domain reference.
- `EPALabels` takes `included_timepoints` (not `included_time_points`) and no `transform`.

## Environment notes (not gfmtools)

- Python 3.13+/networkx 3: `graspologic 0.3.x` references `nx.OrderedGraph` and rejects
  SciPy `csr_array`. Use a graspologic release that supports your Python, or put `tests/compat` on
  `PYTHONPATH` (its `sitecustomize.py` aliases `nx.Ordered*` and converts `sparray -> csr_matrix`).
- OlmoEarth needs `olmoearth_pretrain_minimal` (pip `olmoearth-pretrain-minimal`).
- calflops passes list arguments to some `Conv2d` variants (EfficientNet same-padding);
  `resources/compute_resource_reqs._patch_calflops_conv_args` handles it.
- source.coop returns HTTP 403 without a `User-Agent` header; `scripts/ftw_geo_subset.py` sets one.

## Geolocation (geo_id)

- `FTWRepSet.meta_rows` carries chip-centre lat/lon (from the chip footprint bounds, reprojected to
  EPSG:4326); the prepare stage stores them per condition. EPA and MADOS rows carry coordinates only
  if their gfmtools loaders expose bounds/CRS; otherwise set `geo_id.datasets.<family>.coords_csv`.
