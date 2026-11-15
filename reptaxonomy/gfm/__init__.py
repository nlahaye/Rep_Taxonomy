"""gfmtools integration for rep_taxonomy (FTW / EPA / MADOS rep sets, encoder catalog,
cross-condition analysis, fine-tuning comparison). See README_GFM.md."""

GFMTOOLS_MIN_VERSION = "0.2.0"
GFMTOOLS_REQUIRED_FEATURES = frozenset({
    "epa_time_axis", "dofa_final_norm", "prithvi_num_frames", "meta_schema_v1",
    "precomputed_embeddings_model",
})


def require_gfmtools() -> None:
    """Fail early (instead of with silently wrong embeddings) on an old gfmtools."""
    import gfmtools

    missing = GFMTOOLS_REQUIRED_FEATURES - set(getattr(gfmtools, "__features__", ()))
    if missing:
        raise ImportError(
            f"rep_taxonomy needs gfmtools >= {GFMTOOLS_MIN_VERSION} (installed: "
            f"{getattr(gfmtools, '__version__', 'unknown')}; missing features {sorted(missing)})")
