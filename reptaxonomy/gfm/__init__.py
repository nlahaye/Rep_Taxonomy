"""gfmtools integration for rep_taxonomy (FTW / EPA / MADOS rep sets, encoder catalog,
cross-condition analysis, fine-tuning comparison). See README_GFM.md."""

GFMTOOLS_MIN_VERSION = "0.0.0"


def require_gfmtools() -> None:
    """Fail early (instead of with silently wrong embeddings) on an old gfmtools."""
    import gfmtools
