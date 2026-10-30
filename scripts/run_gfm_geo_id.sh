#!/usr/bin/env bash
# Geolocated intrinsic dimension (GeoINRID-style) for the gfmtools experiments.
#   scripts/run_gfm_geo_id.sh locations [extra args]   location encoders at every FTW site + shift-task site sets
#   scripts/run_gfm_geo_id.sh images    [extra args]   image encoders, FTW per-country test chips (prepare/embed/geo_id)
#   scripts/run_gfm_geo_id.sh stage <cfg.yaml> [args]  geo_id stage only, on an existing gfm_experiment root
# Data: python scripts/ftw_geo_subset.py --root <dir> --countries all --footprints-only   (sites only)
#       python scripts/ftw_geo_subset.py --root <dir> --countries <list> --split test --n-per-country 500 (imagery subset)
# Location encoders need GeoINRID weights: git clone https://github.com/arjunarao619/GeoINRID
set -euo pipefail
WHAT="${1:-locations}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
case "$WHAT" in
  locations) python -m reptaxonomy.pipelines.geo_id_locations -y "${HERE}/reptaxonomy/config/gfm/ftw_geo_id_locations.yaml" "$@" ;;
  images)    python -m reptaxonomy.pipelines.gfm_experiment -y "${HERE}/reptaxonomy/config/gfm/ftw_geo_id.yaml" "$@" ;;
  stage)     CFG="$1"; shift; python -m reptaxonomy.pipelines.gfm_experiment -y "$CFG" --stages geo_id "$@" ;;
  *) echo "usage: $0 locations|images|stage <cfg>"; exit 1 ;;
esac
