#!/usr/bin/env bash
# Pretrained-only rep_taxonomy evaluation of the gfmtools encoder suite.
#   scripts/run_gfm_pretrained.sh [ftw|epa|mados|all] [extra gfm_experiment args...]
# e.g. scripts/run_gfm_pretrained.sh ftw --models resnet50__sentinel2_all_moco dofa_base__dofa_mae
#      scripts/run_gfm_pretrained.sh epa --stages prepare embed          (GPU node)
#      scripts/run_gfm_pretrained.sh epa --stages pipeline cross_condition synthesis   (CPU node)
set -euo pipefail
DS="${1:-all}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${HERE}/reptaxonomy/config/gfm/${DS}_pretrained.yaml"
[[ -f "$CFG" ]] || { echo "no config $CFG"; exit 1; }
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
python -m reptaxonomy.pipelines.gfm_experiment -y "$CFG" --mode pretrained "$@"
