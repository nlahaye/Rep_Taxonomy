#!/usr/bin/env bash
# Pretrained vs fine-tuned comparison (runs the pretrained stages first unless --stages is given).
#   scripts/run_gfm_finetune_compare.sh [ftw|epa|mados|all] [extra gfm_experiment args...]
# Typical split across nodes:
#   ... ftw --stages prepare embed finetune ft_embed                 (GPU)
#   ... ftw --stages pipeline cross_condition synthesis ft_pipeline finetune_compare ft_synthesis   (CPU)
set -euo pipefail
DS="${1:-all}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${HERE}/reptaxonomy/config/gfm/${DS}_finetune_compare.yaml"
[[ -f "$CFG" ]] || { echo "no config $CFG"; exit 1; }
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
python -m reptaxonomy.pipelines.gfm_experiment -y "$CFG" --mode finetune_compare "$@"
