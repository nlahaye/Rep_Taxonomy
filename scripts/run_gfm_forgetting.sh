#!/usr/bin/env bash
# Catastrophic forgetting across FTW / EPA / MADOS (config all_forgetting.yaml).
#   scripts/run_gfm_forgetting.sh full   [args]   every stage (pretrained -> fine-tune variants -> WiSE ->
#                                                 forgetting -> sequential)
#   scripts/run_gfm_forgetting.sh gpu    [args]   prepare embed finetune ft_wise ft_embed ft_sequential
#   scripts/run_gfm_forgetting.sh cpu    [args]   finetune_compare forgetting (embeds retention sets if missing)
#   scripts/run_gfm_forgetting.sh stage <cfg.yaml> [args]   forgetting stage only on an existing root
set -euo pipefail
WHAT="${1:-full}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${HERE}/reptaxonomy/config/gfm/all_forgetting.yaml"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
run() { python -m reptaxonomy.pipelines.gfm_experiment "$@"; }
case "$WHAT" in
  full)  run -y "$CFG" --mode finetune_compare "$@" ;;
  gpu)   run -y "$CFG" --mode finetune_compare --stages prepare embed finetune ft_wise ft_embed ft_sequential "$@" ;;
  cpu)   run -y "$CFG" --mode finetune_compare --stages finetune_compare forgetting "$@" ;;
  stage) C="$1"; shift; run -y "$C" --mode finetune_compare --stages forgetting "$@" ;;
  *) echo "usage: $0 full|gpu|cpu|stage <cfg>"; exit 1 ;;
esac
