#!/usr/bin/env bash
# Pack the manuscript-relevant outputs of a gfm_experiment run into one archive.
#   scripts/gather_gfm_outputs.sh                         # MADOS fast config, four upsampling variants
#   CFG=reptaxonomy/config/gfm/ftw_pretrained_fast.yaml OUT=ftw_outputs.tar.gz scripts/gather_gfm_outputs.sh
#   COMMANDS=my_runs.txt scripts/gather_gfm_outputs.sh    # derive configs / overrides from the run commands
# Extra arguments go to reptaxonomy.pipelines.gather_outputs (e.g. --dry-run, --root-dir, --include-labels).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
CFG="${CFG:-reptaxonomy/config/gfm/mados_pretrained_fast.yaml}"
METHODS="${METHODS:-[pixelshuffle,patch,nearest,bilinear]}"
OUT="${OUT:-mados_outputs_$(date +%Y%m%d).tar.gz}"
if [[ -n "${COMMANDS:-}" ]]; then
  python -m reptaxonomy.pipelines.gather_outputs --commands "${COMMANDS}" -o "${OUT}" "$@"
else
  python -m reptaxonomy.pipelines.gather_outputs -y "${CFG}" --set "pipeline.upsample_methods=${METHODS}" -o "${OUT}" "$@"
fi
