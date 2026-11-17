#!/usr/bin/env bash
# Fine-tuning + forgetting on the reduced 7-model setup, FTW and MADOS side by side on one GPU.
#
#   scripts/run_gfm_ft_fast.sh launch     start the FTW and MADOS fine-tuning drivers concurrently
#                                         (background, logs in logs/), then return
#   scripts/run_gfm_ft_fast.sh joint      after both finished: link the two roots and run the
#                                         cross-family forgetting + sequential stages
#   scripts/run_gfm_ft_fast.sh audit      outstanding units of all three configs
#   scripts/run_gfm_ft_fast.sh status     running drivers and log tails
#
# Each driver runs its stages in order: finetune -> ft_wise -> ft_embed (GPU) -> ft_pipeline
# (CPU, parallel workers) -> finetune_compare -> ft_geo_id -> ft_synthesis. MADOS (1 run) reaches
# its CPU stages early, so most of the time only one fine-tune holds the GPU. Re-running a command
# resumes (finished units are skipped; killed fine-tunes resume from their last epoch).
# Extra arguments are passed to every gfm_experiment call (e.g. --models ..., --set ...).
set -euo pipefail
WHAT="${1:-launch}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${HERE}:${PYTHONPATH:-}"
C="${HERE}/reptaxonomy/config/gfm"
FT_STAGES=(finetune ft_wise ft_embed ft_pipeline finetune_compare ft_geo_id ft_synthesis)
JOINT_ROOT="${JOINT_ROOT:-./gfm_runs/ftw_mados_joint}"
mkdir -p logs
drv() { python -m reptaxonomy.pipelines.gfm_experiment "$@"; }

case "$WHAT" in
  launch)
    # main-process math threads; ft_pipeline workers get threads_per_worker from the configs
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup python -m reptaxonomy.pipelines.gfm_experiment \
      -y "$C/ftw_finetune_fast.yaml" --stages "${FT_STAGES[@]}" "$@" > logs/ftw_ft.log 2>&1 &
    echo "FTW   pid $!  log logs/ftw_ft.log"
    sleep 60   # let FTW claim its GPU memory first
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup python -m reptaxonomy.pipelines.gfm_experiment \
      -y "$C/mados_finetune_fast.yaml" --stages "${FT_STAGES[@]}" "$@" > logs/mados_ft.log 2>&1 &
    echo "MADOS pid $!  log logs/mados_ft.log"
    ;;
  joint)
    python -m reptaxonomy.pipelines.link_roots --dest "$JOINT_ROOT" --src ./gfm_runs/ftw ./gfm_runs/mados
    drv -y "$C/ftw_mados_forgetting_fast.yaml" --root-dir "$JOINT_ROOT" --stages forgetting ft_sequential "$@" \
      2>&1 | tee logs/joint_forgetting.log
    ;;
  audit)
    python -m reptaxonomy.pipelines.audit_experiments \
      -y "$C/ftw_finetune_fast.yaml" "$C/mados_finetune_fast.yaml" --stages "${FT_STAGES[@]}" "$@"
    python -m reptaxonomy.pipelines.audit_experiments -y "$C/ftw_mados_forgetting_fast.yaml" \
      --root-dir "$JOINT_ROOT" --stages forgetting ft_sequential "$@"
    ;;
  status)
    pgrep -af "reptaxonomy.pipelines.gfm_experiment" || echo "no drivers running"
    for f in logs/ftw_ft.log logs/mados_ft.log; do [[ -f $f ]] && { echo "== $f"; tail -n 3 "$f"; }; done
    command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true
    ;;
  *) echo "usage: $0 launch|joint|audit|status [gfm_experiment args]"; exit 1 ;;
esac
