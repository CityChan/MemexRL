#!/usr/bin/env bash
# =============================================================================
# run_30b_production.sh
#
# FULL PRODUCTION training for Qwen3-30B-A3B-Thinking-2507 on ALFWorld.
# This is the real run — multi-day, eats serious AST24021 budget.
#
# Two-run experiment:
#   Run A: lossless_db   — original MemexRL backend (paper-replicated baseline)
#   Run B: graph_db      — typed-edge graph backend (the new contribution)
#
# Budget per run (defaults from train_alfworld_30B_vista.sbatch):
#   Nodes:                   8
#   Walltime:                48h
#   GPU-hours per run:       8 × 48 = 384
#   GPU-hours both runs:     768
#   NUM_ROLLOUT:             3000
#   N_SAMPLES:               8
#   ROLLOUT_BATCH_SIZE:      32  (256 episodes per rollout)
#   GLOBAL_BATCH_SIZE:       128
#   MAX_CONTEXT_LEN:         32000
#   MAX_STEPS:               50
#   SAVE_INTERVAL:           20  (~150 ckpts saved per run)
#   ENABLE_EVAL:             true
#
# Pre-flight (you SHOULD have done these already):
#   1. mini-smoke passed for both modes (see run_30b_mini_smoke.sh)
#      — non-zero pg_loss
#      — graph_db: entity_count > 0 in stats
#      — raw_reward not stuck at -1
#   2. WANDB_KEY set (highly recommended for 48h jobs — checkpoint metrics
#      to the cloud in case the slurm log gets rotated)
#   3. Ckpt dirs cleaned: rm -rf $MODEL_HF-int4_slime_memex_{lossless,graph}_db/
#      (or NOT, if you want to resume from a previous interrupted run)
#
# Usage:
#   bash training/scripts/run_30b_production.sh lossless          # Run A only
#   bash training/scripts/run_30b_production.sh graph             # Run B only
#   bash training/scripts/run_30b_production.sh both              # both, sequential
#   bash training/scripts/run_30b_production.sh both --concurrent # both, parallel (16 nodes)
#   bash training/scripts/run_30b_production.sh both --yes        # skip confirmation
#
# Override defaults:
#   NUM_ROLLOUT=1500 bash training/scripts/run_30b_production.sh both
#   WALLTIME=24:00:00 bash training/scripts/run_30b_production.sh lossless
# =============================================================================

set -euo pipefail

PROJ="${PROJ:-/work/09281/chc_1996/vista/MemexRL}"
SBATCH_FILE="$PROJ/training/scripts/train_alfworld_30B_vista.sbatch"
DATE_TAG="$(date +%Y%m%d)"

[[ -f "$SBATCH_FILE" ]] || { echo "[ERROR] missing $SBATCH_FILE" >&2; exit 1; }

mode="${1:-}"
shift || true
sequential=true
auto_yes=false
for arg in "$@"; do
    case "$arg" in
        --concurrent) sequential=false ;;
        --yes|-y)     auto_yes=true ;;
        *) echo "[ERROR] unknown flag: $arg" >&2; exit 1 ;;
    esac
done
if [[ -z "$mode" ]] || ! [[ "$mode" =~ ^(lossless|graph|both)$ ]]; then
    echo "Usage: $0 {lossless|graph|both} [--concurrent] [--yes]" >&2
    exit 1
fi

# Production budget (overridable via env)
NODES="${NODES:-8}"
WALLTIME="${WALLTIME:-48:00:00}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
N_SAMPLES="${N_SAMPLES:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-32000}"
MAX_STEPS="${MAX_STEPS:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"

# Pre-flight checks
if [[ -z "${WANDB_KEY:-}" ]]; then
    echo "[WARN] WANDB_KEY not set — 48h job with no cloud logging is risky."
    echo "       Set it with: source \$WORK/.wandb_env"
    if ! $auto_yes; then
        read -r -p "       Continue without wandb? [y/N] " ans
        [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
    fi
fi

# Compute budget summary
runs_count=1
[[ "$mode" == "both" ]] && runs_count=2
total_gpu_hours=$(( NODES * (${WALLTIME%%:*}) * runs_count ))
parallelism="sequential (Run B starts after Run A finishes)"
$sequential || parallelism="concurrent (Run A + Run B run in parallel, 2× nodes)"

cat <<BANNER

==============================================================
  PRODUCTION run summary — read carefully
==============================================================
  Mode:              ${mode}
  Runs:              ${runs_count}
  Nodes per run:     ${NODES}
  Walltime per run:  ${WALLTIME}
  Concurrency:       ${parallelism}
  Total GPU-hours:   ~${total_gpu_hours}  (allocation: AST24021)
  WandB:             $([[ -n "${WANDB_KEY:-}" ]] && echo enabled || echo DISABLED)

  Hyperparams:
    NUM_ROLLOUT          ${NUM_ROLLOUT}
    N_SAMPLES            ${N_SAMPLES}    (GRPO group; ≥2 enforced by sbatch)
    ROLLOUT_BATCH_SIZE   ${ROLLOUT_BATCH_SIZE}
    GLOBAL_BATCH_SIZE    ${GLOBAL_BATCH_SIZE}
    MAX_CONTEXT_LEN      ${MAX_CONTEXT_LEN}
    MAX_STEPS            ${MAX_STEPS}
    SAVE_INTERVAL        ${SAVE_INTERVAL}
    EVAL_INTERVAL        ${EVAL_INTERVAL}

  Ckpt dirs (per mode, mode-specific suffix):
$(if [[ "$mode" != graph ]]; then echo "    \$MODEL_HF-int4_slime_memex_lossless_db/"; fi)
$(if [[ "$mode" != lossless ]]; then echo "    \$MODEL_HF-int4_slime_memex_graph_db/"; fi)

==============================================================
BANNER

if ! $auto_yes; then
    read -r -p "  Submit these jobs to slurm? [y/N] " ans
    if [[ ! "$ans" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Common env exports — production values, identical for both runs so
# the only variable between A and B is COMPRESSION_MODE.
COMMON_EXPORT="ALL"
COMMON_EXPORT+=",NUM_ROLLOUT=${NUM_ROLLOUT}"
COMMON_EXPORT+=",N_SAMPLES=${N_SAMPLES}"
COMMON_EXPORT+=",ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}"
COMMON_EXPORT+=",GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
COMMON_EXPORT+=",MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN}"
COMMON_EXPORT+=",MAX_STEPS=${MAX_STEPS}"
COMMON_EXPORT+=",SAVE_INTERVAL=${SAVE_INTERVAL}"
COMMON_EXPORT+=",EVAL_INTERVAL=${EVAL_INTERVAL}"
COMMON_EXPORT+=",ENABLE_EVAL=true"

submit_run() {
    local run_label="$1"     # "A" or "B"
    local mode_value="$2"    # "lossless_db" or "graph_db"
    local depends_on="${3:-}"  # optional jobid this run waits for
    local exp_name="prod_${mode_value}_${DATE_TAG}"
    local exports="${COMMON_EXPORT},COMPRESSION_MODE=${mode_value},EXPERIMENT_NAME=${exp_name}"
    local dep_arg=()
    [[ -n "$depends_on" ]] && dep_arg=(--dependency="afterany:${depends_on}")

    echo
    echo "[Run ${run_label}] submitting ${mode_value}${depends_on:+ (depends on $depends_on)}"
    local jobid
    jobid=$(sbatch --parsable \
        -J "memex_30b_prod_${mode_value}" \
        -N "$NODES" \
        -t "$WALLTIME" \
        --export="$exports" \
        "${dep_arg[@]}" \
        "$SBATCH_FILE")
    echo "[Run ${run_label}] jobid=${jobid}  experiment=${exp_name}"
    echo "$jobid"
}

case "$mode" in
    lossless)
        jid_a=$(submit_run A lossless_db | tail -n1)
        ;;
    graph)
        jid_b=$(submit_run B graph_db | tail -n1)
        ;;
    both)
        jid_a=$(submit_run A lossless_db | tail -n1)
        if $sequential; then
            jid_b=$(submit_run B graph_db "$jid_a" | tail -n1)
        else
            jid_b=$(submit_run B graph_db | tail -n1)
        fi
        ;;
esac

echo
echo "=============================================================="
echo "Submitted. Track with:"
echo "  squeue -u \$USER"
echo "  tail -f $PROJ/logs/memex_30b_<jobid>.out"
echo
echo "WandB:  https://wandb.ai (project: memex-vista-30b)"
echo "=============================================================="
