#!/usr/bin/env bash
# =============================================================================
# run_30b_mini_smoke.sh
#
# Mini-production smoke for Qwen3-30B-A3B-Thinking-2507 on ALFWorld.
# Two runs to compare under identical RL conditions:
#
#   Run A: lossless_db   — original MemexRL backend (baseline)
#   Run B: graph_db      — typed-edge graph backend (the new work)
#
# Goal: validate that 30B-Thinking under GRPO can actually start using the
# memory tools (compress count > 0, raw_reward != -1, pg_loss != 0). The 4B
# Instruct smoke proved the infra works but the model never used the tools;
# 30B-Thinking should — if it doesn't, that's a prompt / reward problem,
# not a code one.
#
# Usage:
#   bash training/scripts/run_30b_mini_smoke.sh lossless   # Run A only
#   bash training/scripts/run_30b_mini_smoke.sh graph      # Run B only
#   bash training/scripts/run_30b_mini_smoke.sh both       # both, back-to-back
#
# Budget per run (4 nodes × 1 GH200, ~2 hours):
#   NUM_ROLLOUT=50           50 RL training cycles
#   N_SAMPLES=4              GRPO group size (≥2 required, half of prod 8)
#   ROLLOUT_BATCH_SIZE=8     8 unique prompts/cycle → 32 episodes/cycle
#   GLOBAL_BATCH_SIZE=32     consume all 32 in one train step
#   MAX_STEPS=30             production-like episode length
#   MAX_CONTEXT_LEN=20000    smaller than prod 32K for safety headroom
#   CONTEXT_THRESHOLD=8000   forces compress when ctx > 8K
#   SAVE_INTERVAL=999        no checkpoint saves
#   ENABLE_EVAL=false        no eval (smoke compares train metrics only)
#
# Decision criteria after both runs finish:
#   ✓ infra: train/pg_loss != 0 across runs (GRPO advantages working)
#   ✓ behavior: episode compress count > 0 in some episodes
#   ✓ signal: rollout/raw_reward not stuck at -1 (model getting partial credit)
#   ✓ Run B specific: graph stats show entity_count > 0, edge_count > 0
#
# If both pass → proceed to add ContextGraph-style mutation tools.
# If Run B fails on graph stats → SFT warm-up first (scripts/build_graph_sft_data.py).
# =============================================================================

set -euo pipefail

PROJ="${PROJ:-/work/09281/chc_1996/vista/MemexRL}"
SBATCH_FILE="$PROJ/training/scripts/train_alfworld_30B_vista.sbatch"
DATE_TAG="$(date +%Y%m%d)"

[[ -f "$SBATCH_FILE" ]] || { echo "[ERROR] missing $SBATCH_FILE" >&2; exit 1; }
[[ -n "${WANDB_KEY:-}" ]] || echo "[WARN] WANDB_KEY not set — runs will not log to wandb"

mode="${1:-}"
if [[ -z "$mode" ]] || ! [[ "$mode" =~ ^(lossless|graph|both)$ ]]; then
    echo "Usage: $0 {lossless|graph|both}" >&2
    exit 1
fi

# Common smoke-budget overrides — same for both runs so the two are comparable.
COMMON_EXPORT="ALL"
COMMON_EXPORT+=",NUM_ROLLOUT=50"
COMMON_EXPORT+=",N_SAMPLES=4"
COMMON_EXPORT+=",ROLLOUT_BATCH_SIZE=8"
COMMON_EXPORT+=",GLOBAL_BATCH_SIZE=32"
COMMON_EXPORT+=",MAX_STEPS=30"
COMMON_EXPORT+=",MAX_CONTEXT_LEN=20000"
COMMON_EXPORT+=",CONTEXT_THRESHOLD=8000"
COMMON_EXPORT+=",SAVE_INTERVAL=999"
COMMON_EXPORT+=",ENABLE_EVAL=false"

submit_run() {
    local run_label="$1"     # "A" or "B"
    local mode_value="$2"    # "lossless_db" or "graph_db"
    local exp_name="mini_smoke_${mode_value}_${DATE_TAG}"
    local exports="${COMMON_EXPORT},COMPRESSION_MODE=${mode_value},EXPERIMENT_NAME=${exp_name}"

    echo "=========================================="
    echo "  Run ${run_label}: ${mode_value}"
    echo "  Experiment: ${exp_name}"
    echo "  Nodes: 4   Walltime: 02:00:00"
    echo "=========================================="
    set -x
    sbatch \
        -J "memex_30b_mini_${mode_value}" \
        -N 4 \
        -t 02:00:00 \
        --export="${exports}" \
        "$SBATCH_FILE"
    set +x
}

case "$mode" in
    lossless) submit_run A lossless_db ;;
    graph)    submit_run B graph_db ;;
    both)
        submit_run A lossless_db
        echo
        submit_run B graph_db
        ;;
esac

echo
echo "Submitted. Track with:  squeue -u \$USER"
echo "Logs land in:           $PROJ/logs/memex_30b_<jobid>.{out,err}"
