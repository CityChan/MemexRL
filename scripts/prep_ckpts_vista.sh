#!/usr/bin/env bash
# =============================================================================
# prep_ckpts_vista.sh
#
# Generic checkpoint converter for Slime training on Vista (GH200/aarch64).
# Replaces the older prep_4b_ckpts_vista.sh — supports any Qwen3 variant
# by auto-detecting the slime model arch script and rope_theta.
#
# Produces:
#   ${MODEL_HF}-int4/                  rollout backend (SGLang)
#   ${MODEL_HF}-int4_torch_dist/       Megatron actor / ref model
#
# Idempotent: re-running skips conversions that are already complete.
#
# Usage (in the slime env):
#
#   # Default (Qwen3-30B-A3B-Thinking-2507 — production target)
#   bash scripts/prep_ckpts_vista.sh
#
#   # Other variants
#   MODEL_HF=$PROJ/models/Qwen3-4B-Instruct-2507 bash scripts/prep_ckpts_vista.sh
#   MODEL_HF=$PROJ/models/Qwen3-8B-Instruct-2507 bash scripts/prep_ckpts_vista.sh
#
#   # Force a specific slime model arch script (rare; auto-detect usually works)
#   SLIME_MODEL_NAME=qwen3-30B-A3B \
#       MODEL_HF=$PROJ/models/some-30B-finetune \
#       bash scripts/prep_ckpts_vista.sh
# =============================================================================

set -eo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SLIME_ROOT="${SLIME_ROOT:-$PROJ/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-$PROJ/Megatron-LM}"
MODEL_HF="${MODEL_HF:-$PROJ/models/Qwen3-30B-A3B-Thinking-2507}"
MODEL_INT4="${MODEL_INT4:-${MODEL_HF}-int4}"
MODEL_TD="${MODEL_TD:-${MODEL_INT4}_torch_dist}"

# Megatron source must be on PYTHONPATH for megatron.training to import.
[[ -d "$MEGATRON_ROOT" ]] || { echo "[ERROR] Megatron-LM repo missing at $MEGATRON_ROOT" >&2; exit 1; }
export PYTHONPATH="$MEGATRON_ROOT:$SLIME_ROOT:${PYTHONPATH:-}"

log()  { printf '\033[1;36m[%s]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*"; }
err()  { printf '\033[1;31m[%s ERROR]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }

# Pre-flight ------------------------------------------------------------------

[[ -d "$MODEL_HF" ]] || { err "BF16 model not found at $MODEL_HF (download from HF first)"; exit 1; }
[[ -d "$SLIME_ROOT" ]] || { err "Slime not found at $SLIME_ROOT"; exit 1; }

# Auto-detect Slime model arch script ----------------------------------------

if [[ -z "${SLIME_MODEL_NAME:-}" ]]; then
    bn=$(basename "$MODEL_HF")
    case "$bn" in
        Qwen3-0.6B-*|Qwen3-0.5B-*) SLIME_MODEL_NAME="qwen3-0.6B" ;;
        Qwen3-1.7B-*|Qwen3-1.8B-*) SLIME_MODEL_NAME="qwen3-1.7B" ;;
        Qwen3-4B-*)                SLIME_MODEL_NAME="qwen3-4B" ;;
        Qwen3-8B-*)                SLIME_MODEL_NAME="qwen3-8B" ;;
        Qwen3-14B-*)               SLIME_MODEL_NAME="qwen3-14B" ;;
        Qwen3-30B-A3B-*)           SLIME_MODEL_NAME="qwen3-30B-A3B" ;;
        Qwen3-235B-A22B-*)         SLIME_MODEL_NAME="qwen3-235B-A22B" ;;
        *)
            err "could not auto-detect slime model arch from '$bn'."
            err "set SLIME_MODEL_NAME explicitly. Available scripts:"
            ls "$SLIME_ROOT/scripts/models/" 2>/dev/null | sed 's/^/    /' >&2
            exit 2
            ;;
    esac
fi

MODEL_ARG_FILE="$SLIME_ROOT/scripts/models/${SLIME_MODEL_NAME}.sh"
[[ -f "$MODEL_ARG_FILE" ]] || {
    err "Slime arch script missing: $MODEL_ARG_FILE"
    err "Available:"
    ls "$SLIME_ROOT/scripts/models/" 2>/dev/null | sed 's/^/    /' >&2
    exit 2
}

log "PROJ              = $PROJ"
log "MODEL_HF          = $MODEL_HF"
log "MODEL_INT4        = $MODEL_INT4"
log "MODEL_TD          = $MODEL_TD"
log "SLIME_MODEL_NAME  = $SLIME_MODEL_NAME"
log "MODEL_ARG_FILE    = $MODEL_ARG_FILE"

# Step 1: BF16 -> INT4 HF -----------------------------------------------------

if [[ -f "$MODEL_INT4/config.json" ]]; then
    log "INT4 ckpt already at $MODEL_INT4 (skip)"
else
    log "==== Step 1: BF16 -> INT4 HF ===="
    log "    expected ~5 min for 4B, ~30 min for 30B"
    python "$SLIME_ROOT/tools/convert_hf_to_int4_direct.py" \
        --model-dir "$MODEL_HF" \
        --save-dir  "$MODEL_INT4" \
        --group-size 128 --is-symmetric True
fi

# Step 2: BF16 -> torch_dist (Megatron) ---------------------------------------

if [[ -d "$MODEL_TD" && -n "$(ls -A "$MODEL_TD" 2>/dev/null)" ]]; then
    log "torch_dist ckpt already at $MODEL_TD (skip)"
else
    log "==== Step 2: BF16 -> torch_dist (Megatron) ===="
    log "    expected ~5-10 min for 4B, ~20-30 min for 30B"

    # Read rope_theta from the actual config.json. Variants disagree
    # (Thinking-2507=1e7, Instruct-2507=5e6, base Qwen3=1e6). Slime's
    # hf_validate_args is strict about this matching.
    MODEL_ARGS_ROTARY_BASE=$(python - <<PY
import json
cfg = json.load(open("$MODEL_HF/config.json"))
print(int(cfg.get("rope_theta", 1000000)))
PY
)
    log "auto-detected rope_theta = $MODEL_ARGS_ROTARY_BASE from config.json"
    export MODEL_ARGS_ROTARY_BASE
    # shellcheck source=/dev/null
    source "$MODEL_ARG_FILE"

    torchrun --nproc-per-node 1 \
        "$SLIME_ROOT/tools/convert_hf_to_torch_dist.py" \
        --hf-checkpoint "$MODEL_HF" \
        --save "$MODEL_TD" \
        "${MODEL_ARGS[@]}"
fi

log "==== checkpoints ready ===="
log "  HF:         $MODEL_HF"
log "  INT4:       $MODEL_INT4  ($(du -sh "$MODEL_INT4" 2>/dev/null | awk '{print $1}'))"
log "  torch_dist: $MODEL_TD    ($(du -sh "$MODEL_TD" 2>/dev/null | awk '{print $1}'))"
log ""
log "next: 4-node mini-production smoke (50 rollouts, ~2 hours)"
log "  sbatch -N 4 -t 02:00:00 \\"
log "      --export=ALL,NUM_ROLLOUT=50,EXPERIMENT_NAME=mini_smoke_30b \\"
log "      training/scripts/train_alfworld_30B_vista.sbatch"
log ""
log "then full production (8 nodes, 48 hours, 3000 rollouts):"
log "  sbatch training/scripts/train_alfworld_30B_vista.sbatch"
