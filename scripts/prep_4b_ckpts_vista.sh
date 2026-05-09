#!/usr/bin/env bash
# =============================================================================
# prep_4b_ckpts_vista.sh
#
# One-shot checkpoint converter for the Qwen3-4B smoke training run on Vista.
# After install_slime_vista.sh + smoke validation, this produces the two
# checkpoint formats Slime needs:
#
#   1. INT4 HF checkpoint  -> rollout backend (SGLang / vLLM)
#   2. torch_dist           -> Megatron reference + actor model
#
# Idempotent: re-running skips work that's already done.
#
# Usage (in the slime env):
#   bash scripts/prep_4b_ckpts_vista.sh
#
# Required env (or use defaults that match install_slime_vista.sh):
#   PROJ        default: /work/$ALLOC/$USER/vista/MemexRL
#   SLIME_ROOT  default: $PROJ/slime
#   MODEL_HF    default: $PROJ/models/Qwen3-4B-Instruct-2507
# =============================================================================

set -eo pipefail

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SLIME_ROOT="${SLIME_ROOT:-$PROJ/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-$PROJ/Megatron-LM}"
MODEL_HF="${MODEL_HF:-$PROJ/models/Qwen3-4B-Instruct-2507}"
MODEL_INT4="${MODEL_INT4:-${MODEL_HF}-int4}"
MODEL_TD="${MODEL_TD:-${MODEL_INT4}_torch_dist}"

# Megatron-LM's `pip install -e .` only exposes megatron.core. The
# convert_hf_to_torch_dist.py script we're about to call imports
# megatron.training.arguments, which lives in the repo source but is NOT
# packaged. Need to put the repo on PYTHONPATH explicitly. Slime's own
# training scripts do this via Ray runtime_env; for our standalone prep
# we set it directly.
[[ -d "$MEGATRON_ROOT" ]] || { echo "[ERROR] Megatron-LM repo missing at $MEGATRON_ROOT" >&2; exit 1; }
export PYTHONPATH="$MEGATRON_ROOT:$SLIME_ROOT:${PYTHONPATH:-}"

log()  { printf '\033[1;36m[%s]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*"; }
err()  { printf '\033[1;31m[%s ERROR]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }

# Pre-flight ------------------------------------------------------------------

[[ -d "$MODEL_HF" ]] || { err "BF16 model not found at $MODEL_HF"; exit 1; }
[[ -d "$SLIME_ROOT" ]] || { err "Slime not found at $SLIME_ROOT"; exit 1; }

log "PROJ        = $PROJ"
log "SLIME_ROOT  = $SLIME_ROOT"
log "MODEL_HF    = $MODEL_HF"
log "MODEL_INT4  = $MODEL_INT4"
log "MODEL_TD    = $MODEL_TD"

# 1. INT4 HF -----------------------------------------------------------------

if [[ -f "$MODEL_INT4/config.json" ]]; then
    log "INT4 ckpt already at $MODEL_INT4"
else
    log "==== Step 1: BF16 -> INT4 HF ===="
    log "    expected ~5 minutes for 4B"
    python "$SLIME_ROOT/tools/convert_hf_to_int4_direct.py" \
        --model-dir "$MODEL_HF" \
        --save-dir  "$MODEL_INT4" \
        --group-size 128 --is-symmetric True
fi

# 2. torch_dist --------------------------------------------------------------
# Slime's torchrun-based converter requires MODEL_ARGS from the model arch
# script. Try Qwen3-4B-specific first, then fall back to a reasonable guess.

MODEL_ARG_FILE="$SLIME_ROOT/scripts/models/qwen3-4B.sh"
if [[ ! -f "$MODEL_ARG_FILE" ]]; then
    log "Slime does not ship $MODEL_ARG_FILE."
    log "Available model configs:"
    ls "$SLIME_ROOT/scripts/models/" 2>/dev/null | grep -i qwen | sed 's/^/    /'
    log ""
    log "Slime usually has qwen3-0.6B / qwen3-1.7B / qwen3-4B / qwen3-8B / qwen3-30B-A3B."
    log "If qwen3-4B.sh is missing on this commit, copy it from a newer slime tag, or"
    log "write one yourself by adapting qwen3-8B.sh (smaller hidden_size, fewer layers)."
    err "abort: missing model arch script for 4B"
    exit 2
fi

if [[ -d "$MODEL_TD" && -n "$(ls -A "$MODEL_TD" 2>/dev/null)" ]]; then
    log "torch_dist ckpt already at $MODEL_TD"
else
    log "==== Step 2: BF16 -> torch_dist (Megatron) ===="
    log "    expected ~5-10 minutes for 4B"

    # Read rope_theta from the actual HF config.json. Qwen3 variants disagree:
    #   Thinking-2507 -> 1e7, Instruct-2507 -> 5e6, base Qwen3 -> 1e6.
    # Slime's hf_validate_args is strict. Read it instead of guessing.
    MODEL_ARGS_ROTARY_BASE=$(python - <<PY
import json
cfg = json.load(open("$MODEL_HF/config.json"))
print(int(cfg.get("rope_theta", 1000000)))
PY
)
    log "auto-detected rope_theta = $MODEL_ARGS_ROTARY_BASE from $MODEL_HF/config.json"
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
log "  INT4:       $MODEL_INT4"
log "  torch_dist: $MODEL_TD"
log ""
log "next: bash scripts/smoke_train_4b_vista.sh"
