#!/usr/bin/env bash
# =============================================================================
# run_smoke_4b.sh
#
# One-shot smoke test orchestrator for the bare-metal Vista install.
# After install_slime_vista.sh has finished, this script:
#
#   1. Activates the (correct) slime micromamba env, ignoring any same-named
#      conda env in miniconda3/. This is the recurring source of confusion.
#   2. Patches + installs textworld and alfworld into the slime env if missing
#      (TextWorld's setup.sh fails on aarch64 because Inform7 has no aarch64
#      binary; we lie and use x86_64 binaries that never get executed when
#      running pre-compiled ALFWorld games).
#   3. Downloads Qwen3-4B-Thinking-2507 from HuggingFace if missing (~8 GB).
#   4. Runs alfworld-download to fetch the .tw-pddl game files (~300 MB).
#   5. Runs scripts/smoke_4b_inference.py and scripts/smoke_4b_alfworld.py
#      with three memory modes (lossless_db / lossy / graph_db).
#
# Each step is idempotent: re-running the script picks up where it left off.
# All smoke logs land under $PROJ/smoke_logs/.
#
# Usage:
#   bash scripts/run_smoke_4b.sh                   # do everything
#   bash scripts/run_smoke_4b.sh --skip-install    # skip pip installs (faster re-run)
#   bash scripts/run_smoke_4b.sh --skip-download   # skip model + alfworld data
#   bash scripts/run_smoke_4b.sh --inference-only  # only run smoke_4b_inference.py
#   bash scripts/run_smoke_4b.sh --modes "lossless_db graph_db"  # subset
# =============================================================================

set -eo pipefail

# --------------------------- configuration -----------------------------------

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-slime}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJ/.micromamba}"

MODEL_REPO="Qwen/Qwen3-4B-Thinking-2507"
MODEL_DIR="$PROJ/models/Qwen3-4B-Thinking-2507"

# Argument parsing
SKIP_INSTALL=0
SKIP_DOWNLOAD=0
INFERENCE_ONLY=0
MODES=("lossless_db" "graph_db")
MAX_STEPS=8

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install)   SKIP_INSTALL=1; shift ;;
        --skip-download)  SKIP_DOWNLOAD=1; shift ;;
        --inference-only) INFERENCE_ONLY=1; shift ;;
        --modes)          read -ra MODES <<<"$2"; shift 2 ;;
        --max-steps)      MAX_STEPS="$2"; shift 2 ;;
        -h|--help)        sed -n '1,40p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

LOG_DIR="$PROJ/smoke_logs"
mkdir -p "$LOG_DIR"

log() { printf '\033[1;36m[%s]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*"; }
err() { printf '\033[1;31m[%s ERROR]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }

# --------------------------- step A: activate env ----------------------------

log "==== A: activate slime env ===="

# Drop out of any conda env we might have inherited (e.g. miniconda3 base/slime).
# The most common gotcha is that miniconda3 ships its own (empty) env named
# "slime", and `conda activate slime` picks that one instead of ours.
type conda &>/dev/null && {
    conda deactivate &>/dev/null || true
    conda deactivate &>/dev/null || true
}
type micromamba &>/dev/null || export PATH="$HOME/.local/bin:$PATH"

export MAMBA_ROOT_PREFIX
eval "$(micromamba shell hook -s bash)"
micromamba activate "$ENV_NAME"

# Sanity: must be the env at $PROJ/.micromamba, not /work/.../miniconda3
expected="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME"
if [[ "$CONDA_PREFIX" != "$expected" ]]; then
    err "active env is $CONDA_PREFIX but expected $expected"
    err "did MAMBA_ROOT_PREFIX get overridden? bashrc points at /home1?"
    exit 1
fi
log "active env: $CONDA_PREFIX"
log "python:     $(which python)"
log "torch:      $(python -c 'import torch; print(torch.__version__)')"

export PATH="$HOME/.local/bin:$PATH"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.alfworld}"

# --------------------------- step B: install env deps ------------------------

if [[ $SKIP_INSTALL -eq 0 ]]; then
    log "==== B: install textworld + alfworld (if missing) ===="

    if python -c "import textworld" &>/dev/null; then
        log "textworld already installed: $(python -c 'import textworld; print(textworld.__version__)')"
    else
        log "textworld missing — building from source with aarch64 patch"
        TW_SRC="$PROJ/tw_src"
        if [[ ! -d "$TW_SRC" ]]; then
            git clone https://github.com/microsoft/TextWorld.git "$TW_SRC"
        fi
        # Patch: force ARCH=x86_64 so setup.sh doesn't try to extract
        # nonexistent aarch64 inform7 binaries. The x86_64 binaries that get
        # extracted are never executed when playing pre-compiled .tw-pddl
        # games (which is all ALFWorld does).
        if grep -q '^\s*ARCH=\$(uname -m)' "$TW_SRC/setup.sh"; then
            log "patching $TW_SRC/setup.sh ARCH=x86_64"
            sed -i 's|ARCH=$(uname -m)|ARCH=x86_64|' "$TW_SRC/setup.sh"
        fi
        # Clean partial extracts from earlier failed attempts
        rm -rf "$TW_SRC/textworld/thirdparty/inform7-6M62"
        rm -f  "$TW_SRC/textworld/thirdparty/I7_6M62_Linux_all.tar.gz"

        pushd "$TW_SRC" >/dev/null
        pip install --user . 2>&1 | tail -5
        popd >/dev/null
    fi

    if python -c "import alfworld" &>/dev/null; then
        log "alfworld already installed"
    else
        pip install --user alfworld 2>&1 | tail -5
    fi
fi

# --------------------------- step C: model + alfworld data -------------------

if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
    log "==== C: model + alfworld data ===="

    if [[ -d "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]]; then
        log "model already present at $MODEL_DIR"
    else
        log "downloading $MODEL_REPO -> $MODEL_DIR"
        python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$MODEL_REPO", local_dir="$MODEL_DIR")
print("model download OK")
PY
    fi

    if find "$ALFWORLD_DATA" -name "*.tw-pddl" -print -quit 2>/dev/null | grep -q .; then
        log "alfworld data already present at $ALFWORLD_DATA"
    else
        log "running alfworld-download -> $ALFWORLD_DATA"
        if ! command -v alfworld-download >/dev/null; then
            err "alfworld-download not found on PATH (\$HOME/.local/bin not in PATH?)"
            exit 1
        fi
        alfworld-download -f
    fi
fi

# --------------------------- step D: smoke tests -----------------------------

log "==== D: run smokes ===="

# D.1 inference smoke
log "D.1: inference smoke"
python "$PROJ/scripts/smoke_4b_inference.py" \
    --model "$MODEL_DIR" \
    2>&1 | tee "$LOG_DIR/inference.log"

INFERENCE_RC=${PIPESTATUS[0]}
if [[ "$INFERENCE_RC" -ne 0 ]]; then
    err "inference smoke failed (rc=$INFERENCE_RC); see $LOG_DIR/inference.log"
    exit "$INFERENCE_RC"
fi

if [[ $INFERENCE_ONLY -eq 1 ]]; then
    log "--inference-only set: stopping after inference smoke"
    exit 0
fi

# D.2 alfworld smokes (one per memory mode)
for mode in "${MODES[@]}"; do
    log "D.2: alfworld smoke (mode=$mode, max_steps=$MAX_STEPS)"
    python "$PROJ/scripts/smoke_4b_alfworld.py" \
        --model "$MODEL_DIR" \
        --alfworld-data "$ALFWORLD_DATA" \
        --mode "$mode" \
        --max-steps "$MAX_STEPS" \
        2>&1 | tee "$LOG_DIR/alfworld_${mode}.log"
    rc=${PIPESTATUS[0]}
    if [[ "$rc" -ne 0 ]]; then
        err "alfworld smoke (mode=$mode) had failures (rc=$rc); continuing"
    fi
done

log "==== ALL SMOKES DONE ===="
log "logs at: $LOG_DIR"
ls -la "$LOG_DIR"
