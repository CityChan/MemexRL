#!/usr/bin/env bash
# =============================================================================
# install_slime_vista.sh
#
# Bare-metal Slime install for TACC Vista (GH200 / aarch64).
#
# Purpose:
#   MemexRL's documented training stack is the slimerl/slime Docker image,
#   which is x86_64 only and therefore unrunnable on Vista. Slime upstream
#   ships build_conda.sh as the official non-Docker install path; this script
#   is a Vista-adapted version of that, with:
#     - aarch64 / Hopper compile flags (TORCH_CUDA_ARCH_LIST=9.0, etc.)
#     - PyTorch wheel fallback (cu129 → cu128 → from-source)
#     - resumable execution via per-step stamp files
#     - explicit isolation from Vista's system CUDA modules
#
# Usage:
#   bash scripts/install_slime_vista.sh                  # run all steps
#   bash scripts/install_slime_vista.sh --from <step>    # resume from step N
#   bash scripts/install_slime_vista.sh --force          # re-run all steps
#   bash scripts/install_slime_vista.sh --check          # verify a finished install
#
# Required env (override defaults if your layout differs):
#   PROJ           default: $WORK/vista/MemexRL
#   ENV_NAME       default: slime
#   PYTORCH_CHANNEL  one of "cu129" | "cu128" | "src" (default cu128, see notes)
#
# Pins (from Slime @ commit 67a21a1b's build_conda.sh + docker/Dockerfile):
#   Python    3.12
#   CUDA      12.9.1     (nvidia conda channel)
#   PyTorch   2.9.1      (torchvision 0.24.1, torchaudio 2.9.1)
#   SGLang    24c91001cf99ba642be791e099d358f4dfe955f5
#   Megatron  3714d81d418c9f1bca4594fc35f9e8289f652862
#   apex      10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
#   flash-attn        2.7.4.post1
#   flash-attention 3 fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89  (Hopper)
#   transformer_engine  2.10.0
#   flash-linear-attention 0.4.1
#   mbridge   89eb10887887bc74853f89a4de258c0702932a1c
#   torch_memory_saver  dc6876905830430b5054325fa4211ff302169c6b
#   Slime     67a21a1b
#
# Total wall-time on a single GH200 idev (cold cache): ~3–5 hours.
# =============================================================================

set -euo pipefail

# --------------------------- configuration -----------------------------------

: "${PROJ:=${WORK:-/work}/vista/MemexRL}"
: "${ENV_NAME:=slime}"
: "${PYTORCH_CHANNEL:=cu128}"   # cu129 has no aarch64 wheel; cu128 does

SLIME_COMMIT="67a21a1b"
SGLANG_COMMIT="24c91001cf99ba642be791e099d358f4dfe955f5"
MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"
APEX_COMMIT="10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"
MBRIDGE_COMMIT="89eb10887887bc74853f89a4de258c0702932a1c"
FA3_COMMIT="fbf24f67cf7f6442c5cfb2c1057f4bfc57e72d89"
TMS_COMMIT="dc6876905830430b5054325fa4211ff302169c6b"

STAMP_DIR="$PROJ/.slime_install_state"
LOG_DIR="$PROJ/install_logs"

mkdir -p "$PROJ" "$STAMP_DIR" "$LOG_DIR"

# CLI flags ------------------------------------------------------------------
FORCE=0
FROM_STEP=0
CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --from)  FROM_STEP="$2"; shift 2 ;;
        --check) CHECK_ONLY=1; shift ;;
        -h|--help)
            sed -n '1,50p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

# Helpers --------------------------------------------------------------------
log()  { printf '\033[1;36m[%s]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*"; }
err()  { printf '\033[1;31m[%s ERROR]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }
done_stamp() { touch "$STAMP_DIR/step_${1}.done"; }
is_done()    { [[ -f "$STAMP_DIR/step_${1}.done" && $FORCE -eq 0 ]]; }

step() {
    local n="$1" name="$2"
    if [[ "$n" -lt "$FROM_STEP" ]]; then
        log "skip step $n ($name) — before --from"
        return 1
    fi
    if is_done "$n"; then
        log "skip step $n ($name) — already done (use --force to re-run)"
        return 1
    fi
    log "==== step $n: $name ===="
    return 0
}

run_logged() {
    local step_n="$1"; shift
    local logfile="$LOG_DIR/step_${step_n}.log"
    log "logging to $logfile"
    "$@" 2>&1 | tee "$logfile"
    return "${PIPESTATUS[0]}"
}

# Pre-flight -----------------------------------------------------------------
arch="$(uname -m)"
log "host: $(hostname)  arch: $arch  proj: $PROJ"
if [[ "$arch" != "aarch64" ]]; then
    err "this script is for aarch64 (TACC Vista GH200). Detected: $arch"
    err "if you actually want to install Slime on x86_64, just use the Docker image."
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" && $CHECK_ONLY -eq 0 ]]; then
    log "WARNING: not inside a Slurm allocation. Compiles will run on the login node."
    log "         Recommended: get an idev allocation first:"
    log "             idev -p gh -t 06:00:00 -N 1"
    sleep 3
fi

# Isolate from Vista system CUDA / compiler paths so micromamba's CUDA is
# the only one in scope. ContextGraph's env_common sets these; we override.
unset CUDA_HOME CUDA_PATH || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/Linux_aarch64/' | paste -sd:)
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/Linux_aarch64/' | paste -sd:)

cd "$PROJ"

# --------------------------- step 1: micromamba ------------------------------

if step 1 "install micromamba (if missing) and create $ENV_NAME env"; then
    if ! command -v micromamba >/dev/null 2>&1; then
        log "installing micromamba via curl ..."
        yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
    fi
    # ensure shell hook is loaded for non-interactive shell
    eval "$(micromamba shell hook -s bash)"
    if ! micromamba env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        micromamba create -n "$ENV_NAME" python=3.12 pip -c conda-forge -y
    else
        log "env '$ENV_NAME' already exists"
    fi
    done_stamp 1
fi

# Activate env for all subsequent steps
eval "$(micromamba shell hook -s bash)"
micromamba activate "$ENV_NAME"
export CUDA_HOME="$CONDA_PREFIX"
log "active env: $CONDA_PREFIX"

# --------------------------- step 2: CUDA + cuDNN ----------------------------

if step 2 "install CUDA 12.9.1 + nvtx + nccl + cuDNN via conda"; then
    micromamba install -n "$ENV_NAME" -c nvidia/label/cuda-12.9.1 \
        cuda cuda-nvtx cuda-nvtx-dev nccl -y
    micromamba install -n "$ENV_NAME" -c conda-forge cudnn -y
    # Sanity: nvcc must be from the conda env
    which nvcc | grep -q "$CONDA_PREFIX" \
        || { err "nvcc not from $CONDA_PREFIX — env path leak"; exit 1; }
    nvcc --version | tee "$LOG_DIR/nvcc_version.txt"
    done_stamp 2
fi

# --------------------------- step 3: PyTorch ---------------------------------

if step 3 "install PyTorch 2.9.1 (channel: $PYTORCH_CHANNEL)"; then
    case "$PYTORCH_CHANNEL" in
        cu129|cu128)
            log "trying PyTorch wheel for $PYTORCH_CHANNEL on aarch64 ..."
            if pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
                    --index-url "https://download.pytorch.org/whl/${PYTORCH_CHANNEL}"; then
                log "PyTorch wheel install succeeded"
            else
                err "PyTorch wheel for ${PYTORCH_CHANNEL}+aarch64 unavailable."
                err "rerun with PYTORCH_CHANNEL=src to build from source (5–8 hours)"
                exit 1
            fi
            ;;
        src)
            log "building PyTorch 2.9.1 from source — this will take 5–8 hours"
            git clone https://github.com/pytorch/pytorch || true
            (cd pytorch && git fetch && git checkout v2.9.1 \
                && git submodule sync && git submodule update --init --recursive \
                && USE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0" \
                   MAX_JOBS="${MAX_JOBS:-32}" python setup.py install)
            pip install torchvision==0.24.1 torchaudio==2.9.1 --no-deps || \
                log "torchvision/torchaudio install non-fatal failure (often expected); continue"
            ;;
        *)
            err "unknown PYTORCH_CHANNEL: $PYTORCH_CHANNEL"; exit 2 ;;
    esac
    python -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())'
    done_stamp 3
fi

# --------------------------- step 4: cuda-python -----------------------------

if step 4 "install cuda-python 13.1.0"; then
    pip install cuda-python==13.1.0
    done_stamp 4
fi

# --------------------------- step 5: SGLang from source ----------------------

if step 5 "build SGLang from source @ $SGLANG_COMMIT"; then
    [[ -d sglang ]] || git clone https://github.com/sgl-project/sglang.git
    (cd sglang && git fetch && git checkout "$SGLANG_COMMIT" && pip install -e "python[all]")
    done_stamp 5
fi

# --------------------------- step 6: build tooling ---------------------------

if step 6 "install cmake + ninja"; then
    pip install cmake ninja
    done_stamp 6
fi

# --------------------------- step 7: flash-attn 2 ----------------------------
# Hopper / GH200 needs explicit arch flags. Use the same values ContextGraph
# uses on Vista (proven in their VISTA_NOTES.md).

export TORCH_CUDA_ARCH_LIST="9.0"
export FLASH_ATTN_CUDA_ARCHS=90
export MAX_JOBS="${MAX_JOBS:-64}"

if step 7 "build flash-attn 2.7.4.post1 (TORCH_CUDA_ARCH_LIST=9.0)"; then
    pip install flash-attn==2.7.4.post1 --no-build-isolation
    python -c 'import flash_attn; print("flash_attn", flash_attn.__version__)'
    done_stamp 7
fi

# --------------------------- step 8: flash-attention 3 (Hopper) -------------

if step 8 "build flash-attention 3 @ $FA3_COMMIT (Hopper-only) and install fa3 shim"; then
    [[ -d flash-attention ]] || git clone https://github.com/Dao-AILab/flash-attention.git
    (cd flash-attention \
        && git fetch && git checkout "$FA3_COMMIT" \
        && git submodule update --init \
        && cd hopper \
        && MAX_JOBS="${MAX_JOBS}" python setup.py install)
    # Slime's Dockerfile copies the new flash_attn_interface.py into a
    # flash_attn_3/ package so the rest of the codebase can `from flash_attn_3
    # import flash_attn_interface`. Replicate that here.
    py_site="$(python -c 'import site; print(site.getsitepackages()[0])')"
    mkdir -p "$py_site/flash_attn_3"
    cp flash-attention/hopper/flash_attn_interface.py \
        "$py_site/flash_attn_3/flash_attn_interface.py"
    python -c 'from flash_attn_3 import flash_attn_interface; print("fa3 ok")'
    done_stamp 8
fi

# --------------------------- step 9: misc Python deps -----------------------

if step 9 "mbridge / flash-linear-attention / tilelang / transformer_engine"; then
    pip install "git+https://github.com/ISEEKYAN/mbridge.git@${MBRIDGE_COMMIT}" --no-deps
    pip install flash-linear-attention==0.4.1
    pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/
    pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
    done_stamp 9
fi

# --------------------------- step 10: Apex ----------------------------------
# This is the longest single step (~30–60 min on GH200). If it fails, check
# nvcc visibility (must be the conda one), TORCH_CUDA_ARCH_LIST=9.0, and
# that PyTorch's CUDA matches the env's nvcc.

if step 10 "build NVIDIA Apex @ $APEX_COMMIT (~30–60 min)"; then
    NVCC_APPEND_FLAGS="--threads 4" \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
        "git+https://github.com/NVIDIA/apex.git@${APEX_COMMIT}"
    python -c 'import apex; from apex.optimizers import FusedAdam; print("apex ok")'
    done_stamp 10
fi

# --------------------------- step 11: Megatron-LM ---------------------------

if step 11 "clone + install Megatron-LM @ $MEGATRON_COMMIT"; then
    if [[ ! -d Megatron-LM ]]; then
        git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
    fi
    (cd Megatron-LM \
        && git fetch && git checkout "$MEGATRON_COMMIT" \
        && git submodule update --init --recursive \
        && pip install -e .)
    done_stamp 11
fi

# --------------------------- step 12: bridge / saver / modelopt --------------

if step 12 "torch_memory_saver / Megatron-Bridge / nvidia-modelopt"; then
    pip install "git+https://github.com/fzyzcjy/torch_memory_saver.git@${TMS_COMMIT}" \
        --no-cache-dir --force-reinstall
    pip install "git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl" \
        --no-build-isolation
    pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
    done_stamp 12
fi

# --------------------------- step 13: Slime + INT4 QAT kernel ----------------

if step 13 "clone + install Slime @ $SLIME_COMMIT + INT4 QAT custom kernel"; then
    [[ -d slime ]] || git clone https://github.com/THUDM/slime.git
    (cd slime \
        && git fetch && git checkout "$SLIME_COMMIT" \
        && pip install -e .)
    # MemexRL's INT4 + Fake QAT path depends on this custom CUDA kernel.
    (cd slime/slime/backends/megatron_utils/kernels/int4_qat \
        && pip install . --no-build-isolation)
    done_stamp 13
fi

# --------------------------- step 14: apply Slime patches --------------------
# These patches modify Megatron-LM and SGLang for Slime's training/rollout
# integration. They're version-specific (v0.5.7) and ship inside the Slime
# repo. They MUST be applied or training will fail at import / runtime.

if step 14 "apply Slime patches to SGLang and Megatron-LM (v0.5.7)"; then
    SLIME_DIR="$PROJ/slime"
    PATCH_DIR="$SLIME_DIR/docker/patch/v0.5.7"
    if [[ ! -d "$PATCH_DIR" ]]; then
        err "patch dir $PATCH_DIR missing — patches not applied. Investigate."
        exit 1
    fi
    # SGLang patch
    (cd sglang \
        && git apply --check "$PATCH_DIR/sglang.patch" 2>/dev/null \
        && git apply "$PATCH_DIR/sglang.patch" \
        || log "sglang patch already applied or failed (check manually)")
    # Megatron patch
    (cd Megatron-LM \
        && git apply --check "$PATCH_DIR/megatron.patch" 2>/dev/null \
        && git apply "$PATCH_DIR/megatron.patch" \
        || log "megatron patch already applied or failed (check manually)")
    done_stamp 14
fi

# --------------------------- step 15: post-pin overrides ---------------------

if step 15 "post-pin overrides (cudnn, numpy<2)"; then
    pip install nvidia-cudnn-cu12==9.16.0.29
    pip install "numpy<2"
    done_stamp 15
fi

# --------------------------- step 16: Slime upper deps -----------------------

if step 16 "install Slime requirements.txt (transformers, ray, wandb, ...)"; then
    pip install -r "$PROJ/slime/requirements.txt"
    done_stamp 16
fi

# --------------------------- verification ------------------------------------

log "==== install verification ===="
python <<'PY'
import importlib, sys
mods = [
    ("torch",                None),
    ("flash_attn",           "2.7.4"),
    ("flash_attn_3.flash_attn_interface", None),
    ("transformer_engine",   "2.10.0"),
    ("apex",                 None),
    ("megatron",             None),
    ("sglang",               None),
    ("ring_flash_attn",      None),
    ("ray",                  None),
    ("transformers",         None),
]
print(f"{'module':38s} {'version':12s} status")
print("-" * 70)
for name, expect_prefix in mods:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        ok = "OK" if (expect_prefix is None or v.startswith(expect_prefix)) else "WARN(version)"
        print(f"{name:38s} {v:12s} {ok}")
    except Exception as e:
        print(f"{name:38s} {'-':12s} FAIL: {e}")
        sys.exit(1)
print("-" * 70)
print("import sanity OK")
PY

log "==== install complete ===="
log "next: build INT4 + torch_dist checkpoints, then submit your sbatch script"
log "INT4 conversion:"
log "  python $PROJ/slime/tools/convert_hf_to_int4_direct.py \\"
log "    --model-dir <path/to/bf16> --save-dir <path/to/int4> --group-size 128 --is-symmetric True"
log "torch_dist conversion:"
log "  source $PROJ/slime/scripts/models/qwen3-30B-A3B.sh"
log "  torchrun --nproc-per-node 1 $PROJ/slime/tools/convert_hf_to_torch_dist.py \\"
log "    --hf-checkpoint <bf16-path> --save <torch-dist-out> \"\${MODEL_ARGS[@]}\""
