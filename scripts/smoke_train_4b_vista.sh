#!/usr/bin/env bash
# =============================================================================
# smoke_train_4b_vista.sh
#
# Single-node single-GPU smoke training for Memex+Slime on Vista (GH200).
# Adapted from training/scripts/run_alfworld_qwen3_30B_A3B_memex.sh:
#   - Qwen3-4B-Instruct-2507 (dense, not MoE)         instead of 30B-A3B
#   - 1 GPU, TP=1, EP=1                                instead of 8 / 4 / 8
#   - tiny smoke params: 3 rollouts, batch 4, no save  instead of production
#   - Vista aarch64 specific env vars                  (vLLM / SGLang quirks)
#
# Goal: validate that the full Slime+Megatron+SGLang training loop boots and
# completes a few RL steps on Vista. Not for actual learning. Once this
# passes you can move to multi-node 30B with the original script.
#
# Usage:
#   bash scripts/smoke_train_4b_vista.sh
#
# Override behaviour with env vars:
#   COMPRESSION_MODE=graph_db bash scripts/smoke_train_4b_vista.sh
#   NUM_ROLLOUT=10 ROLLOUT_BATCH_SIZE=8 bash scripts/smoke_train_4b_vista.sh
#   WANDB_KEY=... bash scripts/smoke_train_4b_vista.sh   # enable wandb
# =============================================================================

# Best-effort cleanup of leftover daemons from a previous attempt
pkill -9 sglang 2>/dev/null; sleep 1
ray stop --force 2>/dev/null; pkill -9 ray 2>/dev/null; sleep 2

set -ex

# ============================================================================
# One-shot Slime monkey-patch:
# MemexRL's generate_with_memex returns list[Sample] for episodes that
# triggered auto-compression (one Sample per segment). Slime's
# compute_metrics_from_samples and other downstream code iterate samples
# expecting flat list[Sample] and break on AttributeError.
#
# Idempotent: only patches if the marker comment isn't already present.
# Touches the vendored slime clone at $PROJ/slime, which we control.
# ============================================================================
SLIME_DIR_PATCH="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/slime"
python <<PY
import os
src = os.path.join("$SLIME_DIR_PATCH", "slime", "ray", "rollout.py")
marker = "# MEMEX_PATCH: flatten nested Sample lists"
with open(src) as f:
    code = f.read()

# We may need to apply multiple patches. The marker is on the helper, which
# is added once. The per-function flatten lines are also idempotent (re-running
# just inserts the line after a def... we guard by checking if the line is
# already present).

if "def _memex_flatten(samples):" not in code:
    helper = (
        "def _memex_flatten(samples):\n"
        f"    {marker}\n"
        "    flat = []\n"
        "    for s in samples:\n"
        "        if isinstance(s, list):\n"
        "            flat.extend(s)\n"
        "        else:\n"
        "            flat.append(s)\n"
        "    return flat\n\n\n"
    )
    # Insert at top of file after imports. Find the first 'def ' or 'class '.
    insertion_point = code.find("\ndef ")
    if insertion_point < 0:
        insertion_point = code.find("\nclass ")
    if insertion_point < 0:
        raise SystemExit(f"[memex-patch] could not find a def/class in {src}")
    code = code[:insertion_point + 1] + helper + code[insertion_point + 1:]
    print(f"[memex-patch] inserted _memex_flatten helper")

# Functions that iterate samples and need flattening at the top.
targets = [
    "def compute_metrics_from_samples(args, samples):",
    "def compute_perf_metrics_from_samples(args, samples, rollout_time):",
    "def _log_rollout_data(rollout_id, args, data, metrics, rollout_time):",
]
flatten_call = "    samples = _memex_flatten(samples)\n"
log_flatten_call = "    data = _memex_flatten(data)\n"

for needle in targets:
    if needle not in code:
        # Some functions may not exist on all slime tags — skip silently
        print(f"[memex-patch] skip (not found): {needle}")
        continue
    if needle == targets[2]:
        # _log_rollout_data uses `data` not `samples`
        injected = needle + "\n" + log_flatten_call
    else:
        injected = needle + "\n" + flatten_call
    if injected in code:
        print(f"[memex-patch] already inline-patched: {needle}")
        continue
    code = code.replace(needle, injected, 1)
    print(f"[memex-patch] patched: {needle}")

with open(src, "w") as f:
    f.write(code)
PY

# ============================================================================
# Paths
# ============================================================================

PROJ="${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SLIME_ROOT="${SLIME_ROOT:-$PROJ/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-$PROJ/Megatron-LM}"
MEMEX_ROOT="${MEMEX_ROOT:-$PROJ}"
MEMEX_SLIME_ROOT="${MEMEX_SLIME_ROOT:-$PROJ/training}"

MODEL_HF="${MODEL_HF:-$PROJ/models/Qwen3-4B-Instruct-2507}"
MODEL_PATH="${MODEL_PATH:-${MODEL_HF}-int4}"
MODEL_TD="${MODEL_TD:-${MODEL_PATH}_torch_dist}"
MODEL_SLIME_CKPT="${MODEL_SLIME_CKPT:-${MODEL_PATH}_slime_memex_smoke}"

DATA_DIR="${DATA_DIR:-$PROJ/data/alfworld}"
ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.alfworld}"

# ============================================================================
# Pre-flight checks
# ============================================================================

for d in "$MODEL_PATH" "$MODEL_TD" "$DATA_DIR" "$ALFWORLD_DATA"; do
    if [[ ! -d "$d" ]]; then
        echo "[ERROR] missing required dir: $d" >&2
        echo "" >&2
        echo "  $MODEL_PATH        : run scripts/prep_4b_ckpts_vista.sh" >&2
        echo "  $MODEL_TD          : run scripts/prep_4b_ckpts_vista.sh" >&2
        echo "  $DATA_DIR          : PYTHONPATH=.:training python training/convert_data.py --output-dir $DATA_DIR" >&2
        echo "  $ALFWORLD_DATA     : alfworld-download -f" >&2
        exit 1
    fi
done

[[ -f "$DATA_DIR/alfworld_train.jsonl" ]] || {
    echo "[ERROR] $DATA_DIR/alfworld_train.jsonl missing — run training/convert_data.py" >&2
    exit 1
}

# ============================================================================
# Vista aarch64 / GH200 environment
# Borrowed from ContextGraph's VISTA_NOTES.md — these are the env vars that
# avoid the Hopper sleep-mode + NVRTC failures on Vista.
# ============================================================================

export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export TORCHDYNAMO_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# vLLM sleep mode + free_cache_engine break on Vista's CUDA driver. Disable
# both so the rollout engine doesn't crash on idle.
export SGLANG_DISABLE_SLEEP_MODE=1
# Vista idev compute nodes have ~212 GB CPU memory; with SGLang scheduler
# (~2 GB) + rollout manager + Megatron actor we can brush 95% Ray's
# default OOM-kill threshold during the rollout->train transition before
# SGLang finishes releasing memory. Bump to 0.98 to give some breathing
# room. (RAY_memory_monitor_refresh_ms=0 would disable the monitor
# entirely; we keep it on, just less aggressive.)
export RAY_memory_usage_threshold=0.98

# ============================================================================
# Memex agent config (smoke-friendly)
# ============================================================================

MEMEX_ENV_TYPE="alfworld"
MEMEX_COMPRESSION_MODE="${COMPRESSION_MODE:-lossless_db}"   # change to graph_db when validated
MEMEX_TOOL_CALL_FORMAT="qwen"
MEMEX_MAX_STEPS="${MAX_STEPS:-12}"                          # short episode for smoke
MEMEX_CONTEXT_THRESHOLD="${CONTEXT_THRESHOLD:-4000}"
MEMEX_AUTO_COMPRESS="true"
MEMEX_DISABLE_RETRIEVE="false"

ALFWORLD_HIDE_ADMISSIBLE_COMMANDS="${HIDE_ADMISSIBLE_COMMANDS:-true}"
ALFWORLD_HIDE_INITIAL_OBS="${HIDE_INITIAL_OBS:-true}"
ALFWORLD_LIMIT_LOOK="${LIMIT_LOOK:-true}"
MEMEX_MAX_SUMMARY_TOKENS="${MAX_SUMMARY_TOKENS:-300}"

MEMEX_MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-16000}"           # smaller than 32K
MEMEX_PARALLEL_ENV="${PARALLEL_ENV:-false}"                 # 1 GPU, no need for parallel envs

MEMEX_REWARD_SHAPER_ENABLE="${REWARD_SHAPER_ENABLE:-true}"
MEMEX_LAMBDA_CTX="${LAMBDA_CTX:-1}"
MEMEX_LAMBDA_RED="${LAMBDA_RED:-0.05}"
MEMEX_LAMBDA_FORMAT="${LAMBDA_FORMAT:-1}"

# ============================================================================
# Slime model config
# ============================================================================

# Read rope_theta from the actual HF config.json instead of hard-coding it.
# The Thinking variant uses 1e7, Instruct-2507 uses 5e6, base Qwen3 uses 1e6,
# and Slime's hf_validate_args strictly checks they match. Reading from the
# model's own config makes the script work for any Qwen3 variant.
if [[ -z "${MODEL_ARGS_ROTARY_BASE:-}" ]]; then
    MODEL_ARGS_ROTARY_BASE=$(python - <<PY
import json, sys
cfg = json.load(open("$MODEL_HF/config.json"))
print(int(cfg.get("rope_theta", 1000000)))
PY
)
    echo "[smoke] auto-detected rope_theta = $MODEL_ARGS_ROTARY_BASE from $MODEL_HF/config.json"
fi
export MODEL_ARGS_ROTARY_BASE

MODEL_ARG_FILE="$SLIME_ROOT/scripts/models/qwen3-4B.sh"
if [[ ! -f "$MODEL_ARG_FILE" ]]; then
    echo "[ERROR] missing $MODEL_ARG_FILE" >&2
    echo "       Slime does not ship a 4B arch script on this commit." >&2
    echo "       Copy/adapt qwen3-8B.sh, or install a newer slime tag." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$MODEL_ARG_FILE"

PRECISION_MODE="${PRECISION_MODE:-int4}"

# ============================================================================
# Slime training args (smoke values)
# ============================================================================

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${MODEL_TD}/"
    --load "${MODEL_SLIME_CKPT}/"
    --save "${MODEL_SLIME_CKPT}/"
    --save-interval 999999            # effectively never save during smoke
)

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/alfworld_train.jsonl"
    --input-key prompt
    --rollout-shuffle
    --num-rollout "${NUM_ROLLOUT:-3}"               # 3 rollouts for smoke
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
    --n-samples-per-prompt "${N_SAMPLES:-2}"
    --rollout-max-response-len "${MEMEX_MAX_CONTEXT_LEN}"
    --rollout-max-context-len "${MEMEX_MAX_CONTEXT_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-8}"
    # Skip the upstream-Slime check_reward_nonzero_std filter for smoke.
    # MemexRL's generate_with_memex returns list[list[Sample]] (two levels)
    # while Slime's filter expects list[Sample]; the two were tested
    # against different Slime tags. Set DYNAMIC_SAMPLING_FILTER=1 to
    # restore it once the filter contract is unified.
    --balance-data
)
if [[ "${DYNAMIC_SAMPLING_FILTER:-0}" == "1" ]]; then
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
fi

EVAL_ARGS=()                                          # skip eval in smoke
if [[ "${ENABLE_EVAL:-false}" == "true" ]]; then
    EVAL_ARGS=(
        --eval-interval 1
        --eval-prompt-data alfworld-test "${DATA_DIR}/alfworld_test.jsonl"
        --n-samples-per-eval-prompt 1
        --eval-max-response-len "${MEMEX_MAX_CONTEXT_LEN}"
        --eval-top-k 1
    )
fi

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef "${KL_COEF:-0.001}"
    --kl-loss-type low_var_kl
    --entropy-coef "${ENTROPY_COEF:-0.002}"
    --eps-clip "${EPS_CLIP:-0.2}"
    --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
)

USE_TIS="${USE_TIS:-true}"
TIS_ARGS=()
if [[ "${USE_TIS}" == "true" ]]; then
    TIS_ARGS=(--use-tis --tis-clip "${TIS_CLIP:-2.0}" --tis-clip-low "${TIS_CLIP_LOW:-0}")
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-5e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --use-precision-aware-optimizer
)
# Optimizer CPU offload: enabled by default in production (30B-A3B + 8 GPUs
# is too big to keep optimizer state on each GPU). Disabled by default for
# this single-GPU smoke because it eats ~118 GB CPU memory for the Adam
# state and Vista idev nodes only allocate ~212 GB CPU memory; SGLang +
# rollout manager + Ray overhead push us past Ray's 0.95 OOM-kill threshold.
# 4B model's optimizer state fits comfortably on a 96 GB GH200, so keeping
# it on GPU is fine. Set USE_CPU_OFFLOAD=1 to re-enable.
if [[ "${USE_CPU_OFFLOAD:-0}" == "1" ]]; then
    OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d)
fi

# Single-GPU dense model: TP=1, EP=1, no MoE expert parallel.
NUM_GPUS="${NUM_GPUS:-1}"
PERF_ARGS=(
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --tensor-model-parallel-size "${TP_SIZE:-1}"
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size "${EP_SIZE:-1}"
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.55         # leave more headroom on a single 96 GB GH200
    --sglang-cuda-graph-bs 1 2 4 8
    --use-slime-router
)

CUSTOM_ARGS=(
    --custom-generate-function-path generate_with_memex.generate
)

WANDB_ARGS=()
if [[ -n "${WANDB_KEY:-}" ]]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project "${PROJECT_NAME:-memex-vista-smoke}"
        --wandb-group "${EXPERIMENT_NAME:-smoke-4b-${MEMEX_COMPRESSION_MODE}}"
        --wandb-key "${WANDB_KEY}"
    )
fi

# ============================================================================
# Banner
# ============================================================================

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([[ "$NVLINK_COUNT" -gt 0 ]] && echo 1 || echo 0)

echo "=========================================="
echo "Memex + Slime smoke training (Vista 1xGH200)"
echo "=========================================="
echo "PROJ          : $PROJ"
echo "Model HF      : $MODEL_HF"
echo "Model INT4    : $MODEL_PATH"
echo "Model TD      : $MODEL_TD"
echo "Slime ckpt    : $MODEL_SLIME_CKPT"
echo "Data          : $DATA_DIR"
echo "ALFWorld data : $ALFWORLD_DATA"
echo "Compression   : $MEMEX_COMPRESSION_MODE"
echo "Rollouts      : ${NUM_ROLLOUT:-3}"
echo "Batch         : rollout=${ROLLOUT_BATCH_SIZE:-4}  global=${GLOBAL_BATCH_SIZE:-8}  samples/prompt=${N_SAMPLES:-2}"
echo "Parallelism   : NUM_GPUS=$NUM_GPUS  TP=${TP_SIZE:-1}  EP=${EP_SIZE:-1}"
echo "WandB         : $([[ -n "${WANDB_KEY:-}" ]] && echo enabled || echo disabled)"
echo "=========================================="

# ============================================================================
# Ray (single node, head only)
# ============================================================================

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=127.0.0.1 \
    --temp-dir "${RAY_TMPDIR:-/tmp/ray_smoke}"

# Give Ray a beat to settle
sleep 3

# ============================================================================
# Runtime env (passed into Ray actors)
# ============================================================================

PRECISION_ENV_VARS=""
if [[ "${PRECISION_MODE}" == "int4" ]]; then
    PRECISION_ENV_VARS=$(cat <<'ENVEOF'
    "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
    "OPEN_TRAINING_INT4_GROUP_SIZE": "128",
ENVEOF
)
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT}:${MEMEX_ROOT}:${MEMEX_SLIME_ROOT}:${SLIME_ROOT}\",
    \"ALFWORLD_DATA\": \"${ALFWORLD_DATA}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"CUDA_HOME\": \"${CONDA_PREFIX:-}\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"VLLM_USE_V1\": \"0\",
    \"VLLM_ATTENTION_BACKEND\": \"FLASH_ATTN\",
    \"TORCHDYNAMO_DISABLE\": \"1\",
    \"SGLANG_DISABLE_SLEEP_MODE\": \"1\",
    ${PRECISION_ENV_VARS}
    \"MEMEX_ENV_TYPE\": \"${MEMEX_ENV_TYPE}\",
    \"MEMEX_COMPRESSION_MODE\": \"${MEMEX_COMPRESSION_MODE}\",
    \"MEMEX_TOOL_CALL_FORMAT\": \"${MEMEX_TOOL_CALL_FORMAT}\",
    \"MEMEX_MAX_STEPS\": \"${MEMEX_MAX_STEPS}\",
    \"MEMEX_CONTEXT_THRESHOLD\": \"${MEMEX_CONTEXT_THRESHOLD}\",
    \"MEMEX_AUTO_COMPRESS\": \"${MEMEX_AUTO_COMPRESS}\",
    \"MEMEX_DISABLE_RETRIEVE\": \"${MEMEX_DISABLE_RETRIEVE}\",
    \"MEMEX_REWARD_SHAPER_ENABLE\": \"${MEMEX_REWARD_SHAPER_ENABLE}\",
    \"MEMEX_LAMBDA_CTX\": \"${MEMEX_LAMBDA_CTX}\",
    \"MEMEX_LAMBDA_RED\": \"${MEMEX_LAMBDA_RED}\",
    \"MEMEX_LAMBDA_FORMAT\": \"${MEMEX_LAMBDA_FORMAT}\",
    \"MEMEX_MAX_CONTEXT_LEN\": \"${MEMEX_MAX_CONTEXT_LEN}\",
    \"MEMEX_PARALLEL_ENV\": \"${MEMEX_PARALLEL_ENV}\",
    \"ALFWORLD_HIDE_ADMISSIBLE_COMMANDS\": \"${ALFWORLD_HIDE_ADMISSIBLE_COMMANDS}\",
    \"ALFWORLD_HIDE_INITIAL_OBS\": \"${ALFWORLD_HIDE_INITIAL_OBS}\",
    \"ALFWORLD_LIMIT_LOOK\": \"${ALFWORLD_LIMIT_LOOK}\",
    \"MEMEX_MAX_SUMMARY_TOKENS\": \"${MEMEX_MAX_SUMMARY_TOKENS}\"
  }
}"

# ============================================================================
# Submit
# ============================================================================

ray job submit \
    --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_ROOT}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    --rollout-num-gpus "${NUM_GPUS}" \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${TIS_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
