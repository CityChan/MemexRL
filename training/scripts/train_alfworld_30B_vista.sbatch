#!/usr/bin/env bash
#SBATCH -J memex_30b_alfworld
#SBATCH -o /work/09281/chc_1996/vista/MemexRL/logs/memex_30b_%j.out
#SBATCH -e /work/09281/chc_1996/vista/MemexRL/logs/memex_30b_%j.err
#SBATCH -t 48:00:00
#SBATCH -N 8
#SBATCH -p gh
#SBATCH -A AST24021
# =============================================================================
# train_alfworld_30B_vista.sbatch
#
# Production multi-node training of Memex+Slime on Vista (GH200).
# Default: 8 nodes × 1 GH200 = 8 GPUs total, 48 hours.
#
# Adapted from training/scripts/run_alfworld_qwen3_30B_A3B_memex.sh:
#   - Multi-node Ray cluster (head on first node, workers on rest) via srun
#   - GH200 single-GPU-per-node parallelism: TP=1, EP=NUM_GPUS (was TP=4 EP=8 for 1x8H100)
#   - All patches we figured out during 4B smoke (sample contract, slime
#     monkey-patch, no expandable_segments, etc.)
#   - aarch64 / Hopper env vars from ContextGraph VISTA_NOTES.md
#
# Usage:
#
#   # Full production (8 nodes, 48h, 3000 rollouts)
#   sbatch training/scripts/train_alfworld_30B_vista.sbatch
#
#   # Mini-production smoke (4 nodes, 2h, 50 rollouts)
#   sbatch -N 4 -t 02:00:00 \
#       --export=ALL,NUM_ROLLOUT=50,EXPERIMENT_NAME=mini_smoke \
#       training/scripts/train_alfworld_30B_vista.sbatch
#
#   # Test graph_db mode
#   sbatch --export=ALL,COMPRESSION_MODE=graph_db \
#       training/scripts/train_alfworld_30B_vista.sbatch
#
# Required env (set via --export=ALL after exporting locally, or hard-code):
#   WANDB_KEY      Weights & Biases API key (required for production tracking)
# =============================================================================

set -e

# ============================================================================
# Paths
# ============================================================================

PROJ="${PROJ:-/work/09281/chc_1996/vista/MemexRL}"
SLIME_ROOT="${SLIME_ROOT:-$PROJ/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-$PROJ/Megatron-LM}"
MEMEX_ROOT="${MEMEX_ROOT:-$PROJ}"
MEMEX_SLIME_ROOT="${MEMEX_SLIME_ROOT:-$PROJ/training}"

MODEL_HF="${MODEL_HF:-$PROJ/models/Qwen3-30B-A3B-Thinking-2507}"
MODEL_PATH="${MODEL_PATH:-${MODEL_HF}-int4}"
MODEL_TD="${MODEL_TD:-${MODEL_PATH}_torch_dist}"
MODEL_SLIME_CKPT="${MODEL_SLIME_CKPT:-${MODEL_PATH}_slime_memex}"

DATA_DIR="${DATA_DIR:-$PROJ/data/alfworld}"
ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.alfworld}"

mkdir -p "$PROJ/logs" "$PROJ/smoke_logs"

# ============================================================================
# Environment activation
# ============================================================================

# Vista login modules pollute PATH and pin nvidia/24.7's nvc++. Purge first.
if command -v module >/dev/null; then
    module purge 2>/dev/null || true
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$PROJ/.micromamba}"
eval "$(micromamba shell hook -s bash)"
micromamba activate slime

export PATH="$HOME/.local/bin:$PATH"
export ALFWORLD_DATA

# ============================================================================
# Pre-flight checks (head node only)
# ============================================================================

for d in "$MODEL_PATH" "$MODEL_TD" "$DATA_DIR" "$ALFWORLD_DATA"; do
    [[ -d "$d" ]] || { echo "[ERROR] missing $d" >&2; exit 1; }
done
[[ -f "$DATA_DIR/alfworld_train.jsonl" ]] || { echo "[ERROR] missing alfworld_train.jsonl" >&2; exit 1; }
[[ -n "${WANDB_KEY:-}" ]] || echo "[WARN] WANDB_KEY not set — wandb logging disabled"

# ============================================================================
# Apply Slime monkey-patch for MemexRL list[Sample] contract.
# (idempotent — only patches if marker absent)
# ============================================================================

python <<PY
import os
src = os.path.join("$SLIME_ROOT", "slime", "ray", "rollout.py")
marker = "# MEMEX_PATCH: flatten nested Sample lists"
with open(src) as f: code = f.read()
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
    insertion_point = code.find("\ndef ")
    code = code[:insertion_point + 1] + helper + code[insertion_point + 1:]
for needle, var in [
    ("def compute_metrics_from_samples(args, samples):", "samples"),
    ("def compute_perf_metrics_from_samples(args, samples, rollout_time):", "samples"),
    ("def _log_rollout_data(rollout_id, args, data, metrics, rollout_time):", "data"),
]:
    if needle in code:
        injected = needle + f"\n    {var} = _memex_flatten({var})\n"
        if injected not in code:
            code = code.replace(needle, injected, 1)
with open(src, "w") as f: f.write(code)
print("[memex-patch] slime/ray/rollout.py patched (or already)")
PY

# ============================================================================
# Vista aarch64 / Hopper environment
# ============================================================================

export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export TORCHDYNAMO_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export SGLANG_DISABLE_SLEEP_MODE=1
export RAY_memory_usage_threshold=0.98
# DO NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments — breaks SGLang CUDA graphs.

# ============================================================================
# Multi-node Ray cluster (Slurm-driven)
# ============================================================================

NUM_NODES="${SLURM_NNODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
NUM_GPUS=$(( NUM_NODES * NUM_GPUS_PER_NODE ))

# Resolve nodelist
nodes=$(scontrol show hostnames "$SLURM_NODELIST")
head_node=$(echo "$nodes" | head -n1)
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

echo "=========================================="
echo "Memex+Slime production training"
echo "=========================================="
echo "Job ID    : ${SLURM_JOB_ID:-?}"
echo "Nodes     : $NUM_NODES (${nodes//$'\n'/ })"
echo "Head      : $head_node ($head_ip)"
echo "GPUs/node : $NUM_GPUS_PER_NODE"
echo "Total GPU : $NUM_GPUS"
echo "Model     : $MODEL_PATH"
echo "Data      : $DATA_DIR"
echo "ALFWORLD  : $ALFWORLD_DATA"
echo "WandB     : $([[ -n ${WANDB_KEY:-} ]] && echo enabled || echo disabled)"
echo "=========================================="

# Stop any leftovers
ray stop --force 2>/dev/null || true
pkill -9 ray sglang 2>/dev/null || true
sleep 2

# Start ray head
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head \
        --node-ip-address "$head_ip" \
        --port 6379 \
        --num-gpus "$NUM_GPUS_PER_NODE" \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --temp-dir "${RAY_TMPDIR:-/tmp/ray_$SLURM_JOB_ID}" \
    &
sleep 10

# Start ray workers on the remaining nodes
worker_nodes=$(echo "$nodes" | tail -n +2)
for node in $worker_nodes; do
    srun --nodes=1 --ntasks=1 -w "$node" \
        ray start --address="$head_ip:6379" \
            --num-gpus "$NUM_GPUS_PER_NODE" \
            --disable-usage-stats \
            --temp-dir "${RAY_TMPDIR:-/tmp/ray_$SLURM_JOB_ID}" \
        &
done
sleep 15

echo "Ray cluster status:"
ray status --address="$head_ip:6379" || true

# ============================================================================
# Memex agent config (production)
# ============================================================================

MEMEX_ENV_TYPE="alfworld"
MEMEX_COMPRESSION_MODE="${COMPRESSION_MODE:-lossless_db}"
MEMEX_TOOL_CALL_FORMAT="qwen"
MEMEX_MAX_STEPS="${MAX_STEPS:-50}"
MEMEX_CONTEXT_THRESHOLD="${CONTEXT_THRESHOLD:-8000}"
MEMEX_AUTO_COMPRESS="true"
MEMEX_DISABLE_RETRIEVE="false"
ALFWORLD_HIDE_ADMISSIBLE_COMMANDS="${HIDE_ADMISSIBLE_COMMANDS:-true}"
ALFWORLD_HIDE_INITIAL_OBS="${HIDE_INITIAL_OBS:-true}"
ALFWORLD_LIMIT_LOOK="${LIMIT_LOOK:-true}"
MEMEX_MAX_SUMMARY_TOKENS="${MAX_SUMMARY_TOKENS:-300}"
MEMEX_MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-32000}"
MEMEX_PARALLEL_ENV="${PARALLEL_ENV:-true}"
MEMEX_REWARD_SHAPER_ENABLE="${REWARD_SHAPER_ENABLE:-true}"
MEMEX_LAMBDA_CTX="${LAMBDA_CTX:-1}"
MEMEX_LAMBDA_RED="${LAMBDA_RED:-0.05}"
MEMEX_LAMBDA_FORMAT="${LAMBDA_FORMAT:-1}"

# ============================================================================
# Slime model config — Qwen3-30B-A3B (MoE)
# ============================================================================

# Auto-detect rope_theta from the actual config (Thinking=1e7, Instruct=5e6)
MODEL_ARGS_ROTARY_BASE=$(python - <<PY
import json
print(int(json.load(open("$MODEL_HF/config.json")).get("rope_theta", 10000000)))
PY
)
echo "[smoke] auto-detected rope_theta = $MODEL_ARGS_ROTARY_BASE"
export MODEL_ARGS_ROTARY_BASE

# shellcheck source=/dev/null
source "$SLIME_ROOT/scripts/models/qwen3-30B-A3B.sh"

PRECISION_MODE="${PRECISION_MODE:-int4}"

# ============================================================================
# Slime args
# ============================================================================

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${MODEL_TD}/"
    --load "${MODEL_SLIME_CKPT}/"
    --save "${MODEL_SLIME_CKPT}/"
    --save-interval "${SAVE_INTERVAL:-20}"
)

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/alfworld_train.jsonl"
    --input-key prompt
    --rollout-shuffle
    --num-rollout "${NUM_ROLLOUT:-3000}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
    --n-samples-per-prompt "${N_SAMPLES:-8}"
    --rollout-max-response-len "${MEMEX_MAX_CONTEXT_LEN}"
    --rollout-max-context-len "${MEMEX_MAX_CONTEXT_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
    --balance-data
)
# Skip the dynamic-sampling filter — its API doesn't match what MemexRL's
# adapter currently returns. Set DYNAMIC_SAMPLING_FILTER=1 to re-enable.
[[ "${DYNAMIC_SAMPLING_FILTER:-0}" == "1" ]] && \
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)

EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL:-5}"
    --eval-prompt-data alfworld-test "${DATA_DIR}/alfworld_test.jsonl"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "${MEMEX_MAX_CONTEXT_LEN}"
    --eval-top-k 1
)

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
# CPU offload required for 30B MoE (~140 GB optimizer state). GH200 has
# unified memory so the cost is small; without offload optimizer + model
# wouldn't fit on one GPU.
if [[ "${USE_CPU_OFFLOAD:-1}" == "1" ]]; then
    OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d)
fi

# Parallelism: GH200 nodes have 1 GPU each. Default to TP=1, EP=NUM_GPUS
# (all experts spread across all GPUs, MoE-friendly). Override for finer
# control.
PERF_ARGS=(
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --tensor-model-parallel-size "${TP_SIZE:-1}"
    --sequence-parallel
    --pipeline-model-parallel-size "${PP_SIZE:-1}"
    --context-parallel-size 1
    --expert-model-parallel-size "${EP_SIZE:-$NUM_GPUS}"
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.70
    --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
    --use-slime-router
)

CUSTOM_ARGS=(--custom-generate-function-path generate_with_memex.generate)

WANDB_ARGS=()
if [[ -n "${WANDB_KEY:-}" ]]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project "${PROJECT_NAME:-memex-vista-30b}"
        --wandb-group "${EXPERIMENT_NAME:-alfworld-grpo-30b-${MEMEX_COMPRESSION_MODE}-$(date +%Y%m%d)}"
        --wandb-key "${WANDB_KEY}"
    )
fi

# ============================================================================
# Runtime env (passed into Ray actors on every node)
# ============================================================================

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([[ "$NVLINK_COUNT" -gt 0 ]] && echo 1 || echo 0)

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
# Submit the training job
# ============================================================================

ray job submit \
    --address="http://$head_ip:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_ROOT}/train.py" \
    --actor-num-nodes "$NUM_NODES" \
    --actor-num-gpus-per-node "$NUM_GPUS_PER_NODE" \
    --rollout-num-gpus "$NUM_GPUS" \
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

# ============================================================================
# Cleanup
# ============================================================================

echo "Stopping Ray cluster..."
for node in $nodes; do
    srun --nodes=1 --ntasks=1 -w "$node" ray stop --force || true &
done
wait
echo "Done."
