#!/bin/bash
# Rollout-only debug run — sources train.sh config and adds --debug-rollout-only.
# Runs the full harbor rollout pipeline (SGLang + ModelInterceptProxy + Trial)
# without any training or gradient updates.
#
# Usage:
#   bash debug_rollout.sh
#   bash debug_rollout.sh --save-debug-rollout-data /tmp/rollout_{}.pkl

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export PYTHONBUFFERED=16
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B-Instruct-2507
   --ref-load /root/Qwen3-4B-Instruct-2507_torch_dist/
)

ROLLOUT_ARGS=(
   --prompt-data /data/harbor/swebench-verified/swebench-verified_2026_03_12.parquet
   --input-key prompt
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 1
   --rollout-batch-size 4
   --n-samples-per-prompt 1
   --rollout-max-response-len 32768
   --rollout-temperature 1
   --global-batch-size 4
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-tool-call-parser qwen
   --use-slime-router
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.harbor.harbor_rollout.generate
   --custom-rm-path examples.harbor.harbor_rollout.reward_func
)

DEBUG_ARGS=(
   --debug-rollout-only
   --save-debug-rollout-data /tmp/harbor_debug_rollout_{rollout_id}.pkl
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
HARBOR_PORT_BASE="${HARBOR_PROXY_PORT_BASE:-19000}"
HARBOR_N_CONCURRENT="${HARBOR_N_CONCURRENT:-16}"
HARBOR_CONFIG_PATH="/tmp/swe_harbor_debug.yaml"

cp "${SCRIPT_DIR}/swe_harbor.yaml" "${HARBOR_CONFIG_PATH}"
cat >> "${HARBOR_CONFIG_PATH}" <<EOF
harbor_proxy_port_base: ${HARBOR_PORT_BASE}
harbor_n_concurrent_tasks: ${HARBOR_N_CONCURRENT}
EOF

CUSTOM_ARGS+=(
   --custom-config-path "${HARBOR_CONFIG_PATH}"
)

# Start Ray head only if not already running on this machine.
if ray status &>/dev/null; then
   echo "Ray already running — reusing existing cluster."
else
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${DEBUG_ARGS[@]} \
   "$@"
