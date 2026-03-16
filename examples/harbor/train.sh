#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B-Instruct-2507
   --ref-load /root/Qwen3-4B-Instruct-2507_torch_dist/
   # --load /root/Qwen3-8B_slime/
   --save /root/Qwen3-4B_slime/
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data /data/harbor/swebench-verified/swebench-verified_2026_03_12.parquet
   --input-key prompt
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 1

   # eval args
   # --eval-interval 25
   # --eval-prompt-data swebench_verified_test /root/harbor-data/swebench_verified/test.parquet@[0:500]
   # --eval-input-key prompt
   # --eval-label-key reward_model
   # --n-samples-per-eval-prompt 1

   --global-batch-size 64
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --log-probs-chunk-size 512

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8196
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # whether enabling TIS
   # --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-team n-alignment
   --wandb-project blackbox-rl-agent
   --wandb-group harbor-qwen3-4b-sweverified-codex-bsz8-onpolicy
   --wandb-key 852fc153cc57c9266416085a2d78f2ca9277e671
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   --sglang-tool-call-parser qwen
   --use-slime-router
   --sglang-router-policy consistent_hashing
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.harbor.harbor_rollout.generate
   --custom-rm-path examples.harbor.harbor_rollout.reward_func
)

# ---------------------------------------------------------------------------
# Multi-node setup
# ---------------------------------------------------------------------------
# For multi-node training set these before running:
#   MASTER_ADDR=<head node IP>
#   NUM_NODES=<total node count>          (default: 1)
#   NODE_RANK=<this node's rank 0..N-1>  (default: 0)
#
# On head node (rank 0):   bash train.sh
# On worker nodes (rank>0): MASTER_ADDR=<head> NODE_RANK=<rank> bash train.sh
#   Worker nodes skip ray start --head and join the existing cluster instead.
# ---------------------------------------------------------------------------

NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_GPUS_PER_NODE=8
HARBOR_N_CONCURRENT="${HARBOR_N_CONCURRENT:-16}"

# Each rollout worker on a node needs its own port range.
# With n_concurrent_tasks=16 per worker, stride by 32 (headroom) per rank.
HARBOR_PORT_BASE=$((19000 + NODE_RANK * 32))
HARBOR_CONFIG_PATH="/tmp/swe_harbor_rank_${NODE_RANK}.yaml"

cp "${SCRIPT_DIR}/swe_harbor_codex.yaml" "${HARBOR_CONFIG_PATH}"
cat >> "${HARBOR_CONFIG_PATH}" <<EOF
harbor_proxy_port_base: ${HARBOR_PORT_BASE}
harbor_n_concurrent_tasks: ${HARBOR_N_CONCURRENT}
EOF

CUSTOM_ARGS+=(
   --custom-config-path "${HARBOR_CONFIG_PATH}"
)

if [ "${NODE_RANK}" = "0" ]; then
    ray start --head --node-ip-address "${MASTER_ADDR}" \
        --num-gpus ${NUM_GPUS_PER_NODE} --disable-usage-stats
else
    # Worker node: join the head's Ray cluster
    ray start --address="${MASTER_ADDR}:6379" \
        --num-gpus ${NUM_GPUS_PER_NODE} --disable-usage-stats
    # Workers don't submit the job — only head node does
    echo "Worker node rank=${NODE_RANK} joined Ray cluster at ${MASTER_ADDR}:6379"
    wait
    exit 0
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"BUILDX_NO_DEFAULT_ATTESTATIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node ${NUM_GPUS_PER_NODE} \
   --rollout-num-gpus $((NUM_NODES * NUM_GPUS_PER_NODE)) \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
