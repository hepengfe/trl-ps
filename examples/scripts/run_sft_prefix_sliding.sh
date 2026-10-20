#!/bin/bash
# Launch prefix-sliding SFT for Qwen3-1.7B on OpenThoughts-114k.
#
# Usage:
#   NUM_GPUS=4 bash examples/scripts/run_sft_prefix_sliding.sh
#   NUM_GPUS=1 bash examples/scripts/run_sft_prefix_sliding.sh [path/to/config.yaml]
set -euo pipefail

CONFIG="${1:-examples/cli_configs/sft_prefix_sliding_qwen3_1.7b.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

# For multi-node / ZeRO-2+ scaling, pass a deepspeed config instead, e.g.:
#   accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml ...
accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    examples/scripts/sft_prefix_sliding.py \
    --config "$CONFIG"
