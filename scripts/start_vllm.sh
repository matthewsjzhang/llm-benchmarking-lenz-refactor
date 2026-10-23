#!/usr/bin/env bash
# Start a local vLLM server for testing the pipeline on this laptop (WSL2, RTX 4060 8 GB).
# The remote desktop (gemma-4-31B-it via SSH tunnel on :11436) remains the main inference host.
#
# Usage (inside WSL):   bash scripts/start_vllm.sh
# Then run the pipeline: python 00_call_models.py --vllm_url http://localhost:8000/v1 ...
#
# Every setting can be overridden with an environment variable, e.g.
#   MAX_MODEL_LEN=24576 bash scripts/start_vllm.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"   # 4-bit AWQ, ~5.5 GB of weights
PORT="${PORT:-8000}"                             # 11436 is reserved for the SSH tunnel
MAX_MODEL_LEN="${MAX_MODEL_LEN:-22016}"          # what fits in the KV cache with fp16 (~22.2k tokens). Largest
                                                 # prompt is 18,775 tokens (German, 10 defs), so German runs
                                                 # need --vllm_max_tokens <= 3200; English fits the default 8192
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.86}"            # only ~6.9 of 8 GiB is free under WSL; 0.90 fails to start
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-2048}"  # prefill in 2k-token chunks to keep peak activation memory low
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"         # fp16. fp8 would double the context but produced garbled
                                                 # output with this AWQ model (no KV scaling factors)
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"  # FlashInfer (auto pick with fp8 KV) needs the full CUDA toolkit (nvcc); Triton only needs gcc
VENV="${VENV:-$HOME/venvs/vllm}"

source "$VENV/bin/activate"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"  # same nvcc requirement; use the PyTorch sampler

# Bind to 127.0.0.1 only: reachable from this laptop (WSL forwards localhost to Windows), not the network.
exec vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --attention-backend "$ATTENTION_BACKEND" \
  --max-num-seqs 1 \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --enforce-eager \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  --seed 0 \
  "$@"
