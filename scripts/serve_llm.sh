#!/usr/bin/env bash
# Serve Gemma 4 E4B (compressed-tensors w4a16) on vLLM 0.23.0 with the Gemma 4
# MTP drafter for speculative decoding. NO TriAttention. Throughput-optimized.

set -euo pipefail

VENV=${VENV:-/home/walkie/robocup2026/triattention/.venv023}
WEIGHTS=/home/walkie/robocup2026/triattention/weights
MODEL=${MODEL:-$WEIGHTS/gemma-4-E4B-it-qat-w4a16-ct}
DRAFT=${DRAFT:-$WEIGHTS/gemma-4-E4B-it-assistant}

PORT=${PORT:-8000}
SERVED_NAME=${SERVED_NAME:-qwen3.5-9b}
MTP=${MTP:-on}
MTP_TOKENS=${MTP_TOKENS:-3}
EAGER=${EAGER:-off}
KV_DTYPE=${KV_DTYPE:-auto}
MAXLEN=${MAXLEN:-8192}
GPU_UTIL=${GPU_UTIL:-0.60}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
TOOLS=${TOOLS:-on}
STRUCTURED=${STRUCTURED:-on}        # enforce JSON/grammar even inside reasoning (needed when --reasoning-parser is on)

GPU_UTIL_CAP=0.75
if awk "BEGIN{exit !($GPU_UTIL > $GPU_UTIL_CAP)}"; then
  echo "⚠️  GPU_UTIL=$GPU_UTIL exceeds 75% cap; clamping to $GPU_UTIL_CAP"
  GPU_UTIL=$GPU_UTIL_CAP
fi

export VLLM_PLUGINS=""
export ENABLE_TRIATTENTION=0

# FlashInfer ships in vLLM 0.23.0 but its top-k/top-p sampler is JIT-compiled and
# there's no matching CUDA-13 nvcc here, so fall back to vLLM's native PyTorch
# sampler. Attention already uses the Triton backend (no flashinfer JIT needed).
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}

EXTRA_ARGS=()

# --- max tensor compression: serve the existing w4a16 compressed-tensors weights ---
EXTRA_ARGS+=(--quantization compressed-tensors)
echo "🔢 weights: compressed-tensors w4a16 (max tensor compression)"

# --- MTP speculative decoding via the Gemma 4 assistant drafter ---
if [ "$MTP" = "on" ]; then
  EXTRA_ARGS+=(--speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":$MTP_TOKENS}")
  echo "⚡ MTP ON  drafter=$DRAFT  num_speculative_tokens=$MTP_TOKENS"
else
  echo "🐢 MTP OFF"
fi

# --- CUDA graphs (throughput). Only force eager when explicitly requested. ---
if [ "$EAGER" = "on" ]; then
  EXTRA_ARGS+=(--enforce-eager)
  echo "🦥 enforce-eager ON (no CUDA graphs)"
else
  echo "🚀 CUDA graphs ON (throughput)"
fi

# --- optional fp8 KV cache (extra memory compression; needs flashinfer, bundled in 0.23.0) ---
if [ "$KV_DTYPE" != "auto" ]; then
  EXTRA_ARGS+=(--kv-cache-dtype "$KV_DTYPE")
  echo "🧊 KV cache dtype: $KV_DTYPE"
fi

# --- tool calling + reasoning (for agent ability tests; pure output-parsing, no throughput cost) ---
if [ "$TOOLS" = "on" ]; then
  EXTRA_ARGS+=(--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4)
  echo "🛠️  tool-calling (gemma4) + reasoning (gemma4) parsers ON"
fi

# --- structured outputs: force grammar enforcement during reasoning ---
# With --reasoning-parser set (above), vLLM suspends the structured-output grammar
# bitmask until the model emits the reasoning-end token (<channel|> for gemma4).
# Gemma often never closes a thinking channel, so reasoning_ended never flips and
# response_format/guided_json/guided_choice are SILENTLY IGNORED (free prose).
# enable_in_reasoning=true (default false) applies the grammar from the first
# token, so json_schema extraction (e.g. langchain with_structured_output) works.
#
# disable_any_whitespace=true forces COMPACT json: without it, the xgrammar grammar
# permits arbitrary whitespace between fields and greedy decoding gets stuck in an
# infinite "\n  " loop after a few fields (never emitting the next key) — a
# deterministic hang on multi-field schemas, independent of MTP. Compact output
# removes the whitespace branch entirely. This flag REQUIRES an explicit backend
# of xgrammar/guidance (it errors with the default "auto").
if [ "$STRUCTURED" = "on" ]; then
  EXTRA_ARGS+=(--structured-outputs-config '{"backend": "xgrammar", "enable_in_reasoning": true, "disable_any_whitespace": true}')
  echo "📐 structured outputs: xgrammar, enable_in_reasoning + compact (json_schema/guided_* enforced)"
fi

# Gemma 4 is multimodal; cap mm inputs to keep the text-serving footprint small.
EXTRA_ARGS+=(--limit-mm-per-prompt '{"image":0,"audio":0,"video":0}')

echo "🟢 Gemma4 serve  port=$PORT  maxlen=$MAXLEN  gpu_util=$GPU_UTIL  max_num_seqs=$MAX_NUM_SEQS"

exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-prefix-caching \
    --trust-remote-code \
    --max-num-batched-tokens 1024 \
    "${EXTRA_ARGS[@]}"