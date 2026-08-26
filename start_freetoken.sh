#!/bin/bash
# 啟動 FreeToken 伺服器
# 用法: ./start_freetoken.sh [模型] [port]
#   模型預設: Qwen/Qwen2.5-32B-Instruct
#   port 預設: 1919

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${2:-1919}"
MEMORY_RATIO="${3:-0.9}"  # 14B 大模型請用 0.05

echo "🚀 啟動 FreeToken..."
echo "   模型: $MODEL"
echo "   端點: http://localhost:$PORT"
echo ""

source "$(dirname "$0")/venv/bin/activate"

# 確認 ft 指令存在
if ! command -v ft &>/dev/null; then
    echo "❌ ft 指令不存在，請先安裝 FreeToken:"
    echo "   pip install 'freetoken[accel]'"
    exit 1
fi

# CUDA 環境（nvcc 12.9 + torch cu130 mismatch override）
export PATH="/usr/local/cuda-12.9/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.9
export TVM_FFI_CUDA_ARCH_LIST=12.0
export FREETOKEN_ALLOW_CUDA_MISMATCH=1

ft serve --model-path "$MODEL" --port "$PORT" --host 0.0.0.0 --memory-ratio "$MEMORY_RATIO"
