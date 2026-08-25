#!/bin/bash
# 啟動 FreeToken 伺服器
# 用法: ./start_freetoken.sh [模型] [port]
#   模型預設: Qwen/Qwen2.5-32B-Instruct
#   port 預設: 1919

MODEL="${1:-Qwen/Qwen2.5-32B-Instruct}"
PORT="${2:-1919}"

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

ft serve --model "$MODEL" --port "$PORT"
