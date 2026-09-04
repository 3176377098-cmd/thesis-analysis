#!/bin/bash
# ============================================================
# Docker 容器入口脚本
# 适配 HuggingFace Spaces (PORT 环境变量) 和 自定义服务器
# ============================================================
set -e

PORT="${PORT:-8501}"

echo "============================================"
echo "  Paper Analyzer - Multi-Agent System"
echo "  Listening on: 0.0.0.0:${PORT}"
echo "  Provider: ${LLM_PROVIDER:-deepseek}"
echo "============================================"

exec streamlit run src/ui/app.py \
    --server.address=0.0.0.0 \
    --server.port="${PORT}" \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableXsrfProtection=true
