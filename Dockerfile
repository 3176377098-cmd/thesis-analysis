# ============================================================
# Multi-Agent 学术论文深度解析系统 - Docker 镜像
# ============================================================

FROM python:3.10-slim

LABEL org.opencontainers.image.title="Paper Analyzer"
LABEL org.opencontainers.image.description="RAG + Multi-Agent 学术论文智能分析平台"

# ---- 系统依赖 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Python 依赖 (利用 Docker 层缓存) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ---- 复制项目代码 ----
COPY . .

# ---- 预创建数据目录 ----
RUN mkdir -p data/papers data/parsed data/chroma data/traces

# ---- 预加载嵌入模型 (可选, 加快首次启动) ----
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-zh-v1.5', device='cpu')" 2>/dev/null || true

# ---- 环境变量 ----
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1
ENV STREAMLIT_CLOUD=1

# ---- 容器端口 (HuggingFace Spaces 会通过 PORT 环境变量覆盖) ----
EXPOSE 8501

# ---- 健康检查 ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# ---- 入口 ----
COPY scripts/docker-entrypoint.sh /app/scripts/docker-entrypoint.sh
RUN chmod +x /app/scripts/docker-entrypoint.sh

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
