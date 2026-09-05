# 基于 Multi-Agent 架构的学术论文深度解析系统

RAG + Multi-Agent 结合的学术论文智能分析平台。

> 🚀 **在线体验**: https://thesis-analysis-nfchfc7rvkg9mnxjxhivwu.streamlit.app

## 功能概述

- **📄 PDF 智能解析**: PyMuPDF 高速文本提取 + 章节结构检测 + 表格识别
- **🔍 多策略检索**: 混合检索(Hybrid)、HyDE、Multi-Query、BM25 + 向量融合
- **🤖 Multi-Agent 协作**: 元数据 → 精读 → 批判 → 综述 → 代码复现（LangGraph 编排）
- **📊 评测仪表盘**: Hit Rate / MRR / NDCG 检索质量指标 + Agent Trace 可视化
- **🌐 多数据源**: 本地上传 / arXiv / Semantic Scholar / 知网 / 万方（合法合规）
- **🖼️ 图片智能分析**: Gemini 免费多模态模型解析论文图表、架构图、实验结果图

## 支持模型

| 模型 | 费用 | 图片分析 | 说明 |
|------|------|:---:|------|
| **DeepSeek** | 付费（约 ¥1/百万 token） | ❌ | 文本分析质量最佳 |
| **OpenAI (GPT-4o)** | 付费 | ✅ | 支持多模态 |
| **Gemini** | 免费（1500 次/天） | ✅ | 免费图片分析首选 |
| **Ollama** | 免费离线 | ❌ | 本地部署，隐私安全 |

## 快速开始

### 1. 环境要求
- Python 3.10+
- Windows / Linux / macOS

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置 API Key
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key（DeepSeek / OpenAI / Gemini 任选）
```

### 4. 启动
```bash
# Web 界面
python run.py ui

# CLI 模式
python run.py cli --pdf path/to/paper.pdf --query "这篇论文的主要贡献是什么？"
```

## 项目结构

```
paper-analyzer/
├── config/          # 配置（LLM、路径、参数）
├── src/
│   ├── parsing/     # PDF 解析层
│   ├── processing/  # 文本预处理 + 学术分块
│   ├── indexing/    # 嵌入 + ChromaDB + BM25 + 检索
│   ├── agents/      # 5个 Agent + LangGraph 状态
│   ├── graph/       # LangGraph 图构建 + 工具
│   ├── tools/       # 搜索/代码执行/引文工具
│   ├── evaluation/  # 评测指标 + Trace 追踪
│   ├── sources/     # arXiv/Semantic Scholar/知网/万方
│   └── ui/          # Streamlit 界面
├── data/            # 运行时数据
├── tests/           # 测试用例
└── run.py           # 启动器
```

## 分析深度

| 模式 | 流程 | 说明 |
|------|------|------|
| **Basic** | 检索 → LLM 生成 | 快速问答 |
| **Advanced** | Metadata → DeepRead → Critique → Synthesis | 多 Agent 深度分析 |
| **Full** | Advanced + Code Reproduce | 含算法复现 |

## 合规声明

- 知网/万方：仅提供接口框架，需用户自行配置昆明理工大学图书馆合法凭证
- arXiv API：遵守 rate limit (1 req/3s)，学术研究用途
- 不内置任何爬虫、paywall 绕过或批量下载功能
- 所有分析基于合法获取的论文内容

## License

MIT - 学术用途
