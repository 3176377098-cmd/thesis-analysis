"""
Streamlit Cloud 入口 — 基于 Multi-Agent 架构的学术论文深度解析系统

部署到 Streamlit Community Cloud (免费):
  1. 推送仓库到 GitHub (git push)
  2. 访问 https://share.streamlit.io
  3. 连接 GitHub 仓库
  4. Main file path 设置为: streamlit_app.py
  5. 可选: 在 Advanced Settings 中设置 Python 版本为 3.11+
  6. 可选: 在 Secrets 中预设 DEEPSEEK_API_KEY

首次启动会自动下载 BGE 嵌入模型 (~24MB)，后续启动使用缓存秒开。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 直接运行主 UI 应用
app_path = PROJECT_ROOT / "src" / "ui" / "app.py"
with open(app_path, encoding="utf-8") as f:
    code = f.read()
exec(compile(code, str(app_path), "exec"))
