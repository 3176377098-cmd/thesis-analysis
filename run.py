"""
系统启动器 - 启动 Streamlit UI 或 CLI 模式
"""

import sys
import os
import argparse
from pathlib import Path

# Windows: 强制 UTF-8 编码，解决中文显示和 emoji 问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="基于 Multi-Agent 架构的学术论文深度解析系统"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="ui",
        choices=["ui", "cli", "test"],
        help="运行模式: ui (Web界面), cli (命令行), test (运行测试)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="CLI 模式: 论文 PDF 路径",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="CLI 模式: 分析查询",
    )
    parser.add_argument(
        "--depth",
        type=str,
        default="full",
        choices=["basic", "advanced", "full"],
        help="分析深度",
    )

    args = parser.parse_args()

    if args.mode == "ui":
        # 启动 Streamlit
        import subprocess
        ui_path = PROJECT_ROOT / "src" / "ui" / "app.py"
        print(f"[UI] Starting Web Interface...")
        print(f"[UI] Project: {PROJECT_ROOT}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        subprocess.run([
            "streamlit", "run", str(ui_path),
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.serverAddress", "localhost",
        ], env=env)

    elif args.mode == "cli":
        if not args.pdf:
            print("[ERROR] CLI mode requires --pdf parameter")
            sys.exit(1)

        print(f"[CLI] Paper: {args.pdf}")
        print(f"[CLI] Query: {args.query or '(interactive)'}")
        print(f"[CLI] Depth: {args.depth}")
        print()
        print("[INFO] CLI mode under development. Use UI: python run.py ui")

    elif args.mode == "test":
        import subprocess
        print("[TEST] Running tests...")
        subprocess.run(["pytest", "tests/", "-v", "--tb=short"])


if __name__ == "__main__":
    main()
