"""
LangGraph Agent 工具定义
包括: 论文内检索工具、章节获取工具、Python 代码沙箱
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from loguru import logger


# ===== 检索工具工厂 =====

def create_retrieval_tool(retrieval_orchestrator, paper_id: str):
    """
    创建绑定到特定论文的检索工具。

    Args:
        retrieval_orchestrator: RetrievalOrchestrator 实例
        paper_id: 当前分析的论文 ID
    """
    orchestrator = retrieval_orchestrator

    @tool
    def retrieve_from_paper(query: str, strategy: str = "hybrid", k: int = 10) -> str:
        """
        从当前论文中检索与查询相关的段落。

        Args:
            query: 检索查询文本
            strategy: 检索策略 (hybrid/dense/sparse/hyde)
            k: 返回结果数量 (默认10)

        Returns:
            相关的论文段落文本，包含章节信息
        """
        try:
            results = orchestrator.retrieve(
                query=query,
                paper_id=paper_id,
                strategy=strategy,  # type: ignore
                k=k,
            )
            if not results:
                return "未找到相关段落。"

            output_parts = []
            for i, doc in enumerate(results[:k]):
                section = doc.metadata.get("section_title", "Unknown")
                score = doc.score
                output_parts.append(
                    f"[Result {i+1}] Score: {score:.3f} | Section: {section}\n{doc.text[:500]}"
                )

            return "\n\n---\n\n".join(output_parts)

        except Exception as e:
            return f"检索失败: {str(e)}"

    return retrieve_from_paper


@tool
def get_paper_section(section_title: str, paper_text: str = "") -> str:
    """
    获取论文中指定章节的完整内容。

    Args:
        section_title: 章节标题关键词（如 "Method", "Experiment", "Introduction"）
        paper_text: 论文全文（由系统自动提供）

    Returns:
        匹配章节的完整文本
    """
    if not paper_text:
        return "错误: 论文文本不可用"

    # 简单章节匹配
    lines = paper_text.split('\n')
    found_section = []
    in_section = False
    for line in lines:
        if section_title.lower() in line.lower() and len(line.strip()) < 150:
            in_section = True
        elif in_section:
            # 检测到下一个章节标题（大写开头 + 短行）
            if line.strip() and line.strip()[0].isupper() and len(line.strip()) < 100:
                if any(kw in line.lower() for kw in ['introduction', 'method', 'experiment',
                                                       'result', 'conclusion', 'reference',
                                                       'abstract', 'related', 'appendix']):
                    break
            found_section.append(line)

    if found_section:
        return '\n'.join(found_section[:100])  # 限制最大 100 行
    return f"未找到包含 '{section_title}' 的章节"


def create_python_repl_tool(timeout: int = 30):
    """
    创建安全的 Python 代码执行工具（沙箱）。

    Args:
        timeout: 执行超时秒数
    """

    @tool
    def python_repl(code: str) -> str:
        """
        在沙箱环境中执行 Python 代码。用于验证论文算法复现。

        Args:
            code: 要执行的 Python 代码

        Returns:
            标准输出结果，或错误信息
        """
        from src.tools.code_executor import execute_and_format
        return execute_and_format(code, timeout=timeout)

    return python_repl
