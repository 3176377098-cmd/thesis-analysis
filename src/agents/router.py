"""
Agent 条件路由器 - 根据论文类型和分析深度决定 Agent 执行路径
"""

from typing import Literal

from loguru import logger

from src.agents.state import PaperAnalysisState


def route_analysis_depth(state: PaperAnalysisState) -> Literal["rag_qa", "metadata", "end"]:
    """
    根据分析深度路由到不同的执行路径。

    - basic: 直接走 RAG 问答（检索 + 一次 LLM 调用）
    - advanced: 走 Agent 编排（Metadata → DeepRead → Critique → Synthesis）
    - full: 完整流程（Advanced + Code Reproduce）
    """
    depth = state.get("analysis_depth", "full")

    if depth == "basic":
        logger.info("路由: basic → RAG Q&A")
        return "rag_qa"
    elif depth in ("advanced", "full"):
        logger.info(f"路由: {depth} → Multi-Agent 分析")
        return "metadata"

    logger.warning(f"未知分析深度: {depth}，默认路由到 Multi-Agent")
    return "metadata"


def should_revise(state: PaperAnalysisState) -> Literal["deepread", "synthesis"]:
    """
    判断 Critique Agent 后是否需要回到 DeepRead 进行修正。

    判断依据: critique_result.needs_revision == True 且 revision_count < max
    """
    critique = state.get("critique_result", {})
    needs = critique.get("needs_revision", False)
    count = state.get("revision_count", 0)
    max_revisions = 2

    if needs and count < max_revisions:
        state["revision_count"] = count + 1
        logger.info(f"Critique → DeepRead 修正循环 (第 {count + 1}/{max_revisions} 次)")
        return "deepread"

    logger.info("Critique 通过，进入 Synthesis")
    return "synthesis"


def should_reproduce_code(state: PaperAnalysisState) -> Literal["code", "end"]:
    """
    判断是否需要触发代码复现。

    条件:
    1. 分析深度为 full
    2. Deep Read 发现论文包含算法
    """
    depth = state.get("analysis_depth", "full")
    deep_read = state.get("deep_read_analysis", {})

    if depth == "full" and deep_read.get("contains_algorithm", False):
        logger.info("触发 Code Reproduction Agent")
        return "code"

    logger.info("跳过代码复现")
    return "end"


def route_retrieval_strategy(state: PaperAnalysisState) -> str:
    """
    根据查询类型推荐检索策略。

    启发式规则:
    - 短查询 (<10词): hybrid (混合检索，覆盖更广)
    - 长查询 (>=10词): dense (语义检索，匹配更精确)
    - 含"对比"/"compare": multi_query (多角度检索)
    """
    query = state.get("query", "")
    strategy = state.get("retrieval_strategy", "hybrid")

    # 如果用户已手动指定，尊重用户选择
    if strategy != "hybrid":
        return strategy

    # 自动选择
    word_count = len(query.split())
    if word_count <= 5:
        return "hybrid"
    elif any(kw in query.lower() for kw in ["compare", "对比", "difference", "区别"]):
        return "multi_query"
    else:
        return "dense"
