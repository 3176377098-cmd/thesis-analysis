"""
LangGraph 图构建器 - 定义 Multi-Agent 工作流拓扑
包含: AnalysisGraph (多Agent分析) + IngestionGraph (论文导入)
"""

from typing import Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from src.agents.state import PaperAnalysisState
from src.agents.base import BaseAgent
from src.graph.nodes import make_agent_node, retrieve_node, parse_pdf_node


# ===== 条件路由函数 =====

def should_revise(state: PaperAnalysisState) -> Literal["deepread", "synthesis"]:
    """
    判断是否需要从 Critique 回到 DeepRead 修正。
    最多 2 次修正循环。
    """
    critique = state.get("critique_result", {})
    needs = critique.get("needs_revision", False)
    count = state.get("revision_count", 0)
    max_revisions = 2  # 可根据配置调整

    if needs and count < max_revisions:
        logger.info(f"触发修正循环 (第 {count + 1}/{max_revisions} 次)")
        return "deepread"
    return "synthesis"


def should_reproduce_code(state: PaperAnalysisState) -> Literal["code", "end"]:
    """判断是否触发代码复现节点"""
    deep_read = state.get("deep_read_analysis", {})
    if deep_read.get("contains_algorithm", False):
        logger.info("检测到算法描述，触发代码复现")
        return "code"
    return "end"


def route_by_depth(state: PaperAnalysisState) -> Literal["rag_qa", "metadata", "end"]:
    """根据分析深度路由"""
    depth = state.get("analysis_depth", "full")
    if depth == "basic":
        return "rag_qa"
    elif depth in ("advanced", "full"):
        return "metadata"
    return "end"


# ===== 图构建 =====

def build_analysis_graph(
    metadata_agent: BaseAgent,
    deepread_agent: BaseAgent,
    critique_agent: BaseAgent,
    synthesis_agent: BaseAgent,
    code_agent: BaseAgent,
    retrieval_orchestrator=None,
    checkpointer=None,
) -> StateGraph:
    """
    构建 Multi-Agent 分析图。

    拓扑:
        START → retrieve → metadata → deepread → critique
              → [needs_revision?] → deepread (loop, max 2x)
              → synthesis → [has_algorithm?] → code → END

    Args:
        metadata_agent: 元数据提取 Agent
        deepread_agent: 精读分析 Agent
        critique_agent: 批判审查 Agent
        synthesis_agent: 综述生成 Agent
        code_agent: 代码复现 Agent
        retrieval_orchestrator: 检索编排器（用于检索节点）
        checkpointer: LangGraph 检查点存储（默认 MemorySaver）

    Returns:
        编译后的 StateGraph
    """
    # 创建图
    workflow = StateGraph(PaperAnalysisState)

    # 注册节点
    if retrieval_orchestrator:
        workflow.add_node(
            "retrieve",
            lambda state: retrieve_node(state, retrieval_orchestrator)
        )

    workflow.add_node("metadata", make_agent_node(metadata_agent))
    workflow.add_node("deepread", make_agent_node(deepread_agent))
    workflow.add_node("critique", make_agent_node(critique_agent))
    workflow.add_node("synthesis", make_agent_node(synthesis_agent))
    workflow.add_node("code", make_agent_node(code_agent))

    # 简易 RAG 问答节点（basic 模式）
    workflow.add_node("rag_qa", _make_rag_qa_node())

    # ===== 边和条件路由 =====

    # 入口
    workflow.set_entry_point("retrieve" if retrieval_orchestrator else "metadata")

    if retrieval_orchestrator:
        # retrieve → metadata → deepread
        workflow.add_edge("retrieve", "metadata")
    workflow.add_edge("metadata", "deepread")

    # deepread → critique
    workflow.add_edge("deepread", "critique")

    # critique → [条件] → deepread (修正循环) or synthesis
    workflow.add_conditional_edges(
        "critique",
        should_revise,
        {
            "deepread": "deepread",
            "synthesis": "synthesis",
        }
    )

    # synthesis → [条件] → code or END
    workflow.add_conditional_edges(
        "synthesis",
        should_reproduce_code,
        {
            "code": "code",
            "end": END,
        }
    )

    workflow.add_edge("code", END)

    # RAG QA 节点直通 END
    workflow.add_edge("rag_qa", END)

    # 编译
    memory = checkpointer or MemorySaver()
    compiled = workflow.compile(checkpointer=memory)

    logger.info("AnalysisGraph 编译完成")
    return compiled


def build_ingestion_graph(ingestion_pipeline) -> StateGraph:
    """
    构建论文导入管线图。

    拓扑: START → parse_pdf → END
    """
    workflow = StateGraph(PaperAnalysisState)

    workflow.add_node(
        "parse_pdf",
        lambda state: parse_pdf_node(state, ingestion_pipeline)
    )

    workflow.set_entry_point("parse_pdf")
    workflow.add_edge("parse_pdf", END)

    compiled = workflow.compile()
    logger.info("IngestionGraph 编译完成")
    return compiled


def _make_rag_qa_node():
    """
    简易 RAG 问答节点 - basic 模式使用。
    不做多 Agent 分析，仅做检索 + 单次 LLM 回答。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    async def rag_qa_node(state: PaperAnalysisState) -> dict:
        from config.llm_config import create_fast_model

        query = state.get("query", "")
        docs = state.get("retrieved_docs", [])

        llm = create_fast_model()

        context = "\n\n---\n\n".join(
            f"[{i+1}] {doc.get('text', '')[:500]}"
            for i, doc in enumerate(docs[:5])
        )

        prompt = f"""基于以下论文章节内容回答问题。如果无法从提供的内容中找到答案，请明确说明。

论文内容:
{context}

问题: {query}

请提供基于论文内容的准确回答。"""

        try:
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            answer = f"RAG 问答失败: {str(e)}"

        return {
            "synthesis_result": {
                "executive_summary": answer[:300],
                "integrated_analysis": answer,
                "key_findings": [],
                "contradictions_resolved": [],
                "practical_implications": [],
                "limitations_acknowledged": [],
                "future_work_suggested": [],
                "citation_map": [],
            },
            "status": "complete",
            "execution_path": ["rag_qa"],
        }

    return rag_qa_node
