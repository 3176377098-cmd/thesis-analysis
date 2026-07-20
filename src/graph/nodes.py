"""
LangGraph 图节点函数 - 将各 Agent 封装为标准节点
"""

from typing import Any, Callable

from src.agents.base import BaseAgent
from src.agents.state import PaperAnalysisState


def make_agent_node(agent: BaseAgent) -> Callable:
    """
    工厂函数：将 Agent 实例转为 LangGraph 节点函数。

    节点函数签名: (state: PaperAnalysisState) -> dict[str, Any]
    返回的 dict 被 LangGraph 自动合并到全局状态。
    """
    async def agent_node(state: PaperAnalysisState) -> dict[str, Any]:
        state["current_step"] = agent.role
        result = await agent.run(state)
        return result

    return agent_node


# ===== 实用节点 =====

def retrieve_node(state: PaperAnalysisState, orchestrator) -> dict[str, Any]:
    """
    检索节点 - 在 Agent 管道开始前执行。

    从 state 获取 query/paper_id/retrieval_strategy，执行检索，
    将结果写入 state.retrieved_docs。
    """
    from src.indexing.retrieval import RetrievalStrategy

    query = state.get("query", "")
    paper_id = state.get("paper_id", "")
    strategy = state.get("retrieval_strategy", "hybrid")

    if not query or not paper_id:
        return {"status": "error", "errors": ["Missing query or paper_id"]}

    try:
        results = orchestrator.retrieve(
            query=query,
            paper_id=paper_id,
            strategy=strategy,  # type: ignore
            k=20,
        )

        retrieved = [
            {
                "chunk_id": doc.chunk_id,
                "text": doc.text,
                "score": doc.score,
                "retrieval_method": doc.retrieval_method,
                "metadata": doc.metadata,
            }
            for doc in results
        ]

        return {
            "retrieved_docs": retrieved,
            "status": "retrieved",
            "execution_path": ["retrieve"],
        }

    except Exception as e:
        return {"status": "error", "errors": [f"Retrieval failed: {str(e)}"]}


def parse_pdf_node(state: PaperAnalysisState, ingestion_pipeline) -> dict[str, Any]:
    """
    PDF 解析节点 - 导入管线专用。
    解析 PDF 并写入 state.paper_text / state.chunks。
    """
    from pathlib import Path

    paper_path = state.get("paper_path", "")
    if not paper_path:
        return {"status": "error", "errors": ["Missing paper_path"]}

    try:
        paper_id = ingestion_pipeline.ingest(Path(paper_path))
        return {
            "paper_id": paper_id,
            "status": "parsed",
            "execution_path": ["parse_pdf"],
        }
    except Exception as e:
        return {"status": "error", "errors": [f"PDF parsing failed: {str(e)}"]}
