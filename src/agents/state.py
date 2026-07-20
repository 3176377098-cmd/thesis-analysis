"""
LangGraph 全局状态定义 - PaperAnalysisState
所有 Agent 节点通过此状态进行数据交换
"""

from typing import TypedDict, Literal, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class ChunkMetadata(TypedDict, total=False):
    """文本块溯源信息"""
    chunk_id: str
    section_title: str
    section_level: int
    page_range: tuple[int, int]
    token_count: int
    contains_table: bool
    contains_equation: bool
    prev_chunk_id: str
    next_chunk_id: str


class ChunkDict(TypedDict):
    """文本块"""
    text: str
    metadata: ChunkMetadata


class RetrievedDoc(TypedDict):
    """检索结果"""
    chunk_id: str
    text: str
    score: float
    retrieval_method: str


class AgentTrace(TypedDict):
    """单次 Agent 执行记录"""
    agent_name: str
    step: str
    started_at: str       # ISO timestamp
    duration_ms: int
    input_summary: str
    output_summary: str
    tool_calls: list[dict]
    token_usage: dict     # {"prompt": N, "completion": N}


class PaperAnalysisState(TypedDict, total=False):
    """
    LangGraph 全局分析状态。

    在整个 Multi-Agent 工作流中流转，各个节点读取并更新此状态。
    所有值必须是 JSON 可序列化的（支持 LangGraph Checkpointing）。
    """

    # ===== 论文标识 =====
    paper_id: str
    paper_path: str
    source: str                             # "upload" | "arxiv" | "semantic_scholar" | "text"

    # ===== 解析结果 =====
    paper_text: str                         # 全文纯文本
    paper_metadata: dict                    # 结构化元数据 (title, authors, abstract, etc.)
    chunks: list[dict]                      # 所有 Chunk
    tables: list[dict]                      # 提取的表格

    # ===== 用户交互 =====
    query: str                              # 当前用户查询
    query_history: list[dict]               # 历史 Q&A
    retrieval_strategy: str                 # "hybrid" | "hyde" | "multi_query"

    # ===== 检索结果 =====
    retrieved_docs: list[RetrievedDoc]      # 当前检索结果

    # ===== Agent 输出 =====
    metadata_analysis: dict                 # Metadata Agent 输出
    deep_read_analysis: dict                # Deep Read Agent 输出
    critique_result: dict                   # Critique Agent 输出
    synthesis_result: dict                  # Synthesis Agent 输出
    code_reproduction: dict                 # Code Agent 输出

    # ===== 流程控制 =====
    analysis_depth: str                     # "basic" | "advanced" | "full"
    revision_count: int                     # Critique→DeepRead 循环计数
    current_step: str                       # 当前执行节点
    messages: Annotated[list[BaseMessage], add_messages]

    # ===== 追踪 =====
    agent_traces: list[AgentTrace]          # 所有 Agent 的执行记录
    execution_path: list[str]               # 已执行的节点路径

    # ===== 状态 =====
    status: str                             # "initialized" | "retrieved" | "analyzing" | "complete" | "error"
    errors: list[str]                       # 错误信息
