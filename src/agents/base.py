"""
Agent 基类 - 定义统一的 Agent 运行模式
包含: 工具绑定、结构化输出、追踪记录、错误处理
"""

import time
import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from loguru import logger

from .state import PaperAnalysisState, AgentTrace


class BaseAgent(ABC):
    """
    Multi-Agent 系统的基础 Agent 抽象类。

    每个 Agent 子类需提供:
    - role: Agent 角色名
    - system_prompt: 系统提示词
    - output_schema: 结构化输出的 Pydantic 模型
    - run_impl(): 核心推理逻辑
    """

    role: str = "base"
    output_key: str = ""  # 写入 state 的 key

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool] | None = None,
        output_schema: Type[BaseModel] | None = None,
    ):
        self.llm = llm
        self.tools = tools or []
        self.output_schema = output_schema

    async def run(self, state: PaperAnalysisState) -> dict:
        """
        执行 Agent 的完整运行流程: 准备 → 推理 → 解析 → 追踪。

        Args:
            state: 当前全局状态

        Returns:
            需合并到 state 的字典 {output_key: result, "agent_traces": [trace], "execution_path": [...]}
        """
        start_time = time.time()
        logger.info(f"[{self.role}] 开始分析...")

        trace = AgentTrace(
            agent_name=self.role,
            step=state.get("current_step", ""),
            started_at=datetime.now().isoformat(),
            duration_ms=0,
            input_summary="",
            output_summary="",
            tool_calls=[],
            token_usage={},
        )

        try:
            # 1. 构建消息
            messages = self._build_messages(state)

            # 2. 调用 LLM (no structured output - get raw text)
            trace["input_summary"] = self._summarize_input(messages)
            response = await self.llm.ainvoke(messages)

            # 3. 提取文本内容
            text = response.content if hasattr(response, 'content') else str(response)
            if text is None:
                text = ""

            # 4. 尝试解析为 JSON，否则保留原文
            output = self._parse_text(text)

            # 5. 记录追踪
            elapsed_ms = int((time.time() - start_time) * 1000)
            trace["duration_ms"] = elapsed_ms
            trace["status"] = "completed"
            trace["output_summary"] = text[:500]

            logger.info(f"[{self.role}] completed ({elapsed_ms}ms)")

            result = {
                self.output_key: output if isinstance(output, dict) else (
                    output.model_dump() if hasattr(output, 'model_dump') else str(output)
                ),
                "agent_traces": [trace],
                "execution_path": [self.role],
                "current_step": self.role,
                "status": "analyzing",
            }

            return result

        except Exception as e:
            logger.error(f"[{self.role}] 执行失败: {e}")
            elapsed_ms = int((time.time() - start_time) * 1000)
            trace["duration_ms"] = elapsed_ms
            trace["output_summary"] = f"ERROR: {str(e)}"

            return {
                self.output_key: {"error": str(e)},
                "agent_traces": [trace],
                "execution_path": [self.role],
                "errors": [f"[{self.role}] {str(e)}"],
                "status": "error",
            }

    def _build_messages(self, state: PaperAnalysisState) -> list:
        """构建发送给 LLM 的消息列表"""
        from langchain_core.messages import SystemMessage, HumanMessage

        system_prompt = self.get_system_prompt(state)
        user_prompt = self._build_user_prompt(state)

        messages = [SystemMessage(content=system_prompt)]

        # 附加上下文（检索到的文本块）
        retrieved = state.get("retrieved_docs", [])
        if retrieved:
            context = "\n\n---\n\n".join(
                f"[Source {i+1}] (Section: {r.get('metadata', {}).get('section_title', 'N/A')})\n{r['text']}"
                for i, r in enumerate(retrieved[:10])
            )
            messages.append(HumanMessage(
                content=f"以下是论文相关段落:\n\n{context}\n\n---\n\n{user_prompt}"
            ))
        else:
            messages.append(HumanMessage(content=user_prompt))

        return messages

    def _build_user_prompt(self, state: PaperAnalysisState) -> str:
        """构建用户提示（子类可覆盖）"""
        query = state.get("query", "")
        paper_text = state.get("paper_text", "")[:4000]  # 前 4000 字符
        return f"论文内容:\n{paper_text}\n\n分析任务: {query}"

    @abstractmethod
    def get_system_prompt(self, state: PaperAnalysisState) -> str:
        """获取 Agent 的系统提示词"""
        ...

    def _parse_text(self, text: str) -> dict:
        """Parse LLM text response. Try JSON, otherwise return as content."""
        if not text:
            return {"content": "(empty response)"}
        # Try to parse as JSON
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        # Try code block extraction
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, TypeError):
                pass
        # Return raw text
        return {"content": text}

    def _summarize_input(self, messages: list) -> str:
        """生成输入摘要（用于追踪）"""
        if not messages:
            return ""
        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        return content[:300] + ("..." if len(content) > 300 else "")

    def _build_context_from_retrieved(self, state: PaperAnalysisState) -> str:
        """从检索结果构建上下文文本"""
        docs = state.get("retrieved_docs", [])
        if not docs:
            return ""
        parts = []
        for i, doc in enumerate(docs[:8]):
            section = doc.get("metadata", {}).get("section_title", "")
            parts.append(f"[Chunk {i+1}] (Section: {section})\n{doc['text']}")
        return "\n\n---\n\n".join(parts)
