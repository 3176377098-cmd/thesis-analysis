"""
Agent 执行追踪器 - 记录每个 Agent 的输入/输出/耗时/Token 用量
用于 Trace 可视化和性能调试
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


class AgentTracer:
    """单个 Agent 执行的上下文追踪器"""

    def __init__(self, agent_name: str, run_id: str = ""):
        self.agent_name = agent_name
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.input_summary = ""
        self.output_summary = ""
        self.tool_calls: list[dict] = []
        self.token_usage: dict = {}
        self.status = "pending"
        self.error: Optional[str] = None

    def start(self, input_summary: str = ""):
        """标记执行开始"""
        self.start_time = time.time()
        self.input_summary = input_summary[:300]
        self.status = "running"

    def end(self, output: dict | str, token_usage: dict | None = None):
        """标记执行结束"""
        self.end_time = time.time()
        if isinstance(output, dict):
            self.output_summary = json.dumps(output, ensure_ascii=False)[:500]
        else:
            self.output_summary = str(output)[:500]
        if token_usage:
            self.token_usage = token_usage
        self.status = "completed"

    def fail(self, error: str):
        """标记执行失败"""
        self.end_time = time.time()
        self.error = error
        self.status = "failed"

    def add_tool_call(self, tool_name: str, args: dict, result: str):
        """记录一次工具调用"""
        self.tool_calls.append({
            "tool": tool_name,
            "args": json.dumps(args, ensure_ascii=False)[:200],
            "result": result[:300],
        })

    def to_dict(self) -> dict:
        """序列化为字典"""
        duration_ms = 0
        if self.start_time and self.end_time:
            duration_ms = int((self.end_time - self.start_time) * 1000)

        return {
            "agent_name": self.agent_name,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else "",
            "duration_ms": duration_ms,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "tool_calls": self.tool_calls,
            "token_usage": self.token_usage,
            "error": self.error,
        }


class TraceCollector:
    """
    Agent 执行追踪收集器。

    收集所有 Agent 在单次分析运行中的执行记录，
    支持持久化到 JSON 文件和查询。
    """

    def __init__(self, traces_dir: Path):
        self.traces_dir = Path(traces_dir)
        self.traces_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, run_id: str, traces: list[dict], metadata: dict | None = None):
        """
        保存一次完整分析的追踪记录。

        Args:
            run_id: 运行唯一标识
            traces: Agent 追踪记录列表
            metadata: 附加元数据 (paper_id, query, etc.)
        """
        record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "traces": traces,
            "agent_count": len(traces),
            "total_duration_ms": sum(t.get("duration_ms", 0) for t in traces),
        }

        filepath = self.traces_dir / f"run_{run_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        logger.info(f"追踪记录已保存: {filepath}")

    def load_run(self, run_id: str) -> Optional[dict]:
        """加载指定运行的追踪记录"""
        filepath = self.traces_dir / f"run_{run_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_runs(self, limit: int = 50) -> list[dict]:
        """列出最近的运行记录"""
        files = sorted(self.traces_dir.glob("run_*.json"), reverse=True)
        runs = []
        for f in files[:limit]:
            with open(f, "r", encoding="utf-8") as fh:
                record = json.load(fh)
                runs.append({
                    "run_id": record["run_id"],
                    "timestamp": record["timestamp"],
                    "agent_count": record["agent_count"],
                    "total_duration_ms": record["total_duration_ms"],
                })
        return runs
