"""
综述生成 Agent - 整合所有前置 Agent 的分析结果，生成结构化学术报告
输出: 执行摘要、整合分析、矛盾解决、实际意义、局限性、引用映射
"""

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.agents.state import PaperAnalysisState


class SynthesisOutput(BaseModel):
    """综述 Agent 的结构化输出"""
    executive_summary: str = Field(description="300字以内的执行摘要")
    integrated_analysis: str = Field(description="整合所有 Agent 分析后的全面解读")
    contradictions_resolved: list[str] = Field(
        default_factory=list,
        description="发现的矛盾点及其解决方式"
    )
    practical_implications: list[str] = Field(default_factory=list, description="研究成果的实际应用价值")
    limitations_acknowledged: list[str] = Field(default_factory=list, description="论文的局限性")
    future_work_suggested: list[str] = Field(default_factory=list, description="建议的未来研究方向")
    key_findings: list[str] = Field(default_factory=list, description="3-5 个核心发现/结论")
    citation_map: list[dict] = Field(
        default_factory=list,
        description="关键论点的原文出处 [{claim, chunk_id, section}]"
    )


class SynthesisAgent(BaseAgent):
    """综述生成 Agent - 整合多 Agent 分析形成最终报告"""
    role = "synthesis"
    output_key = "synthesis_result"

    def __init__(self, llm):
        super().__init__(llm=llm, output_schema=SynthesisOutput)

    def get_system_prompt(self, state: PaperAnalysisState) -> str:
        metadata = state.get("metadata_analysis", {})
        deep_read = state.get("deep_read_analysis", {})
        critique = state.get("critique_result", {})

        return f"""你是一位学术综述专家，擅长将多角度的论文分析整合为连贯、深入、有价值的学术报告。

**论文**: {metadata.get('title', 'N/A')}
**作者**: {', '.join(metadata.get('authors', ['N/A']))}

**前置分析摘要**:
- 元数据: 问题={metadata.get('problem_statement', 'N/A')[:100]}, 方法={metadata.get('methods_summary', 'N/A')[:100]}
- 精读分析: {deep_read.get('overall_answer', 'N/A')[:200]}
- 批判审查: 优点{len(critique.get('strengths', []))}条, 问题{len(critique.get('weaknesses', []))}条, 需修正={critique.get('needs_revision', False)}

**任务**: 基于以上所有分析，生成一份结构化的学术综述报告。每个字段只写该字段专有的内容，不要在不同字段间重复。

**重要格式规则**:
- 每个字段的内容是纯 Markdown 正文，不要在字段内容里写字段名称（如不要在 integrated_analysis 里写"integrated_analysis"这个标签）
- integrated_analysis 只包含整合分析的正文内容，不要把其他字段的内容也塞进来

**报告要求**:
1. **executive_summary**: 300字以内的精简摘要，让读者快速了解论文做什么、核心发现是什么
2. **integrated_analysis**: 将元数据、精读、批判审查的信息有机整合，形成连贯的叙述（仅整合分析正文，不包含其他字段内容）
3. **contradictions_resolved**: 如果前置分析中存在矛盾（如 Metadata 说方法是 A，DeepRead 说是 B），指出并解决
4. **practical_implications**: 研究成果的实际应用场景和价值
5. **limitations_acknowledged**: 客观列出论文的局限性（方法、实验、范围等）
6. **future_work_suggested**: 基于论文局限性和领域趋势，建议未来研究方向
7. **key_findings**: 3-5 个最重要的发现/结论（列表形式）
8. **citation_map**: 为每个关键论点提供原文出处

**风格**: 学术性、客观中立、结构清晰。用 Markdown 格式组织内容。"""
