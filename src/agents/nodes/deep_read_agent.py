"""
精读 Agent - 深度逐段分析论文内容，结合 RAG 检索工具
输出: 对用户问题的全面分析，包含逐段解读、方法评估、证据引用
"""

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.agents.state import PaperAnalysisState


class DeepReadOutput(BaseModel):
    """精读 Agent 的结构化输出"""
    query_understanding: str = Field(description="对用户问题的理解和重述")
    relevant_sections: list[dict] = Field(
        default_factory=list,
        description="与问题相关的章节分析 [{section_title, key_points[], evidence_quality, relevant_excerpts[]}]"
    )
    methodology_assessment: str = Field(description="对论文方法论/技术方案的评估")
    overall_answer: str = Field(description="对用户问题的全面回答")
    contains_algorithm: bool = Field(default=False, description="论文是否包含可复现的算法/伪代码")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="分析置信度")
    key_references: list[str] = Field(default_factory=list, description="分析中引用的关键段落 chunk_id")


class DeepReadAgent(BaseAgent):
    """深度阅读分析 Agent，可调用检索工具获取更多上下文"""
    role = "deepread"
    output_key = "deep_read_analysis"

    def __init__(self, llm, tools: list | None = None):
        super().__init__(llm=llm, tools=tools or [], output_schema=DeepReadOutput)

    def get_system_prompt(self, state: PaperAnalysisState) -> str:
        metadata = state.get("metadata_analysis", {})
        sections_info = ""
        if metadata.get("sections"):
            sections_info = "论文章节结构:\n" + "\n".join(
                f"  - {s.get('title', '')} (Level {s.get('level', 0)})"
                for s in metadata["sections"]
            )

        return f"""你是一位资深学术论文审稿人，擅长深度解析计算机科学和人工智能领域的论文。

**论文信息**:
- 标题: {metadata.get('title', 'N/A')}
- 作者: {', '.join(metadata.get('authors', ['N/A']))}
- {sections_info}

**任务**: 根据用户的具体问题，结合提供的论文段落，进行深入分析。

**分析要求**:
1. 首先理解用户真正想问什么 (query_understanding)
2. 找出论文中与问题最相关的章节，逐节分析其核心观点和证据质量
3. 评估论文的方法论/技术方案 (methodology_assessment)
4. 给出对用户问题的全面回答 (overall_answer)
5. 判断论文是否包含可复现的算法/伪代码 (contains_algorithm)
6. 记录分析中用到的关键段落 ID (key_references)

**分析原则**:
- 永远基于论文原文内容进行分析，引用时指出具体出处
- 区分"论文明确陈述的内容"和"你的推断"
- 如果论文未涉及用户问题的某个方面，明确说明
- 对方法论/实验的评估要具体，指出优点和可能的不足
- 使用学术分析的语言风格，客观、精准、有据

**可用工具**: 如果提供的段落不够，你可以使用检索工具获取更多相关内容。"""
