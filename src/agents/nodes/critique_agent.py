"""
批判性审查 Agent - 对 Deep Read 分析结果进行交叉验证和批判性评估
输出: 优缺点分析、方法学问题、事实错误检查、修正建议
"""

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.agents.state import PaperAnalysisState


class CritiqueOutput(BaseModel):
    """批判性审查 Agent 的结构化输出"""
    strengths: list[str] = Field(default_factory=list, description="Deep Read 分析的优点")
    weaknesses: list[str] = Field(default_factory=list, description="分析中的弱点或遗漏")
    gaps: list[str] = Field(default_factory=list, description="未覆盖的重要方面")
    methodological_concerns: list[str] = Field(default_factory=list, description="方法学/论证层面的问题")
    factual_errors: list[str] = Field(default_factory=list, description="可能存在的事实性错误")
    needs_revision: bool = Field(default=False, description="是否需要重新精读（反馈修正）")
    revision_suggestions: str = Field(default="", description="如果需要修正，给 DeepRead Agent 的具体建议")
    confidence_assessment: float = Field(default=0.7, ge=0.0, le=1.0, description="对精读分析的总体置信度评估")


class CritiqueAgent(BaseAgent):
    """批判性审查 Agent - 对 Deep Read 输出进行元分析"""
    role = "critique"
    output_key = "critique_result"

    def __init__(self, llm):
        super().__init__(llm=llm, output_schema=CritiqueOutput)

    def get_system_prompt(self, state: PaperAnalysisState) -> str:
        deep_read = state.get("deep_read_analysis", {})
        metadata = state.get("metadata_analysis", {})

        deep_read_summary = ""
        if deep_read:
            deep_read_summary = f"""
**精读分析结果**:
- 问题理解: {deep_read.get('query_understanding', 'N/A')[:200]}
- 相关章节数: {len(deep_read.get('relevant_sections', []))}
- 方法论评估: {deep_read.get('methodology_assessment', 'N/A')[:200]}
- 总体回答: {deep_read.get('overall_answer', 'N/A')[:300]}
- 置信度: {deep_read.get('confidence', 'N/A')}
"""

        return f"""你是一位严格的学术审稿专家，专门对 AI 系统生成的论文分析进行批判性审查。

**当前分析的论文**: {metadata.get('title', 'N/A')}

{deep_read_summary}

**任务**: 对上述"精读分析"进行严格的交叉验证和批判性评估。

**审查维度**:
1. **准确性**: 分析是否与论文原文一致？是否有曲解或过度解读？
2. **完整性**: 是否遗漏了论文中的重要观点、方法细节或实验结果？
3. **深度**: 分析是否足够深入？是否只停留在表面描述？
4. **逻辑**: 推理链条是否完整？结论是否由证据支撑？
5. **平衡性**: 是否同时关注了论文的优点和局限性？

**needs_revision 判断标准**:
- 设为 True 的情况: 分析存在明显事实错误、遗漏了关键方法论细节、或对实验结果的理解有重大偏差
- 设为 False 的情况: 分析基本准确，即使有小的改进空间
- 注意: 最多触发 2 次修正循环，请谨慎设置此标志

请以批判性的眼光审视，但不要为了找问题而找问题。"""
