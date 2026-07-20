"""
元数据提取 Agent - 从论文中提取结构化元信息
输出: 标题/作者/摘要/关键词/章节结构/方法/贡献点
"""

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.agents.state import PaperAnalysisState


class MetadataOutput(BaseModel):
    """元数据 Agent 的结构化输出"""
    title: str = Field(description="论文完整标题")
    authors: list[str] = Field(default_factory=list, description="作者列表")
    affiliations: list[str] = Field(default_factory=list, description="作者机构")
    abstract: str = Field(default="", description="摘要全文")
    keywords: list[str] = Field(default_factory=list, description="关键词")
    sections: list[dict] = Field(default_factory=list, description="章节结构 [{title, level, summary}]")
    methods_summary: str = Field(default="", description="研究方法一句话描述")
    contributions: list[str] = Field(default_factory=list, description="主要贡献点")
    problem_statement: str = Field(default="", description="论文要解决的问题")
    venue: str = Field(default="", description="发表会议/期刊")
    year: str = Field(default="", description="发表年份")


class MetadataAgent(BaseAgent):
    """提取论文基本元信息的 Agent"""
    role = "metadata"
    output_key = "metadata_analysis"

    def __init__(self, llm):
        super().__init__(llm=llm, output_schema=MetadataOutput)

    def get_system_prompt(self, state: PaperAnalysisState) -> str:
        return """你是一位学术论文分析专家，专门从学术论文中提取结构化元数据。

**任务**: 仔细阅读提供的论文文本，提取以下结构化信息:

1. **title**: 论文完整标题
2. **authors**: 所有作者姓名列表
3. **affiliations**: 所有作者机构列表（从邮箱或机构声明中推断）
4. **abstract**: 摘要部分的完整原文，不要改写
5. **keywords**: 论文列出的关键词
6. **sections**: 论文的章节结构，每个章节包含:
   - title: 章节标题
   - level: 层级(1=一级标题, 2=二级标题)
   - summary: 该节核心内容的一句话总结
7. **methods_summary**: 研究方法/技术路线的一句话描述
8. **contributions**: 论文明确列出的贡献点（通常出现在 Introduction 末尾）
9. **problem_statement**: 论文要解决的核心问题
10. **venue**: 发表会议或期刊名（如能识别）
11. **year**: 发表年份

**重要规则**:
- 只提取文本中明确出现的信息，绝对不要编造
- 如果某个字段无法确定，使用空值 ([] 或 "")
- 摘要部分提取原文，不要总结或改写
- 章节结构应反映论文的实际组织结构
- 用 JSON 格式返回结果"""
