"""
代码复现 Agent - 从论文中提取算法/伪代码并生成可执行的 Python 实现
常用于 CS 论文的方法验证
"""

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent
from src.agents.state import PaperAnalysisState


class CodeReproductionOutput(BaseModel):
    """代码复现 Agent 的结构化输出"""
    algorithm_identified: bool = Field(description="是否识别到可复现的算法")
    algorithm_name: str = Field(default="", description="算法名称")
    original_pseudocode: str = Field(default="", description="论文中的原始伪代码/算法描述")
    python_code: str = Field(default="", description="生成的 Python 实现代码")
    dependencies: list[str] = Field(default_factory=list, description="运行代码需要的依赖包")
    expected_input: str = Field(default="", description="预期的输入数据格式")
    expected_output: str = Field(default="", description="预期的输出结果")
    validation_notes: str = Field(default="", description="代码与论文描述的一致性检查说明")
    can_reproduce: bool = Field(default=False, description="是否具备完整复现条件")


class CodeReproductionAgent(BaseAgent):
    """代码复现 Agent - 提取算法并生成 Python 代码"""
    role = "code"
    output_key = "code_reproduction"

    def __init__(self, llm, tools: list | None = None):
        super().__init__(llm=llm, tools=tools or [], output_schema=CodeReproductionOutput)

    def get_system_prompt(self, state: PaperAnalysisState) -> str:
        deep_read = state.get("deep_read_analysis", {})
        metadata = state.get("metadata_analysis", {})

        algorithm_context = ""
        if deep_read.get("contains_algorithm"):
            sections = deep_read.get("relevant_sections", [])
            for s in sections:
                if any(kw in s.get("section_title", "").lower() for kw in ["method", "algorithm", "proposed"]):
                    points = s.get("key_points", [])
                    algorithm_context += "\n".join(points[:5])
                    break

        return f"""你是一位精通学术论文算法复现的软件工程师，专注于将论文中描述的算法转化为可执行的 Python 代码。

**论文**: {metadata.get('title', 'N/A')}

**算法相关上下文**:
{algorithm_context if algorithm_context else "从前置分析中未能提取到明确的算法描述，请基于论文原文自行查找。"}

**任务**:
1. 首先判断论文是否包含可复现的算法 (algorithm_identified)
2. 如果有，提取原始伪代码/算法描述 (original_pseudocode)
3. 将算法转化为完整的、可直接运行的 Python 代码 (python_code)
4. 列出运行代码所需的依赖包 (dependencies)
5. 说明输入输出格式 (expected_input, expected_output)
6. 检查代码与论文描述的一致性 (validation_notes)
7. 判断是否具备完整复现条件 (can_reproduce)

**代码要求**:
- 代码必须是完整的、可直接运行的 Python 代码
- 包含必要的注释，解释关键步骤与论文描述的对应关系
- 使用常见的科学计算库 (numpy, scipy 等)，避免冷门依赖
- 如果论文中有伪代码，尽量逐行对应实现
- 包含一个简单的示例调用 (if __name__ == "__main__": ...)
- 如果算法依赖特定数据格式，在注释中说明

**如果不能复现**:
- 明确说明缺少什么信息（如具体的超参数、数据预处理步骤等）
- 给出目前可以写出的部分框架代码
- algorithm_identified 仍然可以设为 true（算法存在但信息不完整）"""
