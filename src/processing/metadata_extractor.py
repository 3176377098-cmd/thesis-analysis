"""
元数据提取器 - 使用 LLM 从论文中提取结构化元信息
标题、作者、摘要、关键词、章节大纲、方法摘要、贡献点
"""

import json
import re
from typing import Optional

from langchain_core.language_models import BaseChatModel
from loguru import logger


class PaperMetadata(dict):
    """论文元数据"""
    pass


class MetadataExtractor:
    """使用 LLM 从论文前几页提取结构化元数据"""

    EXTRACT_PROMPT = """你是一位学术论文分析专家。请从以下论文文本中提取结构化信息。

请严格按 JSON 格式返回，包含以下字段:
{{
    "title": "论文标题",
    "authors": ["作者1", "作者2"],
    "affiliations": ["机构1", "机构2"],
    "abstract": "摘要全文",
    "keywords": ["关键词1", "关键词2"],
    "sections": [
        {{"title": "章节标题", "level": 1, "summary": "该节核心内容的一句话总结"}}
    ],
    "methods_summary": "研究方法的一句话描述",
    "contributions": ["贡献点1", "贡献点2"],
    "problem_statement": "论文要解决的问题",
    "venue": "发表会议/期刊(如果能识别)",
    "year": "发表年份(如果能识别)"
}}

注意:
1. 只提取文本中明确出现的信息，不要编造
2. 如果某个字段无法确定，使用空值(null/[])
3. 章节列表应反映论文的实际结构
4. 保持摘要原文，不要改写

论文文本:
{paper_text}

请直接返回 JSON，不要加任何解释。"""

    def __init__(self, llm: BaseChatModel, max_input_tokens: int = 8000):
        """
        Args:
            llm: 用于提取的 LLM 实例
            max_input_tokens: 传给 LLM 的最大文本长度（取论文前N字符）
        """
        self.llm = llm
        self.max_input_tokens = max_input_tokens

    def extract(self, text: str) -> PaperMetadata:
        """
        从论文文本中提取元数据。

        Args:
            text: 论文全文或前 N 页文本

        Returns:
            PaperMetadata 字典
        """
        # 截取论文前部（元信息通常在前几页）
        truncated_text = text[:self.max_input_tokens * 4]  # 粗略估计: 1 token ≈ 4 chars

        logger.info(f"正在提取元数据 (输入 {len(truncated_text)} 字符)...")

        prompt = self.EXTRACT_PROMPT.format(paper_text=truncated_text)

        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)

            # 从回复中提取 JSON
            metadata = self._parse_json_response(content)
            logger.info(f"元数据提取成功: {metadata.get('title', 'Unknown')[:60]}...")
            return PaperMetadata(metadata)

        except Exception as e:
            logger.warning(f"LLM 元数据提取失败: {e}，使用基础提取")
            return self._fallback_extraction(truncated_text)

    def _parse_json_response(self, content: str) -> dict:
        """从 LLM 回复中提取 JSON"""
        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 代码块
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取 { ... } 最外层
        brace_match = re.search(r'\{[\s\S]*\}', content)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("无法解析 LLM 返回的 JSON，使用空元数据")
        return {}

    def _fallback_extraction(self, text: str) -> PaperMetadata:
        """基于正则表达式的基础元数据提取（不依赖 LLM）"""
        metadata = {
            "title": "",
            "authors": [],
            "affiliations": [],
            "abstract": "",
            "keywords": [],
            "sections": [],
            "methods_summary": "",
            "contributions": [],
            "problem_statement": "",
        }

        # 尝试提取标题（通常是前几行中最大字体的文本，这里简单取第一行非空）
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines:
            metadata["title"] = lines[0][:200]

        # 尝试匹配 Abstract
        abstract_match = re.search(
            r'(?:abstract|摘要)\s*\n+(.+?)(?:\n\s*(?:\d+\.?\s*)?(?:introduction|引言|绪论|\d+\s*$))',
            text, re.IGNORECASE | re.DOTALL
        )
        if abstract_match:
            metadata["abstract"] = abstract_match.group(1).strip()[:2000]

        # 尝试匹配邮箱格式提取作者机构
        email_pattern = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
        # 提取邮箱域名作为机构
        domains = set()
        for email in email_pattern[:20]:
            domain = email.split('@')[1]
            domains.add(domain)
        metadata["affiliations"] = list(domains)

        return PaperMetadata(metadata)
