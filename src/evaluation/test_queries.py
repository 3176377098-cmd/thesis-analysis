"""
评测查询生成器 - 自动从论文分块生成测试 Q&A 对
用于评测检索质量 (Hit Rate / MRR)
"""

import random
import re
from loguru import logger

from src.processing.chunker import Chunk


class QueryGenerator:
    """
    从论文自动生成评测查询集。

    策略:
    1. 随机采样 chunk
    2. 从 chunk 中提取关键实体/概念
    3. 生成"这篇论文关于 X 说了什么？"类型的查询
    4. 以原始 chunk 和相关 chunk 作为 ground truth
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate(
        self,
        chunks: list[Chunk],
        num_queries: int = 10,
    ) -> list[dict]:
        """
        从分块列表生成测试查询。

        Args:
            chunks: 论文分块列表
            num_queries: 生成多少条查询

        Returns:
            [{query, relevant_chunk_ids, source_chunk_id}, ...]
        """
        if not chunks:
            return []

        # 排除太短或只有表格的 chunk
        valid_chunks = [
            c for c in chunks
            if c.metadata.token_count > 50 and not c.metadata.contains_table
        ]

        if len(valid_chunks) < num_queries:
            num_queries = len(valid_chunks)

        sampled = random.sample(valid_chunks, num_queries)
        queries = []

        for chunk in sampled:
            # 提取关键概念生成查询
            key_terms = self._extract_key_terms(chunk.text)
            if not key_terms:
                continue

            # 生成不同风格的查询
            query_style = random.choice([
                "what_is",
                "how_does",
                "what_are",
                "describe",
            ])

            term = random.choice(key_terms[:3])
            query = self._build_query(term, query_style, chunk.metadata.section_title)

            # Ground truth: 当前 chunk + 相邻 chunk
            relevant_ids = {chunk.metadata.chunk_id}
            if chunk.metadata.prev_chunk_id:
                relevant_ids.add(chunk.metadata.prev_chunk_id)
            if chunk.metadata.next_chunk_id:
                relevant_ids.add(chunk.metadata.next_chunk_id)

            queries.append({
                "query": query,
                "relevant_chunk_ids": list(relevant_ids),
                "source_chunk_id": chunk.metadata.chunk_id,
                "section": chunk.metadata.section_title,
            })

        logger.info(f"生成了 {len(queries)} 条测试查询")
        return queries

    def _extract_key_terms(self, text: str) -> list[str]:
        """从文本中提取关键术语"""
        terms = []

        # 匹配大写缩写术语 (CNN, LSTM, BERT...)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        terms.extend(acronyms)

        # 匹配被引号包围的术语
        quoted = re.findall(r'"([^"]{3,30})"', text)
        terms.extend(quoted)

        # 匹配驼峰或下划线连接的技术术语
        technical = re.findall(r'\b[a-z]+(?:[_-][a-z]+)+(?:\s*(?:model|method|algorithm|framework|network|architecture))?\b', text)
        terms.extend(technical)

        # 去重
        seen = set()
        unique = []
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)

        return unique[:10]

    def _build_query(self, term: str, style: str, section: str) -> str:
        """生成自然语言查询"""
        templates = {
            "what_is": f"What is {term} in the context of this paper?",
            "how_does": f"How does {term} work according to the paper?",
            "what_are": f"What are the key components of {term} described in the paper?",
            "describe": f"Describe the {term} approach proposed in the paper.",
        }
        return templates.get(style, templates["what_is"])
