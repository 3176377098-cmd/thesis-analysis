"""
BM25 稀疏检索索引 - 基于关键词的互补检索通路
与向量检索混合使用，提升召回率
"""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from src.processing.chunker import Chunk


class BM25Index:
    """
    基于 rank-bm25 的稀疏检索索引。

    设计: 每篇论文构建一个独立的 BM25 索引（内存中）。
    一篇论文通常 50-200 个 chunk，索引构建 < 100ms。
    """

    def __init__(self):
        self._indices: dict[str, "BM25PaperIndex"] = {}

    def build(self, paper_id: str, chunks: list[Chunk]) -> "BM25PaperIndex":
        """
        为论文构建 BM25 索引。

        Args:
            paper_id: 论文标识
            chunks: 分块列表

        Returns:
            BM25PaperIndex 实例
        """
        index = BM25PaperIndex(paper_id, chunks)
        self._indices[paper_id] = index
        logger.debug(f"BM25 索引已构建: paper={paper_id}, chunks={len(chunks)}")
        return index

    def search(
        self,
        paper_id: str,
        query: str,
        k: int = 20,
    ) -> list[tuple[str, str, float]]:
        """
        对单篇论文执行 BM25 检索。

        Args:
            paper_id: 论文标识
            query: 查询文本
            k: 返回结果数

        Returns:
            [(chunk_id, text, bm25_score), ...]
        """
        if paper_id not in self._indices:
            logger.warning(f"BM25 索引不存在: {paper_id}")
            return []

        return self._indices[paper_id].search(query, k)

    def remove(self, paper_id: str) -> bool:
        """移除论文索引"""
        if paper_id in self._indices:
            del self._indices[paper_id]
            return True
        return False


class BM25PaperIndex:
    """单篇论文的 BM25 索引"""

    def __init__(self, paper_id: str, chunks: list[Chunk]):
        self.paper_id = paper_id
        self.chunks = chunks

        # 准备文档和 tokenizer
        self.documents = [c.text for c in chunks]
        self.chunk_ids = [c.metadata.chunk_id for c in chunks]

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 未安装。请运行: pip install rank-bm25")

        # 分词（支持中英文）
        tokenized = [self._tokenize(doc) for doc in self.documents]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 20) -> list[tuple[str, str, float]]:
        """
        BM25 检索。

        Returns:
            [(chunk_id, text, bm25_score), ...] 按分数降序
        """
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 按分数排序
        top_k = min(k, len(self.documents))
        if top_k == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 过滤零分结果
                results.append((
                    self.chunk_ids[idx],
                    self.documents[idx],
                    float(scores[idx]),
                ))

        return results

    def _tokenize(self, text: str) -> list[str]:
        """中英文混合分词"""
        tokens = []

        # 英文：按空白和标点分词
        import re
        english_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        tokens.extend(english_tokens)

        # 中文：使用 jieba
        try:
            import jieba
            chinese_chars = re.findall(r'[一-鿿]+', text)
            for chars in chinese_chars:
                tokens.extend(jieba.cut(chars))
        except ImportError:
            # jieba 不可用时，按单字切分
            chinese_chars = re.findall(r'[一-鿿]', text)
            tokens.extend(chinese_chars)

        return tokens
