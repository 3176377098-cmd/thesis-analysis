"""
检索策略编排器 - 多策略检索的统一入口
支持: 纯向量 / 纯BM25 / 混合(Hybrid) / HyDE / 多查询(Multi-Query)
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from loguru import logger

from .embedder import Embedder
from .chroma_store import ChromaStore
from .bm25_index import BM25Index

# 检索策略类型
RetrievalStrategy = Literal["dense", "sparse", "hybrid", "hyde", "multi_query"]


@dataclass
class RetrievedDoc:
    """检索结果文档"""
    chunk_id: str
    text: str
    score: float
    retrieval_method: str  # "dense" | "sparse" | "hybrid" | "hyde" | "multi_query"
    metadata: dict = field(default_factory=dict)


class RetrievalOrchestrator:
    """
    检索策略编排器。

    支持 5 种策略:
    - dense: 纯向量语义检索
    - sparse: 纯 BM25 关键词检索
    - hybrid: 向量 + BM25 混合 (RRF 融合) [默认]
    - hyde: 假设性文档嵌入 (先让 LLM 生成理想答案，再检索)
    - multi_query: 多查询变体 (从多角度改写查询，合并结果)
    """

    def __init__(
        self,
        chroma_store: ChromaStore,
        bm25_index: BM25Index,
        embedder: Embedder,
        llm=None,  # 用于 HyDE 和 Multi-Query 策略
        rrf_k: int = 60,
    ):
        self.chroma_store = chroma_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.llm = llm
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        paper_id: str,
        strategy: RetrievalStrategy = "hybrid",
        k: int = 20,
        section_filter: str | None = None,
    ) -> list[RetrievedDoc]:
        """
        执行检索。

        Args:
            query: 用户查询
            paper_id: 目标论文 ID
            strategy: 检索策略
            k: 返回结果数
            section_filter: 可选的章节标题过滤

        Returns:
            按分数降序排列的 RetrievedDoc 列表
        """
        logger.info(f"检索: strategy={strategy}, paper={paper_id}, query='{query[:60]}...'")

        if strategy == "dense":
            return self._dense_retrieve(query, paper_id, k, section_filter)
        elif strategy == "sparse":
            return self._sparse_retrieve(query, paper_id, k)
        elif strategy == "hybrid":
            return self._hybrid_retrieve(query, paper_id, k, section_filter)
        elif strategy == "hyde":
            return self._hyde_retrieve(query, paper_id, k)
        elif strategy == "multi_query":
            return self._multi_query_retrieve(query, paper_id, k)
        else:
            logger.warning(f"未知检索策略: {strategy}, 回退到 hybrid")
            return self._hybrid_retrieve(query, paper_id, k)

    # ===== 各策略实现 =====

    def _dense_retrieve(
        self, query: str, paper_id: str, k: int, section_filter: str | None = None
    ) -> list[RetrievedDoc]:
        """纯向量语义检索"""
        query_embedding = self.embedder.embed_query(query)

        where = None
        if section_filter:
            where = {"section_title": section_filter}

        results = self.chroma_store.query(paper_id, query_embedding, k=k, where=where)

        return [
            RetrievedDoc(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                retrieval_method="dense",
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    def _sparse_retrieve(self, query: str, paper_id: str, k: int) -> list[RetrievedDoc]:
        """纯 BM25 关键词检索"""
        results = self.bm25_index.search(paper_id, query, k=k)

        return [
            RetrievedDoc(
                chunk_id=r[0],
                text=r[1],
                score=r[2],
                retrieval_method="sparse",
            )
            for r in results
        ]

    def _hybrid_retrieve(
        self, query: str, paper_id: str, k: int, section_filter: str | None = None
    ) -> list[RetrievedDoc]:
        """
        混合检索 (Dense + Sparse → RRF 融合)

        RRF (Reciprocal Rank Fusion) 算法:
        score(d) = Σ 1/(k + rank_i(d))
        其中 k=60, rank_i(d) 是文档 d 在第 i 个排序列表中的排名
        """
        # 并行检索（每种召回 2k 个，给融合留足候选）
        dense_results = self._dense_retrieve(query, paper_id, k * 2, section_filter)
        sparse_results = self._sparse_retrieve(query, paper_id, k * 2)

        # RRF 融合
        fused: dict[str, dict] = {}

        for rank, doc in enumerate(dense_results, 1):
            if doc.chunk_id not in fused:
                fused[doc.chunk_id] = {"doc": doc, "rrf": 0}
            fused[doc.chunk_id]["rrf"] += 1.0 / (self.rrf_k + rank)

        for rank, doc in enumerate(sparse_results, 1):
            if doc.chunk_id not in fused:
                fused[doc.chunk_id] = {"doc": doc, "rrf": 0}
            fused[doc.chunk_id]["rrf"] += 1.0 / (self.rrf_k + rank)

        # 按 RRF 分数降序排序
        sorted_results = sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)[:k]

        # 更新 score 并标记方法
        for item in sorted_results:
            item["doc"].score = item["rrf"]
            item["doc"].retrieval_method = "hybrid"

        return [item["doc"] for item in sorted_results]

    def _hyde_retrieve(self, query: str, paper_id: str, k: int) -> list[RetrievedDoc]:
        """
        HyDE (Hypothetical Document Embeddings) 策略:
        1. 让 LLM 生成一个"理想答案段落"
        2. 用这个假答案做向量检索

        优点: 桥接查询和文档之间的语义鸿沟
        缺点: 多一次 LLM 调用
        """
        if self.llm is None:
            logger.warning("HyDE 策略需要 LLM，回退到 dense 检索")
            return self._dense_retrieve(query, paper_id, k)

        # Step 1: 生成假设性文档
        hyde_prompt = f"""你是一位学术论文写作助手。请根据以下问题，生成一段学术论文中可能包含此答案的段落。
不要直接回答问题，而是生成一篇学术论文中可能包含该信息的典型段落。
用英文或中文生成（与问题语言一致）。

问题: {query}

假设性文档段落:"""

        try:
            response = self.llm.invoke(hyde_prompt)
            hypothetical_doc = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"HyDE 生成文档: {hypothetical_doc[:100]}...")
        except Exception as e:
            logger.warning(f"HyDE 生成失败: {e}, 回退到 dense")
            return self._dense_retrieve(query, paper_id, k)

        # Step 2: 用假设性文档做向量检索
        hyde_embedding = self.embedder.embed_query(hypothetical_doc)
        results = self.chroma_store.query(paper_id, hyde_embedding, k=k)

        return [
            RetrievedDoc(
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=r["score"],
                retrieval_method="hyde",
                metadata=r.get("metadata", {}),
            )
            for r in results
        ]

    def _multi_query_retrieve(
        self, query: str, paper_id: str, k: int
    ) -> list[RetrievedDoc]:
        """
        多查询检索策略:
        1. 让 LLM 生成 3-5 个不同角度的查询变体
        2. 对每个变体执行 Dense 检索
        3. 去重合并，取最高分
        """
        if self.llm is None:
            logger.warning("Multi-Query 策略需要 LLM，回退到 dense 检索")
            return self._dense_retrieve(query, paper_id, k)

        # Step 1: 生成查询变体
        mq_prompt = f"""你是一位学术检索专家。请将以下用户问题改写为 3-5 个不同角度的检索查询。
每个查询应从不同角度（方法论、实验结果、理论背景、应用场景等）切入。
每行一个查询，不要编号。

原始问题: {query}

改写查询:"""

        try:
            response = self.llm.invoke(mq_prompt)
            variants_text = response.content if hasattr(response, 'content') else str(response)
            variants = [v.strip() for v in variants_text.strip().split('\n') if v.strip()]
            # 确保包含原始查询
            if query not in variants:
                variants.insert(0, query)
            variants = variants[:5]  # 最多 5 个变体
            logger.debug(f"生成 {len(variants)} 个查询变体")
        except Exception as e:
            logger.warning(f"查询变体生成失败: {e}, 回退到 dense")
            return self._dense_retrieve(query, paper_id, k)

        # Step 2: 对每个变体检索
        all_results: dict[str, RetrievedDoc] = {}

        for variant in variants:
            results = self._dense_retrieve(variant, paper_id, k // len(variants) + 1)
            for doc in results:
                if doc.chunk_id in all_results:
                    # 保留最高分
                    if doc.score > all_results[doc.chunk_id].score:
                        all_results[doc.chunk_id] = doc
                else:
                    all_results[doc.chunk_id] = doc

        # 按分数排序
        sorted_results = sorted(
            all_results.values(), key=lambda x: x.score, reverse=True
        )[:k]

        for doc in sorted_results:
            doc.retrieval_method = "multi_query"

        return sorted_results
