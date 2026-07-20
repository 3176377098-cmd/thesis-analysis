"""
检索质量评测指标 - Hit Rate, MRR, NDCG, Recall
用于对比不同检索策略的效果
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class RetrievalEvalResult:
    """单次检索评测结果"""
    hit_rate_k: dict[int, float]  # {k: hit_rate}
    mrr: float
    ndcg_k: dict[int, float]  # {k: ndcg}
    recall_k: dict[int, float]  # {k: recall}
    precision_k: dict[int, float]  # {k: precision}
    num_queries: int


class RetrievalMetrics:
    """
    检索质量评测器。

    计算多项标准指标:
    - Hit Rate@K: Top-K 结果中命中相关文档的比例
    - MRR (Mean Reciprocal Rank): 第一个相关文档排名的倒数均值
    - NDCG@K: 归一化折损累计增益
    - Recall@K: 相关文档被检索到的比例
    """

    def __init__(self, k_values: list[int] | None = None):
        """
        Args:
            k_values: 评测的 K 值列表，默认 [1, 3, 5, 10, 20]
        """
        self.k_values = k_values or [1, 3, 5, 10, 20]
        self.max_k = max(self.k_values)

    def evaluate(
        self,
        queries: list[str],
        relevant_chunk_ids: list[list[str]],  # 每个查询对应的相关 chunk_id 列表
        retrieved_results: list[list[str]],  # 每个查询检索到的 chunk_id 列表（按排名）
    ) -> RetrievalEvalResult:
        """
        批量评测检索质量。

        Args:
            queries: 查询列表
            relevant_chunk_ids: 每个查询对应的"正确答案" chunk_id 列表
            retrieved_results: 每个查询检索到的 chunk_id 排名列表 (best first)

        Returns:
            RetrievalEvalResult
        """
        assert len(queries) == len(relevant_chunk_ids) == len(retrieved_results), \
            "输入数组长度必须一致"

        n = len(queries)
        if n == 0:
            return RetrievalEvalResult(
                hit_rate_k={k: 0.0 for k in self.k_values},
                mrr=0.0, ndcg_k={k: 0.0 for k in self.k_values},
                recall_k={k: 0.0 for k in self.k_values},
                precision_k={k: 0.0 for k in self.k_values},
                num_queries=0,
            )

        # 累积指标
        hit_rates = {k: 0.0 for k in self.k_values}
        reciprocal_ranks = []
        ndcgs = {k: 0.0 for k in self.k_values}
        recalls = {k: 0.0 for k in self.k_values}
        precisions = {k: 0.0 for k in self.k_values}

        for query, relevant, retrieved in zip(queries, relevant_chunk_ids, retrieved_results):
            relevant_set = set(relevant)

            # Hit Rate @ K
            for k in self.k_values:
                top_k = set(retrieved[:k])
                if top_k & relevant_set:
                    hit_rates[k] += 1.0

            # MRR
            rr = 0.0
            for rank, chunk_id in enumerate(retrieved, 1):
                if chunk_id in relevant_set:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            # NDCG @ K
            for k in self.k_values:
                ndcgs[k] += self._ndcg_at_k(retrieved[:k], relevant_set, k)

            # Recall @ K
            for k in self.k_values:
                top_k = set(retrieved[:k])
                if relevant_set:
                    recalls[k] += len(top_k & relevant_set) / len(relevant_set)
                else:
                    recalls[k] += 0.0

            # Precision @ K
            for k in self.k_values:
                top_k = set(retrieved[:k])
                precisions[k] += len(top_k & relevant_set) / k if k > 0 else 0.0

        # 平均
        for k in self.k_values:
            hit_rates[k] /= n
            ndcgs[k] /= n
            recalls[k] /= n
            precisions[k] /= n

        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

        return RetrievalEvalResult(
            hit_rate_k=hit_rates,
            mrr=mrr,
            ndcg_k=ndcgs,
            recall_k=recalls,
            precision_k=precisions,
            num_queries=n,
        )

    def _ndcg_at_k(self, retrieved: list[str], relevant_set: set[str], k: int) -> float:
        """计算 NDCG@K"""
        if not retrieved:
            return 0.0

        # DCG
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved[:k]):
            rel = 1.0 if chunk_id in relevant_set else 0.0
            dcg += rel / np.log2(i + 2)  # i+2 因为 log2(1)=0

        # IDCG (理想排序: 所有相关文档排在最前面)
        ideal_rel_count = min(len(relevant_set), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_rel_count))

        return dcg / idcg if idcg > 0 else 0.0

    def print_report(self, result: RetrievalEvalResult):
        """打印评测报告"""
        logger.info(f"\n{'='*50}")
        logger.info(f"检索质量评测报告 (Queries: {result.num_queries})")
        logger.info(f"{'='*50}")

        for k in self.k_values:
            logger.info(f"Hit Rate@{k:2d}:  {result.hit_rate_k[k]:.3f}")
        logger.info(f"MRR:         {result.mrr:.3f}")

        for k in self.k_values:
            logger.info(f"NDCG@{k:2d}:     {result.ndcg_k[k]:.3f}")

        for k in self.k_values:
            logger.info(f"Recall@{k:2d}:   {result.recall_k[k]:.3f}")
