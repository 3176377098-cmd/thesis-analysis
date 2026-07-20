"""
ChromaDB 向量存储管理 - 论文分块的持久化和语义检索
每个论文使用独立 Collection，便于增量更新和删除
"""

import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from src.processing.chunker import Chunk, ChunkMetadata


class ChromaStore:
    """
    ChromaDB 向量存储封装。

    设计: 每篇论文一个 Collection (命名为 paper_{paper_id})
    - 优点: O(1) 删除单篇论文, 不污染其他论文的检索结果
    - 缺点: 跨论文检索需要遍历 Collections
    """

    def __init__(self, persist_dir: Path, embedder=None):
        """
        Args:
            persist_dir: ChromaDB 数据持久化目录
            embedder: Embedder 实例（延迟注入，避免循环依赖）
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._client = None

    @property
    def client(self):
        """延迟初始化 ChromaDB 客户端"""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _collection_name(self, paper_id: str) -> str:
        """生成 Collection 名 (符合 ChromaDB 命名规则)"""
        import re
        # Replace illegal chars with underscore
        safe_id = re.sub(r'[^a-zA-Z0-9._-]', '_', paper_id)
        # Remove consecutive underscores
        safe_id = re.sub(r'_+', '_', safe_id)
        # Truncate to 60 chars max (keep start and end)
        if len(safe_id) > 60:
            safe_id = safe_id[:30] + safe_id[-30:]
        # Ensure starts/ends with alphanumeric
        safe_id = safe_id.strip('_.-')
        if not safe_id:
            safe_id = "paper"
        return f"paper_{safe_id}"

    def add_chunks(
        self,
        paper_id: str,
        chunks: list[Chunk],
        embeddings: np.ndarray | None = None,
    ) -> str:
        """
        将论文分块写入 ChromaDB。

        Args:
            paper_id: 论文唯一标识
            chunks: 分块列表
            embeddings: 预计算的嵌入向量（如不提供则在查询时动态计算）

        Returns:
            collection_name
        """
        collection_name = self._collection_name(paper_id)

        # 删除旧 collection（如果存在）
        try:
            self.client.delete_collection(collection_name)
            logger.debug(f"已删除旧 collection: {collection_name}")
        except Exception:
            pass

        collection = self.client.create_collection(
            name=collection_name,
            metadata={
                "paper_id": paper_id,
                "chunk_count": len(chunks),
            },
        )

        # 准备数据
        ids = [c.metadata.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "section_title": c.metadata.section_title,
                "section_level": c.metadata.section_level,
                "page_start": c.metadata.page_range[0],
                "token_count": c.metadata.token_count,
                "contains_table": c.metadata.contains_table,
                "contains_equation": c.metadata.contains_equation,
                "prev_chunk_id": c.metadata.prev_chunk_id or "",
                "next_chunk_id": c.metadata.next_chunk_id or "",
            }
            for c in chunks
        ]

        # 添加
        if embeddings is not None:
            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )
        else:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        logger.info(
            f"ChromaDB 写入完成: collection={collection_name}, "
            f"chunks={len(chunks)}"
        )
        return collection_name

    def query(
        self,
        paper_id: str,
        query_embedding: np.ndarray,
        k: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """
        对指定论文进行语义检索。

        Args:
            paper_id: 论文标识
            query_embedding: 查询嵌入向量
            k: 返回结果数
            where: ChromaDB 元数据过滤条件

        Returns:
            [{chunk_id, text, score, metadata}, ...]
        """
        collection_name = self._collection_name(paper_id)

        try:
            collection = self.client.get_collection(collection_name)
        except Exception:
            logger.warning(f"Collection 不存在: {collection_name}")
            return []

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # 格式化结果
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1.0 / (1.0 + distance)  # 距离 → 相似度
                formatted.append({
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "score": score,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return formatted

    def delete_paper(self, paper_id: str) -> bool:
        """删除某篇论文的所有向量"""
        collection_name = self._collection_name(paper_id)
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"已删除 collection: {collection_name}")
            return True
        except Exception as e:
            logger.warning(f"删除 collection 失败: {e}")
            return False

    def get_paper_chunk_ids(self, paper_id: str) -> list[str]:
        """获取某篇论文的所有 chunk ID"""
        collection_name = self._collection_name(paper_id)
        try:
            collection = self.client.get_collection(collection_name)
            result = collection.get(include=[])
            return result["ids"] if result and result["ids"] else []
        except Exception:
            return []

    def list_papers(self) -> list[str]:
        """列出所有已索引的论文 ID"""
        collections = self.client.list_collections()
        paper_ids = []
        for col in collections:
            if col.name.startswith("paper_"):
                paper_id = col.name[6:]  # 去掉 "paper_" 前缀
                paper_ids.append(paper_id)
        return paper_ids
