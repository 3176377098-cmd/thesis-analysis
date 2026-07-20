"""
论文导入完整管线 - 从 PDF 到可检索向量库的端到端流程
编排: 解析 → 预处理 → 元数据 → 分块 → 嵌入 → 存储
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import get_settings
from src.parsing import ParserFactory, AbstractParser
from src.parsing.base import ParsedDocument
from src.processing import TextPreprocessor, MetadataExtractor, AcademicChunker
from src.processing.chunker import Chunk
from src.indexing.embedder import Embedder
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index import BM25Index


class IngestionPipeline:
    """
    论文导入管线。

    用法:
        pipeline = IngestionPipeline(embedder, chroma_store, bm25_index)
        paper_id = pipeline.ingest("path/to/paper.pdf")
    """

    def __init__(
        self,
        embedder: Embedder,
        chroma_store: ChromaStore,
        bm25_index: BM25Index,
        llm=None,
    ):
        self.embedder = embedder
        self.chroma_store = chroma_store
        self.bm25_index = bm25_index
        self.llm = llm

        settings = get_settings()
        self.parser = ParserFactory.create(preference=settings.PARSER_PREFERENCE)
        self.preprocessor = TextPreprocessor(
            remove_headers_footers=True,
            fix_hyphenation=True,
            normalize_unicode=True,
        )
        self.chunker = AcademicChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        self.metadata_extractor = None
        if llm:
            self.metadata_extractor = MetadataExtractor(llm)

    def ingest(self, pdf_path: str | Path) -> str:
        """
        导入一篇论文的完整流程。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            paper_id: 论文唯一标识

        Raises:
            FileNotFoundError: PDF 不存在
            ValueError: 解析或导入失败
        """
        pdf_path = Path(pdf_path)
        paper_id = pdf_path.stem  # 使用文件名作为 paper_id

        logger.info(f"===== 开始导入论文: {paper_id} =====")

        # Step 1: 解析 PDF
        logger.info("[1/5] 解析 PDF...")
        parsed = self.parser.parse(pdf_path)

        # Step 2: 文本预处理
        logger.info("[2/5] 文本预处理...")
        parsed = self.preprocessor.preprocess(parsed)

        # Step 3: 提取元数据
        logger.info("[3/5] 提取元数据...")
        if self.metadata_extractor:
            metadata = self.metadata_extractor.extract(parsed.text)
            logger.info(f"  标题: {metadata.get('title', 'N/A')[:80]}")
            logger.info(f"  作者: {metadata.get('authors', [])}")

        # Step 4: 学术分块
        logger.info("[4/5] 学术智能分块...")
        chunks = self.chunker.chunk(parsed, paper_id=paper_id)
        logger.info(f"  生成 {len(chunks)} 个文本块")

        # Step 5: 嵌入 + 向量存储
        logger.info("[5/5] 嵌入向量化 + 存储...")
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(chunk_texts)

        # 存 ChromaDB
        self.chroma_store.add_chunks(paper_id, chunks, embeddings)

        # 构建 BM25 索引
        self.bm25_index.build(paper_id, chunks)

        logger.info(f"===== 导入完成: {paper_id} ({len(chunks)} chunks) =====")
        return paper_id

    def ingest_text(
        self,
        text: str,
        paper_id: str,
        title: str = "",
    ) -> str:
        """
        直接导入文本（绕过 PDF 解析，用于测试或 API 获取的文本）。

        Args:
            text: 论文全文
            paper_id: 论文标识
            title: 论文标题

        Returns:
            paper_id
        """
        logger.info(f"从文本导入: {paper_id}")

        # 构建一个最小的 ParsedDocument
        from src.parsing.base import ParsedDocument
        parsed = ParsedDocument(
            text=text,
            metadata={"title": title, "source": "text"},
            source_format="text",
        )

        parsed = self.preprocessor.preprocess(parsed)
        chunks = self.chunker.chunk(parsed, paper_id=paper_id)

        embeddings = self.embedder.embed([c.text for c in chunks])
        self.chroma_store.add_chunks(paper_id, chunks, embeddings)
        self.bm25_index.build(paper_id, chunks)

        logger.info(f"文本导入完成: {paper_id} ({len(chunks)} chunks)")
        return paper_id
