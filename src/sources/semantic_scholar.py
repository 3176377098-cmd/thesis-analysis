"""
Semantic Scholar 数据源连接器
"""

from pathlib import Path

from loguru import logger

from .base import AbstractSourceConnector, PaperRef
from src.tools.search_tools import SemanticScholarSearchTool


class SemanticScholarConnector(AbstractSourceConnector):
    """Semantic Scholar 连接器 - 免费 tier 可用"""

    def __init__(self, api_key: str = ""):
        self.search_tool = SemanticScholarSearchTool(api_key=api_key)

    def search(self, query: str, limit: int = 10) -> list[PaperRef]:
        results = self.search_tool.search(query, limit=limit)

        return [
            PaperRef(
                title=r.title,
                authors=r.authors,
                year=r.year,
                source_id=r.arxiv_id or r.doi,
                doi=r.doi,
                abstract=r.abstract,
                venue=r.venue,
                source="semantic_scholar",
            )
            for r in results
        ]

    def download(self, paper_id: str, target_dir: Path) -> Path:
        """Semantic Scholar 不直接提供 PDF，尝试通过 arXiv ID 下载"""
        logger.warning("Semantic Scholar 不直接托管 PDF，如果 paper_id 是 arxiv ID 将尝试从 arXiv 下载")

        from .arxiv_connector import ArxivConnector
        arxiv = ArxivConnector()
        return arxiv.download(paper_id, target_dir)

    def get_metadata(self, paper_id: str) -> dict:
        # Semantic Scholar 的搜索本身就返回元数据
        results = self.search_tool.search(paper_id, limit=1)
        if results:
            r = results[0]
            return {
                "title": r.title,
                "authors": r.authors,
                "year": r.year,
                "abstract": r.abstract,
                "venue": r.venue,
                "doi": r.doi,
            }
        return {}
