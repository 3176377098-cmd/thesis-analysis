"""
arXiv 数据源连接器 - 公开学术论文库
"""

import urllib.request
from pathlib import Path

from loguru import logger

from .base import AbstractSourceConnector, PaperRef
from src.tools.search_tools import ArxivSearchTool


class ArxivConnector(AbstractSourceConnector):
    """arXiv 连接器 - 免费开放，无需认证"""

    def __init__(self):
        self.search_tool = ArxivSearchTool()

    def search(self, query: str, limit: int = 10) -> list[PaperRef]:
        results = self.search_tool.search(query, max_results=limit)

        return [
            PaperRef(
                title=r.title,
                authors=r.authors,
                year=r.year,
                source_id=r.arxiv_id,
                abstract=r.abstract,
                pdf_url=r.pdf_url,
                source="arxiv",
            )
            for r in results
        ]

    def download(self, paper_id: str, target_dir: Path) -> Path:
        """下载 arXiv PDF"""
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        target_path = target_dir / f"{paper_id}.pdf"

        logger.info(f"下载 arXiv PDF: {paper_id}")

        try:
            req = urllib.request.Request(pdf_url, headers={"User-Agent": "PaperAnalyzer/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                target_path.write_bytes(response.read())

            logger.info(f"下载完成: {target_path}")
            return target_path

        except Exception as e:
            logger.error(f"下载失败: {e}")
            raise

    def get_metadata(self, paper_id: str) -> dict:
        result = self.search_tool.get_paper(paper_id)
        if result:
            return {
                "title": result.title,
                "authors": result.authors,
                "year": result.year,
                "abstract": result.abstract,
                "arxiv_id": result.arxiv_id,
            }
        return {}
