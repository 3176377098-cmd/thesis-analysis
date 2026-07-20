"""
学术搜索工具 - arXiv API / Semantic Scholar API
合规使用，遵守各平台的 Rate Limit
"""

import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from typing import Optional
from dataclasses import dataclass

from loguru import logger


@dataclass
class PaperRef:
    """搜索结果论文引用"""
    title: str
    authors: list[str]
    year: str
    arxiv_id: str = ""
    doi: str = ""
    abstract: str = ""
    pdf_url: str = ""
    venue: str = ""


class ArxivSearchTool:
    """
    arXiv API 搜索工具。

    使用官方 arXiv API (atom+xml)，无需认证。
    遵守 rate limit: 1 request / 3 seconds。
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, rate_limit: float = 3.0):
        """
        Args:
            rate_limit: 请求间隔（秒），遵守 arXiv 的 1 req/3s 限制
        """
        self.rate_limit = rate_limit
        self._last_request = 0.0

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
    ) -> list[PaperRef]:
        """
        搜索 arXiv 论文。

        Args:
            query: 搜索查询 (支持 arXiv 查询语法)
            max_results: 最大结果数
            sort_by: "relevance" | "lastUpdatedDate" | "submittedDate"

        Returns:
            PaperRef 列表
        """
        self._respect_rate_limit()

        params = {
            "search_query": query,
            "max_results": min(max_results, 50),
            "sortBy": sort_by,
        }
        url = f"{self.BASE_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"

        logger.info(f"arXiv 搜索: {query[:60]}...")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PaperAnalyzer/1.0 (Academic Use)"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read().decode("utf-8")

            self._last_request = time.time()
            return self._parse_response(data)

        except urllib.error.URLError as e:
            logger.error(f"arXiv API 请求失败: {e}")
            return []

    def get_paper(self, arxiv_id: str) -> Optional[PaperRef]:
        """根据 arXiv ID 获取论文详情"""
        results = self.search(f"id:{arxiv_id}", max_results=1)
        return results[0] if results else None

    def _respect_rate_limit(self):
        """确保请求间隔"""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

    def _parse_response(self, xml_data: str) -> list[PaperRef]:
        """解析 arXiv API XML 响应"""
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_data)

        papers = []
        for entry in root.findall("atom:entry", namespace):
            title = self._get_text(entry, "atom:title", namespace).replace("\n", " ").strip()
            abstract = self._get_text(entry, "atom:summary", namespace).replace("\n", " ").strip()

            authors = [
                author.find("atom:name", namespace).text or ""
                for author in entry.findall("atom:author", namespace)
            ]

            arxiv_id = ""
            for link in entry.findall("atom:id", namespace):
                if link.text:
                    arxiv_id = link.text.split("/abs/")[-1]
                    break

            pdf_url = ""
            for link in entry.findall("atom:link", namespace):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href", "")
                    break

            year = entry.find("atom:published", namespace)
            year = year.text[:4] if year is not None and year.text else ""

            papers.append(PaperRef(
                title=title,
                authors=authors,
                year=year,
                arxiv_id=arxiv_id,
                abstract=abstract,
                pdf_url=pdf_url,
            ))

        return papers

    @staticmethod
    def _get_text(element, tag: str, namespace: dict) -> str:
        child = element.find(tag, namespace)
        return child.text or "" if child is not None else ""


class SemanticScholarSearchTool:
    """
    Semantic Scholar API 搜索工具。

    免费 tier: 100 req / 5 min，无需认证。
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def search(
        self,
        query: str,
        limit: int = 10,
        fields: str = "title,authors,year,abstract,externalIds,venue",
    ) -> list[PaperRef]:
        """搜索 Semantic Scholar"""
        import json
        import urllib.request

        url = f"{self.BASE_URL}/paper/search?query={urllib.request.quote(query)}&limit={min(limit, 100)}&fields={fields}"

        headers = {"User-Agent": "PaperAnalyzer/1.0 (Academic Use)"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            papers = []
            for item in data.get("data", []):
                papers.append(PaperRef(
                    title=item.get("title", ""),
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    year=str(item.get("year", "")),
                    arxiv_id=item.get("externalIds", {}).get("ArXiv", ""),
                    doi=item.get("externalIds", {}).get("DOI", ""),
                    abstract=item.get("abstract", ""),
                    venue=item.get("venue", ""),
                ))
            return papers

        except Exception as e:
            logger.error(f"Semantic Scholar API 请求失败: {e}")
            return []
