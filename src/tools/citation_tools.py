"""
引文图谱工具 - 查询论文的引用关系和参考文献
基于 Semantic Scholar API
"""

from loguru import logger


class CitationGraphTool:
    """引文关系查询工具"""

    def __init__(self, semantic_scholar_tool=None):
        """
        Args:
            semantic_scholar_tool: SemanticScholarSearchTool 实例
        """
        self.ss_tool = semantic_scholar_tool

    def get_citations(self, paper_id: str, limit: int = 20) -> list[dict]:
        """获取引用该论文的论文列表"""
        import json
        import urllib.request

        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
        params = f"?fields=title,authors,year,venue&limit={min(limit, 100)}"

        try:
            req = urllib.request.Request(
                url + params,
                headers={"User-Agent": "PaperAnalyzer/1.0 (Academic Use)"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            return [
                {
                    "title": item.get("citingPaper", {}).get("title", ""),
                    "authors": [a.get("name", "") for a in item.get("citingPaper", {}).get("authors", [])],
                    "year": item.get("citingPaper", {}).get("year", ""),
                    "venue": item.get("citingPaper", {}).get("venue", ""),
                }
                for item in data.get("data", [])
            ]
        except Exception as e:
            logger.error(f"引文查询失败: {e}")
            return []

    def get_references(self, paper_id: str, limit: int = 20) -> list[dict]:
        """获取该论文引用的参考文献"""
        import json
        import urllib.request

        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
        params = f"?fields=title,authors,year,venue&limit={min(limit, 100)}"

        try:
            req = urllib.request.Request(
                url + params,
                headers={"User-Agent": "PaperAnalyzer/1.0 (Academic Use)"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            return [
                {
                    "title": item.get("citedPaper", {}).get("title", ""),
                    "authors": [a.get("name", "") for a in item.get("citedPaper", {}).get("authors", [])],
                    "year": item.get("citedPaper", {}).get("year", ""),
                    "venue": item.get("citedPaper", {}).get("venue", ""),
                }
                for item in data.get("data", [])
            ]
        except Exception as e:
            logger.error(f"参考文献查询失败: {e}")
            return []
