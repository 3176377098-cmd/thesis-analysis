"""
合规网络搜索工具 - DuckDuckGo (免费, 无需API Key)
用于补充论文背景信息、相关研究工作查找
"""

from loguru import logger


class WebSearchTool:
    """
    合规的网络搜索工具。

    默认使用 DuckDuckGo（免费、匿名、无追踪）。
    可选配置 Tavily Search API 获得更精准的学术搜索结果。
    """

    def __init__(self, tavily_api_key: str = ""):
        self.tavily_api_key = tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        执行网络搜索。

        优先使用 Tavily（如有 API Key），否则使用 DuckDuckGo。
        """
        if self.tavily_api_key:
            return self._search_tavily(query, max_results)
        return self._search_duckduckgo(query, max_results)

    def _search_duckduckgo(self, query: str, max_results: int = 5) -> list[dict]:
        """使用 DuckDuckGo Instant Answer API (非官方但公开可用)"""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo-search 未安装，搜索功能不可用")
            return []

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    }
                    for r in results
                ]
        except Exception as e:
            logger.error(f"DuckDuckGo 搜索失败: {e}")
            return []

    def _search_tavily(self, query: str, max_results: int = 5) -> list[dict]:
        """使用 Tavily Search API"""
        import json
        import urllib.request

        url = "https://api.tavily.com/search"
        payload = json.dumps({
            "api_key": self.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                }
                for r in data.get("results", [])
            ]
        except Exception as e:
            logger.error(f"Tavily 搜索失败: {e}")
            return []
