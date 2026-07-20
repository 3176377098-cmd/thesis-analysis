"""
万方数据源连接器 - 昆明理工大学图书馆接口

!!! 重要合规声明 !!!
同知网连接器，本接口仅提供框架，不内置任何凭证或绕过机制。
用户必须通过昆明理工大学图书馆正规途径访问万方。
"""

from pathlib import Path
from loguru import logger

from .base import AbstractSourceConnector, PaperRef


class WanfangConnector(AbstractSourceConnector):
    """
    万方连接器接口。

    需要用户在 .env 中配置:
    - KUST_LIBRARY_PROXY_URL: 学校图书馆代理地址
    - WANFANG_USERNAME: 万方用户名
    - WANFANG_PASSWORD: 万方密码
    """

    def __init__(
        self,
        proxy_url: str = "",
        username: str = "",
        password: str = "",
    ):
        self.proxy_url = proxy_url
        self.username = username
        self.password = password
        self._authenticated = False

    def search(self, query: str, limit: int = 10) -> list[PaperRef]:
        if not self.proxy_url or not self.username:
            logger.warning(
                "万方连接器未配置。请在 .env 中设置:\n"
                "  KUST_LIBRARY_PROXY_URL=你的学校图书馆地址\n"
                "  WANFANG_USERNAME=你的万方用户名\n"
                "  WANFANG_PASSWORD=你的万方密码"
            )
            return []

        logger.info(f"万方搜索: {query[:60]}... (需要有效的 KUST 凭证)")
        return []

    def download(self, paper_id: str, target_dir: Path) -> Path:
        if not self._authenticated:
            raise PermissionError(
                "万方未认证。请配置 KUST 图书馆凭证后重试。"
            )

        raise NotImplementedError(
            "万方 PDF 下载功能待实现。请先通过浏览器访问 KUST 图书馆下载论文，"
            "然后使用本地上传功能导入。"
        )

    def get_metadata(self, paper_id: str) -> dict:
        return {
            "source": "wanfang",
            "paper_id": paper_id,
            "note": "请通过 KUST 图书馆网站获取完整元数据",
        }
