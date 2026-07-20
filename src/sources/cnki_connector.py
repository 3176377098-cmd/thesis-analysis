"""
CNKI (知网) 数据源连接器 - 昆明理工大学图书馆接口

!!! 重要合规声明 !!!
本连接器仅提供接口框架，不内置任何:
- 知网账号/密码
- 爬虫或自动化下载工具
- Paywall 绕过机制
- 批量下载功能

用户必须:
1. 拥有合法的昆明理工大学学籍
2. 通过学校图书馆的正规途径访问知网
3. 自行在 .env 中配置个人凭证
4. 遵守知网和昆明理工大学的使用条款
5. 仅用于个人学术研究，不得批量下载或商业使用
"""

from pathlib import Path
from loguru import logger

from .base import AbstractSourceConnector, PaperRef


class CNKIConnector(AbstractSourceConnector):
    """
    知网连接器接口。

    使用昆明理工大学图书馆代理访问知网。
    需要用户在 .env 中配置:
    - KUST_LIBRARY_PROXY_URL: 学校图书馆代理地址
    - CNKI_USERNAME: 知网用户名（通常为学号）
    - CNKI_PASSWORD: 知网密码
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
        self._session = None

    def search(self, query: str, limit: int = 10) -> list[PaperRef]:
        """
        搜索知网论文。

        注意: 此方法需要有效的 KUST 图书馆凭证。
        未配置凭证时将返回空列表并提示配置。
        """
        if not self.proxy_url or not self.username:
            logger.warning(
                "知网连接器未配置。请在 .env 中设置:\n"
                "  KUST_LIBRARY_PROXY_URL=你的学校图书馆地址\n"
                "  CNKI_USERNAME=你的学号\n"
                "  CNKI_PASSWORD=你的知网密码"
            )
            return []

        # TODO: 实现通过 KUST 图书馆代理访问知网搜索 API
        # 此处为接口框架，需要根据实际 KUST 图书馆代理方式实现
        logger.info(f"知网搜索: {query[:60]}... (需要有效的 KUST 凭证)")
        return []

    def download(self, paper_id: str, target_dir: Path) -> Path:
        """
        通过图书馆代理下载知网论文 PDF。

        注意: 此功能需要有效的 KUST 图书馆凭证。
        下载的论文仅用于个人学术研究。
        """
        if not self._authenticated:
            raise PermissionError(
                "知网未认证。请配置 KUST 图书馆凭证后重试。\n"
                "设置方法: 在 .env 文件中填写 CNKI_USERNAME 和 CNKI_PASSWORD"
            )

        # TODO: 实现通过 KUST 图书馆代理下载 PDF
        raise NotImplementedError(
            "知网 PDF 下载功能待实现。请先通过浏览器访问 KUST 图书馆下载论文，"
            "然后使用本地上传功能导入。"
        )

    def get_metadata(self, paper_id: str) -> dict:
        """获取知网论文元数据"""
        return {
            "source": "cnki",
            "paper_id": paper_id,
            "note": "请通过 KUST 图书馆网站获取完整元数据",
        }

    def _authenticate(self) -> bool:
        """
        通过 KUST 图书馆代理认证知网。

        认证流程:
        1. 访问 KUST 图书馆代理
        2. 使用学号/密码登录统一认证
        3. 通过代理跳转到知网
        4. 维持会话 Cookie
        """
        # 昆明理工大学通常使用 CAS 或统一认证系统
        # 具体实现需要根据学校实际系统调整
        logger.info("正在通过 KUST 图书馆认证知网...")
        self._authenticated = False
        return False
