"""
数据源连接器抽象基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PaperRef:
    """论文引用信息"""
    title: str
    authors: list[str]
    year: str = ""
    source_id: str = ""  # 来源系统内的唯一标识
    doi: str = ""
    abstract: str = ""
    pdf_url: str = ""
    venue: str = ""
    source: str = ""  # "arxiv" | "semantic_scholar" | "cnki" | "wanfang"


class AbstractSourceConnector(ABC):
    """论文数据源连接器抽象基类"""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[PaperRef]:
        """
        按关键词搜索论文。

        Args:
            query: 搜索查询
            limit: 返回结果数

        Returns:
            PaperRef 列表
        """
        ...

    @abstractmethod
    def download(self, paper_id: str, target_dir: Path) -> Path:
        """
        下载论文 PDF。

        Args:
            paper_id: 来源系统内的论文标识
            target_dir: 下载目标目录

        Returns:
            下载的 PDF 文件路径
        """
        ...

    @abstractmethod
    def get_metadata(self, paper_id: str) -> dict:
        """
        获取论文元数据。

        Args:
            paper_id: 来源系统内的论文标识

        Returns:
            元数据字典
        """
        ...
