"""
PDF 解析器抽象基类和数据模型
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PageBlock:
    """单页文本块"""
    page_number: int
    text: str
    blocks: list[dict] = field(default_factory=list)  # 每块的 bbox + text + font 信息


@dataclass
class Table:
    """提取的表格"""
    page_number: int
    caption: str = ""
    data: list[list[str]] = field(default_factory=list)  # 二维数组
    markdown: str = ""  # Markdown 表格表示
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)


@dataclass
class Section:
    """论文章节"""
    title: str
    level: int  # 标题层级 (1=Abstract, 2=Introduction, 3=subsection)
    start_char: int  # 在全文中的起始字符位置
    end_char: int = -1  # 结束位置（下一节开始或文末）
    page_start: int = 0


@dataclass
class ParsedDocument:
    """PDF 解析结果"""
    text: str  # 完整纯文本
    pages: list[PageBlock] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # PDF 元信息
    source_format: str = "unknown"  # "pymupdf" | "mineru"

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def table_count(self) -> int:
        return len(self.tables)

    def get_section_text(self, section: Section) -> str:
        """获取某个章节的完整文本"""
        if section.end_char > section.start_char:
            return self.text[section.start_char:section.end_char]
        return self.text[section.start_char:]


class AbstractParser(ABC):
    """PDF 解析器抽象基类"""

    @abstractmethod
    def parse(self, pdf_path: Path) -> ParsedDocument:
        """
        解析 PDF 文件，返回结构化文档。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            ParsedDocument 包含文本、表格、章节结构

        Raises:
            FileNotFoundError: PDF 文件不存在
            ValueError: PDF 文件损坏或无法解析
        """
        ...
