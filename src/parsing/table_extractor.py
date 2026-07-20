"""
pdfplumber 表格提取器 - 专注于学术论文中的实验结果表格
"""

import re
from pathlib import Path

import pdfplumber
from loguru import logger

from .base import Table


class TableExtractor:
    """使用 pdfplumber 从 PDF 中提取表格"""

    # 表格标题关键词
    TABLE_CAPTION_PATTERNS = [
        re.compile(r'(?:^|\n)\s*(?:Table|表)\s*\d+[.:]\s*(.+?)(?:\n|$)', re.IGNORECASE),
        re.compile(r'(?:^|\n)\s*(.+?)\s*\n\s*(?:Table|表)\s*\d+', re.IGNORECASE),
    ]

    def __init__(self, min_rows: int = 2, min_columns: int = 2):
        """
        Args:
            min_rows: 最少行数（过滤非表格元素）
            min_columns: 最少列数
        """
        self.min_rows = min_rows
        self.min_columns = min_columns

    def extract(self, pdf_path: Path, pages: list[int] | None = None) -> list[Table]:
        """
        提取指定页面的表格。

        Args:
            pdf_path: PDF 文件路径
            pages: 要提取的页码列表（1-indexed），None 表示全部页面

        Returns:
            提取到的 Table 列表
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        tables: list[Table] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            target_pages = pages or list(range(1, len(pdf.pages) + 1))

            for page_num in target_pages:
                if page_num < 1 or page_num > len(pdf.pages):
                    continue

                pdf_page = pdf.pages[page_num - 1]

                # 提取该页所有表格
                extracted = pdf_page.extract_tables()
                if not extracted:
                    continue

                for table_data in extracted:
                    if not table_data or len(table_data) < self.min_rows:
                        continue

                    # 过滤空列
                    cleaned_data = self._clean_table(table_data)
                    if not cleaned_data or len(cleaned_data[0]) < self.min_columns:
                        continue

                    # 尝试找到表格标题
                    caption = self._find_caption(pdf_page)

                    # 生成 Markdown 表示
                    markdown = self._to_markdown(cleaned_data)

                    tables.append(Table(
                        page_number=page_num,
                        caption=caption,
                        data=cleaned_data,
                        markdown=markdown,
                    ))

        logger.info(f"从 {pdf_path.name} 提取了 {len(tables)} 个表格")
        return tables

    def _clean_table(self, data: list[list[str | None]]) -> list[list[str]]:
        """清洗表格数据"""
        cleaned = []
        for row in data:
            str_row = [str(cell).strip() if cell is not None else "" for cell in row]
            # 跳过全空行
            if any(str_row):
                cleaned.append(str_row)
        return cleaned

    def _find_caption(self, page) -> str:
        """在页面文本中查找表格标题"""
        text = page.extract_text()
        if not text:
            return ""

        for pattern in self.TABLE_CAPTION_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(0).strip()

        return ""

    def _to_markdown(self, data: list[list[str]]) -> str:
        """将二维数组转为 Markdown 表格格式"""
        if not data:
            return ""

        # 确定每列最大宽度
        col_widths = [0] * len(data[0])
        for row in data:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))

        lines = []
        # 表头
        header = data[0]
        lines.append("| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(header)) + " |")
        # 分隔线
        lines.append("|" + "|".join("-" * (w + 2) for w in col_widths) + "|")
        # 数据行
        for row in data[1:]:
            padded = []
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    padded.append(cell.ljust(col_widths[i]))
            # 补齐列数不一致的情况
            while len(padded) < len(col_widths):
                padded.append("")
            lines.append("| " + " | ".join(padded) + " |")

        return "\n".join(lines)
