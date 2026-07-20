"""
PyMuPDF 主解析器 - 高速 PDF 文本/结构提取
支持字体大小启发式章节检测、文本块定位、表格识别
"""

import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from loguru import logger

from .base import AbstractParser, ParsedDocument, PageBlock, Section


class PyMuPDFParser(AbstractParser):
    """基于 PyMuPDF (fitz) 的 PDF 解析器"""

    # 常见的论文章节标题模式
    SECTION_PATTERNS = [
        re.compile(r'^(?:abstract|摘要)\s*$', re.IGNORECASE),
        re.compile(r'^(?:introduction|引言|绪论)\s*$', re.IGNORECASE),
        re.compile(r'^(?:related\s*work|相关工作|文献综述)\s*$', re.IGNORECASE),
        re.compile(r'^(?:method|方法|methodology|approach|方案|our\s+approach)\s*$', re.IGNORECASE),
        re.compile(r'^(?:experiment|实验|evaluation|评估|results?|结果)\s*$', re.IGNORECASE),
        re.compile(r'^(?:discussion|讨论|analysis|分析)\s*$', re.IGNORECASE),
        re.compile(r'^(?:conclusion|结论|future\s*work|未来工作|limitations?)\s*$', re.IGNORECASE),
        re.compile(r'^(?:reference|参考文献|bibliography)\s*$', re.IGNORECASE),
        re.compile(r'^(?:appendix|附录)\s*$', re.IGNORECASE),
    ]

    # 编号标题模式：1. / 1.1 / I. / A. / 一、/ (1)
    NUMBERED_HEADING_PATTERN = re.compile(
        r'^(?:(?:\d+\.?)+(?:\s+|\.)?|'  # 1. 1.1 1.1.1
        r'[IVX]+\.\s+|'                   # I. II.
        r'[A-Z]\.\s+|'                    # A. B.
        r'[一二三四五六七八九十]+[、．.]|'   # 一、
        r'\(\d+\))\s*'                    # (1)
    )

    # 标题候选的最大长度（纯标题不应该太长）
    MAX_HEADING_LENGTH = 200

    def __init__(self, detect_sections: bool = True, extract_images: bool = False):
        """
        Args:
            detect_sections: 是否自动检测章节结构
            extract_images: 是否提取页面图片（增加内存占用）
        """
        self.detect_sections = detect_sections
        self.extract_images = extract_images

    def parse(self, pdf_path: Path) -> ParsedDocument:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))

        logger.info(f"开始解析 PDF: {pdf_path.name} ({doc.page_count} 页)")

        pages: list[PageBlock] = []
        all_text_parts: list[str] = []

        for page_num in range(doc.page_count):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            page_blocks = []
            page_text_parts = []

            for block in blocks:
                if block["type"] == 0:  # 文本块
                    text_parts = []
                    block_fonts = []
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_parts.append(span["text"])
                            block_fonts.append({
                                "size": span.get("size", 0),
                                "font": span.get("font", ""),
                                "bold": "Bold" in span.get("font", ""),
                            })

                    block_text = " ".join(text_parts).strip()
                    if block_text:
                        avg_font_size = sum(f["size"] for f in block_fonts) / max(len(block_fonts), 1)
                        is_bold = any(f["bold"] for f in block_fonts)

                        page_blocks.append({
                            "text": block_text,
                            "bbox": block["bbox"],
                            "font_size": avg_font_size,
                            "is_bold": is_bold,
                            "block_num": block.get("number", 0),
                        })
                        page_text_parts.append(block_text)

            page_text = "\n".join(page_text_parts)
            all_text_parts.append(page_text)

            pages.append(PageBlock(
                page_number=page_num + 1,
                text=page_text,
                blocks=page_blocks,
            ))

        full_text = "\n\n".join(all_text_parts)

        # 检测章节结构
        sections: list[Section] = []
        if self.detect_sections:
            sections = self._detect_sections(pages, full_text)

        # 提取 PDF 元信息
        metadata = {
            "filename": pdf_path.name,
            "page_count": doc.page_count,
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "format": doc.metadata.get("format", "PDF"),
        }

        doc.close()

        parsed = ParsedDocument(
            text=full_text,
            pages=pages,
            sections=sections,
            metadata=metadata,
            source_format="pymupdf",
        )

        logger.info(
            f"PDF 解析完成: {parsed.page_count} 页, "
            f"{len(sections)} 个章节, "
            f"{len(full_text)} 字符"
        )
        return parsed

    def _detect_sections(self, pages: list[PageBlock], full_text: str) -> list[Section]:
        """
        通过字体大小启发式 + 关键词匹配检测章节结构。

        算法:
        1. 收集所有文本块的字体大小
        2. 找出标题字体（最大的 1-3 个字号簇）
        3. 对每个候选标题文本块进行关键词/编号模式匹配
        4. 构建章节树
        """
        # 收集所有字号
        all_sizes = []
        for page in pages:
            for block in page.blocks:
                all_sizes.append(block["font_size"])

        if not all_sizes:
            return []

        # 找出标题字号范围（最大的前 20% 字号）
        sorted_sizes = sorted(set(all_sizes), reverse=True)
        heading_size_threshold = sorted_sizes[min(len(sorted_sizes) // 5, len(sorted_sizes) - 1)]

        # 遍历所有文本块，找出标题候选
        heading_candidates: list[dict] = []
        char_offset = 0

        for page in pages:
            for block in page.blocks:
                text = block["text"].strip()
                if not text:
                    char_offset += len(text) + 1
                    continue

                block_start = full_text.find(text, char_offset - len(text), char_offset + len(text))
                if block_start < 0:
                    block_start = char_offset

                is_heading = False
                level = 3  # 默认子节级别

                # 字号启发式
                if block["font_size"] >= heading_size_threshold:
                    is_heading = True
                    level = 2
                if block["font_size"] >= heading_size_threshold * 1.2:
                    level = 1

                # 加粗 + 短文本启发式
                if block["is_bold"] and len(text) < self.MAX_HEADING_LENGTH:
                    # 结尾无标点 => 更像标题
                    if not text.rstrip().endswith(('.', ',', ';', ':', '?', '!', '。', '，', '；', '：')):
                        is_heading = True

                # 关键词匹配
                for i, pattern in enumerate(self.SECTION_PATTERNS):
                    if pattern.match(text.strip()):
                        is_heading = True
                        level = min(level, 2)
                        break

                # 编号匹配
                if self.NUMBERED_HEADING_PATTERN.match(text.strip()) and len(text) < self.MAX_HEADING_LENGTH:
                    is_heading = True

                if is_heading and len(text) < self.MAX_HEADING_LENGTH:
                    heading_candidates.append({
                        "title": text,
                        "level": level,
                        "start_char": block_start,
                        "page": page.page_number,
                        "font_size": block["font_size"],
                        "is_bold": block["is_bold"],
                    })

                char_offset = block_start + len(text) + 1

        # 去重合并（相邻同标题）
        merged: list[Section] = []
        for h in heading_candidates:
            if merged and abs(h["start_char"] - merged[-1].start_char) < 50:
                continue  # 太近的跳过（重复检测）
            merged.append(Section(
                title=h["title"],
                level=h["level"],
                start_char=h["start_char"],
                page_start=h["page"],
            ))

        # 设置每个 section 的 end_char (下一个 section 的 start_char 或文末)
        for i in range(len(merged)):
            if i + 1 < len(merged):
                merged[i].end_char = merged[i + 1].start_char
            else:
                merged[i].end_char = len(full_text)

        return merged
