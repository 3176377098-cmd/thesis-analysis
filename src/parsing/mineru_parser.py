"""
MinerU (magic-pdf) 解析器 - 可选增强
提供卓越的学术 PDF 解析（公式/表格/结构保留），但依赖较重
"""

from pathlib import Path

from loguru import logger

from .base import AbstractParser, ParsedDocument, Section


class MinerUParser(AbstractParser):
    """
    基于 MinerU (magic-pdf) 的学术 PDF 解析器。

    注意: MinerU 依赖 PyTorch 和 CUDA 环境，安装约 2GB。
    仅在处理公式密集型论文时推荐使用。
    日常使用请选择 PyMuPDF。
    """

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """
        使用 magic-pdf CLI 解析 PDF，读取其输出的 Markdown/JSON 结果。

        MinerU 输出结构:
        - {output_dir}/{name}.md   # 完整 Markdown
        - {output_dir}/{name}/     # 按页分割的资源(图片/公式/表格)
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        logger.info(f"[MinerU] 开始解析: {pdf_path.name}")

        try:
            from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
            from magic_pdf.data.dataset import PymuDocDataset
            from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
            from magic_pdf.config.enums import SupportedPdfParseMethod

            import tempfile
            import os

            # 创建临时输出目录
            with tempfile.TemporaryDirectory() as output_dir:
                # 写入器
                image_writer = FileBasedDataWriter(output_dir)
                md_writer = FileBasedDataWriter(output_dir)

                # 读取 PDF
                reader = FileBasedDataReader()
                pdf_bytes = reader.read(str(pdf_path))

                # 创建数据集
                ds = PymuDocDataset(pdf_bytes)

                # 判断解析模式
                if ds.classify() == SupportedPdfParseMethod.OCR:
                    infer_result = ds.apply(doc_analyze, ocr=True)
                    pipe_result = infer_result.apply(doc_analyze, ocr=True)
                    pipe_result.dump_md(md_writer, f"{pdf_path.stem}", "auto")
                else:
                    infer_result = ds.apply(doc_analyze, ocr=False)
                    pipe_result = infer_result.apply(doc_analyze, ocr=False)
                    pipe_result.dump_md(md_writer, f"{pdf_path.stem}", "txt")

                # 读取生成的 Markdown
                md_path = Path(output_dir) / f"{pdf_path.stem}.md"
                if md_path.exists():
                    full_text = md_path.read_text(encoding="utf-8")
                else:
                    # 尝试其他路径格式
                    md_files = list(Path(output_dir).glob("*.md"))
                    full_text = md_files[0].read_text(encoding="utf-8") if md_files else ""

                # MinerU 的 Markdown 已有章节结构 (# ## ###)
                sections = self._parse_markdown_sections(full_text)

                logger.info(f"[MinerU] 解析完成: {len(full_text)} 字符, {len(sections)} 个章节")
                return ParsedDocument(
                    text=full_text,
                    sections=sections,
                    metadata={"filename": pdf_path.name, "parser": "mineru"},
                    source_format="mineru",
                )

        except ImportError:
            raise ImportError(
                "MinerU 未正确安装。请确保:\n"
                "1. pip install magic-pdf\n"
                "2. PyTorch 环境已配置\n"
                "3. CUDA 可用（或使用 CPU 模式）"
            )

    def _parse_markdown_sections(self, text: str) -> list[Section]:
        """从 Markdown 标题解析章节结构"""
        import re
        sections = []
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

        for match in heading_pattern.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.start()
            sections.append(Section(
                title=title,
                level=min(level, 4),  # 限制最大层级
                start_char=start,
            ))

        # 设置 end_char
        for i in range(len(sections)):
            if i + 1 < len(sections):
                sections[i].end_char = sections[i + 1].start_char
            else:
                sections[i].end_char = len(text)

        return sections
