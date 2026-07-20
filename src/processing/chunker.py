"""
学术智能分块器 - 保持论文章节逻辑结构的文本切分
特点: 章节感知 + 表格保护 + 重叠保留 + 元数据追踪
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

import tiktoken
from loguru import logger

from src.parsing.base import ParsedDocument, Section, Table


# ===== 数据模型 =====

@dataclass
class ChunkMetadata:
    """文本块的溯源元数据"""
    chunk_id: str
    paper_id: str = ""
    section_title: str = ""
    section_level: int = 0
    page_range: tuple[int, int] = (0, 0)
    token_count: int = 0
    contains_table: bool = False
    contains_equation: bool = False
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None


@dataclass
class Chunk:
    """智能分块结果"""
    text: str
    metadata: ChunkMetadata


# ===== 分块策略 =====

class AcademicChunker:
    """
    学术论文智能分块器。

    策略:
    1. 以章节为边界，绝不跨节切分
    2. 表格作为独立 chunk（附带标题）
    3. 按 token 数（非字符数）计，目标 800 token
    4. 块间 10% 重叠，保持论证连贯
    5. 每块记录完整的溯源元数据
    """

    # 中英文句子分割
    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?。！？])\s+(?=[A-Z一-鿿])|'  # 标点后 + 大写/中文
        r'(?<=\n)(?=[A-Z一-鿿])'                # 换行后 + 大写/中文
    )

    # LaTeX 公式检测
    EQUATION_PATTERNS = [
        re.compile(r'\$\$.*?\$\$', re.DOTALL),
        re.compile(r'\$.*?\$'),
        re.compile(r'\\begin\{equation\}.*?\\end\{equation\}', re.DOTALL),
    ]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 80,
        min_chunk_size: int = 100,
        token_encoding: str = "cl100k_base",
    ):
        """
        Args:
            chunk_size: 目标 chunk 大小 (tokens)
            chunk_overlap: 重叠 token 数
            min_chunk_size: 最小 chunk token 数（小于此的合并到前一个）
            token_encoding: tiktoken 编码名称
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        try:
            self.tokenizer = tiktoken.get_encoding(token_encoding)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(
        self,
        parsed: ParsedDocument,
        paper_id: str = "",
        tables: list[Table] | None = None,
    ) -> list[Chunk]:
        """
        对解析后的文档执行学术智能分块。

        Args:
            parsed: 解析后的文档
            paper_id: 论文唯一标识
            tables: 附加表格列表（可覆盖 parsed.tables）

        Returns:
            按阅读顺序排列的 Chunk 列表
        """
        all_tables = tables or parsed.tables
        sections = parsed.sections
        text = parsed.text

        chunks: list[Chunk] = []

        if not sections:
            # 无章节结构，回退到纯滑动窗口分块
            logger.warning("未检测到章节结构，使用滑动窗口分块")
            chunks = self._sliding_window_chunk(text, paper_id)
        else:
            # 按章节分块
            for section in sections:
                section_text = parsed.get_section_text(section)
                section_chunks = self._chunk_section(
                    section_text=section_text,
                    section=section,
                    paper_id=paper_id,
                    text_offset=section.start_char,
                    all_text=text,
                )
                chunks.extend(section_chunks)

        # 合并过小的 chunk
        chunks = self._merge_small_chunks(chunks)

        # 设置 chunk 间的双向链表
        chunks = self._link_chunks(chunks)

        # 检测每个 chunk 是否包含公式/表格
        for chunk in chunks:
            chunk.metadata.contains_equation = self._detect_equations(chunk.text)

        # 将表格插入到对应位置
        if all_tables:
            chunks = self._insert_tables(chunks, all_tables, parsed, paper_id)

        logger.info(f"分块完成: {len(chunks)} 个块 (平均 {self._avg_tokens(chunks):.0f} tokens/块)")
        return chunks

    def _chunk_section(
        self,
        section_text: str,
        section: Section,
        paper_id: str,
        text_offset: int = 0,
        all_text: str = "",
    ) -> list[Chunk]:
        """对单个章节执行分块"""
        if not section_text.strip():
            return []

        chunks: list[Chunk] = []
        sentences = self._split_sentences(section_text)
        buffer: list[str] = []
        buffer_tokens = 0

        for sent in sentences:
            sent_tokens = self._count_tokens(sent)

            if buffer_tokens + sent_tokens > self.chunk_size and buffer:
                # 创建 chunk
                chunk = self._create_chunk(
                    text="\n".join(buffer),
                    section=section,
                    paper_id=paper_id,
                )
                chunks.append(chunk)

                # 保留重叠部分
                overlap_text = self._extract_overlap(buffer, self.chunk_overlap)
                buffer = [overlap_text] if overlap_text else []
                buffer_tokens = self._count_tokens(overlap_text)

            buffer.append(sent)
            buffer_tokens += sent_tokens

        # 最后一个 chunk
        if buffer:
            chunk = self._create_chunk(
                text="\n".join(buffer),
                section=section,
                paper_id=paper_id,
            )
            chunks.append(chunk)

        return chunks

    def _sliding_window_chunk(self, text: str, paper_id: str) -> list[Chunk]:
        """无章节结构时的滑动窗口分块"""
        chunks: list[Chunk] = []
        sentences = self._split_sentences(text)
        buffer: list[str] = []
        buffer_tokens = 0

        for sent in sentences:
            sent_tokens = self._count_tokens(sent)
            if buffer_tokens + sent_tokens > self.chunk_size and buffer:
                chunks.append(self._create_chunk("\n".join(buffer), None, paper_id))
                overlap_text = self._extract_overlap(buffer, self.chunk_overlap)
                buffer = [overlap_text] if overlap_text else []
                buffer_tokens = self._count_tokens(overlap_text)
            buffer.append(sent)
            buffer_tokens += sent_tokens

        if buffer:
            chunks.append(self._create_chunk("\n".join(buffer), None, paper_id))

        return chunks

    def _create_chunk(
        self,
        text: str,
        section: Section | None,
        paper_id: str,
    ) -> Chunk:
        """创建一个 Chunk 实例"""
        chunk_id = str(uuid.uuid4())[:8]
        token_count = self._count_tokens(text)

        return Chunk(
            text=text,
            metadata=ChunkMetadata(
                chunk_id=chunk_id,
                paper_id=paper_id,
                section_title=section.title if section else "",
                section_level=section.level if section else 0,
                page_range=(section.page_start if section else 0, 0),
                token_count=token_count,
                contains_table=False,
                contains_equation=False,
            ),
        )

    def _insert_tables(
        self,
        chunks: list[Chunk],
        tables: list[Table],
        parsed: ParsedDocument,
        paper_id: str,
    ) -> list[Chunk]:
        """将表格作为独立 chunk 插入到对应位置"""
        # 简化实现: 表格追加到文末，标记关联章节
        for table in tables:
            if table.markdown and table.data:
                table_chunk = Chunk(
                    text=f"**{table.caption}**\n\n{table.markdown}",
                    metadata=ChunkMetadata(
                        chunk_id=str(uuid.uuid4())[:8],
                        paper_id=paper_id,
                        section_title=table.caption or f"Table (Page {table.page_number})",
                        section_level=3,
                        page_range=(table.page_number, table.page_number),
                        token_count=self._count_tokens(table.markdown),
                        contains_table=True,
                        contains_equation=False,
                    ),
                )
                chunks.append(table_chunk)

        return chunks

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """合并过小的 chunk 到前一个"""
        if not chunks:
            return chunks

        merged = []
        for chunk in chunks:
            if merged and chunk.metadata.token_count < self.min_chunk_size:
                # 合并到前一个
                prev = merged[-1]
                prev.text = prev.text + "\n\n" + chunk.text
                prev.metadata.token_count = self._count_tokens(prev.text)
                prev.metadata.contains_table = prev.metadata.contains_table or chunk.metadata.contains_table
            else:
                merged.append(chunk)

        return merged

    def _link_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """设置双向链表关系"""
        for i in range(len(chunks)):
            if i > 0:
                chunks[i].metadata.prev_chunk_id = chunks[i - 1].metadata.chunk_id
            if i < len(chunks) - 1:
                chunks[i].metadata.next_chunk_id = chunks[i + 1].metadata.chunk_id
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """分句 - 支持中英文混合"""
        # 先用简单正则，保留换行结构
        raw = re.split(r'(?<=[.!?。！？])\s+', text)
        result = []
        for part in raw:
            # 进一步按换行分割（保留段落结构）
            sub_parts = re.split(r'\n{2,}', part)
            result.extend(p for p in sub_parts if p.strip())
        return [r.strip() for r in result if r.strip()]

    def _extract_overlap(self, buffer: list[str], overlap_tokens: int) -> str:
        """从 buffer 尾部提取指定 token 数的重叠文本"""
        if not buffer or overlap_tokens <= 0:
            return ""

        accumulated = ""
        for line in reversed(buffer):
            candidate = line + "\n" + accumulated
            if self._count_tokens(candidate) > overlap_tokens * 2:
                break
            accumulated = candidate

        return accumulated.strip()

    def _detect_equations(self, text: str) -> bool:
        """检测文本中是否包含 LaTeX 公式"""
        for pattern in self.EQUATION_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _count_tokens(self, text: str) -> int:
        """计算文本的 token 数"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # 回退：粗略估计 (1 token ≈ 3-4 chars for English, 1-2 chars for Chinese)
            return len(text) // 3

    def _avg_tokens(self, chunks: list[Chunk]) -> float:
        """计算平均 token 数"""
        if not chunks:
            return 0.0
        return sum(c.metadata.token_count for c in chunks) / len(chunks)
