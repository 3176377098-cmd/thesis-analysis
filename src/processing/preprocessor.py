"""
文本预处理器 - 清洗和规范化从 PDF 提取的原始文本
处理换行去连字符、Unicode 规范化、页眉页脚去除、LaTeX 公式保护
"""

import re
import unicodedata

from loguru import logger

from src.parsing.base import ParsedDocument


class TextPreprocessor:
    """PDF 文本清洗预处理器"""

    # 页眉页脚常见模式（页码、期刊名、会议名等）
    HEADER_FOOTER_PATTERNS = [
        re.compile(r'^\d{1,4}\s*$', re.MULTILINE),  # 单独页码
        re.compile(r'^(?:IEEE|ACM|Springer|Elsevier|arXiv).*$', re.MULTILINE | re.IGNORECASE),
        re.compile(r'^\d{4}-\d{4}/\d{2}/\d{2}.*$', re.MULTILINE),  # 日期
        re.compile(r'^©\s*\d{4}.*$', re.MULTILINE),  # 版权声明
    ]

    # LaTeX 公式模式（需要保留不处理）
    LATEX_PATTERNS = [
        re.compile(r'\$\$.+?\$\$', re.DOTALL),  # 行间公式 $$...$$
        re.compile(r'\$.+?\$'),                   # 行内公式 $...$
        re.compile(r'\\begin\{equation\}.+?\\end\{equation\}', re.DOTALL),
        re.compile(r'\\begin\{align\}.+?\\end\{align\}', re.DOTALL),
    ]

    def __init__(
        self,
        remove_headers_footers: bool = True,
        fix_hyphenation: bool = True,
        normalize_unicode: bool = True,
        preserve_latex: bool = True,
    ):
        self.remove_headers_footers = remove_headers_footers
        self.fix_hyphenation = fix_hyphenation
        self.normalize_unicode = normalize_unicode
        self.preserve_latex = preserve_latex

    def preprocess(self, parsed: ParsedDocument) -> ParsedDocument:
        """
        对解析后的文档执行清洗流水线。

        Args:
            parsed: 原始解析文档

        Returns:
            处理后的 ParsedDocument (原地修改)
        """
        text = parsed.text

        # Step 1: 保护 LaTeX 公式（用占位符替换）
        latex_blocks = {}
        if self.preserve_latex:
            text, latex_blocks = self._protect_latex(text)

        # Step 2: Unicode 规范化
        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        # Step 3: 修复换行连字符
        if self.fix_hyphenation:
            text = self._fix_hyphenation(text)

        # Step 4: 去除页眉页脚
        if self.remove_headers_footers:
            text = self._remove_headers_footers(text)

        # Step 5: 清理多余空白
        text = self._clean_whitespace(text)

        # Step 6: 恢复 LaTeX 公式
        if self.preserve_latex and latex_blocks:
            text = self._restore_latex(text, latex_blocks)

        parsed.text = text
        logger.debug(f"文本预处理完成: {len(parsed.text)} 字符")
        return parsed

    def _protect_latex(self, text: str) -> tuple[str, dict[str, str]]:
        """用占位符替换 LaTeX 公式"""
        blocks = {}
        counter = 0

        for pattern in self.LATEX_PATTERNS:
            def replace_match(m, cnt=[counter]):
                key = f"__LATEX_{cnt[0]}__"
                blocks[key] = m.group(0)
                cnt[0] += 1
                return key

            # 使用闭包计数器
            result = []
            last_end = 0
            for match in pattern.finditer(text):
                result.append(text[last_end:match.start()])
                key = f"__LATEX_{counter}__"
                blocks[key] = match.group(0)
                result.append(key)
                last_end = match.end()
                counter += 1
            result.append(text[last_end:])
            text = "".join(result)

        return text, blocks

    def _restore_latex(self, text: str, blocks: dict[str, str]) -> str:
        """恢复 LaTeX 公式占位符"""
        for key, value in blocks.items():
            text = text.replace(key, value)
        return text

    def _normalize_unicode(self, text: str) -> str:
        """Unicode 规范化: 全角→半角, 兼容字符清理"""
        # NFKC 规范化（兼容性组合）
        text = unicodedata.normalize("NFKC", text)
        # 替换特殊空白字符
        text = text.replace(" ", " ")  # non-breaking space
        text = text.replace("​", "")   # zero-width space
        text = text.replace(" ", " ")  # em space
        text = text.replace(" ", " ")  # en space
        return text

    def _fix_hyphenation(self, text: str) -> str:
        """修复 PDF 换行导致的单词断开: "con-\nclusion" → "conclusion" """
        # 行尾连字符 + 换行
        pattern = re.compile(r'(\w+)-\n(\w+)')
        return pattern.sub(r'\1\2', text)

    def _remove_headers_footers(self, text: str) -> str:
        """去除疑似页眉页脚的文本行"""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            should_remove = False
            for pattern in self.HEADER_FOOTER_PATTERNS:
                if pattern.match(stripped):
                    should_remove = True
                    break
            if not should_remove:
                cleaned.append(line)
        return '\n'.join(cleaned)

    def _clean_whitespace(self, text: str) -> str:
        """清理多余空白"""
        # 多个连续空行 → 最多两个
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 多个连续空格 → 一个
        text = re.sub(r' {2,}', ' ', text)
        # 行首尾空格
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text
