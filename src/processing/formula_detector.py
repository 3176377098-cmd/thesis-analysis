"""
公式检测与保护 - 识别论文中的数学公式并转为 LaTeX 标记
支持行内公式 ($...$)、行间公式 ($$...$$)、环境公式
"""

import re
from dataclasses import dataclass, field


@dataclass
class Formula:
    """检测到的公式"""
    text: str           # 公式文本 (纯文本或 LaTeX)
    latex: str          # LaTeX 格式 (如果可转换)
    formula_type: str   # "inline" | "display" | "environment"
    char_start: int = 0
    char_end: int = 0


class FormulaDetector:
    """
    学术论文公式检测器。

    策略:
    1. 正则匹配 LaTeX 公式模式
    2. 启发式检测非 LaTeX 公式 (密集数学符号)
    3. 标记公式区域，生成 LaTeX 占位符
    """

    # LaTeX 公式模式
    DISPLAY_FORMULA = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
    INLINE_FORMULA = re.compile(r'\$(.+?)\$')
    ENV_PATTERNS = [
        re.compile(r'\\begin\{(equation|align|gather|multline)\*?\}(.+?)\\end\{\1\*?\}', re.DOTALL),
        re.compile(r'\\\[(.+?)\\\]', re.DOTALL),
        re.compile(r'\\\((.+?)\\\)', re.DOTALL),
    ]

    # 非 LaTeX 公式的启发式特征
    MATH_SYMBOLS = re.compile(
        r'[αβγδεζηθικλμνξπρστυφχψω]|'          # 希腊字母
        r'[∑∏∫∮∂∇√∞≈≠≤≥±×÷]|'                  # 数学符号
        r'[→⇒⇔↔↑↓↕]|'                            # 箭头
        r'[∈∉⊂⊃⊆⊇∪∩∅]'                           # 集合符号
    )

    # 纯数学公式行 (以数学符号为主)
    MATH_LINE_PATTERN = re.compile(
        r'^[\s\d\w+\-*/=<>()\[\]{}|&^~.,:;!?@#$%\\αβγδεζηθικλμνξπρστυφχψω∑∏∫∂∇√∞≈≠≤≥±×÷→⇒∈⊂⊆∪∩]+$'
    )

    def detect(self, text: str) -> list[Formula]:
        """
        检测文本中的所有公式。

        Returns:
            Formula 列表，按出现位置排序
        """
        formulas: list[Formula] = []

        # 1. 检测行间公式 $$...$$
        for match in self.DISPLAY_FORMULA.finditer(text):
            formulas.append(Formula(
                text=match.group(0),
                latex=match.group(0),
                formula_type="display",
                char_start=match.start(),
                char_end=match.end(),
            ))

        # 2. 检测环境公式
        for pattern in self.ENV_PATTERNS:
            for match in pattern.finditer(text):
                formulas.append(Formula(
                    text=match.group(0),
                    latex=match.group(0),
                    formula_type="environment",
                    char_start=match.start(),
                    char_end=match.end(),
                ))

        # 3. 检测行内公式 $...$ (避免与美元符号冲突)
        for match in self.INLINE_FORMULA.finditer(text):
            inner = match.group(1)
            # 过滤掉明显不是公式的 (纯数字、纯文本)
            if self._is_likely_formula(inner):
                formulas.append(Formula(
                    text=match.group(0),
                    latex=match.group(0),
                    formula_type="inline",
                    char_start=match.start(),
                    char_end=match.end(),
                ))

        # 4. 启发式检测无 LaTeX 标记的公式行
        for line_match in re.finditer(r'^.*$', text, re.MULTILINE):
            line = line_match.group(0).strip()
            if self._is_plain_math_line(line):
                # 避免与已检测的 LaTeX 公式重叠
                start, end = line_match.start(), line_match.end()
                if not any(f.char_start <= start < f.char_end for f in formulas):
                    formulas.append(Formula(
                        text=line,
                        latex=self._plain_to_latex(line),
                        formula_type="display",
                        char_start=start,
                        char_end=end,
                    ))

        # 按位置排序
        formulas.sort(key=lambda f: f.char_start)
        return formulas

    def protect_formulas(self, text: str) -> tuple[str, dict[str, str]]:
        """
        用占位符替换公式，返回 (处理后文本, {占位符: 原公式})。

        用于分块时保护公式不被拆散。
        """
        formulas = self.detect(text)
        placeholders: dict[str, str] = {}
        result_parts: list[str] = []
        last_end = 0

        for i, formula in enumerate(formulas):
            key = f"__FORMULA_{i}__"
            result_parts.append(text[last_end:formula.char_start])
            result_parts.append(f" [{key}] ")
            placeholders[key] = formula.latex or formula.text
            last_end = formula.char_end

        result_parts.append(text[last_end:])
        return "".join(result_parts), placeholders

    def _is_likely_formula(self, text: str) -> bool:
        """判断 $...$ 内部是否像是公式"""
        if not text or len(text) < 2:
            return False
        # 包含数学符号或 LaTeX 命令
        if re.search(r'[\\^_{}]', text):
            return True
        if self.MATH_SYMBOLS.search(text):
            return True
        # 包含算式特征: = + - * / 且不是纯文本
        if re.search(r'[=+\-*/]', text) and len(text) < 200:
            return True
        return False

    def _is_plain_math_line(self, line: str) -> bool:
        """判断一行是否是无标记的数学公式"""
        if len(line) < 10 or len(line) > 500:
            return False
        # 包含密集的数学符号
        math_count = len(self.MATH_SYMBOLS.findall(line))
        if math_count >= 3:
            return True
        # 以等号或希腊字母为主的行
        if re.match(r'^\s*[αβγδεζηθλμπστφχψωA-Za-z0-9\s+\-*/=<>()\[\]{}|&^~.,:;!?@#$%\\]+\s*$', line):
            if re.search(r'[=<>]', line) and math_count >= 2:
                return True
        return False

    def _plain_to_latex(self, line: str) -> str:
        """将纯文本数学行包装为 LaTeX"""
        # 基本清理和转义
        latex = line.strip()
        # 希腊字母转 LaTeX
        greek_map = {
            'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta',
            'ε': '\\epsilon', 'ζ': '\\zeta', 'η': '\\eta', 'θ': '\\theta',
            'λ': '\\lambda', 'μ': '\\mu', 'π': '\\pi', 'σ': '\\sigma',
            'τ': '\\tau', 'φ': '\\phi', 'χ': '\\chi', 'ψ': '\\psi', 'ω': '\\omega',
            '∑': '\\sum', '∏': '\\prod', '∫': '\\int', '∂': '\\partial',
            '∇': '\\nabla', '√': '\\sqrt', '∞': '\\infty', '≈': '\\approx',
            '≠': '\\neq', '≤': '\\leq', '≥': '\\geq', '±': '\\pm',
            '×': '\\times', '÷': '\\div', '→': '\\rightarrow', '⇒': '\\Rightarrow',
            '∈': '\\in', '⊂': '\\subset', '⊆': '\\subseteq', '∪': '\\cup', '∩': '\\cap',
        }
        for glyph, tex in greek_map.items():
            latex = latex.replace(glyph, tex)
        return f"$${latex}$$"
