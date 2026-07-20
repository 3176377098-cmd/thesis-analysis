"""Quick test script for paper analysis - output to file to avoid GBK issues"""
import sys, os, io
sys.path.insert(0, '.')
from pathlib import Path
from src.parsing.pymupdf_parser import PyMuPDFParser
from src.processing.formula_detector import FormulaDetector
from src.processing.image_extractor import ImageExtractor

# Write to UTF-8 file instead of stdout
out = io.StringIO()

pdf_path = Path(r"d:\AI_Projects\基于Multi-Agent架构的学术论文深度解析系统\pdf\Weighted squared envelope diversity entropy as a nonlinear dynamic prognostic measure of rolling element bearing.pdf")

# 1. Parse
parser = PyMuPDFParser()
parsed = parser.parse(pdf_path)
out.write(f"Pages: {parsed.page_count}\n")
out.write(f"Text: {len(parsed.text)} chars\n")
out.write(f"Sections: {len(parsed.sections)}\n")

# 2. Show a section with formula content
text = parsed.text
idx = text.find("2   Theory of DE")
if idx > 0:
    sample = text[idx:idx+2000]
    out.write("\n=== Section 2 Sample (cleaned) ===\n")
    # Clean non-ASCII to avoid encoding issues
    cleaned = ''.join(ch if ord(ch) < 128 or ch in '\n\r\t ' else ' ' for ch in sample)
    out.write(cleaned)
    out.write("\n")

# 3. Formula analysis
out.write("\n=== Formula Detection ===\n")
fd = FormulaDetector()
formulas = fd.detect(text)
out.write(f"Formulas found: {len(formulas)}\n")
for f in formulas[:5]:
    clean_text = ''.join(ch if ord(ch) < 128 else '?' for ch in f.text)
    out.write(f"  [{f.formula_type}] {clean_text[:120]}\n")

# 4. Key character counts
out.write("\n=== Math Character Counts ===\n")
for ch, name in [('=', 'equals'), ('_', 'underscore'), ('^', 'caret'),
                 ('{', 'lbrace'), ('}', 'rbrace'), ('(', 'lparen'),
                 (')', 'rparen'), ('[', 'lbrack'), (']', 'rbrack')]:
    out.write(f"  {name}: {text.count(ch)}\n")

out.write(f"  dollar $: {text.count('$')}\n")
out.write(f"  backslash: {text.count(chr(92))}\n")

# 5. Check for Unicode math symbols
math_unicode = ['α', 'β', 'λ', 'σ', 'θ',  # Greek
                '≤', '≥', '≠', '≈',            # relations
                '∑', '∫', '∞']                       # sum, integral, infinity
out.write("\n=== Unicode Math Symbols ===\n")
for sym in math_unicode:
    cnt = text.count(sym)
    if cnt > 0:
        out.write(f"  U+{ord(sym):04X}: {cnt} occurrences\n")

# 6. Images
ie = ImageExtractor()
images = ie.extract(pdf_path)
out.write(f"\n=== Images ===\n")
out.write(f"Total: {len(images)}\n")
for img in images:
    out.write(f"  Page {img.page_number}: {img.width}x{img.height}px\n")
    if img.caption:
        caption = ''.join(ch if ord(ch) < 128 else '?' for ch in img.caption)
        out.write(f"    Caption: {caption[:120]}\n")

# Save output
result = out.getvalue()
output_path = Path(r"d:\AI_Projects\基于Multi-Agent架构的学术论文深度解析系统\test_output.txt")
output_path.write_text(result, encoding='utf-8')
# Write raw utf8 to stdout bypassing GBK
sys.stdout.buffer.write(result.encode('utf-8') + b'\n')
