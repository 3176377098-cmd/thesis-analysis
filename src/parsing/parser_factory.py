"""
解析器工厂 - 自动选择最佳 PDF 解析器
优先 MinerU > PyMuPDF，支持自动降级
"""

from typing import Literal

from loguru import logger

from .base import AbstractParser
from .pymupdf_parser import PyMuPDFParser


class ParserFactory:
    """PDF 解析器工厂"""

    _mineru_available: bool | None = None  # 缓存 MinerU 可用性检查

    @classmethod
    def create(
        cls,
        preference: Literal["auto", "pymupdf", "mineru"] = "auto",
        detect_sections: bool = True,
    ) -> AbstractParser:
        """
        创建 PDF 解析器。

        选择策略:
        - "auto": 优先尝试 MinerU，不可用时降级到 PyMuPDF
        - "pymupdf": 强制使用 PyMuPDF
        - "mineru": 强制使用 MinerU（不可用时抛出异常）

        Args:
            preference: 解析器偏好
            detect_sections: 是否检测章节结构

        Returns:
            AbstractParser 实例

        Raises:
            ImportError: 当 preference="mineru" 但 MinerU 不可用时
        """
        if preference == "pymupdf":
            logger.info("使用 PyMuPDF 解析器")
            return PyMuPDFParser(detect_sections=detect_sections)

        if preference == "mineru":
            if not cls._check_mineru():
                raise ImportError(
                    "MinerU (magic-pdf) 未安装。请安装: pip install magic-pdf\n"
                    "注意: MinerU 需要 PyTorch+CUDA 环境，安装包约 2GB。"
                )
            logger.info("使用 MinerU 解析器")
            return cls._create_mineru()

        # "auto" 模式
        if cls._check_mineru():
            logger.info("自动选择 MinerU 解析器")
            return cls._create_mineru()

        logger.info("MinerU 不可用，降级到 PyMuPDF 解析器")
        return PyMuPDFParser(detect_sections=detect_sections)

    @classmethod
    def _check_mineru(cls) -> bool:
        """检查 MinerU 是否可用（缓存结果）"""
        if cls._mineru_available is not None:
            return cls._mineru_available
        try:
            import importlib
            importlib.import_module("magic_pdf")
            cls._mineru_available = True
        except ImportError:
            cls._mineru_available = False
        return cls._mineru_available

    @classmethod
    def _create_mineru(cls) -> AbstractParser:
        """创建 MinerU 解析器（延迟导入，避免未安装时报错）"""
        from .mineru_parser import MinerUParser
        return MinerUParser()
