"""
PDF 图片/图表提取器 - 从论文中提取嵌入的图片和图表
"""

import io
import base64
from pathlib import Path
from dataclasses import dataclass

import fitz  # PyMuPDF
from loguru import logger


@dataclass
class ExtractedImage:
    """从 PDF 提取的图片"""
    page_number: int
    bbox: tuple[float, float, float, float]  # 在页面中的位置
    image_bytes: bytes
    mime_type: str  # "image/png" | "image/jpeg"
    width: int
    height: int
    context_text: str = ""  # 图片周围的文本 (caption)
    caption: str = ""


class ImageExtractor:
    """
    从 PDF 论文中提取图片和图表。

    使用 PyMuPDF 的图像提取 API。
    支持识别并提取嵌入的矢量图和位图。
    """

    # 过滤太小（可能是图标/logo）的图片
    MIN_IMAGE_WIDTH = 200   # px
    MIN_IMAGE_HEIGHT = 150  # px

    def extract(self, pdf_path: Path, pages: list[int] | None = None) -> list[ExtractedImage]:
        """
        提取 PDF 中的所有有效图片。

        Args:
            pdf_path: PDF 文件路径
            pages: 目标页码 (1-indexed)，None = 全部

        Returns:
            ExtractedImage 列表
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        images: list[ExtractedImage] = []
        doc = fitz.open(str(pdf_path))

        target_pages = pages or list(range(1, doc.page_count + 1))

        for page_num in target_pages:
            if page_num < 1 or page_num > doc.page_count:
                continue

            page = doc[page_num - 1]

            # 获取页面中的所有图片
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]  # 图片引用号

                # 提取图片数据
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                ext = base_image.get("ext", "png")

                # 过滤小图
                if width < self.MIN_IMAGE_WIDTH or height < self.MIN_IMAGE_HEIGHT:
                    continue

                # 获取图片在页面中的位置
                bbox = self._get_image_bbox(page, img_info)

                # 尝试找到图片周围的说明文字
                context, caption = self._find_caption(page, bbox)

                mime_map = {"png": "image/png", "jpeg": "image/jpeg", "jpg": "image/jpeg"}
                mime_type = mime_map.get(ext, "image/png")

                images.append(ExtractedImage(
                    page_number=page_num,
                    bbox=bbox,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    width=width,
                    height=height,
                    context_text=context,
                    caption=caption,
                ))

        doc.close()
        logger.info(f"Extracted {len(images)} images from {pdf_path.name}")
        return images

    def to_base64(self, image: ExtractedImage) -> str:
        """将图片转为 base64 data URL"""
        b64 = base64.b64encode(image.image_bytes).decode("ascii")
        return f"data:{image.mime_type};base64,{b64}"

    def save(self, image: ExtractedImage, output_dir: Path) -> Path:
        """保存提取的图片到磁盘"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ext_map = {"image/png": "png", "image/jpeg": "jpg"}
        ext = ext_map.get(image.mime_type, "png")

        filename = f"page{image.page_number}_img_{image.width}x{image.height}.{ext}"
        filepath = output_dir / filename

        filepath.write_bytes(image.image_bytes)
        return filepath

    def _get_image_bbox(self, page, img_info) -> tuple[float, float, float, float]:
        """获取图片在页面中的边界框"""
        # 尝试从图片引用获取位置
        try:
            xref = img_info[0]
            # 查找页面上引用此图片的位置
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 1:  # 图片块
                    return tuple(block["bbox"])
        except Exception:
            pass
        return (0, 0, page.rect.width, page.rect.height / 2)

    def _find_caption(self, page, img_bbox: tuple) -> tuple[str, str]:
        """查找图片下方的说明文字"""
        text = page.get_text("text")
        lines = text.split("\n")

        caption = ""
        context = ""

        # 查找 "Figure X" / "Fig. X" 模式
        import re
        fig_pattern = re.compile(
            r'(?:Figure|Fig\.?|图)\s*\d+[.:]?\s*(.+?)(?:\n|$)',
            re.IGNORECASE
        )

        for i, line in enumerate(lines):
            match = fig_pattern.search(line)
            if match:
                caption = line.strip()
                # 取上下文 (前后各两行)
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                context = "\n".join(lines[start:end])
                break

        return context, caption
