"""
多模态图表分析器 - 使用 Vision LLM 理解论文中的图片和图表
支持 GPT-4V、Claude Vision、DeepSeek-VL 等多模态模型
"""

import base64
from typing import Optional

from loguru import logger

from .image_extractor import ExtractedImage


class VisionAnalyzer:
    """
    使用多模态 LLM 分析论文中的图表。

    对每个提取的图片，生成结构化的文字描述:
    - 架构图: "这是一个 Transformer 架构图，包含 encoder-decoder 结构..."
    - 实验结果图: "这是一个准确率对比柱状图，横轴是模型名称，纵轴是准确率..."
    - 公式截图: 转写为 LaTeX
    """

    ANALYSIS_PROMPT = """You are analyzing a figure/chart from an academic paper.
Provide a detailed, structured description in English:

1. **Type**: What kind of figure? (architecture diagram, bar chart, line plot, table,
   algorithm flowchart, photo, formula screenshot, etc.)

2. **Content**: What does it show? Describe the key elements, labels, axes, curves,
   components, and relationships.

3. **Key Insight**: What is the main takeaway or finding this figure communicates?

4. **Labels/Text**: Read any visible text, labels, legends, axis titles.

If this figure contains mathematical formulas, transcribe them in LaTeX format.

Be specific and precise. This description will be used for academic paper analysis."""

    def __init__(self, llm=None, model_name: str = "gpt-4o"):
        """
        Args:
            llm: LangChain ChatModel (需支持多模态)
            model_name: 模型名 (用于日志)
        """
        self.llm = llm
        self.model_name = model_name

    def analyze(self, image: ExtractedImage) -> str:
        """
        分析单张图片，返回文字描述。

        Args:
            image: 提取的图片

        Returns:
            结构化的图片描述文本
        """
        if self.llm is None:
            return self._fallback_description(image)

        data_url = f"data:{image.mime_type};base64,{base64.b64encode(image.image_bytes).decode('ascii')}"

        try:
            from langchain_core.messages import HumanMessage

            caption_hint = f"\nFigure caption from paper: {image.caption}" if image.caption else ""

            message = HumanMessage(content=[
                {"type": "text", "text": self.ANALYSIS_PROMPT + caption_hint},
                {"type": "image_url", "image_url": {"url": data_url}},
            ])

            response = self.llm.invoke([message])
            description = response.content if hasattr(response, 'content') else str(response)

            logger.debug(f"Vision analysis: {description[:100]}...")
            return description

        except Exception as e:
            logger.warning(f"Vision analysis failed: {e}, using fallback")
            return self._fallback_description(image)

    def analyze_batch(self, images: list[ExtractedImage]) -> list[str]:
        """批量分析多张图片"""
        descriptions = []
        for i, img in enumerate(images):
            logger.info(f"Analyzing image {i+1}/{len(images)} (page {img.page_number})...")
            desc = self.analyze(img)
            descriptions.append(desc)
        return descriptions

    def _fallback_description(self, image: ExtractedImage) -> str:
        """当 LLM 不可用时，返回基本图片信息"""
        caption = f" | Caption: {image.caption}" if image.caption else ""
        return (
            f"[Figure on page {image.page_number}] "
            f"Size: {image.width}x{image.height}px{caption}. "
            f"(Install a multimodal LLM to get detailed analysis of this figure.)"
        )
