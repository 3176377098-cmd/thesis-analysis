from .base import AbstractSourceConnector, PaperRef
from .arxiv_connector import ArxivConnector
from .semantic_scholar import SemanticScholarConnector
from .cnki_connector import CNKIConnector
from .wanfang_connector import WanfangConnector

__all__ = [
    "AbstractSourceConnector",
    "PaperRef",
    "ArxivConnector",
    "SemanticScholarConnector",
    "CNKIConnector",
    "WanfangConnector",
]
