"""
文本嵌入器 - 双层策略
1. 主: BGE 中文语义模型 (sentence-transformers, 走 hf-mirror.com)
2. Fallback: TF-IDF (纯本地, 零网络依赖)

自动检测可用性，优先使用语义模型，失败时回退 TF-IDF
"""

import os
import numpy as np
from typing import Optional
from loguru import logger

# 设置 HuggingFace 镜像 + 离线优化，必须在导入 sentence_transformers 之前
_HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
if _HF_ENDPOINT and "hf-mirror" not in os.environ.get("HF_ENDPOINT", ""):
    os.environ["HF_ENDPOINT"] = _HF_ENDPOINT

# 减少 HF hub 的网络检查，优先使用缓存
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 禁止网络请求
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Embedder:
    """
    双层嵌入器: sentence-transformers 优先, TF-IDF 兜底

    用法:
        embedder = Embedder(model_name="BAAI/bge-small-zh-v1.5")
        # 如果模型能下载，自动使用语义模型
        embedder.fit(texts)          # TF-IDF 需要, 语义模型是 no-op
        vectors = embedder.embed(texts)
        query_vec = embedder.embed_query("查询文本")
    """

    _instance: Optional["Embedder"] = None

    # 已知的好用的嵌入模型 (按优先级)
    _SEMANTIC_MODELS = [
        "BAAI/bge-small-zh-v1.5",    # 中文, ~24MB, 512维
        "BAAI/bge-base-zh-v1.5",     # 中文, ~408MB, 768维
        "BAAI/bge-small-en-v1.5",    # 英文, ~134MB, 384维
        "sentence-transformers/all-MiniLM-L6-v2",  # 英文, ~90MB, 384维
    ]

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None           # sentence-transformers model
        self._tfidf = None           # TF-IDF fallback
        self._dimension = 512
        self._model_type = "unknown"  # "semantic" | "tfidf"
        self._load_attempted = False

    # ================================================================
    #  Public API
    # ================================================================

    @property
    def model(self):
        return self.model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def is_fitted(self) -> bool:
        """语义模型无需 fit, TF-IDF 需要"""
        if self._model_type == "semantic":
            return self._model is not None
        return self._tfidf is not None

    def fit(self, texts: list[str]):
        """
        在文档集上拟合嵌入器。
        - 语义模型: 触发首次加载 (下载+初始化)
        - TF-IDF: 在 texts 上训练向量器
        """
        if self._model_type == "unknown":
            self._init_model(texts)
        elif self._model_type == "tfidf" and self._tfidf is None:
            self._fit_tfidf(texts)

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """嵌入一段或多段文本, 返回 numpy array (N, dim)"""
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([])

        # Lazy init
        if self._model_type == "unknown":
            self._init_model(texts)

        if self._model_type == "semantic" and self._model is not None:
            return self._embed_semantic(texts)
        else:
            return self._embed_tfidf(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """嵌入单条查询"""
        return self.embed([query])[0]

    # ================================================================
    #  Init logic
    # ================================================================

    def _init_model(self, sample_texts: list[str] | None = None):
        """
        尝试加载语义模型，失败则初始化 TF-IDF。
        只在第一次调用 fit/embed 时执行（lazy init）。
        """
        if self._load_attempted:
            return
        self._load_attempted = True

        # 尝试 1: 按优先级加载语义模型
        for candidate in self._model_candidates():
            logger.info(f"Trying semantic model: {candidate}")
            try:
                self._load_semantic_model(candidate)
                self._model_type = "semantic"
                logger.info(f"Semantic model loaded: {candidate} (dim={self._dimension})")
                return
            except Exception as e:
                logger.warning(f"Failed to load {candidate}: {e}")

        # 尝试 2: TF-IDF fallback
        logger.warning("All semantic models failed, falling back to TF-IDF")
        self._model_type = "tfidf"
        if sample_texts:
            self._fit_tfidf(sample_texts)
        logger.info("TF-IDF embedder ready (offline)")

    def _model_candidates(self):
        """候选模型列表: 用户指定的 + 备选"""
        yield self.model_name
        for m in self._SEMANTIC_MODELS:
            if m != self.model_name:
                yield m

    # ================================================================
    #  Semantic (sentence-transformers)
    # ================================================================

    def _load_semantic_model(self, model_name: str):
        """加载 sentence-transformers 模型 (优先从缓存，失败时下载)"""
        from sentence_transformers import SentenceTransformer

        # 先尝试从本地缓存加载 (秒级)
        try:
            logger.info(f"Loading {model_name} from local cache...")
            self._model = SentenceTransformer(
                model_name, device=self.device, local_files_only=True
            )
        except Exception:
            # 缓存没有，需要下载 (分钟级)
            logger.info(f"Not cached, downloading {model_name} via {os.environ.get('HF_ENDPOINT', 'HF official')} ...")
            old_offline = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "0"  # 暂时允许网络
            try:
                self._model = SentenceTransformer(model_name, device=self.device)
            finally:
                if old_offline is not None:
                    os.environ["HF_HUB_OFFLINE"] = old_offline

        self.model_name = model_name
        self._dimension = self._model.get_embedding_dimension()
        logger.info(f"Model ready: {model_name}, dim={self._dimension}")

    def _embed_semantic(self, texts: list[str]) -> np.ndarray:
        """用语义模型编码"""
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embeddings)

    # ================================================================
    #  TF-IDF fallback
    # ================================================================

    def _fit_tfidf(self, texts: list[str]):
        """在文档集上拟合 TF-IDF 向量器"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        logger.info(f"Fitting TF-IDF on {len(texts)} documents...")
        self._tfidf = TfidfVectorizer(
            max_features=512,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True,
        )
        self._tfidf.fit(texts)
        self._dimension = 512
        self._model_type = "tfidf"
        logger.info(f"TF-IDF fitted: vocab={len(self._tfidf.vocabulary_)}, dim={self._dimension}")

    def _embed_tfidf(self, texts: list[str]) -> np.ndarray:
        """用 TF-IDF 编码"""
        if self._tfidf is None:
            logger.warning("TF-IDF not fitted, auto-fitting on input texts")
            self._fit_tfidf(texts)

        from sklearn.preprocessing import normalize
        vectors = self._tfidf.transform(texts)
        vectors = normalize(vectors, norm='l2')
        return vectors.toarray()

    # ================================================================
    #  Singleton
    # ================================================================

    @classmethod
    def get_instance(cls, **kwargs) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance
