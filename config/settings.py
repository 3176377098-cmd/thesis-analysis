"""
全局配置管理 - 使用 Pydantic BaseSettings
从 .env 文件和环境变量加载配置
"""

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用全局配置"""

    # ===== 项目路径 =====
    PROJECT_ROOT: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.resolve()
    )
    DATA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.resolve() / "data"
    )
    PAPERS_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.resolve() / "data" / "papers"
    )
    PARSED_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.resolve() / "data" / "parsed"
    )
    CHROMA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.resolve() / "data" / "chroma"
    )
    TRACES_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.resolve() / "data" / "traces"
    )

    # ===== LLM 配置 =====
    LLM_PROVIDER: Literal["deepseek", "openai", "ollama", "gemini"] = "ollama"

    # DeepSeek 云端
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # OpenAI 云端
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o"

    # Google Gemini 云端 (免费额度: 1500次/天, 支持图片分析)
    GEMINI_API_KEY: str = ""
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GEMINI_MODEL: str = "gemini-2.0-flash"  # 免费多模态模型

    # Ollama 本地 (免费，无需 API Key)
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_MODEL: str = "qwen3:14b"  # 推荐: qwen3:14b, llama3.1:8b, deepseek-r1:8b

    # ===== 嵌入模型 =====
    # BGE 中文语义模型（走 hf-mirror.com 下载），失败时自动回退 TF-IDF
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DEVICE: str = "cpu"  # "cpu" | "cuda"
    HF_ENDPOINT: str = "https://hf-mirror.com"  # 国内镜像，设为空字符串用官方 HF

    # ===== PDF 解析 =====
    PARSER_PREFERENCE: Literal["auto", "pymupdf", "mineru"] = "auto"
    PDF_MAX_PAGES: int = 100  # 超过此页数警告

    # ===== 分块策略 =====
    CHUNK_SIZE: int = 800  # 目标 token 数
    CHUNK_OVERLAP: int = 80  # 重叠 token 数
    CHUNK_MIN_SIZE: int = 100  # 最小 chunk token 数

    # ===== 检索配置 =====
    RETRIEVAL_DEFAULT_K: int = 20  # 默认检索返回数
    RETRIEVAL_MAX_K: int = 50  # 最大检索返回数
    RETRIEVAL_STRATEGY: Literal["hybrid", "dense", "sparse", "hyde", "multi_query"] = "hybrid"
    RRF_K: int = 60  # Reciprocal Rank Fusion 常数

    # ===== Agent 配置 =====
    AGENT_MAX_REVISIONS: int = 2  # Critique→DeepRead 最大循环次数
    AGENT_MODEL_TEMPERATURE: float = 0.1  # Agent LLM 温度
    AGENT_TIMEOUT_SECONDS: int = 120  # 单 Agent 超时时间

    # ===== 代码执行沙箱 =====
    CODE_SANDBOX_TIMEOUT: int = 30  # 代码执行超时(秒)
    CODE_SANDBOX_MAX_MEMORY_MB: int = 256  # 最大内存(MB)

    # ===== 昆明理工大学图书馆 (可选) =====
    KUST_LIBRARY_PROXY_URL: str = ""
    CNKI_USERNAME: str = ""
    CNKI_PASSWORD: str = ""
    WANFANG_USERNAME: str = ""
    WANFANG_PASSWORD: str = ""

    # ===== 可选 API Keys =====
    SEMANTIC_SCHOLAR_API_KEY: str = ""
    TAVILY_API_KEY: str = ""

    # ===== 日志 =====
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保数据目录存在
        for dir_path in [self.DATA_DIR, self.PAPERS_DIR, self.PARSED_DIR,
                         self.CHROMA_DIR, self.TRACES_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @property
    def llm_api_key(self) -> str:
        """获取当前 LLM 提供商的 API Key (Ollama 不需要)"""
        if self.LLM_PROVIDER == "ollama":
            return "ollama"  # Ollama 不需要真实 Key，但 ChatOpenAI 要求非空
        if self.LLM_PROVIDER == "deepseek":
            return self.DEEPSEEK_API_KEY
        if self.LLM_PROVIDER == "gemini":
            return self.GEMINI_API_KEY
        return self.OPENAI_API_KEY

    @property
    def llm_base_url(self) -> str:
        """获取当前 LLM 提供商的 Base URL"""
        if self.LLM_PROVIDER == "ollama":
            return self.OLLAMA_BASE_URL
        if self.LLM_PROVIDER == "deepseek":
            return self.DEEPSEEK_BASE_URL
        if self.LLM_PROVIDER == "gemini":
            return self.GEMINI_BASE_URL
        return self.OPENAI_BASE_URL

    @property
    def llm_model(self) -> str:
        """获取当前 LLM 模型名"""
        if self.LLM_PROVIDER == "ollama":
            return self.OLLAMA_MODEL
        if self.LLM_PROVIDER == "deepseek":
            return self.DEEPSEEK_MODEL
        if self.LLM_PROVIDER == "gemini":
            return self.GEMINI_MODEL
        return self.OPENAI_MODEL

    @property
    def is_local_model(self) -> bool:
        """是否为本地模型 (离线可用，无 API 费用)"""
        return self.LLM_PROVIDER == "ollama"


# 全局单例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取 Settings 单例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
