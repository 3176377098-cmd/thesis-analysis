"""
LLM 配置工厂 - 创建 OpenAI 兼容的 ChatModel 实例
支持 DeepSeek、OpenAI、Ollama 本地模型 (一键切换)
"""

from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from loguru import logger

from .settings import get_settings

# ===== UI → 后端 Key 桥接 =====
# Streamlit UI 通过 set_ui_credentials() 设置，
# create_chat_model() 自动读取，优先级高于 .env

_ui_credentials: dict = {}  # {"provider": str, "api_key": str, "base_url": str | None}


def set_ui_credentials(provider: str, api_key: str, base_url: Optional[str] = None):
    """UI 侧边栏调用此函数注入用户输入的凭证（仅存内存）"""
    global _ui_credentials
    _ui_credentials = {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
    }
    if provider:
        logger.info(f"[UI] {provider} 凭证已从侧边栏加载")


def _get_ui_override(provider: str) -> tuple[Optional[str], Optional[str]]:
    """获取 UI 输入的 Key/URL 覆盖"""
    if _ui_credentials.get("provider") == provider:
        return _ui_credentials.get("api_key"), _ui_credentials.get("base_url")
    return None, None


def create_chat_model(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
    api_key_override: Optional[str] = None,
    base_url_override: Optional[str] = None,
) -> BaseChatModel:
    """
    根据配置创建 LLM 聊天模型实例。

    支持三种提供商，通过 UI 侧边栏切换:
    - ollama:   本地 (默认，免费离线，无需 Key)
    - deepseek: 云端 (需要 API Key，可在 UI 直接输入)
    - openai:   云端 (需要 API Key，可在 UI 直接输入)

    Args:
        temperature: 温度参数
        max_tokens: 最大输出 token 数
        streaming: 是否启用流式输出
        api_key_override: UI 传入的 API Key (优先级高于 .env)
        base_url_override: UI 传入的 Base URL (优先级高于 .env)

    Returns:
        ChatOpenAI 实例 (Ollama 使用 OpenAI 兼容端点)
    """
    settings = get_settings()

    if temperature is None:
        temperature = settings.AGENT_MODEL_TEMPERATURE

    # Key 优先级: 函数参数 > UI 侧边栏输入 > .env 配置
    ui_key, ui_url = _get_ui_override(settings.LLM_PROVIDER)
    api_key = api_key_override or ui_key or settings.llm_api_key
    base_url = base_url_override or ui_url or settings.llm_base_url

    # 云端模式需要 Key 检查
    if not settings.is_local_model and (not api_key or api_key == "ollama"):
        logger.warning(
            f"[云端模式] API Key 未设置! "
            f"请在 UI 侧边栏输入 API Key，或切换回 Ollama 本地模型"
        )

    # Ollama 本地模型
    if settings.is_local_model:
        logger.info(
            f"[本地模型] Ollama: {settings.llm_model} "
            f"(离线免费，隐私安全)"
        )
        if temperature is None or temperature < 0.1:
            temperature = 0.3
        timeout = max(settings.AGENT_TIMEOUT_SECONDS, 300)
    else:
        logger.info(
            f"[云端模型] provider={settings.LLM_PROVIDER}, "
            f"model={settings.llm_model}"
        )
        timeout = settings.AGENT_TIMEOUT_SECONDS

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=api_key or "sk-placeholder",
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        timeout=timeout,
    )


def create_fast_model() -> BaseChatModel:
    """创建快速/低成本模型 (用于元数据提取等简单任务)"""
    return create_chat_model(temperature=0.0, max_tokens=2048)


def create_deep_model() -> BaseChatModel:
    """创建深度分析模型 (用于批判性审查、综述等复杂任务)"""
    return create_chat_model(temperature=0.2, max_tokens=4096)


def check_ollama_available() -> tuple[bool, str]:
    """
    检测本地 Ollama 服务是否可用。

    Returns:
        (available, message): 是否可用 + 说明信息
    """
    import urllib.request
    import json

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"User-Agent": "PaperAnalyzer"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            return True, f"Ollama 已连接，可用模型: {', '.join(models[:5])}"
    except Exception as e:
        return False, (
            f"Ollama 未检测到 ({str(e)[:60]}...)\n"
            f"请先安装: https://ollama.com\n"
            f"然后运行: ollama pull qwen3:14b"
        )
