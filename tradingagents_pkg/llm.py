"""Chat model construction for the TradingAgents framework.

Wraps `langchain.chat_models.init_chat_model` so the project can specify
LLMs via an explicit `provider` (one of `LLMProvider`) plus a model name
(e.g. `gpt-5.4`, `claude-sonnet-4-6`, `gemini-3.1-pro-preview`) while still
mapping a unified `reasoning_effort` knob onto each provider's native
parameter.

The concrete provider classes are imported explicitly so `ChatModel` is a
visible union of what the project actually supports, instead of leaning on
`BaseChatModel` everywhere.
"""

from typing import Any, Literal, cast

from langchain_ollama import ChatOllama
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel

# Optional providers — imported lazily to avoid hard dependency errors
def _try_import(module: str, cls: str):
    try:
        import importlib
        return getattr(importlib.import_module(module), cls)
    except Exception:
        return None

ChatXAI            = _try_import("langchain_xai",         "ChatXAI")
ChatOpenAI         = _try_import("langchain_openai",       "ChatOpenAI")
ChatLiteLLM        = _try_import("langchain_litellm",      "ChatLiteLLM")
ChatAnthropic      = _try_import("langchain_anthropic",    "ChatAnthropic")
ChatOpenRouter     = _try_import("langchain_openrouter",   "ChatOpenRouter")
ChatHuggingFace    = _try_import("langchain_huggingface",  "ChatHuggingFace")
ChatGoogleGenAI    = _try_import("langchain_google_genai", "ChatGoogleGenerativeAI")
ChatGoogleGenerativeAI = ChatGoogleGenAI

try:
    from langchain.chat_models import init_chat_model as _init_chat_model
except Exception:
    _init_chat_model = None

ChatModel = BaseChatModel

LLMProvider = Literal[
    "openai", "anthropic", "google_genai", "xai", "huggingface", "openrouter", "ollama", "litellm"
]

ReasoningEffort = Literal["low", "medium", "high", "xhigh", "max"]


if ChatGoogleGenAI is not None:
    class NormalizedChatGoogleGenerativeAI(ChatGoogleGenAI):
        """Flatten Gemini 3 list-content responses to a plain string."""

        def invoke(self, prompt_input: object, config: object = None, **kwargs: object) -> object:
            response = super().invoke(prompt_input, config, **kwargs)
            content = response.content
            if isinstance(content, list):
                response.content = "\n".join(
                    item["text"]
                    if isinstance(item, dict) and item.get("type") == "text"
                    else item
                    if isinstance(item, str)
                    else ""
                    for item in content
                ).strip()
            return response
else:
    NormalizedChatGoogleGenerativeAI = None


def build_chat_model(
    provider: LLMProvider,
    model: str,
    *,
    reasoning_effort: ReasoningEffort | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> ChatModel:
    """Construct a chat model from an explicit provider + model name.

    Args:
        provider (LLMProvider): Langchain `init_chat_model` registry key (one of `LLMProvider`).
        model (str): Model name as accepted by the provider (e.g. `gpt-5.4`,
            `claude-sonnet-4-6`, `gemini-3.1-pro-preview`). Any model name
            containing `gemini` or `google` is routed through
            `NormalizedChatGoogleGenerativeAI` regardless of provider.
        reasoning_effort (ReasoningEffort | None, optional): Unified reasoning
            level mapped per provider:
            Anthropic -> `effort` (native low/medium/high/xhigh/max),
            OpenAI -> `reasoning_effort` (max -> xhigh; xhigh native),
            Google -> `thinking_level` (xhigh and max both clamped to high).
            Other providers do not expose a unified knob and ignore this.
        callbacks (list[BaseCallbackHandler] | None, optional): LangChain
            callback handlers attached to the model. Defaults to None.

    Returns:
        ChatModel: The constructed chat model instance.
    """
    kwargs: dict[str, Any] = {}
    if callbacks:
        kwargs["callbacks"] = callbacks
    if reasoning_effort:
        _apply_reasoning(provider, reasoning_effort, kwargs)

    model_lower = model.lower()
    if ("gemini" in model_lower or "google" in model_lower) and NormalizedChatGoogleGenerativeAI is not None:
        return NormalizedChatGoogleGenerativeAI(model=model, **kwargs)

    # Ollama: bypass init_chat_model and construct directly so we can cap token generation.
    # qwen3 extended-thinking mode generates 100k+ tokens without a num_predict limit.
    if provider == "ollama":
        import os
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return ChatOllama(model=model, base_url=host, num_predict=4096, **kwargs)

    if _init_chat_model is None:
        raise ImportError(f"langchain init_chat_model not available for provider={provider}")
    return cast("ChatModel", _init_chat_model(model, model_provider=provider, **kwargs))


def _apply_reasoning(
    provider: LLMProvider, effort: ReasoningEffort, kwargs: dict[str, Any]
) -> None:
    """Apply the corresponding reasoning effort configuration for a given provider.

    Args:
        provider (LLMProvider): The LLM provider key.
        effort (ReasoningEffort): The unified reasoning effort level.
        kwargs (dict[str, Any]): The configuration dictionary to be updated in-place.
    """
    e = effort.lower()
    if provider == "anthropic":
        kwargs["effort"] = e
    elif provider == "openai":
        kwargs["reasoning_effort"] = "xhigh" if e == "max" else e
    elif provider == "google_genai":
        kwargs["thinking_level"] = "high" if e in ("xhigh", "max") else e
