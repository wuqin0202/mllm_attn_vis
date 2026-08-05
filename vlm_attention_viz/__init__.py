"""Interactive attention analysis for Qwen-family vision-language models."""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "AttentionSession",
    "LayerAttention",
    "SessionCache",
    "TokenInfo",
    "build_app",
    "extract_attention",
    "load_model",
]


def __getattr__(name: str):
    if name in {"AttentionSession", "LayerAttention", "TokenInfo", "extract_attention", "load_model"}:
        from . import attention

        return getattr(attention, name)
    if name in {"SessionCache", "build_app"}:
        from . import app

        return getattr(app, name)
    raise AttributeError(name)
