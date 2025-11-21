"""Helpers for selecting the LiteRT interpreter when available."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import tensorflow as tf

try:
    from ai_edge_litert.interpreter import Interpreter as _LiteRtInterpreter
except ImportError:  # pragma: no cover - optional dependency
    _LiteRtInterpreter = None

_LOGGER = logging.getLogger(__name__)
_FALLBACK_WARNING_EMITTED = False


def create_tflite_interpreter(model_path: Path, num_threads: int):
    """Return the LiteRT interpreter when installed, otherwise fall back."""
    global _FALLBACK_WARNING_EMITTED

    if _LiteRtInterpreter is not None:
        return _LiteRtInterpreter(model_path=str(model_path), num_threads=num_threads)

    if not _FALLBACK_WARNING_EMITTED:
        base_message = (
            "ai_edge_litert is not installed; falling back to deprecated tf.lite.Interpreter."
        )
        if sys.platform.startswith("win"):
            # Wheels are not published for Windows yet; nothing actionable.
            _LOGGER.info(
                "%s LiteRT wheels are not currently available on Windows; continuing with tf.lite.",
                base_message,
            )
        else:
            _LOGGER.warning(
                "%s Install ai-edge-litert>=2.20.0 to silence this warning.",
                base_message,
            )
        _FALLBACK_WARNING_EMITTED = True

    return tf.lite.Interpreter(model_path=str(model_path), num_threads=num_threads)
