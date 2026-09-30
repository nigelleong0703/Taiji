"""Native vLLM token-embedding model registration for Taiji.

This follows the serving pattern used by mode-io/vllm-jev: let vLLM run its native
Qwen3.5 backbone with ``task='token_embed'``, then apply Taiji's trained decision head
to the returned marker and query states in ``vllm_backend.py``. This keeps the backbone
on vLLM's normal scheduler and pooling runner without replacing its architecture or
pretending the model is a binary sequence classifier.

This module is loaded as a ``vllm.general_plugins`` entry point before the engine starts.
"""

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.adapters import as_embedding_model
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

ARCH = "TaijiForVLLM"


class TaijiForVLLM(as_embedding_model(Qwen3_5ForConditionalGeneration)):
    """Qwen3.5's native token states for Taiji's external trained decision head."""


def register():
    """Register the Taiji architecture with vLLM's model registry."""
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, "vllm_model:TaijiForVLLM")
    return ARCH


__all__ = ["ARCH", "TaijiForVLLM", "register"]
