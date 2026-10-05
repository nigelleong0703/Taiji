"""Taiji model registration for the single shared vLLM generate engine.

Qwen3.5 generation model with a per-request decision readout. TaijiForVLLM below is a
pure subclass of the native Qwen3.5 conditional-generation model: no pooling wrapper,
no architecture replacement. Generation keeps the native LM head, so the same weights
the patched worker scores readout requests against also write field text. The decision
head itself lives in the worker (loaded once from the model folder's
taiji_config.json/taiji_head.pt by the source patch in inference/vllm_patch/apply.py),
not in this class.

This module is loaded as a ``vllm.general_plugins`` entry point before the engine starts.
"""

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

ARCH = "TaijiForVLLM"


class TaijiForVLLM(Qwen3_5ForConditionalGeneration):
    """Native Qwen3.5 generation model for the shared Taiji engine.

    A pure subclass: the generate runner runs the stock generation path (native LM
    head, sampling, prefix caching) unchanged. Decision scoring is a per-request
    behaviour added by the installed source patch (inference/vllm_patch/apply.py),
    which reads the SamplingParams readout spec and returns head logits on
    RequestOutput.taiji_scores.
    """


def register():
    """Register the Taiji architecture with vLLM's model registry."""
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, "vllm_model:TaijiForVLLM")
    return ARCH


__all__ = ["ARCH", "TaijiForVLLM", "register"]
