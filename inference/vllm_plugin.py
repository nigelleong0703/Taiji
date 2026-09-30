"""General vLLM plugin entry point for Taiji's Qwen3.5 token embedding model.

Install this package so vLLM discovers the model registration before the token-embedding
engine starts. Taiji's trained decision head is applied to the returned states by
`vllm_backend.py`.

Register it via the entry point in `inference/pyproject.toml`:

    [project.entry-points."vllm.general_plugins"]
    taiji = "vllm_plugin:register"

or call `register()` yourself before building the engine.
"""


def register():
    from vllm_model import register as _register

    return _register()


__all__ = ["register"]
