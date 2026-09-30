# Taiji vLLM integration

Registers Taiji's Qwen3.5 backbone with vLLM's token-embedding runner and applies the
trained decision head to the compiled option-marker and query token states.

See [`../docs/vllm.md`](../docs/vllm.md) for checkpoint export, installation, serving, and
the current verification limits. The runtime package is installed in a Linux environment
with NVIDIA GPU support and a compatible vLLM release.
