# Taiji vLLM integration

Registers Taiji's LoRA-merged Qwen3.5 model with one vLLM generation engine. The source
patch reads trained decision scores at option markers during prefill; normal requests
use the same weights for text generation. Both paths share the engine and prefix cache.

See [`../docs/vllm.md`](../docs/vllm.md) for checkpoint export, installation, serving, and
the current verification limits. The runtime package is installed in a Linux environment
with NVIDIA GPU support and a compatible vLLM release.

For deployment on compact GPUs (8 GB VRAM) with FP8 quantization and performance tuning,
see [Serve in FP8 on an 8 GB GPU](../docs/vllm.md#serve-in-fp8-on-an-8-gb-gpu).
