# Inference engine

Taiji includes its own inference path; the repository is not only a weight pointer. The model implementation and loader are in `inference/s1.py`, offline inference is in `inference/infer.py`, and the authenticated HTTP server is in `inference/serve.py`.

## Offline JSONL inference

Install the inference dependencies, download the model weights, then run:

```bash
pip install -r inference/requirements.txt
hf download nigelleong0703/Taiji-2B --local-dir Taiji-2B
python inference/infer.py --base Qwen/Qwen3.5-2B --adapter Taiji-2B \
  --data examples/requests.jsonl --output answers.jsonl
```

Each input line is either a decision request (`state` plus named `questions`) or a text-writing request (`write`). Each output line contains the corresponding answers and probabilities or generated text. `./examples/run_demo.sh` runs the same engine on the checked-in examples and prints its results.

### Apple Silicon (MPS)

On an Apple Silicon Mac, use a native arm64 Python 3.12 environment and the Mac dependency set. The model files stay under the ignored `hf/` directory:

```bash
uv venv --python 3.12 .venv-mac-inference
uv pip install --python .venv-mac-inference/bin/python -r inference/requirements-macos.txt
hf download Qwen/Qwen3.5-2B --local-dir hf/macos/Qwen3.5-2B
hf download nigelleong0703/Taiji-2B --local-dir hf/macos/Taiji-2B --exclude 'wheels/*'
.venv-mac-inference/bin/python inference/infer.py \
  --base hf/macos/Qwen3.5-2B --adapter hf/macos/Taiji-2B \
  --data examples/requests.jsonl --output answers.jsonl
```

The loader selects MPS automatically on Apple Silicon. Run `inference/check_cache.py` before enabling `--shared-prefix`; cached-prefix numerical parity depends on the device kernels.

## HTTP serving

```bash
S1_API_KEY='<long-random-secret>' python inference/serve.py \
  --base Qwen/Qwen3.5-2B --adapter Taiji-2B --host 0.0.0.0 --port 8000
```

The server requires `Authorization: Bearer $S1_API_KEY` for both POST routes. `GET /health` is an unauthenticated health check.

- `POST /v1/systemone`: decision requests with a state and one or more `choice`, `noul`, or `score` questions. A screenshot may be supplied as a data URI/base64 field.
- `POST /v1/chat/completions`: OpenAI-compatible message envelope. Set `model` to `s1` to generate field text with Taiji. Plain base-model chat is optional and requires `--plain-chat`.

A Python example that combines Taiji with an optional OpenAI-compatible System Two model is in [`examples/s1_s2.py`](../examples/s1_s2.py). The optional [browser integration](../examples/browser/README.md) shows how the decision and text endpoints fit into an agent.

For a shared agent loop that discovers tools from configured MCP servers, see the [general agent runtime](../agent_runtime/README.md). It uses Taiji for fast tool selection and an OpenAI-compatible S2 model for planning and native tool calls; the included browser MCP adapter runs through this same loop.

For an agent run, the runtime assigns one private cache session to its S1 requests. With `--shared-prefix`, the Transformers server can keep the stable goal prefix and evaluate the changing recent-step history plus decision question on each turn. This fast path applies to text-only agent state; image requests use the regular shared-state path. It is opt-in because cached and full-prefix scores must pass the numerical check below.

The browser agent can use vLLM as its System Two planner through vLLM's OpenAI-compatible chat endpoint. This is the mainstream serving path for the planner. Taiji's S1 decision head scores hidden states at each option marker against a final query token, so replacing it with ordinary next-token or generic sequence-classification scores would change the trained decision function. The experimental backend below uses vLLM's native token-embedding runner and applies Taiji's trained head to those exact states; compare it with the Transformers runtime before using it as the default. See the [vLLM browser-agent setup](../examples/browser/README.md#run-system-two-with-vllm).

An experimental `--backend vllm` mode now follows the native token-embedding and external trained-head pattern used by [vLLM-jev](https://github.com/mode-io/vllm-jev). It registers Taiji's Qwen3.5 model with vLLM, then applies the exact `YesNoHead` to option-marker and query token states through the existing `inference/serve.py` API. Export steps, CPU parity tests, and current limits are in [docs/vllm.md](vllm.md). For serving on an 8 GB GPU with FP8 quantization and tuned batching, see [Serve in FP8 on an 8 GB GPU](vllm.md#serve-in-fp8-on-an-8-gb-gpu) in the same file.

Before enabling prefix caching, run `python inference/check_cache.py <base> <adapter> <screenshot.png>` on the target inference host. It checks image, text, and incremental agent-state requests against full recomputation; only use `--shared-prefix` if it reports that the path is safe on that host.

## Runtime flow

For decisions, the engine tokenizes the shared state once and scores each supplied question's options in a batched forward pass. The decision head returns logits that are converted to probabilities using the fitted temperature. For writing, it uses deterministic generation with a bounded token budget. When enabled and verified for the target environment, the HTTP server can reuse a cached state prefix across requests.

The inference engine uses Transformers and PyTorch; the base model's license and hardware requirements also apply. Published latency and memory measurements in the [evaluation notes](reproduction.md) are specific to their stated hardware and request sizes.
