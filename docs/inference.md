# Inference engine

Taiji includes its own inference path; the repository is not only a weight pointer. The model implementation and loader are in `s1.py`, offline inference is in `infer.py`, and the authenticated HTTP server is in `serve.py`.

## Offline JSONL inference

Install the dependencies listed in the root `requirements.txt`, download the base and adapter, then run:

```bash
hf download nigelleong0703/Taiji-2B --local-dir Taiji-2B
python infer.py --base Qwen/Qwen3.5-2B --adapter Taiji-2B \
  --data examples/requests.jsonl --output answers.jsonl
```

Each input line is either a decision request (`state` plus named `questions`) or a text-writing request (`write`). Each output line contains the corresponding probabilities/answers or generated text. `./examples/run_demo.sh` runs the same engine on the checked-in examples and prints its results.

## HTTP serving

```bash
S1_API_KEY='<long-random-secret>' python serve.py \
  --base Qwen/Qwen3.5-2B --adapter Taiji-2B --host 0.0.0.0 --port 8000
```

The server requires `Authorization: Bearer $S1_API_KEY` for both POST routes. `GET /health` is an unauthenticated health check.

- `POST /v1/systemone`: decision requests with a state and one or more `choice`, `noul`, or `score` questions. A screenshot may be supplied as a data URI/base64 field.
- `POST /v1/chat/completions`: OpenAI-compatible message envelope. Set `model` to `s1` to generate field text with Taiji. Plain base-model chat is optional and requires `--plain-chat`.

A direct client example that combines Taiji with an optional OpenAI-compatible System Two model is in [`examples/s1_s2.py`](../examples/s1_s2.py). The optional [browser integration](../examples/browser/README.md) shows how the decision and text endpoints fit into an agent.

## Runtime flow

For decisions, the engine tokenizes the shared state once and scores each supplied question's options in a batched forward pass. The decision head returns logits that are converted to probabilities using the fitted temperature. For writing, it uses deterministic generation with a bounded token budget. When enabled and verified for the target environment, the HTTP server can reuse a cached state prefix across requests.

The inference engine uses Transformers and PyTorch; the base model's license and hardware requirements also apply. The published latency and memory measurements in the [reproduction notes](reproduction.md) are specific to their stated hardware and request sizes.
