# Taiji 太极

**Taiji-2B is a System One model for fast, bounded-choice decisions and short structured text.** It scores the options an application supplies, can write a field value, and can hand uncertain decisions to a separate System Two model.

Taiji-2B is based on Qwen3.5-2B with LoRA adapters and a decision head. Download the [model weights](https://huggingface.co/nigelleong0703/Taiji-2B). This repository is the public inference release: it includes the model card, inference engine, runnable demos, and integration examples. Taiji is an independent project and is not affiliated with TypeSafe or Jev.

## Quick start

Linux, Python 3.10+, and an NVIDIA GPU with at least 8 GB of memory:

```bash
git clone https://github.com/nigelleong0703/Taiji.git
cd Taiji
pip install -r inference/requirements.txt
hf download nigelleong0703/Taiji-2B --local-dir Taiji-2B
./examples/run_demo.sh
```

The offline demo runs [`inference/infer.py`](inference/infer.py) on [`examples/requests.jsonl`](examples/requests.jsonl) and prints decision and field-writing results. For a long-running HTTP service, see the [inference guide](docs/inference.md).

## Serve with vLLM in FP8 (8 GB GPU)

The fast path for agents: one patched vLLM 0.29.0 engine serves decisions (`/v1/systemone`) and field writing (`/v1/chat/completions`) from FP8 weights. Measured on an RTX 4060 Laptop: ~6.9 GB GPU memory, decision latency p50/p90 297/458 ms cold and ~213/240 ms with a cached prefix. Weights: [nigelleong0703/Taiji-2B](https://huggingface.co/nigelleong0703/Taiji-2B) (LoRA adapter + decision head on [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)).

Use two Python environments: the exporter needs `transformers`/`peft`, while vLLM pins its own torch and transformers.

```bash
# 1. Export: merge the LoRA into the base model and package the decision head.
python3 -m venv .venv-export
.venv-export/bin/pip install -r inference/requirements.txt
.venv-export/bin/hf download nigelleong0703/Taiji-2B --local-dir Taiji-2B --exclude 'wheels/*'
.venv-export/bin/python inference/vllm_export.py --base Qwen/Qwen3.5-2B --adapter Taiji-2B \
  --out taiji-vllm --dtype bfloat16
# Check: taiji-vllm/ contains taiji_head.pt and taiji_config.json.

# 2. Runtime: vLLM, the Taiji model plugin, and the source patch (engine stopped).
python3.12 -m venv .venv-vllm
.venv-vllm/bin/pip install -r inference/requirements-vllm.txt   # CUDA 13.0 hosts: requirements-vllm-cu130.txt
.venv-vllm/bin/pip install --no-deps -e ./inference
.venv-vllm/bin/python inference/vllm_patch/apply.py
# Check: it exits without error; it refuses any vLLM other than 0.29.0. Undo with --restore.

# 3. Serve in FP8 (vision tower and lm_head stay unquantized; CUDA graphs on).
export S1_API_KEY='<long-random-secret>'   # required; clients send Authorization: Bearer $S1_API_KEY
.venv-vllm/bin/python -u inference/serve.py --backend vllm --model taiji-vllm \
  --host 0.0.0.0 --port 8010 --max-len 262144 \
  --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192 --max-num-seqs 4 \
  --quantization fp8_per_tensor --quantization-config '{"ignore":["*visual*","*lm_head*"]}' \
  --mm-max-pixels 409600

# 4. Verify.
curl http://localhost:8010/health
# -> {"ok": true, "model": ..., "max_model_len": 262144, "quantization": "fp8_per_tensor", "cpu_offload_gb": 0}
# Optional, with the server stopped: cold/cached decisions and generation in one engine.
head -1 examples/requests.jsonl > request.json
.venv-vllm/bin/python inference/vllm_patch/integration_test.py --model taiji-vllm --request request.json
```

CUDA 13.0 toolkit setup (`CUDA_HOME`, nvcc/ninja on PATH), what the patch changes, the request contract, and tuning notes are in [docs/vllm.md](docs/vllm.md).

## General S1/S2 agent

Taiji includes a shared [MCP agent runtime](agent_runtime/README.md): S1 makes fast tool choices, S2 handles planning and native tool calls, and registered MCP servers supply the tools. Browser control is available through an MCP adapter in the same runtime, alongside other tool servers.

## Browser demo

The browser agent is an optional integration, not a model requirement. Its recorded demo and run instructions are in [`examples/browser`](examples/browser/README.md).

![Recorded Taiji browser-agent demo](examples/browser/harness/docs/demo.gif)

The agent passed 6/6 simple-site tasks. Google Flights completion detection remains weak; see the [model card](MODEL_CARD.md) and [evaluation notes](docs/reproduction.md).

## Repository layout

- [`MODEL_CARD.md`](MODEL_CARD.md): intended use, limitations, data, evaluation, and licensing.
- [`inference/`](inference/): model implementation, offline runner, HTTP service, Python client, and runtime dependencies.
- [`agent_runtime/`](agent_runtime/): MCP tool discovery, the shared S1/S2 loop, and browser adapter.
- [`examples/`](examples/): runnable JSONL requests, S1/S2 integration, and optional browser integration.
- [`docs/`](docs/): inference instructions and reported evaluation details.
- [`CITATION.cff`](CITATION.cff): citation metadata.

This release focuses on inference and evaluation. Dataset preparation and training code remain in the private development repository.

## Reported results

Project measurements include 0.882 accuracy on 490 held-out Jev-format questions versus 0.859 for the hosted baseline, 0.86 accuracy and 0.02 ECE on a separate 2,945-row decision set after calibration, and 0.76 exact match on field text. The Jev comparison uses domains represented in training and is not an independent generalization benchmark. Full measurement notes and limitations are in [`docs/reproduction.md`](docs/reproduction.md).

The repository code is Apache-2.0. The released weights are CC BY-NC-SA 4.0 because the training mix includes WebLINX. Upstream datasets retain their own terms. Read the [model card](MODEL_CARD.md) and [`LICENSE`](LICENSE) before reuse.
