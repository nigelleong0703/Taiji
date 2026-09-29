# Taiji 太极

**Taiji-2B is a System One model for fast, bounded-choice decisions and short structured text.** It scores the options an application supplies, can write a field value, and can hand uncertain decisions to a separate System Two model.

Taiji-2B is based on Qwen3.5-2B with LoRA adapters and a decision head. Download the [model weights](https://huggingface.co/nigelleong0703/Taiji-2B); this repository includes the **inference engine**, offline runner, HTTP server, examples, and training recipes. Taiji is an independent project and is not affiliated with TypeSafe or Jev.

## Try it

Linux, Python 3.10+, and an NVIDIA GPU with at least 8 GB of memory:

```bash
git clone https://github.com/nigelleong0703/Taiji.git
cd Taiji
pip install -r requirements.txt
hf download nigelleong0703/Taiji-2B --local-dir Taiji-2B
./examples/run_demo.sh
```

The demo runs the included inference engine against [`examples/requests.jsonl`](examples/requests.jsonl), which contains choice, yes/no, and field-writing examples. For direct JSONL inference, use `python infer.py --adapter Taiji-2B --data examples/requests.jsonl --output answers.jsonl`.

## Browser demo

The browser agent is an optional integration, not a model requirement. Its recorded demo and run instructions are in [`examples/browser`](examples/browser/README.md).

![Recorded Taiji browser-agent demo](examples/browser/harness/docs/demo.gif)

The agent passed 6/6 simple-site tasks. Google Flights completion detection remains weak; see the [model card](MODEL_CARD.md) and [evaluation notes](docs/reproduction.md).

## Inference engine

- [`s1.py`](s1.py): loads the base model and adapter, scores supplied options, and writes short text.
- [`infer.py`](infer.py): runs decision and writing requests offline from JSONL.
- [`serve.py`](serve.py): authenticated HTTP serving at `/v1/systemone` and `/v1/chat/completions`.
- [`client.py`](client.py): Python client with optional escalation to an OpenAI-compatible S2 model.

See the [inference guide](docs/inference.md) for server setup, request formats, and integration details.

## Repository map

- `examples/`: offline requests, S1/S2 client example, and browser integration.
- `docs/`: inference guide and training/evaluation reproduction notes.
- `MODEL_CARD.md`: intended use, limitations, training data, evaluation, and licensing.
- `CITATION.cff`: citation metadata.
- `train.py`, `train_v3.sh`, `train_v4.sh`, `prepare_*.py`: training and data preparation.

## Results and licenses

Project-reported measurements include 0.882 accuracy on 490 held-out Jev-format questions versus 0.859 for the hosted baseline, 0.86 accuracy and 0.02 ECE on a separate 2,945-row decision set after calibration, and 0.76 exact match on field text. The Jev comparison uses domains represented in training and is not an independent generalization benchmark. Full setup and caveats are in [`docs/reproduction.md`](docs/reproduction.md).

Repository code is Apache-2.0. The released weights are CC BY-NC-SA 4.0 because the v3 training mix includes WebLINX. Upstream datasets retain their own terms. Read the [model card](MODEL_CARD.md) and [`LICENSE`](LICENSE) before reuse.
