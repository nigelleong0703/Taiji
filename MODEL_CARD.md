---
language:
  - en
license: cc-by-nc-sa-4.0
base_model: Qwen/Qwen3.5-2B
library_name: transformers
datasets:
  - McGill-NLP/WebLINX-full
  - osunlp/Multimodal-Mind2Web
  - ZefanCai/Open-Jev
  - Agent-Ark/Toucan-1.5M
tags:
  - system-one
  - decision-making
  - multimodal
  - structured-generation
---

# Taiji-2B

Taiji-2B is a fine-tuned Qwen3.5-2B model for bounded-choice decisions and short structured text. It scores the options supplied by the caller and can write a field value. Applications may route uncertain decisions to a separate System Two model.

- **Weights:** [nigelleong0703/Taiji-2B](https://huggingface.co/nigelleong0703/Taiji-2B)
- **Code and inference engine:** [nigelleong0703/Taiji](https://github.com/nigelleong0703/Taiji)
- **Base model:** [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- **Weights license:** CC BY-NC-SA 4.0. **Code license:** Apache-2.0.

## Intended use

Research and prototyping for choice, yes/no, ordered scoring, and short structured text generation. Taiji returns probabilities over caller-supplied choices and can serve as a fast first-stage policy before escalation to a larger model.

## Limitations and risks

Do not treat confidence or a `DONE` decision as proof of correctness. Verify consequential actions independently. This model is not established for high-stakes use. Its browser integration has weak multi-step completion detection. Reported evaluation questions overlap with domains and public datasets represented in training, so the Jev comparison is not an independent generalization benchmark. Long page states and screenshots affect memory use and latency.

## Training data

The v3 weights were fine-tuned on a mixture of WebLINX, Multimodal-Mind2Web, Open-Jev, Toucan, general decision datasets, and synthetic examples. The mixture contains datasets with different licenses and terms. Review the original dataset cards and terms before reusing data or weights. WebLINX is CC BY-NC-SA 4.0; the released weights therefore use CC BY-NC-SA 4.0. The v4 recipe and its additional data sources are documented separately and are not represented as part of these published v3 weights.

## Training procedure

The model uses LoRA rank 16 on the language model, a task-specific decision head, and a frozen vision encoder. Training combines cross-entropy and a Brier loss term, label smoothing, and weighted text rows. Temperature calibration is applied after fine-tuning. Training used one epoch over 60,395 mixed rows, followed by temperature calibration. The public release contains inference and evaluation material; dataset preparation and training scripts are not included here.

## Evaluation

Project-reported measurements for the published weights:

- **Jev-format decisions:** 0.882 accuracy on 490 held-out questions versus 0.859 for the hosted Jev baseline. The questions come from public datasets represented in training; the result is not evidence of a general-purpose advantage.
- **Held-out decisions:** 0.86 accuracy and 0.02 expected calibration error on 2,945 rows after calibration.
- **Field text:** 0.76 exact match.
- **Browser example:** 6/6 simple-site tasks passed. Google Flights reached the results page but did not reliably identify task completion.

These are project measurements, not independently audited results. See [reproduction notes](https://github.com/nigelleong0703/Taiji/blob/main/docs/reproduction.md) for setup, metrics, and caveats.

## Inference

The repository includes the model-loading and decision engine in `s1.py`, offline JSONL inference in `infer.py`, and an HTTP service in `serve.py`. Start with the [inference guide](https://github.com/nigelleong0703/Taiji/blob/main/docs/inference.md) or run the [offline demo](examples/run_demo.sh).

## Ethical and licensing considerations

The model can make incorrect decisions while assigning high confidence. Applications should provide constrained choices, validate outputs, and independently verify completed actions. Training data has mixed terms; downstream users are responsible for checking source dataset licenses. The released weights are for non-commercial use under CC BY-NC-SA 4.0; code is separately licensed under Apache-2.0.
