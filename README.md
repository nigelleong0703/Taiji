# Taiji 太极

**Fast intuition and slow reasoning, fused into one agent system.** Taiji-2B is an open System One model that decides
and writes in a single 2B model, and hands the hard cases to any System Two LLM, the way yin and yang form one circle.

Taiji-2B is Qwen3.5-2B with a LoRA and a decision head. Give it a state (text, JSON, a screenshot) and
questions with the options you allow; it returns calibrated probabilities over exactly those options in one forward
pass, without generating answer tokens. The same model also writes short text, such as the value for a form field or
a tool argument. When it is unsure, route the question to any larger LLM (System Two).

Weights: [nigelleong0703/Taiji-2B](https://huggingface.co/nigelleong0703/Taiji-2B) ·
Request format: compatible with the `/v1/systemone` choice / noul / score questions of
[TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (independent project, not affiliated).

```
             state + questions (the allowed options)
                            │
                  ┌─────────▼─────────┐
                  │     Taiji-2B      │  one forward pass per decision
                  │  S1: decide/write │──► probabilities, confidence, text
                  └─────────┬─────────┘
                confidence < threshold?
                            │ yes
                  ┌─────────▼─────────┐
                  │ S2: any LLM       │  picks among the same options;
                  │ (OpenAI-compat.)  │  an invalid answer keeps S1's
                  └───────────────────┘
```

## What it does

- **Decide**: `choice` (1-255 options), `noul` (true/false) and `score` (2-10 ordered levels). Every option is scored
  in the same pass, so the answer is always one of the options you offered, with a probability for each.
- **Write**: the text for one field or argument (`{"text": ...}`, or `null` when the context lacks the value), from the
  same weights. An agent needs no second model to type "Singapore" into the field its decision picked.
- **Escalate**: probabilities are temperature-calibrated (held-out ECE 0.02), so confidence is a usable signal for when
  to ask System Two. `client.py` does this for any question.
- **See**: an optional screenshot goes through Qwen3.5's vision encoder next to the text state.

## Results

All numbers are our own measurements; setups are listed so they can be reproduced.

**General decisions vs hosted Jev.** Same 490 held-out questions (intent classification, yes/no, scoring, tool choice),
same script (`jev_baseline.py`), same grading:

| Questions | n | Jev (hosted) | Taiji-2B |
|---|---|---|---|
| All | 490 | 0.859 | **0.882** |
| General choice | 248 | 0.891 | 0.899 |
| General yes/no | 33 | 0.970 | 0.970 |
| General score | 19 | 0.632 | 0.684 |
| Tool choice | 116 | 0.871 | 0.914 |
| Tool yes/no | 74 | 0.743 | 0.784 |

The models disagree on 57 questions (Taiji right on 34, Jev on 23; McNemar p ≈ 0.19): **on par**, not a
proven win. These questions come from the same public datasets as part of our training data (held out), which favors
our model.

**Held-out decisions** (2,945 rows): accuracy 0.86, ECE 0.02 after calibration. Field text exact match 0.76.

**Speed.** Server time per decision on a real Google Flights page (~6,500 tokens: page, screenshot and three questions),
one H100: **219 ms** median (171 ms model, 59 ms input preparation), measured on the server; network time comes on
top. Serving memory: about 6 GB of VRAM.

**Browser agent (one example integration).** Simple sites (two Wikipedia tasks, books.toscrape) passed 6/6 with a
correct completion. Google Flights with a screenshot: it fills the route, pages the date picker, picks both dates and
sorts by price, but does not yet declare the task done (p(DONE) stays below 0.1 on the finished results page).
Without a screenshot it cannot operate that date picker. Browsing in this format is under-represented in the training
data; see *Limitations*.

## Quick start

Linux, Python 3.10+, an NVIDIA GPU with 8 GB or more.

```bash
git clone https://github.com/nigelleong0703/Taiji && cd Taiji
pip install -r requirements.txt
hf download nigelleong0703/Taiji-2B --local-dir Taiji-2B
# Optional speed-up for Qwen3.5's convolution layers (torch 2.14 + CUDA 12.6 build, sm 80/86/89/90):
pip install --no-deps Taiji-2B/wheels/causal_conv1d-*.whl
```

**Offline**, one JSONL line per request, decisions and text mixed:

```bash
python infer.py --adapter Taiji-2B --data examples/requests.jsonl --output answers.jsonl
```

```json
{"state": "The traffic light is red.", "questions": {"go": {"type": "noul", "instructions": "Rule: crossing is allowed only when the light is green. Is crossing allowed now?", "criteria": {"true": "Crossing is allowed", "false": "Crossing is not allowed"}}}}
{"write": {"goal": "Find flights from Singapore to Bali on December 27, 2026.", "field": {"label": "Where from?", "role": "combobox", "value": "Bangkok"}}}
```

**Server**: `POST /v1/systemone` decides, `POST /v1/chat/completions` with `"model": "s1"` writes.

```bash
python check_cache.py Qwen/Qwen3.5-2B Taiji-2B some.png   # prints whether --shared-prefix is safe here
S1_API_KEY=<long random string> python serve.py --base Qwen/Qwen3.5-2B --adapter Taiji-2B --shared-prefix
```

`--shared-prefix` reads the state once and answers every question of a request in one batch on it; `check_cache.py`
compares that against full recomputation on your GPU (it passed on the RTX 3090 and H100 we used).

**S1 + S2 from Python** (`client.py`, any OpenAI-compatible S2):

```python
from client import S1, S2
s1 = S1("http://localhost:8000", key)
s2 = S2("https://api.deepseek.com/v1", key2, "deepseek-chat")
answers = s1.decide(state, questions, s2=s2, threshold=0.5)   # unsure choices are re-asked to S2
text = s1.write({"goal": "...", "field": {"label": "Where from?"}})
```

`examples/s1_s2.py` runs a support-ticket example end to end.

## Examples

- `examples/requests.jsonl`: decide and write requests for `infer.py`.
- `examples/s1_s2.py`: route a ticket, flag urgency, write a reply subject; escalate unsure choices.
- `examples/browser/`: a browser agent on this model, built on
  [Browser Use's jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (MIT) with System Two escalation when
  the policy is stuck, and the model writing the field values. One integration among many; the model does not depend
  on it.

## Training

- Base: Qwen/Qwen3.5-2B. LoRA rank 16 on the language model, vision encoder frozen, `yesno` decision head
  (Yes-minus-No readout at each option marker plus a gated candidate head). Loss: cross-entropy + 0.1 Brier, label
  smoothing 0.05, text rows weighted 3x.
- One epoch over 60,395 rows on one H100, about 5.4 hours (`train_v3.sh`), then temperature calibration
  (`calibrate.py`).
- Data (`data.sh`, `data_v3.sh`, `prepare_*.py`): WebLINX 12,000 and Multimodal-Mind2Web 9,000 web steps, Open-Jev
  browser 13,814 and general 10,000 decisions, tool selection 3,000, general public decision sets 3,000, synthetic
  WAIT/BLOCKED 1,500 and SELECT 2,952, completion rows derived from WebLINX 1,420, with rare operations repeated up to
  8% of operation rows.

### Next round (in progress, not trained yet)

- More agent data in the same request format: `prepare_axtree.py` converts released trajectories with accessibility
  trees, [NNetNav-live](https://huggingface.co/datasets/stanfordnlp/nnetnav-live) (Apache-2.0, 54k rows) and
  [AgentTrek](https://huggingface.co/datasets/xlangai/AgentTrek) (no license stated; research use, 46k rows), including
  about 9,800 completion (DONE) steps; `prepare_valen.py` adds screen-reading questions from Valen's RICO-ScreenQA and
  ShowUI desktop sets (28.5k rows with screenshots).
- `mix_round.sh` builds a mix; `train_v4.sh` continues training from this model (`train.py --init`) and calibrates.
- Training speed: `train.py --length-group` gives each step rows of similar length. On real rows it cut padding from
  about 45% to 15-20% of the tokens and raised throughput about 1.5x (measured on an A40).
- `runpod_job.py` runs any of the `*_round.sh` / `speed_probe.sh` jobs on RunPod (creates the pod, saves its logs,
  deletes it). Results will be added here only once measured.

## Limitations

- **Research use only.** WebLINX is CC BY-NC-SA 4.0, so the weights are released under CC BY-NC-SA 4.0. The code in
  this repository is Apache-2.0.
- Recognizing that a multi-step web task is finished is the weakest skill (held-out completion recall 0.44 on
  WebLINX-derived rows; see the browser result above). Trajectories in the agent's own page format, including the
  completion step, are the next training data.
- A DONE choice is not proof of success: verify outcomes independently.
- The browser example's requests repeat each question's instructions and every page element; large pages reach
  6,000+ tokens. A more compact request format is planned for the next training run.

## License

Code: Apache-2.0 (`LICENSE`). `examples/browser/harness`: MIT, Copyright (c) 2026 Browser Use. Weights: CC BY-NC-SA
4.0, on top of Qwen3.5-2B (Apache-2.0). Upstream datasets keep their own licenses.
