"""Decides whether serve.py may use --shared-prefix: its scores must match the full sequence used in training.

python inference/check_cache.py <base> <adapter> <screenshot.png>
Runs an image request, then a text-only one (catches a stale M-RoPE offset), and prints both timings.
Compares calibrated probabilities and the top choice, the values serving acts on.
Measured locally: identical in float32; in bf16 with PyTorch fallback kernels logits differed by up to 0.5,
enough to flip near-ties, so the server defaults to full recompute. Rerun on the GPU box with the fast kernels.
"""

import sys
import time

import torch
from s1 import encode, encode_prefix, encode_question, load

base, adapter, image = sys.argv[1], sys.argv[2], sys.argv[3]
processor, model = load(base, adapter)
model.eval()
questions = {
    "operation": {"type": "choice", "criteria": {"CLICK": "Click", "TYPE_TEXT": "Type", "DONE": "Done"},
                  "instructions": {"goal": "Find flights"}},
    "click_target": {"type": "choice", "criteria": {str(i): {"element": f"[{i}] Button {i}"} for i in range(1, 30)},
                     "instructions": {"goal": "Find flights", "operation": "CLICK"}},
}


def timed(fn):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000


def check(state):
    suffixes = [encode_question(processor.tokenizer, q)[:2] for q in questions.values()]
    longest = max(len(suffix) for suffix, _ in suffixes)
    with torch.inference_mode():
        for _ in range(2):  # the second pass is the timed one
            full, t_full = timed(lambda: [model(encode(processor, {"state": state, "question": q}))
                                          for q in questions.values()])
            cached, t_cached = timed(lambda: model.score_questions(encode_prefix(processor, state, 8192 - longest),
                                                                   suffixes))
    # What serving acts on: the calibrated probabilities and the top choice, not raw logits.
    probs = [[torch.softmax(x.float() / model.temperature, -1) for x in pair] for pair in zip(full, cached)]
    worst = max((f - c).abs().max().item() for f, c in probs)
    same = all(f.argmax() == c.argmax() for f, c in probs)
    print(f"max probability diff {worst:.4f}, same top choice {same}; "
          f"two questions: full {t_full:.0f} ms, shared prefix {t_cached:.0f} ms")
    return worst if same else 1.0


# About the size of a real results page (Google Flights: ~6000 characters of text and ~150 controls).
page = {"page": {"url": "https://a.test/", "title": "Flights", "text": "Where from? Where to? Search " * 200},
        "elements": [{"id": f"e{i}", "role": "button", "label": f"Flight option {i}, 9641 Thai baht"} for i in range(150)]}
worst = max(check({**page, "screenshot": image}), check(page))
# bf16 kernels round differently when the suffix runs on a cached prefix. Accept it while the top choice is the same
# and no probability moves more than 2 points, far inside the agent's 0.25 tie margin.
print("OK: --shared-prefix is safe" if worst < 0.02 else "Keep full recompute (do not pass --shared-prefix)")
