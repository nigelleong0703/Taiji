"""Shared fixtures: put `inference/` on the path, and a tokenizer stand-in.

`s1.py` needs transformers + torch; the tests that import it use `pytest.importorskip`, so
the `vllm_head` / `vllm_prompt` / `vllm_backend` unit tests still run in a bare environment.
"""

import sys
from pathlib import Path

import pytest
import torch

INFERENCE = Path(__file__).resolve().parents[1]
if str(INFERENCE) not in sys.path:
    sys.path.insert(0, str(INFERENCE))

SPECIALS = {"<|quad_end|>": 2_000_000, "<|image_pad|>": 2_000_001}


class FakeTokenizer:
    """Char-level tokenizer with the small surface `s1.py` and `vllm_prompt.py` use.

    `split_special_tokens=True` keeps a literal "<|quad_end|>" as plain characters, exactly
    like the real tokenizer, which is the property the marker security depends on.
    """

    def __init__(self, specials=None):
        self.specials = dict(specials or SPECIALS)

    def convert_tokens_to_ids(self, token):
        return self.specials.get(token)

    def __call__(self, text, add_special_tokens=False, split_special_tokens=False, **kwargs):
        ids, i = [], 0
        while i < len(text):
            match = None
            if not split_special_tokens:
                for token, token_id in self.specials.items():
                    if text.startswith(token, i):
                        match = (token, token_id)
                        break
            if match:
                ids.append(match[1])
                i += len(match[0])
            else:
                ids.append(ord(text[i]))
                i += 1
        return {"input_ids": ids}

    def decode(self, ids):
        reverse = {v: k for k, v in self.specials.items()}
        return "".join(reverse.get(i, chr(i)) for i in ids)


class FakeProcessor:
    """Text-only AutoProcessor stand-in: `"State: "` expands to its ids, no image placeholder.

    Mirrors the real Qwen3.5 processor's text path closely enough that `s1.encode_prefix` runs
    against it, which is what the prefix-parity tests compare to.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or FakeTokenizer()
        self.image_token = "<|image_pad|>"

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        if images is not None:
            raise NotImplementedError("FakeProcessor is text-only")
        rows = [self.tokenizer(t, add_special_tokens=False)["input_ids"] for t in text]
        if return_tensors == "pt":
            width = max((len(r) for r in rows), default=0)
            padded = torch.tensor([r + [0] * (width - len(r)) for r in rows])
            return {"input_ids": padded, "mm_token_type_ids": torch.zeros_like(padded)}
        return {"input_ids": rows}


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def fake_processor():
    return FakeProcessor()


@pytest.fixture
def choice_question():
    return {"type": "choice",
            "criteria": {"1": "Open the page", "2": "Click flights", "3": "Wait"},
            "instructions": {"goal": "Find flights"}}
