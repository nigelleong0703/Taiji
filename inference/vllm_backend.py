"""Taiji decisions through the vLLM token-embedding runner.

vLLM returns native Qwen3.5 token states; this module applies Taiji's exact trained head at
the compiled option markers and final query token, then turns those logits into the same
calibrated `choice` / `noul` / `score` answers as the S1 Transformers server. Prompts are
passed as `prompt_token_ids`, preserving the compiled read-out positions.

The pure helpers (`answer_from`, `answers_from_scores`) are CPU-testable;
only `TaijiVLLM` touches vLLM, which is imported lazily so this module and its tests run
without CUDA. Image questions need the multimodal path and are not handled here yet.
"""

import json
from pathlib import Path

import torch
from vllm_head import read_head_bundle, score_at, softmax_with_temperature
from vllm_prompt import compile_prompts

DEFAULT_TEMPERATURE = 1.0


def confidence(top, k):
    """Normalized maximum probability, (K*max(p) - 1)/(K - 1). Same formula as `s1.confidence`."""
    return 1.0 if k < 2 else max(0.0, (k * top - 1) / (k - 1))


def answer_from(question, keys, probs):
    """Probabilities over options -> Jev's typed answer. Mirrors `s1.answer_from` exactly."""
    kind = question.get("type", "choice")
    by_key = dict(zip(keys, probs))
    best = max(by_key, key=by_key.get)
    if kind == "noul":
        return {"type": "noul", "noul": by_key["true"]}
    if kind == "score":
        return {"type": "score", "score": sum(int(k) * p for k, p in by_key.items()),
                "probabilities": by_key,
                "legend": {str(i): level for i, level in enumerate(question["criteria"])},
                "confidence": confidence(by_key[best], len(keys))}
    return {"type": "choice", "choice": best, "probabilities": by_key,
            "confidence": confidence(by_key[best], len(keys))}


def temperature_for(kind, temperature=DEFAULT_TEMPERATURE, by_kind=None):
    """Per-question-kind fitted temperature, falling back to the global one."""
    return (by_kind or {}).get(kind, temperature)


def answers_from_scores(questions, keys_and_logits, temperature=DEFAULT_TEMPERATURE, by_kind=None):
    """Pure: {name: question} + {name: (keys, logits)} -> {name: typed answer}.

    This is the exact step the server runs after vLLM returns logits, and is what
    `vllm_tests/test_vllm_backend.py` exercises without a GPU.
    """
    answers = {}
    for name, question in questions.items():
        keys, logits = keys_and_logits[name]
        probs = softmax_with_temperature(torch.as_tensor(logits, dtype=torch.float32),
                                         temperature_for(question.get("type", "choice"),
                                                         temperature, by_kind)).tolist()
        answers[name] = answer_from(question, keys, probs)
    return answers


def score_token_states(head, hidden_states, compiled, hidden_size):
    """Apply Taiji's trained head to vLLM's complete per-token hidden-state output."""
    hidden = torch.as_tensor(hidden_states)
    expected = (int(compiled["num_tokens"]), int(hidden_size))
    if hidden.ndim != 2 or tuple(hidden.shape) != expected:
        raise RuntimeError(
            f"vLLM returned incomplete token states: expected {expected}, got {tuple(hidden.shape)}"
        )
    with torch.inference_mode():
        logits = score_at(
            head,
            hidden,
            compiled["positions"],
            int(compiled["num_tokens"]) - 1,
        ).float().cpu().tolist()
    return logits


class TaijiVLLM:
    """Thin vLLM engine wrapper that answers Taiji-style decision requests."""

    def __init__(self, model, head_bundle=None, temperature=None, by_kind=None,
                 max_len=8192, engine_kwargs=None):
        self.model = model
        self.max_len = max_len
        self.engine_kwargs = dict(engine_kwargs or {})
        if head_bundle:
            self.engine_kwargs.setdefault("head_bundle", head_bundle)

        from transformers import AutoProcessor

        # The tokenizer alone cannot expand an image placeholder the way s1.encode_prefix does;
        # keep the processor so the \`State: \` prefix is reproduced exactly.
        self.processor = AutoProcessor.from_pretrained(model)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        bundle = head_bundle or self.engine_kwargs.get("head_bundle")
        if bundle is None:
            bundle = model
        self.head, self.head_meta = read_head_bundle(bundle)
        self.head.to(torch.float32).eval()
        config_path = Path(bundle) / "taiji_config.json" if bundle else None
        meta = json.loads(config_path.read_text()) if config_path and config_path.exists() else {}
        self.temperature = temperature if temperature is not None else meta.get("temperature", DEFAULT_TEMPERATURE)
        self.by_kind = by_kind if by_kind is not None else meta.get("temperature_by_kind", {})
        self._llm = None

    def load_engine(self):
        """Register Taiji's model and start the vLLM pooling engine once."""
        if self._llm is not None:
            return self._llm
        from vllm import LLM
        from vllm.pooling_params import PoolingParams
        from vllm_model import register

        register()
        self._pooling_params = PoolingParams(task="token_embed", use_activation=False)
        kwargs = {
            "runner": "pooling",
            "task": "token_embed",
            "trust_remote_code": True,
            "enable_prefix_caching": True,
        }
        kwargs.update({k: v for k, v in self.engine_kwargs.items() if k != "head_bundle"})
        self._llm = LLM(model=self.model, **kwargs)
        return self._llm

    def compile(self, questions, state=""):
        """One or more questions sharing a state -> compiled prompts (see vllm_prompt)."""
        return compile_prompts(self.processor, questions, state, self.max_len)

    def score(self, questions, state=""):
        """A list of questions sharing one state -> [(keys, logits, num_tokens)] in order.

        The state is fitted and encoded once and shared by every question, exactly as
        `serve.answer` builds its prefix for one request.
        """
        compiled = self.compile(questions, state)
        prompts = [{"prompt_token_ids": c["input_ids"]} for c in compiled]
        outputs = self.load_engine().encode(prompts, pooling_params=self._pooling_params)
        if len(outputs) != len(compiled):
            raise RuntimeError("vLLM returned a different number of outputs than prompts")
        results = []
        for item, output in zip(compiled, outputs, strict=True):
            if list(output.prompt_token_ids) != item["input_ids"]:
                raise RuntimeError("vLLM returned token ids that differ from the compiled prompt")
            logits = score_token_states(
                self.head, output.outputs.data, item, self.head_meta["hidden_size"]
            )
            results.append((item["keys"], logits, item["num_tokens"]))
        return results

    def decide(self, state, questions):
        """State + {name: question} -> {name: typed answer}; the /v1/systemone payload's core."""
        # Pass the raw state so compile_prefix can fit it like s1.encode_prefix (page text, elements).
        scored = self.score(list(questions.values()), state)
        keys_and_logits = {name: (keys, logits)
                           for name, (keys, logits, _tokens) in zip(questions, scored, strict=True)}
        answers = answers_from_scores(questions, keys_and_logits, self.temperature, self.by_kind)
        tokens = sum(tokens for _keys, _logits, tokens in scored)
        return {"model": "taiji-vllm", "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0}}


__all__ = [
    "DEFAULT_TEMPERATURE",
    "TaijiVLLM",
    "answer_from",
    "answers_from_scores",
    "confidence",
    "score_token_states",
    "temperature_for",
]
