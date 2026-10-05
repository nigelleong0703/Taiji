"""Taiji decisions through a single shared vLLM generate engine.

Each decision request carries a plain-JSON readout spec on SamplingParams extra_args
(taiji_readout: positions, query_index, num_tokens, keys); the installed vLLM 0.29
source patch (inference/vllm_patch/apply.py) applies the model folder's trained head at
those rows during the generate prefill and returns the logits on
RequestOutput.taiji_scores with num_cached_tokens. The same engine generates field text,
so one copy of the weights and one prefix cache serves both /v1/systemone and
/v1/chat/completions. Prompts are passed as prompt_token_ids, preserving the compiled
read-out positions.

The pure helpers (answer_from, answers_from_scores, score_token_states) are CPU-testable;
only TaijiVLLM touches vLLM, which is imported lazily so this module and its tests run
without CUDA. Image questions need the multimodal path and are not handled here yet.
"""

import json
from pathlib import Path

import torch
from s1 import TEXT_TOKEN_CAP
from vllm_head import score_at, softmax_with_temperature
from vllm_prompt import compile_prompts

DEFAULT_TEMPERATURE = 1.0
TAIJI_READOUT_KEY = "taiji_readout"


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
    """Apply the trained head to a complete per-token hidden-state matrix (CPU tests).

    The GPU path no longer pulls hidden states across the API: the worker applies the
    same head at the same rows and returns logits on RequestOutput.taiji_scores.
    This helper keeps the pure head/offset plumbing CPU-testable against compiled prompts.
    """
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


def readout_spec(item):
    """Compiled prompt -> plain-JSON readout dict for SamplingParams extra_args.

    positions are absolute prompt-token indices in option order; query_index is the
    final Decision token; num_tokens is the prompt length; keys ride along so the
    caller can zip logits back onto its options.
    """
    prompt_len = int(item["num_tokens"])
    return {
        "positions": [int(p) for p in item["positions"]],
        "query_index": prompt_len - 1,
        "num_tokens": prompt_len,
        "keys": list(item["keys"]),
    }


def score_sampling_params(item):
    """One decision request's SamplingParams: readout spec, one token, no detokenize."""
    from vllm import SamplingParams

    return SamplingParams(
        extra_args={TAIJI_READOUT_KEY: readout_spec(item)},
        max_tokens=1,
        temperature=0.0,
        detokenize=False,
    )


class TaijiVLLM:
    """One shared vLLM generate engine serving both decisions and field text."""

    image_tokens = 400  # class default so partially built instances still compile a state

    def __init__(self, model, head_bundle=None, temperature=None, by_kind=None,
                 max_len=8192, image_tokens=400, engine_kwargs=None):
        self.model = model
        self.max_len = max_len
        self.image_tokens = image_tokens
        self.engine_kwargs = dict(engine_kwargs or {})

        from transformers import AutoProcessor

        # The tokenizer alone cannot expand an image placeholder the way s1.encode_prefix does;
        # keep the processor so the "State: " prefix is reproduced exactly.
        self.processor = AutoProcessor.from_pretrained(model)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

        # The worker loads the head weights once from the model folder's
        # taiji_config.json/taiji_head.pt at engine start. A separate head bundle is
        # not supported: the worker would silently score with the wrong head.
        bundle_raw = head_bundle or self.engine_kwargs.get("head_bundle")
        if bundle_raw is None:
            same_bundle = True
        else:
            candidate = Path(bundle_raw)
            try:
                same_bundle = (candidate.resolve() == Path(model).resolve()
                               or candidate.parent.resolve() == Path(model).resolve())
            except OSError:
                same_bundle = False
        if not same_bundle:
            raise ValueError(
                "separate head bundles are not supported: the worker loads the head "
                "from the model folder's taiji_config.json/taiji_head.pt, so head_bundle "
                "must resolve to the model directory itself"
            )
        config_path = Path(model) / "taiji_config.json"
        meta = json.loads(config_path.read_text()) if config_path.exists() else {}
        self.temperature = temperature if temperature is not None else meta.get("temperature", DEFAULT_TEMPERATURE)
        self.by_kind = by_kind if by_kind is not None else meta.get("temperature_by_kind", {})
        self._llm = None
        self._last_cached_tokens = 0

    def load_engine(self):
        """Register Taiji's model and start the single shared generate engine once."""
        if self._llm is not None:
            return self._llm
        from vllm import LLM
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as v2_runner
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as v1_runner
        from vllm_model import register

        if any(getattr(runner, "TAIJI_READOUT_PATCH_VERSION", None) != 1
               for runner in (v2_runner, v1_runner)):
            raise RuntimeError(
                "Taiji vLLM readout patch v1 is not installed in this vLLM: "
                "TAIJI_READOUT_PATCH_VERSION != 1 on the V1 or V2 runner. Install it "
                "with python inference/vllm_patch/apply.py in the vLLM environment, "
                "then restart the engine."
            )
        register()
        # The readout patch only works on this exact engine shape: a generate runner,
        # synchronous scheduling and an explicit FlashAttention backend. A caller may
        # tune memory and batching, but must not quietly switch any of these on or off.
        required = {
            "runner": "generate",
            "enable_prefix_caching": True,
            "async_scheduling": False,
            "attention_config": {"backend": "FLASH_ATTN"},
        }
        kwargs = {
            "runner": "generate",
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "async_scheduling": False,
            "attention_config": {"backend": "FLASH_ATTN"},
            # A screenshot state is one image; the checkpoint carries its vision tower, so the
            # engine may run it. A caller can still pass image: 0 to keep a run text-only.
            "limit_mm_per_prompt": {"image": 1},
            "max_model_len": self.max_len,
        }
        overrides = {k: v for k, v in self.engine_kwargs.items() if k != "head_bundle"}
        conflicting = {k: v for k, v in overrides.items() if k in required and v != required[k]}
        if conflicting:
            raise ValueError(
                f"engine_kwargs cannot override the shared-engine flags: {conflicting}"
            )
        # Chunked prefill and other scheduling knobs (max_num_batched_tokens, dtype,
        # gpu_memory_utilization) pass through untouched; head_bundle never reaches vLLM,
        # because the worker reads the head from the model folder itself.
        kwargs.update(overrides)
        self._llm = LLM(model=self.model, **kwargs)
        return self._llm

    def write(self, context, max_len=8192):
        """Field context -> the exact text value, reproducing s1.write's prompt construction.

        Generation runs on the same shared engine as scoring. The configured processor
        encodes the training prompt (s1.encode_text); the caller passes only the context.
        """
        from s1 import encode_text
        from vllm import SamplingParams

        encoded = encode_text(self.processor, context, max_len=max_len)
        prompt_ids = [int(i) for i in encoded["input_ids"].tolist()]
        params = SamplingParams(
            temperature=0.0, max_tokens=TEXT_TOKEN_CAP,
            stop_token_ids=[self.tokenizer.eos_token_id])
        outputs = self.load_engine().generate(
            [{"prompt_token_ids": prompt_ids}], params, use_tqdm=False)
        return self.tokenizer.decode(
            outputs[0].outputs[0].token_ids, skip_special_tokens=True).strip()

    def compile(self, questions, state=""):
        """One or more questions sharing a state -> compiled prompts (see vllm_prompt)."""
        return compile_prompts(self.processor, questions, state, self.max_len,
                               image_tokens=self.image_tokens, strict_state=True)

    def score(self, questions, state=""):
        """A list of questions sharing one state -> [(keys, logits, num_tokens)] in order.

        The state is fitted and encoded once and shared by every question, exactly as
        serve.answer builds its prefix for one request. Each question becomes one
        generate request carrying its plain-JSON readout spec; the patched worker
        returns the head logits on RequestOutput.taiji_scores with num_cached_tokens.
        """
        compiled = self.compile(questions, state)
        prompts = []
        for item in compiled:
            request = {"prompt_token_ids": item.get("prompt_token_ids", item["input_ids"])}
            if item.get("image") is not None:
                # The same resized screenshot the prompt was compiled against: vLLM runs the
                # vision tower over it and puts the image embeddings into the placeholder rows,
                # which is why the read-out positions still index this request's own tokens.
                request["multi_modal_data"] = {"image": item["image"]}
            prompts.append(request)
        params = [score_sampling_params(c) for c in compiled]
        outputs = self.load_engine().generate(prompts, params, use_tqdm=False)
        if len(outputs) != len(compiled):
            raise RuntimeError("vLLM returned a different number of outputs than prompts")
        results = []
        cached = 0
        for item, output in zip(compiled, outputs, strict=True):
            scores = getattr(output, "taiji_scores", None)
            if scores is None:
                raise RuntimeError(
                    "vLLM output is missing taiji_scores; is readout patch v1 installed? "
                    "Run python inference/vllm_patch/apply.py in the vLLM environment."
                )
            logits = [float(v) for v in scores]
            if len(logits) != len(item["keys"]):
                raise RuntimeError(
                    f"vLLM returned {len(logits)} readout scores "
                    f"for {len(item['keys'])} options"
                )
            cached += int(getattr(output, "num_cached_tokens", 0) or 0)
            results.append((item["keys"], logits, item["num_tokens"]))
        self._last_cached_tokens = cached
        return results

    def decide(self, state, questions):
        """State + {name: question} -> {name: typed answer}; the /v1/systemone payload's core."""
        # Pass the raw state so compile_prefix can fit it like s1.encode_prefix (page text, elements).
        self._last_cached_tokens = 0
        scored = self.score(list(questions.values()), state)
        keys_and_logits = {name: (keys, logits)
                           for name, (keys, logits, _tokens) in zip(questions, scored, strict=True)}
        answers = answers_from_scores(questions, keys_and_logits, self.temperature, self.by_kind)
        tokens = sum(tokens for _keys, _logits, tokens in scored)
        usage = {"input_tokens": tokens, "output_tokens": 0}
        cached = int(getattr(self, "_last_cached_tokens", 0) or 0)
        if cached:
            usage["cached_tokens"] = cached
        return {"model": "taiji-vllm", "answers": answers, "usage": usage}


__all__ = [
    "DEFAULT_TEMPERATURE",
    "TAIJI_READOUT_KEY",
    "TaijiVLLM",
    "answer_from",
    "answers_from_scores",
    "confidence",
    "readout_spec",
    "score_sampling_params",
    "score_token_states",
    "temperature_for",
]
