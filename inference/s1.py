"""S1: one Qwen3.5 model (vision + language) + one LoRA, two tasks.

decide: Jev's three question types (choice, noul, score) all become "options": every option ends with the
        MARKER token, and the question ends with "Decision:", whose hidden state has read every option (the
        global query). CandidateHead scores each option endpoint against that query; a softmax gives calibrated
        probabilities. Nothing is generated, so option keys never need to be single tokens, and any serving stack
        finds the read-out positions from input_ids alone (every MARKER, then the last token).
text:   the model's own LM head writes a value: a TYPE_TEXT field ("Zurich") or a tool argument.

Untrusted text (page, options, goal) is tokenized with split_special_tokens=True, so a page containing the
literal marker string cannot forge a read-out position; only code inserts MARKER.
"""

import base64
import copy
import io
import json
import math
import os
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

# Language-model projections only; the vision encoder stays frozen.
LORA_TARGETS = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)"
)
OPTION_TOKEN_CAP = 96  # a single long option cannot starve the page state
STATE_TOKEN_FLOOR = 1024  # options shrink before the page state gets fewer tokens than this
# Training and serving share this ceiling: raising TAIJI_MAX_CRITERIA lets both sides see larger action
# spaces, so an experiment never trains and serves against different option counts.
MAX_CRITERIA = int(os.environ.get("TAIJI_MAX_CRITERIA", "255"))
MARKER = "<|quad_end|>"  # closes every option; the decision head reads its hidden state
HEADERS = {
    "choice": "Choose exactly one option.",
    "noul": "Decide whether the statement is true.",
    "score": "Rate the state on the ordered levels below (0 is the lowest).",
}
TEXT_TOKEN_CAP = 64
TEXT_PROMPT = ("Write the exact value for the field or tool argument described below. "
               "Reply with the value only.\nContext: ")


def load_image(value, image_tokens):
    """Path, data URI or bare base64 -> RGB image resized to about `image_tokens` vision tokens."""
    if isinstance(value, str) and (value.startswith("data:") or len(value) > 1024):
        image = Image.open(io.BytesIO(base64.b64decode(value.split(",", 1)[-1])))
    else:
        image = Image.open(value)
    image = image.convert("RGB")
    scale = math.sqrt(image_tokens * 32 * 32 / (image.width * image.height))  # 16px patches, 2x2 merge
    if scale < 1:
        image = image.resize((max(32, int(image.width * scale)), max(32, int(image.height * scale))))
    return image


def plain(tokenizer, text):
    """Token ids for untrusted text: special-token strings stay literal text."""
    return tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]


def as_options(question):
    """Jev question -> {option key: criterion}. noul: true/false; score: levels "0".."n-1"."""
    kind, criteria = question.get("type", "choice"), question.get("criteria")
    if kind == "choice" and isinstance(criteria, dict) and 1 <= len(criteria) <= MAX_CRITERIA:
        return criteria
    if kind == "noul" and isinstance(criteria, dict) and set(criteria) == {"true", "false"}:
        return {"true": criteria["true"], "false": criteria["false"]}
    if kind == "score" and isinstance(criteria, list) and 2 <= len(criteria) <= 10:
        return {str(i): level for i, level in enumerate(criteria)}
    raise ValueError(
        f"choice needs 1-{MAX_CRITERIA} criteria, noul needs criteria true/false, score needs 2-10 levels")


def encode_question(tokenizer, question, shuffle=False, max_tokens=None):
    """Question suffix: type header, instructions, then options each closed by MARKER.

    Returns ids, the MARKER offsets and the option keys in the order they appear. Score levels keep their order
    (the order is the meaning); choice and noul options are shuffled in training so position carries no signal.
    Each option keeps at most OPTION_TOKEN_CAP tokens; with max_tokens, that cap shrinks (not below 8) until the
    suffix fits, so a page with 250 targets still gets an answer instead of an error. Train and serve pass the same.
    """
    options = as_options(question)
    keys = list(options)
    if shuffle and question.get("type", "choice") != "score":
        random.shuffle(keys)
    marker = tokenizer.convert_tokens_to_ids(MARKER)
    newline = plain(tokenizer, "\n")
    instructions = as_text(question.get("instructions", ""))
    head = plain(tokenizer, f"\n{HEADERS[question.get('type', 'choice')]}\nInstructions: {instructions}\nOptions:\n")
    tail = plain(tokenizer, "\nSelect the single option best supported by the state and instructions.\nDecision:")
    bodies = [plain(tokenizer, f"[{key}] {as_text(options[key])}") for key in keys]
    cap = OPTION_TOKEN_CAP
    while max_tokens and cap > 8 and len(head) + len(tail) + sum(min(len(b), cap) + 1 + len(newline)
                                                                  for b in bodies) > max_tokens:
        cap = max(8, cap * 3 // 4)
    ids, positions = list(head), []
    for body in bodies:
        ids += body[:cap]
        positions.append(len(ids))
        ids += [marker] + newline
    ids += tail
    return torch.tensor(ids), torch.tensor(positions), keys  # the global query is always the last token


def as_text(value):
    """Text is used verbatim; objects and lists become JSON. Train and serve with the same form."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def fit_state(tokenizer, state, budget):
    """State tokens within budget: shorten page text, then drop elements from the end, so recent actions survive.
    A hard cut of the end remains only for states that are still too long."""
    ids = plain(tokenizer, as_text(state))
    page = state.get("page") if isinstance(state, dict) else None
    if len(ids) > budget and isinstance(page, dict) and isinstance(page.get("text"), str):
        text_ids = plain(tokenizer, page["text"])
        keep = max(0, len(text_ids) - (len(ids) - budget) - 32)  # 32: JSON escaping can re-tokenise differently
        state = {**state, "page": {**page, "text": tokenizer.decode(text_ids[:keep])}}
        ids = plain(tokenizer, as_text(state))
    elements = state.get("elements") if isinstance(state, dict) else None
    for _ in range(3):  # then drop elements from the end (target questions still list their own options)
        if len(ids) <= budget or not isinstance(elements, list) or not elements:
            break
        elements = elements[: int(len(elements) * budget / len(ids) * 0.97)]
        state = {**state, "elements": elements}
        ids = plain(tokenizer, as_text(state))
    return ids[:budget]


def encode_prefix(processor, state, max_tokens, image_tokens=400, screenshot=None):
    """Shared prefix: optional screenshot, then page state (any text or JSON). Same for every question."""
    tokenizer = processor.tokenizer
    if isinstance(state, dict) and "screenshot" in state:
        state = dict(state)
        screenshot = state.pop("screenshot")
    image_value = screenshot
    image = load_image(image_value, image_tokens) if image_value else None
    # Only this fixed text goes through the processor (it expands the image placeholder); page text never does.
    head = processor(text=[f"Screen: {processor.image_token}\nState: " if image else "State: "],
                     images=[image] if image else None, return_tensors="pt")
    budget = max_tokens - head["input_ids"].shape[1] - 8
    if budget < 64:
        raise ValueError("Too many or too long options for max_len")
    state_ids = torch.tensor(fit_state(tokenizer, state, budget), dtype=torch.long)
    input_ids = torch.cat([head["input_ids"][0], state_ids])
    token_types = torch.zeros_like(input_ids)
    if head.get("mm_token_type_ids") is not None:
        token_types[: head["mm_token_type_ids"].shape[1]] = head["mm_token_type_ids"][0]
    return {"input_ids": input_ids, "mm_token_type_ids": token_types,
            "pixel_values": head.get("pixel_values"), "image_grid_thw": head.get("image_grid_thw")}


def encode_agent_prefix(processor, state, max_tokens):
    """Split the agent's trained-format state into an immutable goal prefix and changing history tail.

    Concatenating `prefix.input_ids` and `tail` exactly reproduces `encode_prefix` for the same state.
    This lets a serving cache retain the goal computation across agent turns while the recent tool history
    and decision suffix are still evaluated for each new state. Other state shapes should use encode_prefix.
    """
    if (not isinstance(state, dict) or set(state) != {"goal", "recent_steps"}
            or not isinstance(state["goal"], str) or not isinstance(state["recent_steps"], list)):
        return None

    tokenizer = processor.tokenizer
    head = processor(text=["State: "], images=None, return_tensors="pt")
    budget = max_tokens - head["input_ids"].shape[1] - 8
    if budget < 64:
        raise ValueError("Too many or too long options for max_len")

    state_ids = fit_state(tokenizer, state, budget)
    stable_text = as_text({"goal": state["goal"], "recent_steps": []})[:-2]  # leave the list open
    stable_ids = plain(tokenizer, stable_text)
    common = 0
    while common < min(len(stable_ids), len(state_ids)) and stable_ids[common] == state_ids[common]:
        common += 1

    base_ids = torch.cat([head["input_ids"][0], torch.tensor(state_ids[:common], dtype=torch.long)])
    token_types = torch.zeros_like(base_ids)
    if head.get("mm_token_type_ids") is not None:
        token_types[: head["mm_token_type_ids"].shape[1]] = head["mm_token_type_ids"][0]
    prefix = {"input_ids": base_ids, "mm_token_type_ids": token_types,
              "pixel_values": None, "image_grid_thw": None}
    tail = torch.tensor(state_ids[common:], dtype=torch.long)
    return prefix, tail


def prepend_context(questions, context):
    """Append changing state tokens before each decision suffix and shift option readout positions."""
    if not len(context):
        return questions
    return [(torch.cat([context, suffix]), positions + len(context)) for suffix, positions in questions]


def encode(processor, row, max_len=8192, image_tokens=400, shuffle=False):
    """One decision row (training) -> shared-prefix inputs + question suffix, laid out exactly as serve.py runs them.

    Labels: "label" (one option key) or "target" ({option key: probability}, soft labels such as Open-Jev's).
    """
    suffix, positions, keys = encode_question(processor.tokenizer, row["question"], shuffle,
                                              max_tokens=max_len - STATE_TOKEN_FLOOR)
    prefix = encode_prefix(processor, row["state"], max_len - len(suffix), image_tokens, row.get("image"))
    encoded = {"prefix": prefix,
               "suffix": suffix, "positions": positions, "keys": keys}
    encoded["input_ids"] = torch.cat([encoded["prefix"]["input_ids"], suffix])  # for token counting
    if "target" in row:
        target = torch.tensor([float(row["target"].get(k, 0.0)) for k in keys])
        encoded["target"] = target / target.sum()
        encoded["label"] = int(target.argmax())
    elif "label" in row:
        encoded["label"] = keys.index(str(row["label"]))
    return encoded


def answer_from(question, keys, probs):
    """Probabilities over options -> Jev's typed answer for this question type."""
    kind = question.get("type", "choice")
    by_key = dict(zip(keys, probs))
    best = max(by_key, key=by_key.get)
    if kind == "noul":
        return {"type": "noul", "noul": by_key["true"]}
    if kind == "score":
        return {"type": "score", "score": sum(int(k) * p for k, p in by_key.items()), "probabilities": by_key,
                "legend": {str(i): level for i, level in enumerate(question["criteria"])},
                "confidence": confidence(by_key[best], len(keys))}
    return {"type": "choice", "choice": best, "probabilities": by_key,
            "confidence": confidence(by_key[best], len(keys))}


def confidence(top, k):
    """Normalized maximum probability, (K*max(p) - 1)/(K - 1): 0 for a uniform guess, 1 for certainty.
    Same published formula as the Decision-1.0 models; Jev's own statistic is unpublished."""
    return 1.0 if k < 2 else max(0.0, (k * top - 1) / (k - 1))


def encode_text(processor, context, target=None, max_len=8192):
    """Field context (the agent's field_context()) -> prompt ids, plus target ids when training."""
    tokenizer = processor.tokenizer
    target_ids = []
    if target is not None:
        target_ids = plain(tokenizer, target)[:TEXT_TOKEN_CAP]
        target_ids.append(tokenizer.eos_token_id)
    head = tokenizer(TEXT_PROMPT, add_special_tokens=False)["input_ids"]
    tail = tokenizer("\nText: ", add_special_tokens=False)["input_ids"]
    body = plain(tokenizer, as_text(context))
    body = body[: max_len - len(head) - len(tail) - len(target_ids)]
    prompt = head + body + tail
    return {"input_ids": torch.tensor(prompt + target_ids), "prompt_len": len(prompt)}


class CandidateHead(torch.nn.Module):
    """Scores each option endpoint against the global query (the design of Decision-1.0-Lux, Apache-2.0).

    Causal attention means an early option's endpoint never saw the later options; the final query token saw
    all of them, so every score is relative to the whole set. bilinear: how well option and question match;
    MLP: non-linear interactions. Runs in fp32: in bf16, near-ties flipped with numerical noise.
    """

    def __init__(self, hidden, dim=256):
        super().__init__()
        self.dim = dim
        self.candidate_norm, self.query_norm = torch.nn.LayerNorm(hidden), torch.nn.LayerNorm(hidden)
        self.key = torch.nn.Linear(hidden, dim, bias=False)
        self.query = torch.nn.Linear(hidden, dim, bias=False)
        self.candidate_mlp = torch.nn.Linear(hidden, dim)
        self.query_mlp = torch.nn.Linear(hidden, dim, bias=False)
        self.scalar = torch.nn.Linear(dim, 1, bias=False)
        torch.nn.init.normal_(self.scalar.weight, std=0.01)  # start near uniform

    def forward(self, candidates, query):
        """candidates [n, hidden], query [hidden] -> scores [n]."""
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            c, q = self.candidate_norm(candidates.float()), self.query_norm(query.float())
            bilinear = (self.key(c) * self.query(q)).sum(-1) / math.sqrt(self.dim)
            interaction = self.scalar(torch.nn.functional.gelu(self.candidate_mlp(c) + self.query_mlp(q))).squeeze(-1)
            return bilinear + interaction


class YesNoHead(torch.nn.Module):
    """Open-Jev's readout on S1's single-sequence layout: each option's score starts as the base model's own
    Yes-minus-No logit at that option's marker (init_from_lm), so an untrained head already reads the model's
    judgement; a CandidateHead residual behind a zero gate adds the comparison with the whole option set as training
    finds it useful. Runs in fp32 like CandidateHead."""

    def __init__(self, hidden):
        super().__init__()
        self.scalar = torch.nn.Linear(hidden, 1)  # `scalar` also names the device for callers, as in CandidateHead
        self.residual = CandidateHead(hidden)
        self.gate = torch.nn.Parameter(torch.zeros(()))

    @torch.no_grad()
    def init_from_lm(self, lm_head_weight, yes_id, no_id):
        self.scalar.weight.copy_((lm_head_weight[yes_id] - lm_head_weight[no_id]).float()[None])
        self.scalar.bias.zero_()

    def forward(self, candidates, query):
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            readout = self.scalar(candidates.float()).squeeze(-1)
        return readout + self.gate * self.residual(candidates, query)


HEADS = {"candidate": CandidateHead, "yesno": YesNoHead}


def yes_no_ids(tokenizer):
    """Token ids of "Yes" and "No" as the answer word right after a prompt (each must be one token)."""
    ids = [tokenizer.encode(word, add_special_tokens=False) for word in ("Yes", "No")]
    assert all(len(i) == 1 for i in ids), ids
    return ids[0][0], ids[1][0]


class S1(torch.nn.Module):
    def __init__(self, lm, head="candidate"):
        super().__init__()
        self.lm = lm  # Qwen3_5ForConditionalGeneration, possibly wrapped by PEFT
        self.head = HEADS[head](self.generator().config.text_config.hidden_size)
        self.temperature, self.temperatures = 1.0, {}  # global, and per question kind (calibrate.py)

    def generator(self):
        return self.lm.get_base_model() if hasattr(self.lm, "get_base_model") else self.lm

    def forward(self, encoded):
        """Option scores (logits) for a choice row; the LM loss for a text row (one entry point, so DDP syncs both).
        A list of rows runs as one padded batch (forward_batch) and returns one output per row."""
        if isinstance(encoded, list):
            return self.forward_batch(encoded)
        if "prompt_len" in encoded:
            return self.text_loss(encoded)
        prefix, device = encoded["prefix"], self.head.scalar.weight.device
        ids = torch.cat([prefix["input_ids"], encoded["suffix"]])[None].to(device)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if prefix["pixel_values"] is not None:
            token_types = torch.cat([prefix["mm_token_type_ids"], torch.zeros_like(encoded["suffix"])])
            inputs.update(mm_token_type_ids=token_types[None].to(device),
                          pixel_values=prefix["pixel_values"].to(device),
                          image_grid_thw=prefix["image_grid_thw"].to(device))
        hidden = self.generator().model(**inputs).last_hidden_state[0]
        positions = encoded["positions"].to(device) + len(prefix["input_ids"])
        return self.head(hidden[positions], hidden[-1])

    def forward_batch(self, rows):
        """Choice and text rows, right-padded into one forward pass; each row's output as forward() gives it alone.
        Attention is causal and the linear-attention recurrence and short convolution only run forward, so padding
        after a row cannot reach its positions: each row is read at its own markers and its own last token."""
        device = self.head.scalar.weight.device
        seqs, types, pixels, grids = [], [], [], []
        for row in rows:
            if "prompt_len" in row:
                seqs.append(row["input_ids"])
                types.append(torch.zeros_like(row["input_ids"]))
                continue
            prefix = row["prefix"]
            seqs.append(torch.cat([prefix["input_ids"], row["suffix"]]))
            types.append(torch.cat([prefix["mm_token_type_ids"], torch.zeros_like(row["suffix"])]))
            if prefix["pixel_values"] is not None:  # images line up with image tokens in batch order
                pixels.append(prefix["pixel_values"])
                grids.append(prefix["image_grid_thw"])
        width = -(-max(len(ids) for ids in seqs) // 256) * 256  # few distinct shapes: kernels compile once per shape
        ids = torch.zeros(len(seqs), width, dtype=torch.long)
        mask, type_ids = torch.zeros_like(ids), torch.zeros_like(ids)
        for i, (seq, kind) in enumerate(zip(seqs, types, strict=True)):
            ids[i, : len(seq)], mask[i, : len(seq)], type_ids[i, : len(seq)] = seq, 1, kind
        inputs = {"input_ids": ids.to(device), "attention_mask": mask.to(device)}
        if pixels:
            inputs.update(mm_token_type_ids=type_ids.to(device), pixel_values=torch.cat(pixels).to(device),
                          image_grid_thw=torch.cat(grids).to(device))
        hidden = self.generator().model(**inputs).last_hidden_state
        outputs = []
        for i, (row, seq) in enumerate(zip(rows, seqs, strict=True)):
            if "prompt_len" in row:  # LM loss on the target tokens only
                start = row["prompt_len"]
                logits = self.generator().lm_head(hidden[i, start - 1 : len(seq) - 1]).float()
                outputs.append(torch.nn.functional.cross_entropy(logits, seq[start:].to(device)))
            else:
                positions = row["positions"].to(device) + len(row["prefix"]["input_ids"])
                outputs.append(self.head(hidden[i, positions], hidden[i, len(seq) - 1]))
        return outputs

    @torch.inference_mode()
    def prefix_cache(self, prefix):
        """Serving: the page-state cache for one prefix, with the M-RoPE offset the suffix positions continue from."""
        device = self.head.scalar.weight.device
        self.generator().model.rope_deltas = None  # M-RoPE offset is cached on the model; never reuse another request's
        ids = prefix["input_ids"][None].to(device)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "use_cache": True}
        if prefix["pixel_values"] is not None:
            inputs.update(mm_token_type_ids=prefix["mm_token_type_ids"][None].to(device),
                          pixel_values=prefix["pixel_values"].to(device),
                          image_grid_thw=prefix["image_grid_thw"].to(device))
        cache = self.generator().model(**inputs).past_key_values
        return cache, self.generator().model.rope_deltas

    @torch.inference_mode()
    def score_questions(self, prefix, questions, cached=None):
        """Serving: the shared prefix once (or `cached`, its prefix_cache() from an identical earlier request), then
        every (suffix, positions) in one right-padded batch on a copy of that cache. Attention is causal and the
        linear-attention recurrence and short convolution only run forward, so padding after a question cannot reach
        it: each row is read at its own markers and its own last token, as it would be alone."""
        device = self.head.scalar.weight.device
        cache, rope_deltas = cached or self.prefix_cache(prefix)
        batch = copy.deepcopy(cache)  # `cached` may serve a later request, so it stays one unmodified row
        batch.reorder_cache(torch.zeros(len(questions), dtype=torch.long, device=device))  # one row per question
        ids = torch.zeros(len(questions), max(len(suffix) for suffix, _ in questions), dtype=torch.long)
        for i, (suffix, _) in enumerate(questions):
            ids[i, : len(suffix)] = suffix
        self.generator().model.rope_deltas = rope_deltas
        # No attention_mask: with one, transformers derives positions from the full length instead of the suffix.
        hidden = self.generator().model(input_ids=ids.to(device), past_key_values=batch, use_cache=True).last_hidden_state
        return [self.head(hidden[i, positions.to(device)], hidden[i, len(suffix) - 1])
                for i, (suffix, positions) in enumerate(questions)]

    def text_loss(self, encoded):
        """LM loss on the target tokens only; logits are computed just there (the vocab is 248K)."""
        device = self.head.scalar.weight.device
        ids = encoded["input_ids"][None].to(device)
        hidden = self.generator().model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0]
        start = encoded["prompt_len"]
        logits = self.generator().lm_head(hidden[start - 1 : -1]).float()
        return torch.nn.functional.cross_entropy(logits, ids[0, start:])

    @torch.inference_mode()
    def write(self, processor, context, max_len=8192):
        encoded = encode_text(processor, context, max_len=max_len)
        self.generator().model.rope_deltas = None
        ids = encoded["input_ids"][None].to(self.head.scalar.weight.device)
        out = self.generator().generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                        max_new_tokens=TEXT_TOKEN_CAP, eos_token_id=processor.tokenizer.eos_token_id)
        return processor.tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def resolve_device(device="auto"):
    """Select an explicit accelerator, preferring CUDA then Apple MPS for local inference."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if selected.type == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available; use an Apple Silicon Mac and an MPS-enabled PyTorch")
    return selected


def load(base, adapter=None, merge=True, device="auto", dtype=None):
    """Processor + model; with `adapter` (a LoRA or a --full run's folder), the trained head and fitted temperature."""
    device = resolve_device(device)
    if dtype is None:
        dtype = (torch.bfloat16 if device.type == "cuda" else
                 torch.float16 if device.type == "mps" else torch.float32)
    processor = AutoProcessor.from_pretrained(base)
    full = adapter and not (Path(adapter) / "adapter_config.json").exists()
    # flash-attn is CUDA-only and is never auto-selected: without this argument the model silently
    # runs the slower attention path even when the package is installed.
    # TAIJI_ATTN=sdpa forces the generic path, which is how the flash-attn contribution is measured.
    attention = os.environ.get("TAIJI_ATTN") or None
    if attention is None and device.type == "cuda":
        try:
            import flash_attn  # noqa: F401
            attention = "flash_attention_2"
        except ImportError:
            attention = None
    lm = Qwen3_5ForConditionalGeneration.from_pretrained(
        adapter if full else base, dtype=dtype,
        **({"attn_implementation": attention} if attention else {}))
    if adapter and not full:
        from peft import PeftModel

        lm = PeftModel.from_pretrained(lm, adapter)
        if merge:
            lm = lm.merge_and_unload()
    trained = Path(adapter or "", "train_args.json")
    head = json.loads(trained.read_text()).get("head", "candidate") if adapter and trained.exists() else "candidate"
    model = S1(lm, head).to(device)
    if adapter:
        model.head.load_state_dict(torch.load(Path(adapter) / "head.pt", map_location=device))
        calibration = Path(adapter) / "temperature.json"
        if calibration.exists():
            fitted = json.loads(calibration.read_text())
            model.temperature, model.temperatures = fitted["temperature"], fitted.get("by_kind", {})
    return processor, model


def kind(row):
    """What a row asks: text, operation, target, or the question type (choice/noul/score) of any other decision."""
    if row.get("task") == "text":
        return "text"
    question = row["question"]
    if "DONE" in question.get("criteria", {}):
        return "operation"
    instructions = question.get("instructions")
    if isinstance(instructions, dict) and "operation" in instructions:
        return "target"
    return question.get("type", "choice")


def read_rows(path):
    """Byte offsets of JSONL rows, so 270K rows with page text do not sit in memory."""
    offsets, position = [], 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                offsets.append(position)
            position += len(line)
    return offsets


def row_at(path, offset):
    with open(path, "rb") as f:
        f.seek(offset)
        return json.loads(f.readline())
