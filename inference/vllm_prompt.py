"""Taiji prompt compilation, producing the exact token ids the trained read-out expects.

vLLM is given the sequence as `prompt_token_ids` directly. That is the only way to guarantee
the option markers land on the same token ids and offsets as training: re-tokenizing a
serialized string can shift offsets (delimiter collisions, context-dependent tokenization).
The read-out metadata (marker offsets, option keys, question type) stays beside the ids in
the local compiled prompt object; vLLM receives the exact ids and Taiji applies its head to
those offsets after the token states return.

The layout is exactly the one `s1.py` built at training time:

    <prefix>  State: <state tokens>\n<head>\nInstructions: <instructions>\nOptions:\n
    [key] <option body> MARKER "\n" ... [key] <option body> MARKER "\n"
    "\nSelect the single option best supported by the state and instructions.\nDecision:"

The global query is the final token, and every option endpoint is a MARKER token; the head
scores the hidden state at each MARKER against the hidden state at that final token
(see `vllm_head.head_logits`). Option keys are returned in appearance order
(no shuffle: serving must not permute the caller's options).

`vllm_tests/test_vllm_prompt.py` asserts this compiles to the same ids and marker offsets as
`s1.encode_question`.
"""

import json
import os

MARKER = "<|quad_end|>"
HEADERS = {
    "choice": "Choose exactly one option.",
    "noul": "Decide whether the statement is true.",
    "score": "Rate the state on the ordered levels below (0 is the lowest).",
}
OPTION_TOKEN_CAP = 96
STATE_TOKEN_FLOOR = 1024  # the state prefix keeps at least this many tokens; options shrink instead
# A choice question carries at most this many options by default. The format contract is one byte per
# option, so the ceiling is where the trained distribution ends, not where compute does: TAIJI_MAX_CRITERIA
# raises it for experiments with larger action spaces.
MAX_CRITERIA = int(os.environ.get("TAIJI_MAX_CRITERIA", "255"))
QUESTION_TAIL = "\nSelect the single option best supported by the state and instructions.\nDecision:"


def as_text(value):
    """Text is used verbatim; objects and lists become JSON. Same form as `s1.as_text`."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def options_for(question):
    """A decision question -> {option key: criterion}; mirrors `s1.as_options` exactly."""
    kind, criteria = question.get("type", "choice"), question.get("criteria")
    if kind == "choice" and isinstance(criteria, dict) and 1 <= len(criteria) <= MAX_CRITERIA:
        return criteria
    if kind == "noul" and isinstance(criteria, dict) and set(criteria) == {"true", "false"}:
        return {"true": criteria["true"], "false": criteria["false"]}
    if kind == "score" and isinstance(criteria, list) and 2 <= len(criteria) <= 10:
        return {str(i): level for i, level in enumerate(criteria)}
    raise ValueError(
        f"choice needs 1-{MAX_CRITERIA} criteria, noul needs criteria true/false, score needs 2-10 levels"
    )


def compile_question(tokenizer, question, max_tokens=None):
    """Question suffix -> (ids, marker_offsets, keys), matching `s1.encode_question`."""
    options = options_for(question)
    keys = list(options)
    marker = tokenizer.convert_tokens_to_ids(MARKER)
    newline = plain(tokenizer, "\n")
    instructions = as_text(question.get("instructions", ""))
    head = plain(
        tokenizer,
        f"\n{HEADERS[question.get('type', 'choice')]}\nInstructions: {instructions}\nOptions:\n",
    )
    tail = plain(tokenizer, QUESTION_TAIL)
    bodies = [plain(tokenizer, f"[{key}] {as_text(options[key])}") for key in keys]
    cap = OPTION_TOKEN_CAP
    while max_tokens and cap > 8 and (
        len(head) + len(tail) + sum(min(len(b), cap) + 1 + len(newline) for b in bodies) > max_tokens
    ):
        cap = max(8, cap * 3 // 4)
    ids, positions = list(head), []
    for body in bodies:
        ids += body[:cap]
        positions.append(len(ids))
        ids += [marker] + newline
    ids += tail
    return ids, positions, keys


def tokenizer_of(processor):
    """A tokenizer or an AutoProcessor -> its tokenizer."""
    return getattr(processor, "tokenizer", processor)


def prefix_head_ids(processor):
    """The `"State: "` prefix ids, produced by the processor exactly as `s1.encode_prefix` does.

    With a real AutoProcessor this calls the processor (so an image placeholder would be
    expanded the same way); with a bare tokenizer it falls back to `plain`, which is
    text-identical. Both yield `[1349, 25, 220]` for Qwen3.5.
    """
    if hasattr(processor, "tokenizer") or hasattr(processor, "image_token"):
        head = processor(text=["State: "], images=None, return_tensors="pt")["input_ids"]
        return [int(token) for token in head[0]]
    return plain(processor, "State: ")


def fit_state(tokenizer, state, budget, strict=False):
    """State tokens within budget; identical to `s1.fit_state` (shorten page, then drop elements)."""
    ids = plain(tokenizer, as_text(state))
    if strict:
        if len(ids) > budget:
            raise ValueError(f"S1 token budget exceeded: state requires {len(ids)} tokens; available {budget}. No state was truncated.")
        return ids
    page = state.get("page") if isinstance(state, dict) else None
    if len(ids) > budget and isinstance(page, dict) and isinstance(page.get("text"), str):
        text_ids = plain(tokenizer, page["text"])
        keep = max(0, len(text_ids) - (len(ids) - budget) - 32)  # 32: JSON escaping can re-tokenise
        state = {**state, "page": {**page, "text": tokenizer.decode(text_ids[:keep])}}
        ids = plain(tokenizer, as_text(state))
    elements = state.get("elements") if isinstance(state, dict) else None
    for _ in range(3):
        if len(ids) <= budget or not isinstance(elements, list) or not elements:
            break
        elements = elements[: int(len(elements) * budget / len(ids) * 0.97)]
        state = {**state, "elements": elements}
        ids = plain(tokenizer, as_text(state))
    return ids[:budget]


def request_token_ids(processor, ids, image):
    """Compiled ids -> the prompt_token_ids vLLM wants for this request.

    vLLM expands the image itself: it replaces one placeholder token with the whole span the
    processor computed. Sending the already-expanded span makes it expand a second time and
    the prompt overflows max_model_len, so the span is collapsed back to a single placeholder
    here. The expanded ids stay the coordinate system for the read-out offsets.
    """
    if image is None:
        return list(ids)
    tokenizer = tokenizer_of(processor)
    placeholder = tokenizer.convert_tokens_to_ids(processor.image_token)
    if placeholder is None:
        raise ValueError("the processor has no image placeholder token")
    span = [index for index, token in enumerate(ids) if token == placeholder]
    if not span:
        raise ValueError("the compiled prompt has no image placeholder to send")
    if span != list(range(span[0], span[0] + len(span))):
        raise ValueError("the image placeholder span is not contiguous")
    return [*ids[: span[0]], placeholder, *ids[span[-1] + 1 :]]


def load_screenshot(value, image_tokens):
    """Screenshot (path, data URI or bare base64) -> the resized image the processor expects.

    Kept as its own function so the image branch is testable without PIL or the real processor.
    """
    from s1 import load_image

    return load_image(value, image_tokens)


def compile_prefix(processor, state, max_tokens, screenshot=None, image_tokens=400, strict_state=False):
    """Shared state prefix, matching `s1.encode_prefix` for a text or image state.

    `max_tokens` is the whole prefix budget (head + state), exactly as `s1.encode_prefix`
    takes it: the state gets `max_tokens - len(head) - 8`. Returns `(ids, image)`, where
    `image` is the resized screenshot to hand to vLLM as multi-modal data (or None).
    """
    tokenizer = tokenizer_of(processor)
    if isinstance(state, dict) and "screenshot" in state:
        state, screenshot = {k: v for k, v in state.items() if k != "screenshot"}, state["screenshot"]
    image = load_screenshot(screenshot, image_tokens) if screenshot else None
    if image is None:
        head = prefix_head_ids(processor)
    else:
        # Same fixed text as `s1.encode_prefix`: only this goes through the processor, so the
        # image placeholder expands exactly as it did in training. Page text never does.
        ids = processor(text=[f"Screen: {processor.image_token}\nState: "], images=[image],
                        return_tensors="pt")["input_ids"]
        head = [int(token) for token in ids[0]]
    budget = max_tokens - len(head) - 8
    if budget < 64:
        raise ValueError("Too many or too long options for max_len")
    return head + fit_state(tokenizer, state, budget, strict=strict_state), image


def compile_prompts(processor, questions, state="", max_len=8192, image_tokens=400, strict_state=False):
    """A batch of questions sharing one state -> one compiled prompt each, mirroring serve.py.

    The suffix budget and the shared prefix budget follow `s1.encode` and `serve.answer`:
    each suffix is capped at `max_len - STATE_TOKEN_FLOOR`, and the prefix is built once for
    the longest suffix so every question shares the identical state encoding. A screenshot
    rides along on every compiled prompt so each request carries its own multi-modal data.
    """
    tokenizer = tokenizer_of(processor)
    suffix_budget = max_len - STATE_TOKEN_FLOOR
    compiled = [compile_question(tokenizer, question, suffix_budget) for question in questions]
    longest = max((len(ids) for ids, _positions, _keys in compiled), default=0)
    prefix, image = compile_prefix(processor, state, max_len - longest,
                                   image_tokens=image_tokens, strict_state=strict_state)
    shift = len(prefix)
    return [
        {**finish_prompt(tokenizer, prefix + ids, [shift + p for p in positions], keys,
                         question.get("type", "choice")), "image": image,
         "prompt_token_ids": request_token_ids(processor, prefix + ids, image)}
        for question, (ids, positions, keys) in zip(questions, compiled, strict=True)
    ]


def compile_prompt(processor, question, state="", max_len=8192):
    """One full decision sequence: the exact id list plus the read-out metadata to score it."""
    return compile_prompts(processor, [question], state, max_len)[0]


def finish_prompt(tokenizer, ids, positions, keys, question_type):
    """Verify the marker offsets and the final `Decision` token, then build the read-out dict."""
    marker_id = tokenizer.convert_tokens_to_ids(MARKER)
    if any(ids[o] != marker_id for o in positions):
        raise ValueError(
            "marker offset does not point at MARKER; a literal marker string in untrusted text "
            "was tokenised as a special token"
        )
    # The last token is the query. On a char-level tokenizer it is always ":"; enforce it so
    # the read-out can never silently index the wrong row.
    if tokenizer.decode([ids[-1]]).strip() != ":":
        raise ValueError(f"final Decision token is not ':': {tokenizer.decode([ids[-1]])!r}")
    return {
        "input_ids": ids,
        "positions": positions,
        "keys": keys,
        "question_type": question_type,
        "marker_token_id": marker_id,
        "num_tokens": len(ids),
    }


def plain(tokenizer, text):
    """Untrusted text (page state, options, instructions): special-token strings stay literal."""
    return tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]


__all__ = [
    "HEADERS",
    "MARKER",
    "OPTION_TOKEN_CAP",
    "QUESTION_TAIL",
    "STATE_TOKEN_FLOOR",
    "as_text",
    "compile_prefix",
    "compile_prompt",
    "compile_prompts",
    "compile_question",
    "finish_prompt",
    "fit_state",
    "options_for",
    "plain",
    "prefix_head_ids",
    "tokenizer_of",
]
