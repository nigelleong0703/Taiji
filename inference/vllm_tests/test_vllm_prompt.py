"""The compiled prompt must reproduce the training-time layout in s1.py exactly."""

import pytest
from vllm_prompt import (
    MARKER,
    STATE_TOKEN_FLOOR,
    compile_prefix,
    compile_prompt,
    compile_prompts,
    compile_question,
    plain,
)


def _long_state():
    return {
        "page": {"url": "https://y/", "title": "Big", "text": "Where from? Where to? Search " * 200},
        "elements": [{"id": f"e{i}", "role": "button", "label": f"Option {i}"} for i in range(80)],
    }


def test_compile_question_matches_s1_encode_question(fake_tokenizer):
    s1 = pytest.importorskip("s1")
    questions = [
        {"type": "choice", "criteria": {"1": "a", "2": "b", "3": "c"}, "instructions": {"goal": "g"}},
        {"type": "noul", "criteria": {"true": "yes", "false": "no"}, "instructions": {}},
        {"type": "score", "criteria": ["low", "mid", "high"], "instructions": ""},
    ]
    for question in questions:
        expected_ids, expected_positions, expected_keys = s1.encode_question(fake_tokenizer, question)
        ids, positions, keys = compile_question(fake_tokenizer, question)
        assert ids == expected_ids.tolist()
        assert positions == expected_positions.tolist()
        assert keys == expected_keys


def test_compile_question_matches_with_option_cap(fake_tokenizer):
    s1 = pytest.importorskip("s1")
    questions = [
        {"type": "choice", "criteria": {"1": "x" * 400, "2": "y" * 400}, "instructions": "long"},
        {"type": "score", "criteria": ["a" * 300, "b" * 300, "c" * 300], "instructions": ""},
    ]
    for question in questions:
        got = compile_question(fake_tokenizer, question, max_tokens=120)
        expected = s1.encode_question(fake_tokenizer, question, max_tokens=120)
        assert got[0] == expected[0].tolist()
        assert got[1] == expected[1].tolist()
        assert got[2] == expected[2]


@pytest.mark.parametrize("max_tokens", [2000, 600, 256])
def test_compile_prefix_matches_s1_encode_prefix(fake_processor, max_tokens):
    """The critical parity: head ids + fit_state must equal s1.encode_prefix on a text state."""
    s1 = pytest.importorskip("s1")
    for state in ("raw state text", {"page": {"text": "short"}, "elements": []}, _long_state()):
        expected = s1.encode_prefix(fake_processor, state, max_tokens, 400, None)["input_ids"].tolist()
        assert compile_prefix(fake_processor, state, max_tokens) == expected


def test_compile_prompts_matches_s1_encode(fake_processor):
    """Full sequence: ids equal s1.encode, and positions are s1's suffix positions + prefix."""
    s1 = pytest.importorskip("s1")
    row = {"state": _long_state(),
           "question": {"type": "choice", "criteria": {"1": "a", "2": "b", "3": "c"},
                        "instructions": {"goal": "g"}}}
    encoded = s1.encode(fake_processor, row, max_len=8192, image_tokens=400, shuffle=False)
    compiled = compile_prompts(fake_processor, [row["question"]], row["state"], max_len=8192)[0]
    assert encoded["input_ids"].tolist() == compiled["input_ids"]
    prefix_len = len(compiled["input_ids"]) - len(encoded["suffix"])
    assert [p + prefix_len for p in encoded["positions"].tolist()] == compiled["positions"]
    assert encoded["keys"] == compiled["keys"]


def test_prefix_head_and_offsets(fake_tokenizer):
    question = {"type": "noul", "criteria": {"true": "yes", "false": "no"}, "instructions": "x"}
    no_state = compile_prompt(fake_tokenizer, question, "")
    with_state = compile_prompt(fake_tokenizer, question, "Hello world")
    # Both prompts carry the fixed "State: " head; adding a state shifts offsets by the state length only.
    assert with_state["positions"] == [p + len("Hello world") for p in no_state["positions"]]
    head = plain(fake_tokenizer, "State: ")
    assert with_state["input_ids"][: len(head)] == head
    assert fake_tokenizer.decode([with_state["input_ids"][-1]]).strip() == ":"


def test_long_state_is_fitted_to_max_len(fake_processor):
    """A state longer than the budget is fitted (matching s1), and the whole prompt fits max_len."""
    s1 = pytest.importorskip("s1")
    state = _long_state()
    question = {"type": "choice", "criteria": {"1": "a"}, "instructions": ""}
    longest_suffix = len(compile_question(fake_processor.tokenizer, question, 2048 - STATE_TOKEN_FLOOR)[0])
    prefix_budget = 2048 - longest_suffix
    assert compile_prefix(fake_processor, state, prefix_budget) == \
        s1.encode_prefix(fake_processor, state, prefix_budget, 400, None)["input_ids"].tolist()
    assert compile_prompt(fake_processor, question, state, max_len=2048)["num_tokens"] <= 2048


def test_marker_ids_are_the_only_ones(fake_processor):
    question = {"type": "choice", "criteria": {"1": "b", "2": "c"}, "instructions": "g"}
    compiled = compile_prompt(fake_processor, question, "some state")
    found = [i for i, t in enumerate(compiled["input_ids"]) if t == compiled["marker_token_id"]]
    assert found == compiled["positions"]
    assert len(found) == 2


def test_literal_marker_string_in_untrusted_text_is_not_special(fake_processor):
    question = {"type": "choice", "criteria": {"1": f"click {MARKER} here", "2": "b"}, "instructions": "g"}
    compiled = compile_prompt(fake_processor, question, f"page says {MARKER} too")
    assert len(compiled["positions"]) == 2  # only the two real markers


def test_screenshot_state_is_rejected_in_text_only_backend(fake_processor):
    with pytest.raises(NotImplementedError):
        compile_prefix(fake_processor, {"screenshot": "data:image/png;base64,AAAA"}, 1024)


def test_invalid_questions_raise(fake_tokenizer):
    with pytest.raises(ValueError):
        compile_question(fake_tokenizer, {"type": "noul", "criteria": {"yes": "a"}})
    with pytest.raises(ValueError):
        compile_question(fake_tokenizer, {"type": "score", "criteria": ["only one"]})
