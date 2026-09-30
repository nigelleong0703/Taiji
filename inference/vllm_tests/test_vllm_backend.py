"""Answer shaping and pooling-output plumbing, CPU-only."""

from types import SimpleNamespace

import pytest
import torch
from vllm_backend import (
    answer_from,
    answers_from_scores,
    confidence,
    score_token_states,
    temperature_for,
)

CHOICE = {"type": "choice", "criteria": {"1": "a", "2": "b", "3": "c"}, "instructions": {"goal": "g"}}


def test_answer_from_matches_s1():
    s1 = pytest.importorskip("s1")
    probs = [0.1, 0.7, 0.2]
    assert answer_from(CHOICE, ["1", "2", "3"], probs) == s1.answer_from(CHOICE, ["1", "2", "3"], probs)
    noul = {"type": "noul", "criteria": {"true": "y", "false": "n"}, "instructions": ""}
    assert answer_from(noul, ["true", "false"], [0.3, 0.7]) == s1.answer_from(noul, ["true", "false"], [0.3, 0.7])
    score = {"type": "score", "criteria": ["lo", "mid", "hi"], "instructions": ""}
    assert answer_from(score, ["0", "1", "2"], [0.2, 0.5, 0.3]) == s1.answer_from(score, ["0", "1", "2"], [0.2, 0.5, 0.3])


def test_confidence_matches_s1():
    s1 = pytest.importorskip("s1")
    for top, k in [(0.5, 3), (1.0, 1), (0.34, 4), (0.9, 2)]:
        assert confidence(top, k) == s1.confidence(top, k)


def test_answers_from_scores_shapes_all_types():
    questions = {
        "operation": CHOICE,
        "is_true": {"type": "noul", "criteria": {"true": "y", "false": "n"}, "instructions": ""},
        "quality": {"type": "score", "criteria": ["lo", "hi"], "instructions": ""},
    }
    keys_and_logits = {
        "operation": (["1", "2", "3"], [0.0, 3.0, 1.0]),
        "is_true": (["true", "false"], [2.0, 0.0]),
        "quality": (["0", "1"], [0.0, 2.0]),
    }
    answers = answers_from_scores(questions, keys_and_logits, temperature=1.0)
    assert answers["operation"]["type"] == "choice" and answers["operation"]["choice"] == "2"
    assert answers["is_true"]["type"] == "noul" and answers["is_true"]["noul"] > 0.8
    # score is the probability-weighted level: softmax([0, 2]) = [0.119, 0.881] -> 0.881
    assert answers["quality"]["type"] == "score" and answers["quality"]["score"] == pytest.approx(0.8808, abs=1e-3)


def test_temperature_by_kind_overrides_global():
    questions = {"q": CHOICE}
    scores = {"q": (["1", "2", "3"], [0.0, 0.0, 0.0])}
    base = answers_from_scores(questions, scores, temperature=1.0)["q"]["probabilities"]
    assert base == {"1": pytest.approx(1 / 3), "2": pytest.approx(1 / 3), "3": pytest.approx(1 / 3)}
    assert temperature_for("choice", 1.0, {"choice": 0.5}) == 0.5
    assert temperature_for("noul", 1.0, {"choice": 0.5}) == 1.0


def test_trained_head_scores_vllm_token_states_at_compiled_offsets():
    """The vLLM token_embed output is scored at all candidates and the Decision query."""
    from vllm_head import YesNoHead
    from vllm_prompt import compile_prompt

    from vllm_tests.conftest import FakeTokenizer

    tk = FakeTokenizer()
    question = {"type": "choice", "criteria": {"1": "hello", "2": "world"}, "instructions": "g"}
    compiled = compile_prompt(tk, question, "state text")
    head = YesNoHead(12)
    torch.manual_seed(0)
    with torch.no_grad():
        for p in head.parameters():
            p.copy_(torch.randn(p.shape) * 0.05)
    head.eval()
    hidden_states = torch.randn(compiled["num_tokens"], 12)
    query = compiled["num_tokens"] - 1  # the final Decision token
    logits = score_token_states(head, hidden_states, compiled, hidden_size=12)
    expected = head(hidden_states[compiled["positions"]], hidden_states[query])
    assert len(logits) == 2
    assert logits == pytest.approx(expected.tolist())


def test_score_token_states_rejects_incomplete_vllm_output():
    with pytest.raises(RuntimeError, match="incomplete token states"):
        score_token_states(
            torch.nn.Identity(), torch.zeros(3, 4), {"num_tokens": 4}, hidden_size=4
        )


def test_backend_compiles_dict_state_like_s1(fake_processor):
    """TaijiVLLM must pass the raw state so compile_prefix fits it like s1.encode_prefix."""
    import vllm_backend

    s1 = pytest.importorskip("s1")
    question = {"type": "choice", "criteria": {"1": "a", "2": "b"}, "instructions": {"goal": "g"}}
    state = {"page": {"url": "u", "title": "t", "text": "word " * 3000},
             "elements": [{"id": f"e{i}", "label": f"L{i}"} for i in range(60)]}
    backend = vllm_backend.TaijiVLLM.__new__(vllm_backend.TaijiVLLM)  # skip __init__ (needs transformers)
    backend.processor, backend.max_len = fake_processor, 8192
    compiled = backend.compile([question], state)[0]
    expected = s1.encode(fake_processor, {"state": state, "question": question}, max_len=8192,
                         image_tokens=400, shuffle=False)
    assert compiled["input_ids"] == expected["input_ids"].tolist()
    assert compiled["num_tokens"] <= 8192


def test_backend_scores_batched_token_embed_outputs(fake_processor):
    from vllm_backend import TaijiVLLM
    from vllm_head import YesNoHead, score_at

    question = {"type": "choice", "criteria": {"a": "A", "b": "B"}, "instructions": "pick"}
    backend = TaijiVLLM.__new__(TaijiVLLM)
    backend.processor, backend.max_len = fake_processor, 8192
    backend.head = YesNoHead(12).eval()
    backend.head_meta = {"hidden_size": 12}
    backend._pooling_params = object()
    backend._llm = None
    item = backend.compile([question], "state")[0]
    hidden = torch.randn(item["num_tokens"], 12)

    class FakeLLM:
        def encode(self, prompts, pooling_params):
            assert pooling_params is backend._pooling_params
            assert prompts[0]["prompt_token_ids"] == item["input_ids"]
            return [SimpleNamespace(
                prompt_token_ids=item["input_ids"],
                outputs=SimpleNamespace(data=hidden),
            )]

    backend._llm = FakeLLM()
    keys, logits, token_count = backend.score([question], "state")[0]
    expected = score_at(backend.head, hidden, item["positions"], item["num_tokens"] - 1)
    assert keys == item["keys"]
    assert token_count == item["num_tokens"]
    assert logits == pytest.approx(expected.tolist())
