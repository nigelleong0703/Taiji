import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from s1 import encode_agent_prefix, encode_prefix, encode_question, prepend_context


def test_agent_prefix_split_reconstructs_the_training_state(fake_processor):
    states = [
        {"goal": "Find flights", "recent_steps": []},
        {"goal": "Find flights", "recent_steps": [
            {"step": 1, "tool": "browser__open", "status": "ok", "result": "Opened page"},
            {"step": 2, "tool": "browser__observe", "status": "ok", "result": "Search form"},
        ]},
    ]
    for state in states:
        prefix, tail = encode_agent_prefix(fake_processor, state, 8192)
        full = encode_prefix(fake_processor, state, 8192)
        assert torch.equal(torch.cat([prefix["input_ids"], tail]), full["input_ids"])


def test_agent_prefix_respects_budget_and_shifts_option_positions(fake_processor, choice_question):
    state = {"goal": "Find flights", "recent_steps": [
        {"step": i, "tool": "browser__observe", "status": "ok", "result": "Search form " * 8}
        for i in range(4)
    ]}
    prefix, tail = encode_agent_prefix(fake_processor, state, 2048)
    full = encode_prefix(fake_processor, state, 2048)
    assert len(prefix["input_ids"]) + len(tail) == len(full["input_ids"])

    suffix, positions, _ = encode_question(fake_processor.tokenizer, choice_question)
    shifted = prepend_context([(suffix, positions)], tail)[0]
    assert torch.equal(shifted[0], torch.cat([tail, suffix]))
    assert torch.equal(shifted[1], positions + len(tail))
