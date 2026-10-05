"""The vLLM copy of the head must produce the same logits as the trained `s1.YesNoHead`."""

import json

import pytest
import torch
from vllm_head import (
    CandidateHead,
    YesNoHead,
    build_head,
    head_logits,
    marker_positions,
    read_head_bundle,
    score_at,
    score_sequence,
    softmax_with_temperature,
)

MARKER_ID = 2_000_000


def _randomize(module, seed):
    """Fill every tensor so two heads are compared on non-trivial weights."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return module


@pytest.mark.parametrize("head_class", [CandidateHead, YesNoHead])
def test_head_matches_s1_headly(head_class):
    s1 = pytest.importorskip("s1")
    hidden = 16
    torch.manual_seed(0)
    theirs = _randomize(s1.HEADS[head_class is YesNoHead and "yesno" or "candidate"](hidden).eval(), 7)
    ours = _randomize(head_class(hidden).eval(), 7)
    assert set(ours.state_dict()) == set(theirs.state_dict())
    hidden_states = torch.randn(11, hidden)
    positions = [2, 5, 9]
    theirs_out = theirs(hidden_states[positions], hidden_states[-1])
    ours_out = ours(hidden_states[positions], hidden_states[-1])
    assert torch.allclose(theirs_out, ours_out, atol=1e-6, rtol=1e-5)


def test_yesno_residual_gate_is_learned():
    """The trained head is not binary-only: the CandidateHead residual contributes when gate != 0."""
    head = _randomize(YesNoHead(12).eval(), 4)
    hidden_states = torch.randn(5, 12)
    positions = [0, 2, 4]
    with torch.no_grad():
        head.gate.zero_()
        no_residual = head(hidden_states[positions], hidden_states[-1])
        head.gate.fill_(0.5)
        with_residual = head(hidden_states[positions], hidden_states[-1])
    assert not torch.allclose(no_residual, with_residual)


def test_score_sequence_and_score_at_agree():
    head = _randomize(YesNoHead(12).eval(), 3)
    ids = [7, 8, MARKER_ID, 9, 10, MARKER_ID, 11]
    hidden_states = torch.randn(len(ids), 12)
    assert marker_positions(ids, MARKER_ID) == [2, 5]
    a = score_sequence(head, hidden_states, ids, MARKER_ID)
    b = score_at(head, hidden_states, [2, 5], len(ids) - 1)
    assert torch.allclose(a, b, atol=1e-7)


def test_head_logits_matches_index_select_and_query():
    head = _randomize(YesNoHead(10).eval(), 5)
    hidden_states = torch.randn(9, 10)
    positions = torch.tensor([1, 4, 8])
    assert torch.allclose(head_logits(head, hidden_states, positions, len(hidden_states) - 1),
                          head(hidden_states[positions], hidden_states[-1]), atol=1e-7)


def test_softmax_temperature_matches_manual():
    logits = torch.tensor([0.5, -1.0, 2.0])
    assert torch.allclose(softmax_with_temperature(logits, 0.7), torch.softmax(logits / 0.7, -1))


def test_build_head_and_bundle_round_trip(tmp_path):
    head = _randomize(YesNoHead(8).eval(), 9)
    torch.save(head.state_dict(), tmp_path / "taiji_head.pt")
    (tmp_path / "taiji_config.json").write_text(json.dumps(
        {"head": "yesno", "hidden_size": 8, "head_file": "taiji_head.pt"}))
    loaded, meta = read_head_bundle(tmp_path)
    assert meta["head"] == "yesno"
    hidden_states = torch.randn(6, 8)
    positions = torch.tensor([0, 3])
    assert torch.allclose(head(hidden_states[positions], hidden_states[-1]),
                          loaded(hidden_states[positions], hidden_states[-1]), atol=1e-7)
    assert isinstance(build_head("candidate", 8), CandidateHead)
    with pytest.raises(ValueError):
        build_head("bogus", 8)
