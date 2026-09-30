"""Taiji's trained decision head, importable without vLLM.

This is the exact read-out the S1 runtime in `s1.py` trained and serves: a Yes-minus-No
language-model logit read at each option marker plus a gated CandidateHead residual that
compares every option endpoint against the global query (the final `Decision:` token).

`YesNoHead` / `CandidateHead` below are behaviour-identical copies of the modules in
`inference/s1.py`. They are duplicated on purpose so the vLLM backend depends only on a
small, framework-free module, and `vllm_tests/test_vllm_parity.py` asserts numerically that
they match the originals. Do not "simplify" this into a generic linear classifier: the
trained checkpoint's `train_args.json` records `head: "yesno"`, and a plain linear or
next-token score is a different model.
"""

import json
import math
from pathlib import Path

import torch

DEFAULT_HEAD = "yesno"


class CandidateHead(torch.nn.Module):
    """Scores each option endpoint against the global query (Decision-1.0-Lux design).

    Runs in fp32: in bf16 near-ties flip with numerical noise, which changes the served choice.
    """

    def __init__(self, hidden, dim=256):
        super().__init__()
        self.dim = dim
        self.candidate_norm = torch.nn.LayerNorm(hidden)
        self.query_norm = torch.nn.LayerNorm(hidden)
        self.key = torch.nn.Linear(hidden, dim, bias=False)
        self.query = torch.nn.Linear(hidden, dim, bias=False)
        self.candidate_mlp = torch.nn.Linear(hidden, dim)
        self.query_mlp = torch.nn.Linear(hidden, dim, bias=False)
        self.scalar = torch.nn.Linear(dim, 1, bias=False)
        torch.nn.init.normal_(self.scalar.weight, std=0.01)

    def forward(self, candidates, query):
        """candidates [n, hidden], query [hidden] -> scores [n]."""
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            c, q = self.candidate_norm(candidates.float()), self.query_norm(query.float())
            bilinear = (self.key(c) * self.query(q)).sum(-1) / math.sqrt(self.dim)
            interaction = self.scalar(
                torch.nn.functional.gelu(self.candidate_mlp(c) + self.query_mlp(q))
            ).squeeze(-1)
            return bilinear + interaction


class YesNoHead(torch.nn.Module):
    """The trained Taiji read-out: LM Yes-minus-No logit at each marker, plus a gated residual.

    `scalar` is initialised from the base model's LM head (Yes row minus No row) by
    `init_from_lm`; the shipped `head.pt` already carries those trained weights, so vLLM
    only has to load them. `gate` is a learned scalar; at 0 the residual contributes nothing.
    """

    def __init__(self, hidden):
        super().__init__()
        self.scalar = torch.nn.Linear(hidden, 1)
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


def build_head(kind, hidden):
    """Construct the head named in a checkpoint's `train_args.json`."""
    if kind not in HEADS:
        raise ValueError(f"unknown head {kind!r}; expected one of {sorted(HEADS)}")
    return HEADS[kind](hidden)


def head_logits(head, hidden_states, positions, query_index):
    """Option logits for one sequence, exactly as `S1.forward` computes them.

    `hidden_states` is the full sequence's final hidden state [seq, hidden]; `positions`
    are the option-marker indices; `query_index` is the final `Decision:` token index.
    Both the existing runtime and this helper index the same hidden rows, so the returned
    logits match `S1.forward` bit-for-bit up to device/dtype (asserted in the parity test).
    """
    candidates = hidden_states.index_select(0, positions.to(hidden_states.device))
    query = hidden_states[query_index]
    return head(candidates, query)


def softmax_with_temperature(logits, temperature):
    """Calibrated option probabilities; temperature < 1 sharpens, > 1 flattens."""
    return torch.softmax(logits.float() / float(temperature), -1)


def marker_positions(ids, marker_id):
    """Every index in `ids` whose token is the option marker; the read-out positions."""
    return [i for i, token in enumerate(ids) if token == marker_id]


def score_sequence(head, hidden_states, ids, marker_id):
    """Head logits for one full decision sequence, from its ids alone.

    This is the exact computation `S1.forward`/vLLM perform: read each marker's hidden row,
    read the final `Decision:` row as the query, and run the trained head. Every marker index
    in `ids` is scored, so a literal marker string that slipped into untrusted text would add
    a spurious candidate; `compile_prompt` prevents that by tokenising untrusted text with
    `split_special_tokens=True`, and the scorer cross-checks the found positions against the
    offsets recorded at compile time.
    """
    return score_at(head, hidden_states, marker_positions(ids, marker_id), len(ids) - 1)


def score_at(head, hidden_states, positions, query_index):
    """Head logits at explicit candidate indices and one query index (the serving path)."""
    indices = torch.as_tensor(positions, dtype=torch.long, device=hidden_states.device)
    candidates = hidden_states.index_select(0, indices)
    return head(candidates, hidden_states[query_index])


def split_packed_hidden_states(hidden_states, lengths):
    """Split vLLM's contiguous, variable-length sequence rows by token counts."""
    lengths = [int(length) for length in lengths]
    if any(length < 0 for length in lengths):
        raise ValueError("hidden-state lengths must be non-negative")
    if sum(lengths) != hidden_states.shape[0]:
        raise ValueError(
            f"packed hidden-state rows ({hidden_states.shape[0]}) do not match "
            f"the supplied sequence lengths ({sum(lengths)})"
        )
    return list(torch.split(hidden_states, lengths, dim=0))


def read_head_bundle(path):
    """Load a Taiji head bundle written by `vllm_export.py`.

    Accepts either a directory (containing `taiji_config.json` and `taiji_head.pt`) or the
    `taiji_config.json` path directly. Returns the head module plus its sidecar metadata.
    """
    path = Path(path)
    config_path = path if path.is_file() else path / "taiji_config.json"
    meta = json.loads(config_path.read_text())
    head_path = Path(meta.get("head_file", "taiji_head.pt"))
    if not head_path.is_absolute():
        head_path = config_path.parent / head_path
    head = build_head(meta.get("head", DEFAULT_HEAD), int(meta["hidden_size"]))
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    head.load_state_dict(state)
    head.eval()
    return head, meta


__all__ = [
    "DEFAULT_HEAD",
    "HEADS",
    "CandidateHead",
    "YesNoHead",
    "build_head",
    "head_logits",
    "marker_positions",
    "read_head_bundle",
    "score_at",
    "score_sequence",
    "softmax_with_temperature",
    "split_packed_hidden_states",
]
