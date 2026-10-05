# SPDX-License-Identifier: Apache-2.0
"""Worker-side Taiji readout installed alongside the vLLM source patch.

Generate requests carry plain dictionaries in SamplingParams.extra_args. The worker
keeps only the selected marker states across prefill chunks and returns head scores
through the added taiji_scores output field. Ordinary generation uses the same model.
The KV manager caps prefix lookup before the earliest required state.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path

import torch

TAIJI_READOUT_KEY = "taiji_readout"

HEAD_KINDS = ("yesno", "candidate")


@dataclass
class TaijiReadoutSpec:
    """Where to read a decision head out of one request's prefill.

    Attributes:
        positions: Absolute prompt-token indices of the option markers, in option order.
            These index the request's own token slice, not the packed batch buffer.
        query_index: Absolute prompt-token index of the final `Decision:` token, read as
            the global query row.
        head: Which read-out the checkpoint was trained with (`train_args.json: head`).
        num_tokens: Prompt length, asserted against the request so a chunked or truncated
            prefill is rejected instead of silently scoring the wrong rows.
        keys: Option keys in the same order as `positions`, carried through so the client
            can zip logits back onto its own options without a second lookup.
    """

    positions: list[int]
    query_index: int
    head: str = "yesno"
    num_tokens: int = 0
    keys: list[str] = field(default_factory=list)

    def validate(self, prompt_token_ids):
        """Reject a spec that does not describe this request's prompt."""
        if not self.positions:
            raise ValueError("taiji_readout requires at least one marker position")
        if self.head not in HEAD_KINDS:
            raise ValueError(
                f"unknown taiji head {self.head!r}; expected one of {list(HEAD_KINDS)}"
            )
        if self.keys and len(self.keys) != len(self.positions):
            raise ValueError(
                "taiji_readout keys must line up with marker positions: "
                f"{len(self.keys)} != {len(self.positions)}"
            )
        if prompt_token_ids is not None and self.num_tokens:
            if len(prompt_token_ids) != self.num_tokens:
                raise ValueError(
                    "taiji_readout num_tokens does not match the prompt: "
                    f"{self.num_tokens} != {len(prompt_token_ids)}"
                )
        highest = max(max(self.positions), self.query_index)
        if self.num_tokens and highest >= self.num_tokens:
            raise ValueError(
                f"taiji_readout index {highest} is outside the "
                f"{self.num_tokens}-token prompt"
            )


class TaijiHead(torch.nn.Module):
    """Taiji's trained read-out: the LM Yes-minus-No logit plus a gated residual.

    Behaviour-identical to `inference/vllm_head.py`'s `YesNoHead`, restated here so the
    runner carries no dependency on the Taiji repository. It runs in fp32: in bf16,
    near-ties flip with numerical noise, which changes the served choice.
    """

    def __init__(self, hidden, dim=256):
        super().__init__()
        self.scalar = torch.nn.Linear(hidden, 1)
        self.residual = _ResidualHead(hidden, dim)
        self.gate = torch.nn.Parameter(torch.zeros(()))

    def forward(self, candidates, query):
        """candidates [n, hidden], query [n, hidden] -> option scores [n]."""
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            readout = self.scalar(candidates.float()).squeeze(-1)
            residual = self.residual(candidates, query)
        return readout + self.gate * residual


class _ResidualHead(torch.nn.Module):
    """The gated residual: bilinear plus an MLP interaction, scored per option.

    A submodule named ``residual`` so its parameters serialise as
    ``residual.candidate_norm.weight`` and friends, exactly the layout the trained
    checkpoint in ``vllm_export.py`` writes.
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

    def forward(self, candidates, query):
        c = self.candidate_norm(candidates.float())
        q = self.query_norm(query.float())
        bilinear = (self.key(c) * self.query(q)).sum(-1) / (self.dim**0.5)
        interaction = self.scalar(
            torch.nn.functional.gelu(self.candidate_mlp(c) + self.query_mlp(q))
        ).squeeze(-1)
        return bilinear + interaction


def load_head(state_dict, hidden, kind="yesno"):
    """Build a head from a checkpoint's tensor dict.

    Raises rather than silently loading a partial head: a mismatch here would serve wrong
    decisions with no error anywhere downstream.
    """
    if kind != "yesno":
        raise ValueError(f"runner-side read-out only implements 'yesno', got {kind!r}")
    head = TaijiHead(hidden)
    missing, unexpected = head.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"taiji head checkpoint has unexpected tensors: {unexpected}")
    if missing:
        raise ValueError(f"taiji head checkpoint is missing tensors: {missing}")
    head.eval()
    return head


def read_out(head, hidden_states, starts, specs):
    """Apply the head to each request's marker rows -> concatenated option logits.

    `hidden_states` is the packed prefill output and `starts[i]` is where request i
    begins inside it. Rows are gathered per request so a batch of read-outs costs one head
    evaluation rather than one per request.
    """
    device = hidden_states.device
    counts = [len(spec.positions) for spec in specs]
    rows = []
    queries = []
    for start, spec in zip(starts, specs, strict=True):
        idx = torch.as_tensor(spec.positions, device=device, dtype=torch.long) + start
        rows.append(hidden_states.index_select(0, idx))
        queries.append(hidden_states[start + spec.query_index].reshape(1, -1))
    all_candidates = torch.cat(rows, dim=0)
    # One query row per request, repeated so each request's option rows see its own query.
    all_query = torch.cat(queries, dim=0).repeat_interleave(
        torch.as_tensor(counts, device=device), dim=0
    )
    with torch.inference_mode():
        return head(all_candidates, all_query).float()


def validate_spec(spec, prompt_token_ids, prompt_len=None):
    """Reject a readout spec that cannot describe a prompt of this length.

    The V1 runner holds the full prompt token ids; the V2 runner only exposes the
    prompt length, so it validates against that instead.
    """
    if not isinstance(spec, dict):
        raise ValueError("taiji_readout must be a JSON dictionary")
    positions = spec.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError("taiji_readout requires a nonempty positions list")
    query_index = spec.get("query_index")
    indices = [*positions, query_index]
    if any(type(position) is not int or position < 0 for position in indices):
        raise ValueError("taiji_readout indices must be nonnegative integers")
    if prompt_token_ids is not None:
        num_tokens = len(prompt_token_ids)
    elif prompt_len is not None:
        num_tokens = int(prompt_len)
    else:
        raise ValueError("taiji_readout requires prompt_token_ids or prompt_len")
    if spec.get("num_tokens") != num_tokens or max(indices) >= num_tokens:
        raise ValueError("taiji_readout positions do not match the prompt length")
    if query_index != num_tokens - 1:
        raise ValueError("taiji_readout query_index must be the last prompt token")
    if spec.get("head", "yesno") != "yesno":
        raise ValueError("taiji_readout currently supports the trained yesno head")
    keys = spec.get("keys")
    if keys is not None and len(keys) != len(positions):
        raise ValueError("taiji_readout keys must match positions")


class TaijiReadout:
    def __init__(self, head):
        self.head = head
        self.rows = {}
        self.progress = {}
        self.specs = {}

    @classmethod
    def load(cls, model_path, device):
        config_path = Path(model_path) / "taiji_config.json"
        if not config_path.is_file():
            return None
        metadata = json.loads(config_path.read_text())
        head_path = Path(metadata.get("head_file", "taiji_head.pt"))
        if not head_path.is_absolute():
            head_path = config_path.parent / head_path
        state = torch.load(head_path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        head = load_head(state, int(metadata["hidden_size"]), metadata.get("head", "yesno"))
        return cls(head.to(device=device, dtype=torch.float32))

    def remove(self, request_id):
        self.rows.pop(request_id, None)
        self.progress.pop(request_id, None)
        self.specs.pop(request_id, None)

    def register(self, request_id, spec):
        """V2 runner: remember a request's already-validated readout spec."""
        self.specs[request_id] = spec

    def collect(self, hidden_states, input_batch, requests, starts, scheduled):
        """V1 runner: read each request's spec off its SamplingParams."""

        def spec_for(request_id):
            request = requests[request_id]
            params = request.sampling_params
            spec = (params.extra_args or {}).get(TAIJI_READOUT_KEY) if params else None
            if spec is None:
                return None
            validate_spec(spec, request.prompt_token_ids)
            if params.n != 1:
                raise ValueError("taiji_readout requires n=1")
            return spec

        return self._collect(
            hidden_states,
            input_batch.req_ids,
            spec_for,
            lambda index: int(input_batch.num_computed_tokens_cpu[index]),
            lambda index: int(starts[index]),
            lambda index: int(scheduled[index]),
        )

    def collect_v2(self, hidden_states, input_batch):
        """V2 runner: specs were registered at add time; offsets come from InputBatch.

        num_computed_tokens_np[i] is the absolute prompt index of request i's first
        scheduled row and query_start_loc_np[i] is where that row starts inside the
        packed buffer, which is exactly the (computed, start, scheduled) triple the
        shared collector needs.
        """
        if not self.specs:
            return {}, set()
        return self._collect(
            hidden_states,
            input_batch.req_ids,
            self.specs.get,
            lambda index: int(input_batch.num_computed_tokens_np[index]),
            lambda index: int(input_batch.query_start_loc_np[index]),
            lambda index: int(input_batch.num_scheduled_tokens[index]),
        )

    def _collect(self, hidden_states, req_ids, spec_for, computed_of, start_of,
                 scheduled_of):
        readout_request_ids = set()
        completed = []
        for request_index, request_id in enumerate(req_ids):
            spec = spec_for(request_id)
            if spec is None:
                continue
            readout_request_ids.add(request_id)
            computed = computed_of(request_index)
            chunk_end = computed + scheduled_of(request_index)
            if computed < self.progress.get(request_id, computed):
                self.remove(request_id)
            rows = self.rows.setdefault(request_id, {})
            required = list(dict.fromkeys([*spec["positions"], spec["query_index"]]))
            available = [position for position in required if computed <= position < chunk_end]
            if available:
                indices = torch.tensor(
                    [start_of(request_index) + position - computed for position in available],
                    dtype=torch.long, device=hidden_states.device,
                )
                selected = hidden_states.index_select(0, indices).clone()
                rows.update(zip(available, selected.unbind(0), strict=True))
            self.progress[request_id] = chunk_end
            if chunk_end < spec["num_tokens"]:
                continue
            missing = set(required) - rows.keys()
            if missing:
                raise RuntimeError(
                    f"Taiji readout missing marker states {sorted(missing)}; "
                    "prefix lookup must stop before the earliest marker"
                )
            candidates = torch.stack([rows[position] for position in spec["positions"]])
            query = rows[spec["query_index"]].expand(len(spec["positions"]), -1)
            completed.append((request_id, candidates, query))

        scores = {}
        if completed:
            candidates = torch.cat([item[1] for item in completed])
            queries = torch.cat([item[2] for item in completed])
            with torch.inference_mode():
                logits = self.head(candidates, queries).float().cpu().tolist()
            offset = 0
            for request_id, candidates, _query in completed:
                count = candidates.shape[0]
                scores[request_id] = logits[offset:offset + count]
                offset += count
                self.remove(request_id)
        return scores, readout_request_ids


__all__ = [
    "HEAD_KINDS",
    "TAIJI_READOUT_KEY",
    "TaijiHead",
    "TaijiReadoutSpec",
    "load_head",
    "read_out",
    "TaijiReadout",
    "validate_spec",
]
