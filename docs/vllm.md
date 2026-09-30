# vLLM inference backend

Taiji can run its decision head on token states produced by vLLM instead of the Transformers server in `inference/serve.py`. It preserves the trained read-out exactly: Taiji applies the same `YesNoHead` — the language model's Yes-minus-No logit at each option marker plus the learned gated `CandidateHead` residual — at the same marker and final `Decision:` positions. It returns one logit per supplied option, then calibrates those logits into `choice`, `noul`, or `score` answers. It does not reduce arbitrary choices to a binary output or substitute next-token scores.

## How this follows vLLM-jev

This uses the same architecture as [mode-io/vllm-jev](https://github.com/mode-io/vllm-jev): register a vLLM token-embedding model, use vLLM's scheduler and pooling runner to produce the backbone's token states, then apply the checkpoint's trained decision head to the required positions. Taiji owns its prompt compiler, head loader, checkpoint exporter, and `/v1/systemone` service in this repository. It does not patch vLLM core or vendor the full vLLM-jev repository.

## Prepare a checkpoint

The exporter merges the LoRA, packages the head, and marks the model so vLLM selects the Taiji class:

```bash
# In the export environment (keep this separate from the vLLM runtime environment):
pip install -r inference/requirements.txt
python inference/vllm_export.py --base Qwen/Qwen3.5-2B --adapter Taiji-2B \
  --out taiji-vllm --dtype bfloat16
```

This writes a standard Hugging Face model folder plus two Taiji files:

- `taiji_head.pt` — the trained decision head in fp32.
- `taiji_config.json` — head type, marker token and id, fitted temperatures, shard list.

It also sets `architectures = ["TaijiForVLLM"]` in `config.json`, so the Taiji plugin maps that architecture to vLLM's native Qwen3.5 token-embedding wrapper. The export includes the tokenizer and processor needed by the runtime.

The head is loaded from the adapter's `head.pt`; the head type is read from its `train_args.json` (`head: "yesno"` for both the v3 and v4 checkpoints), so the export never silently substitutes a different head.

## Install the vLLM model registration

In a separate Linux + NVIDIA vLLM environment, install the pinned vLLM runtime and Taiji plugin:

```bash
pip install -r inference/requirements-vllm.txt
pip install ./inference
```

This installs a `vllm.general_plugins` entry point that registers the embedding wrapper before the engine starts:

```python
from vllm_model import register
register()
```

## Serve decisions

`inference/vllm_serve.py` exposes the same `POST /v1/systemone` contract as the Transformers server, so a browser agent can point `TYPESAFE_URL` at this port unchanged:

```bash
S1_API_KEY='<long-random-secret>' python inference/vllm_serve.py \
  --model taiji-vllm --head-bundle taiji-vllm --host 0.0.0.0 --port 8000
```

The Taiji server uses vLLM's `LLM.encode` token-embedding task, checks that returned prompt ids and hidden-state shapes match the compiled requests, then runs `YesNoHead` over the marker and query rows. This extra scoring step is required because each request can have a different number of candidate options and Taiji's trained head has a learned residual network.

## Reading out, and how parity is checked

The prompt layout the head reads is exactly the training layout, from `inference/s1.py`:

```
<prefix> State: <state tokens> "\n" <type header> "\nInstructions: " <instr> "\nOptions:\n"
[key] <option body> <|quad_end|> "\n" ...
"\nSelect the single option best supported by the state and instructions.\nDecision:"
```

Each option endpoint is a `<|quad_end|>` marker; the global query is the final `Decision` token. Untrusted state and option text is tokenized with `split_special_tokens=True`, so a literal marker string in a page cannot forge a read-out position. The exact token ids are passed to vLLM as `prompt_token_ids` (never a re-tokenized string), and the marker offsets stay alongside them in the Taiji request, so nothing can shift between compile and score.

The state prefix is reproduced from `s1.encode_prefix`, not approximated: the fixed `"State: "` head is produced by the model's `AutoProcessor` (so an image placeholder would expand identically), and the state is fitted to the budget exactly as training did — page text shortened first, then elements dropped — before the question suffix is appended. The backend therefore loads the merged model's processor as well as its tokenizer, and one state is fitted and encoded once and shared by every question in a call.

`inference/vllm_tests/` checks the parts that can be checked on a CPU:

- `test_vllm_head.py` — `YesNoHead`/`CandidateHead` logits equal `s1.py`'s, and the residual actually contributes when the gate is nonzero.
- `test_vllm_prompt.py` — `compile_question` matches `s1.encode_question` (including the option cap), `compile_prefix` matches `s1.encode_prefix` on raw, short and long-fitted states, `compile_prompts` matches `s1.encode` id-for-id, and a literal marker string stays literal.
- `test_vllm_backend.py` — the head scores returned token states at the compiled offsets, and answer shaping and confidence match `s1.py`.

Run them with any environment that has torch:

```bash
python -m pytest inference/vllm_tests -q
```

## Scope and limitations

- **Text decisions.** `compile_prompt` builds text prompts. Image questions need the multimodal path (a screenshot in `state["screenshot"]`); the S1 Transformers server handles those today, and this backend does not yet feed images to vLLM.
- **One sequence per question.** Questions from a `/v1/systemone` request are submitted together in one vLLM encode batch. Prefix caching is enabled to reuse their shared state prefix and repeated prefixes across requests.
- **GPU verification pending.** The environment that produced these files has no CUDA or vLLM, so the head/prompt/answer parity tests run on CPU and the vLLM path itself is unverified end to end. Run a fixed request through both servers on the GPU host and compare probabilities and top choice before relying on it.
- **Field text and plain chat** stay on `inference/serve.py`. The vLLM path serves decisions only.
