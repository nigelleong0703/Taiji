# vLLM inference backend

Taiji runs its decision head and its field writer on one vLLM generation engine, one copy
of the merged weights, and one prefix cache. A small source patch applied to the pinned
vLLM install reads the trained decision head at each option marker during prefill and
returns the logits alongside the normal generation path, so `/v1/systemone` and
`/v1/chat/completions` are served by the same engine process.

The read-out is preserved exactly: the same `YesNoHead` — the language model's Yes-minus-No
logit at each option marker plus the learned gated `CandidateHead` residual — is applied at
the same marker and final `Decision:` positions, then calibrated into `choice`, `noul`, or
`score` answers. Nothing is reduced to a binary or next-token score.

## Prepare a checkpoint

The exporter merges the LoRA, packages the head, and marks the model so vLLM selects the
Taiji class:

```bash
# In the export environment (keep this separate from the vLLM runtime environment):
pip install -r inference/requirements.txt
python inference/vllm_export.py --base Qwen/Qwen3.5-2B --adapter Taiji-2B \
  --out taiji-vllm --dtype bfloat16
```

This writes a standard Hugging Face model folder plus two Taiji files:

- `taiji_head.pt` — the trained decision head in fp32.
- `taiji_config.json` — head type, hidden size, head file, marker token and id, fitted
  temperatures.

It also sets `architectures = ["TaijiForVLLM"]` in `config.json`. `inference/vllm_model.py`
``TaijiForVLLM` is a pure subclass of vLLM's native Qwen3.5 conditional-generation model, so
generation keeps the stock LM head and prefix cache; it is not a pooling or classifier
wrapper.

## Install the vLLM source patch

The read-out runs inside the worker, so the engine environment needs the pinned vLLM
source changes. This patch is required for all deployments.

**Version:** vLLM 0.29.0 only. The installer refuses any other version.

**Installation:**

```bash
pip install -r inference/requirements-vllm.txt
pip install --no-deps -e ./inference        # registers the Taiji architecture plugin
python inference/vllm_patch/apply.py        # applies the pinned vLLM 0.29.0 patch
```

Run `apply.py` with the vLLM environment's Python while the engine is stopped. The
installer checks every patch hunk before modifying any source file, backs up the originals
to `.taiji-source-patch-backup/`, and writes a manifest to `.taiji-source-patch.json`
alongside the vLLM package. On subsequent runs, `apply.py` detects the patch, verifies
file hashes, and exits cleanly without re-applying.

To restore: `python inference/vllm_patch/apply.py --restore` (requires engine stopped).
The restoration removes the backed-up files, the manifest, and all applied changes.

**Verification:**

Unit tests run on CPU (they need `torch` and `pytest`, no GPU):

```bash
cd inference && python -m pytest vllm_tests -q
```

On a GPU host, after the patch is installed, the end-to-end test exercises cold and
cached decisions, mixed batches, and field generation:

```bash
head -1 examples/requests.jsonl > request.json   # any single {"state", "questions"} request
python inference/vllm_patch/integration_test.py \
  --model /path/to/taiji-vllm --request request.json
```

The installed marker is `GPUModelRunner.TAIJI_READOUT_PATCH_VERSION == 1`, set on both
the V1 and the V2 model runner. The backend refuses to start without it.

vLLM 0.29 defaults to its V2 model runner on CUDA (`vllm/v1/worker/gpu/model_runner.py`), which is
a separate implementation from the older V1 runner rather than a subclass. The read-out is
installed in both, so the default configuration works; the V1 path is not covered by the
end-to-end run below.

The patch touches seven vLLM files and adds one helper:

- `vllm/outputs.py`, `vllm/v1/engine/__init__.py`, `vllm/v1/outputs.py` — carry a
  `taiji_scores` field through `RequestOutput`, `EngineCoreOutput` and
  `ModelRunnerOutput`.
- `vllm/v1/engine/output_processor.py` — turns a scored decision into a finished
  `RequestOutput` with the scores and an empty `token_ids`.
- `vllm/v1/core/kv_cache_manager.py` — caps prefix lookup before the earliest required
  marker, so the worker always recomputes the rows it must read.
- `vllm/v1/core/sched/scheduler.py` — finishes a scored request instead of treating it as
  text generation.
- `vllm/v1/worker/gpu/model_runner.py` (V2, the default) and `vllm/v1/worker/gpu_model_runner.py`
  (V1) — load the head once from the model folder, read the marker states after prefill
  (across chunks), and return the scores on `ModelRunnerOutput.taiji_scores`.
- `vllm/model_executor/layers/taiji_readout.py` — the installed helper holding the head and
  the per-request state collection, with one entry point per runner (`collect` for V1,
  `collect_v2` for V2).

Request and output contract:

- A decision request is a normal generate request whose `SamplingParams.extra_args` carries
  `{"taiji_readout": {"positions": [...], "query_index": n-1, "num_tokens": n, "keys": [...]}}`.
  It is a plain JSON dictionary, not a custom dataclass. Positions are absolute prompt-token
  indices of the option markers; `query_index` is the final `Decision:` token.
- The result arrives on `RequestOutput.taiji_scores: list[float]` with `num_cached_tokens`;
  `outputs[0].token_ids` is empty and the request is finished.
- A text request carries no `taiji_readout` and takes the ordinary generation path on the
  same engine.

## Serve decisions and writing

The existing server drives both routes from one engine:

```bash
S1_API_KEY='<long-random-secret>' python inference/serve.py --backend vllm \
  --model taiji-vllm --host 0.0.0.0 --port 8000
```

`/v1/systemone` calls `model.decide`, which compiles the questions, submits one readout
request per question, and shapes the returned logits into answers. `/v1/chat/completions`
calls `model.write`, which reproduces `s1.encode_text`'s field prompt and generates on the
same engine. `--plain-chat` and `--shared-prefix` do not apply to the vLLM backend; prefix
reuse comes from vLLM's own prefix cache.

Engine flags are fixed for the read-out and cannot be overridden through `engine_kwargs`:
`runner="generate"`, `enable_prefix_caching=True`, `async_scheduling=False`, and
`attention_config={"backend": "FLASH_ATTN"}`. A caller may still set `dtype`,
`gpu_memory_utilization` and `max_num_batched_tokens` (chunked prefill).

## Reading out, and how parity is checked

The prompt layout the head reads is exactly the training layout, from `inference/s1.py`:

```
<prefix> State: <state tokens> "\n" <type header> "\nInstructions: " <instr> "\nOptions:\n"
[key] <option body> <|quad_end|> "\n" ...
"\nSelect the single option best supported by the state and instructions.\nDecision:"
```

Each option endpoint is a `<|quad_end|>` marker; the global query is the final `Decision`
token. Untrusted state and option text is tokenized with `split_special_tokens=True`, so a
literal marker string in a page cannot forge a read-out position. The exact token ids are
passed as `prompt_token_ids` (never a re-tokenized string), and the marker offsets stay
alongside them in the readout spec, so nothing can shift between compile and score.

The state prefix is reproduced from `s1.encode_prefix`, not approximated: the fixed
`"State: "` head is produced by the model's `AutoProcessor` (so an image placeholder would
expand identically), and the state is fitted to the budget exactly as training did before
the question suffix is appended. The backend loads the merged model's processor as well as
its tokenizer, and one state is fitted and encoded once and shared by every question in a
call.

`inference/vllm_tests/` checks the parts that can be checked on a CPU:

- `test_vllm_head.py` — `YesNoHead`/`CandidateHead` logits equal `s1.py`'s, and the residual
  actually contributes when the gate is nonzero.
- `test_vllm_prompt.py` — `compile_question` matches `s1.encode_question` (including the
  option cap), `compile_prefix` matches `s1.encode_prefix` on raw, short and long-fitted
  states, `compile_prompts` matches `s1.encode` id-for-id, and a literal marker string stays
  literal.
- `test_vllm_backend.py` — answer shaping and confidence match `s1.py`, the readout spec
  describes the compiled prompt, and `score` submits readout requests and reads
  `RequestOutput.taiji_scores` (and fails loudly when the scores are missing).
- `test_shared_readout.py` — the installed worker helper collects marker states correctly
  across cached prefixes and prefill chunks, offsets a mixed batch per request, rejects a
  spec whose markers were skipped, and discards stale rows after recomputation.

```bash
python -m pytest inference/vllm_tests -q
```

## End-to-end check on a GPU host

Unit tests never load the backbone, so they do not prove the GPU path. `inference/vllm_patch/integration_test.py`
loads one engine and exercises cold and cached decisions, a mixed decision/text batch, and
the field writer, asserting the runner type, the cache hit, zero generation tokens for the
decision, and that no second engine exists:

```bash
vllm-env/bin/python inference/vllm_patch/integration_test.py \
  --model hf/taiji-vllm --request request.json --chunk-tokens 512 \
  --output shared-engine-result.json
```

On an RTX 4090D running vLLM 0.29.0 with the V2 model runner, FlashAttention 2 and prefix
caching, that script passed with:

- three decisions on one engine from a cold and two warm runs: 2176 ms / 784 ms / 830 ms,
  with 3168 then 10368 cached prompt tokens and the same top choices in every run;
- a mixed decision + text batch in 75 ms: 3456 cached tokens, no generation tokens on the
  decision, and the text request generated "Paris";
- a field write in 55 ms returning "Singapore", from the same engine instance.

Image states were checked the same way on that host: the same three questions with
`state["screenshot"]` added compile to 3516 / 7965 / 3536 tokens with 408 image tokens each,
are sent as 3109 / 7558 / 3129 ids with a single placeholder, and answer
`CLICK / 3 / 6` in 545 ms against `TYPE_TEXT / 7 / 6` in 598 ms without the image - the
screenshot changes the decision, so the vision tower is really in the path. Repeating the
image request takes 380 ms and hits 10080 cached tokens.

The same configuration also served both HTTP routes from one process, and drove a
`taiji-agent` browser S1/S2 loop: 4 S1 decisions plus a System Two confirmation for a task
that completed in 11 s, and a 14-decision Google Flights run where the fast policy's field
writer returned "Bali" and System Two answered a stuck escalation with a subgoal.

Two host settings were needed and are not part of the patch. The box has no `ninja` on PATH,
so FlashInfer's JIT-compiled sampler cannot build; start the engine with
`VLLM_USE_FLASHINFER_SAMPLER=0` to use vLLM's native sampler (this does not change the
attention backend, which stays FlashAttention). And the OCX planner rejects
`reasoning_effort: "none"` for `muse-spark-1.3-contributor`; `TAIJI_S2_REASONING_EFFORT=minimal`
makes it return JSON.

## Scope and limitations

- **Image states.** A screenshot in `state["screenshot"]` (path, data URI or bare base64) is
  resized, written into the prompt through the processor exactly as `s1.encode_prefix` does,
  and sent to the engine as multi-modal data, so the vision tower sees it. The read-out
  offsets stay in the expanded coordinate system. vLLM expands one placeholder token per
  image itself, so the request carries a single placeholder (see `request_token_ids`) while
  the local prompt keeps the whole span; both counts come from the same processor, and the
  worker's own length check proves they agree on every request.
- **Chunked prefill with images.** Setting `max_num_batched_tokens` low enough to force
  chunked prefill while images are allowed crashes vLLM 0.29's VLM profile run
  (`profile_run` -> `_dummy_run` -> `x.size()` on None), independently of this patch. Leave
  the token budget at its default when serving image states.
- **Engine shape.** The patch requires `async_scheduling=False`, no speculative decoding,
  `pipeline_parallel_size == 1`, and decision requests with `n=1`. Other configurations are
  not supported and are rejected.
- **Runner coverage.** The V2 model runner (vLLM 0.29's default on CUDA) is the verified path.
  The V1 hunks are installed and unit-tested, but have not been through the end-to-end run.
- **Prefix cache.** The read-out caps its own prefix lookup before the earliest marker, so a
  decision request can never reuse a block that contains a marker row it must read. Cached
  tokens are therefore bounded by the first option marker, which is the correct trade for
  correctness.

## Serve in FP8 on an 8 GB GPU

On an RTX 4060 Laptop with vLLM 0.29.0 and the patched engine, FP8 per-tensor
quantization fits the whole model on the GPU. Weights are
quantized online at load time; the vision tower and generation head remain
unquantized, KV cache is BF16, and the decision head is FP32. No CPU offload
is needed. The checkpoint's full context length (262,144 tokens) is supported.

The recommended launch command:

```bash
python -u inference/serve.py --backend vllm --model /path/to/taiji-vllm \
  --host 0.0.0.0 --port 8010 --max-len 262144 \
  --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192 --max-num-seqs 4 \
  --quantization fp8_per_tensor --quantization-config '{"ignore":["*visual*","*lm_head*"]}' \
  --mm-max-pixels 409600
```

Replace `/path/to/taiji-vllm` with the exported checkpoint path. Set `S1_API_KEY`
before running. GPU memory utilization is ~6.9 GB under this configuration.

CUDA graphs are ON (the default; do not pass `--enforce-eager`). Relative to the
earlier conservative setting (`--max-num-batched-tokens 512 --max-num-seqs 1
--enforce-eager`), decision latency improves: cold decisions ~480–760 ms become
297–458 ms (38–40% faster p50/p90), cached repeats settle at ~213–240 ms. Memory
footprint is slightly lower. Small decision differences appear only on near-tie
options: FP8 accumulation order changes with batch size. This is not an accuracy
drop.

Check the health endpoint:

```bash
curl http://<host>:8010/health
```

It returns `{"ok": true, "model": "taiji-vllm", "max_model_len": 262144,
"quantization": "fp8_per_tensor", "cpu_offload_gb": 0}`.

Prefix caching is enabled and shared by decisions and generation. The model is a
hybrid Qwen3.5 (Gated DeltaNet + attention). vLLM caches prefixes at block
granularity, so put text that stays the same across steps (task, rules) first and
the changing page text last to maximize cache hits.

For comparison, the plain base model (stock Qwen3.5-2B with no Taiji head or
patch) can be served for generation-only workloads with:

```bash
vllm serve Qwen/Qwen3.5-2B --max-model-len 32768 \
  --gpu-memory-utilization 0.92 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --quantization fp8 --language-model-only
```

## CUDA 13.0 environment

The RTX 4060 Laptop deployment uses a separate Linux Python environment; the
copied macOS `.venv` cannot run there. Install matching JIT compiler components
with `pip install -r inference/requirements-vllm-cu130.txt`, then install the Taiji
plugin with `pip install --no-deps -e ./inference` and apply
`python inference/vllm_patch/apply.py`. The plugin must be installed so spawned
workers register the architecture too. Python needs its matching `Python.h`;
uv-managed Python 3.12 supplies it. Set `CUDA_HOME` to the environment's
`lib/python3.12/site-packages/nvidia/cu13` and put both the environment's `bin`
and `$CUDA_HOME/bin` on PATH for nvcc and ninja. Unconstrained toolkit extras
selected nvcc 13.4 against cu130 headers and broke FlashInfer sampler JIT.

`serve.py` accepts `--max-num-seqs` and `--enforce-eager` for limited VRAM; see
[Serve in FP8 on an 8 GB GPU](#serve-in-fp8-on-an-8-gb-gpu) for the tested launch.
`--enforce-eager` saves CUDA graph memory at a latency cost; with FP8 weights it
is not needed on 8 GB. Image support remains enabled.
Attention is FLASH_ATTN (version 2 on this GPU); FlashInfer can separately be
used for sampling and other kernels, so the two names are not exclusive.

For the pip CUDA toolkit layout, the isolated environment additionally needed
`lib64 -> lib` and `lib/libcudart.so -> libcudart.so.13` before JIT linking.
Verify sampling compilation/loading on its own before full engine startup.
