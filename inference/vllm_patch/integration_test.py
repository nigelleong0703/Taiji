"""Verify decisions, generation and caching in one patched vLLM engine."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--output")
    args = parser.parse_args()

    import torch
    from vllm import SamplingParams
    from vllm_backend import TaijiVLLM

    model = TaijiVLLM(
        args.model, max_len=args.max_model_len,
        engine_kwargs={"gpu_memory_utilization": 0.85,
                       "max_num_batched_tokens": args.chunk_tokens,
                       "enable_chunked_prefill": True,
                       "limit_mm_per_prompt": {"image": 0, "video": 0}},
    )
    engine = model.load_engine()
    assert engine.model_config.runner_type == "generate"
    print("ENGINE loaded once; runner_type=generate", flush=True)
    body = json.loads(Path(args.request).read_text())
    questions = list(body["questions"].values())
    timings = []
    baseline = None
    for label in ("cold", "cached", "cached-repeat"):
        started = time.perf_counter()
        result = model.decide(body["state"], body["questions"])
        elapsed = (time.perf_counter() - started) * 1000
        answers = result["answers"]
        if baseline is None:
            baseline = answers
        else:
            for name, answer in answers.items():
                if "choice" in answer:
                    assert answer["choice"] == baseline[name]["choice"], (name, answer, baseline[name])
        print(f"DECISION {label} {elapsed:.1f} ms: {json.dumps(result)}", flush=True)
        timings.append({"label": label, "ms": elapsed, "result": result})

    compiled = model.compile(questions, body["state"])[0]
    decision_prompt = {"prompt_token_ids": compiled["input_ids"]}
    decision_params = SamplingParams(
        temperature=0, max_tokens=1, detokenize=False,
        extra_args={"taiji_readout": {
            "positions": compiled["positions"],
            "query_index": compiled["num_tokens"] - 1,
            "num_tokens": compiled["num_tokens"], "keys": compiled["keys"],
        }},
    )
    messages = [{"role": "user", "content": "Reply with only the word Paris."}]
    write_ids = model.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        return_dict=False,
    )
    started = time.perf_counter()
    mixed = engine.generate(
        [decision_prompt, {"prompt_token_ids": write_ids}],
        [decision_params, SamplingParams(temperature=0, max_tokens=16)],
        use_tqdm=False,
    )
    assert mixed[0].taiji_scores is not None
    assert not mixed[0].outputs[0].token_ids
    assert mixed[0].num_cached_tokens > 0, "Decision prefix cache did not hit"
    assert mixed[0].num_cached_tokens <= min(compiled["positions"])
    assert mixed[1].taiji_scores is None
    assert mixed[1].outputs[0].text.strip(), "Normal generation produced no text"
    mixed_ms = (time.perf_counter() - started) * 1000
    print(f"MIXED batch {mixed_ms:.1f} ms; cached_tokens={mixed[0].num_cached_tokens}; "
          f"GENERATE={mixed[1].outputs[0].text!r}", flush=True)

    started = time.perf_counter()
    written = model.write({"goal": "Fill the departure city with Singapore",
                           "field": {"label": "Departure city"},
                           "state": "Search flights from Singapore to Bali"})
    write_ms = (time.perf_counter() - started) * 1000
    assert written.strip(), "LoRA field writer produced no text"
    assert model.load_engine() is engine
    assert not hasattr(model, "_generate_llm"), "A second engine is still present"
    print(f"FIELD WRITE {write_ms:.1f} ms: {written!r}", flush=True)
    print("BOTH MODES ON ONE ENGINE: PASS", flush=True)
    report = {"passed": True, "decisions": timings, "mixed_ms": mixed_ms,
              "cached_tokens": mixed[0].num_cached_tokens,
              "generation": mixed[1].outputs[0].text,
              "field_write": written, "write_ms": write_ms,
              "torch": torch.__version__}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
