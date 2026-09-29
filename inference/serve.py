"""S1 server, both jobs from one model, on existing interfaces only:

POST /v1/systemone         Jev-compatible choice/noul/score answers. Agent: TYPESAFE_URL=http://<host>:<port>/v1/systemone
POST /v1/chat/completions  OpenAI-compatible. "model": "s1" (the --name) writes the agent's field text
                           (TEXT_MODEL_BASE_URL=http://<host>:<port>/v1, TEXT_MODEL=s1); any other model name is plain
                           chat with the LoRA switched off, i.e. the untouched base model, in the same process (--plain-chat).
Tool calling needs no endpoint of its own: tools.call_tool() asks /v1/systemone and /v1/chat/completions.
Both take Authorization: Bearer $S1_API_KEY (set TYPESAFE_API_KEY and TEXT_MODEL_API_KEY to it).
A screenshot may ride along as state["screenshot"] (base64 or data URI); it is fed to the vision encoder.
"""

import argparse
import hashlib
import hmac
import json
import os
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from s1 import STATE_TOKEN_FLOOR, answer_from, encode_prefix, encode_question, kind, load

MAX_BODY = 32 * 1024 * 1024
PREFIX_CACHE = 8  # recent page-state caches (--shared-prefix); each is tens of MB of GPU memory
PREFIXES = OrderedDict()


def answer(processor, model, body, args):
    state = body["state"]  # text, object or list; a screenshot rides in state["screenshot"] or body["screenshot"]
    if not isinstance(state, (str, dict, list)) or not isinstance(body["questions"], dict):
        raise ValueError("state must be text, an object or a list; questions must be an object")
    started = time.perf_counter()
    suffixes = {}
    for name, question in body["questions"].items():
        try:
            suffixes[name] = encode_question(processor.tokenizer, question, max_tokens=args.max_len - STATE_TOKEN_FLOOR)
        except (ValueError, AttributeError) as error:
            raise ValueError(f"{name}: {error}") from None
    longest = max(len(suffix) for suffix, _, _ in suffixes.values())
    budget, screenshot = args.max_len - longest, body.get("screenshot")
    questions = [(suffix, positions) for suffix, positions, _ in suffixes.values()]
    hit = False
    if args.shared_prefix:
        # The same page state recurs (a tie-break, a re-ask on an unchanged page, a loop): reuse its prefix cache.
        key = hashlib.sha256(json.dumps([state, screenshot, budget, args.image_tokens]).encode()).digest()
        hit = key in PREFIXES
        if hit:
            PREFIXES.move_to_end(key)
        encoded = time.perf_counter()
        if not hit:
            prefix = encode_prefix(processor, state, budget, args.image_tokens, screenshot)
            encoded = time.perf_counter()
            PREFIXES[key] = (*model.prefix_cache(prefix), len(prefix["input_ids"]))
            while len(PREFIXES) > PREFIX_CACHE:
                PREFIXES.popitem(last=False)
        *cached, prefix_tokens = PREFIXES[key]
        scores = model.score_questions(None, questions, cached)
    else:  # the exact computation used in training
        prefix = encode_prefix(processor, state, budget, args.image_tokens, screenshot)
        encoded, prefix_tokens = time.perf_counter(), len(prefix["input_ids"])
        scores = [model({"prefix": prefix, "suffix": suffix, "positions": positions}) for suffix, positions in questions]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    scored = time.perf_counter()
    answers = {}
    for (name, (_, _, keys)), logits in zip(suffixes.items(), scores):
        temperature = model.temperatures.get(kind({"question": body["questions"][name]}), model.temperature)
        probs = torch.softmax(logits.float() / temperature, -1).tolist()
        answers[name] = answer_from(body["questions"][name], keys, probs)
    tokens = prefix_tokens + sum(len(suffix) for suffix, _, _ in suffixes.values())
    # Where the time goes: CPU encoding (image resize, tokenising) vs the model, and prefix vs per-question tokens.
    timing = {"encode_ms": round((encoded - started) * 1000), "model_ms": round((scored - encoded) * 1000),
              "prefix_hit": hit, "prefix_tokens": prefix_tokens,
              "question_tokens": {name: len(suffix) for name, (suffix, _, _) in suffixes.items()}}
    return {"model": args.name, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}, "timing": timing}


def chat(processor, model, body, args):
    """Plain chat on the base model: the same weights with the LoRA disabled for this request only."""
    if not hasattr(model.lm, "disable_adapter"):
        raise ValueError("plain chat needs a LoRA run served with --plain-chat")
    messages = [{"role": m["role"], "content": m["content"]} for m in body["messages"]]
    ids = processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                                  return_dict=True,
                                                  enable_thinking=bool(body.get("enable_thinking", False)))["input_ids"]
    ids = ids.to(model.head.scalar.weight.device)
    temperature = float(body.get("temperature", 0.7))
    with model.lm.disable_adapter():
        model.generator().model.rope_deltas = None
        out = model.generator().generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         max_new_tokens=int(body.get("max_tokens", 1024)),
                                         do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                                         eos_token_id=processor.tokenizer.eos_token_id)
    text = processor.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return {"model": body.get("model"), "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": ids.shape[1], "completion_tokens": out.shape[1] - ids.shape[1]}}


def complete(processor, model, body, args):
    """The agent's field_text() sends [system prompt, user: JSON field context] and expects {"text": ...}."""
    if body.get("model", args.name) != args.name:
        return chat(processor, model, body, args)
    user = next(m["content"] for m in reversed(body["messages"]) if m.get("role") == "user")
    try:
        context = json.loads(user)
    except json.JSONDecodeError:
        context = user
    text = model.write(processor, context, args.max_len)
    content = json.dumps({"text": text or None})  # null lets the agent escalate to its S2 model
    return {"model": args.name, "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--image-tokens", type=int, default=400)
    p.add_argument("--name", default="s1")
    p.add_argument("--plain-chat", action="store_true",
                   help="keep the LoRA unmerged so other model names get the base model (slower decisions)")
    p.add_argument("--shared-prefix", action="store_true",
                   help="reuse the page-state cache across questions; enable only if check_cache.py passes")
    args = p.parse_args()
    key = os.environ.get("S1_API_KEY")
    if not key:
        raise SystemExit("Set S1_API_KEY; this endpoint is reachable from the internet on Vast.")

    # Merged LoRA runs no extra adapter matmuls; only plain chat needs it unmerged, to switch it off per request.
    processor, model = load(args.base, args.adapter, merge=not args.plain_chat)
    model.eval()
    lock = threading.Lock()  # one GPU, one request at a time

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send(200, {"ok": True, "model": args.name}) if self.path == "/health" else self.send(404, {})

        def do_POST(self):
            routes = {"/v1/systemone": answer, "/v1/chat/completions": complete}
            if self.path not in routes:
                return self.send(404, {"error": "not found"})
            if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {key}"):
                return self.send(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                return self.send(413, {"error": "body too large or empty"})
            try:
                body = json.loads(self.rfile.read(length))
                started = time.perf_counter()
                with lock, torch.inference_mode():
                    result = routes[self.path](processor, model, body, args)
                result["latency_ms"] = round((time.perf_counter() - started) * 1000)
                self.send(200, result)
            except (ValueError, KeyError, TypeError, StopIteration, json.JSONDecodeError) as error:
                self.send(400, {"error": str(error)})

        def log_message(self, *_):
            pass

    print(f"S1 listening on {args.host}:{args.port} (temperature {model.temperature})", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
