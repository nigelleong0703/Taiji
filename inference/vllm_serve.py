"""Taiji decisions served from vLLM on the same contract as `serve.py`.

    python inference/vllm_export.py --base Qwen/Qwen3.5-2B --adapter Taiji-2B --out taiji-vllm
    S1_API_KEY='<secret>' python inference/vllm_serve.py \
        --model taiji-vllm --head-bundle taiji-vllm --host 0.0.0.0 --port 8000

POST /v1/systemone takes {"state": ..., "questions": {...}} and returns the same
`choice` / `noul` / `score` answers as the S1 transformers server, so a browser agent can
point TYPESAFE_URL at this port. GET /health is unauthenticated.

This backend serves decisions. Field-text writing and plain chat stay on `serve.py` (the
generation path), which this module does not replace.
"""

import argparse
import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vllm_backend import TaijiVLLM

MAX_BODY = 32 * 1024 * 1024


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="merged Taiji model folder (from vllm_export.py)")
    p.add_argument("--head-bundle", help="folder with taiji_head.pt; defaults to --model")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192,
                   help="keep prompts in one schedule (prefill chunks would otherwise cut the read-out)")
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--name", default="taiji-vllm")
    args = p.parse_args()
    key = os.environ.get("S1_API_KEY")
    if not key:
        raise SystemExit("Set S1_API_KEY; this endpoint is reachable from the internet on Vast.")

    engine_kwargs = {"max_model_len": args.max_len, "max_num_batched_tokens": args.max_num_batched_tokens,
                     "dtype": args.dtype}
    if args.gpu_memory_utilization is not None:
        engine_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    backend = TaijiVLLM(args.model, head_bundle=args.head_bundle or args.model, max_len=args.max_len,
                        engine_kwargs=engine_kwargs)
    backend._build()  # load the engine once, before the first request
    request_lock = threading.Lock()

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
            if self.path != "/v1/systemone":
                return self.send(404, {"error": "not found"})
            if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {key}"):
                return self.send(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                return self.send(413, {"error": "body too large or empty"})
            try:
                body = json.loads(self.rfile.read(length))
                state, questions = body["state"], body["questions"]
                if not isinstance(state, (str, dict, list)) or not isinstance(questions, dict):
                    raise TypeError("state must be text, an object or a list; questions must be an object")
                started = time.perf_counter()
                # The synchronous LLM API is shared across HTTP handler threads; serialize
                # calls while allowing vLLM to batch every question in each request.
                with request_lock:
                    result = backend.decide(state, questions)
                result["model"] = args.name
                result["latency_ms"] = round((time.perf_counter() - started) * 1000)
                self.send(200, result)
            except (ValueError, KeyError, TypeError, NotImplementedError, json.JSONDecodeError) as error:
                self.send(400, {"error": str(error)})

        def log_message(self, *_):
            pass

    print(f"Taiji vLLM listening on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
