"""Offline inference, no server: one model decides and writes field text.

python infer.py --adapter <weights folder> --data requests.jsonl --output answers.jsonl [--base Qwen/Qwen3.5-2B]

Each input line is one of:
  {"state": ..., "questions": {...}}      decide: Jev choice / noul / score questions, as POST /v1/systemone
  {"write": {...field context...}}        write: the text for one field, as the agent's field_text() context
Each output line is the server's answer for that line: {"answers": ..., "usage": ..., "timing": ...} or {"text": ...}.
"""

import argparse
import json
from argparse import Namespace

import torch
from s1 import load
from serve import answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--base", default="Qwen/Qwen3.5-2B")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--image-tokens", type=int, default=400)
    args = p.parse_args()
    processor, model = load(args.base, args.adapter)
    model.eval()
    serving = Namespace(max_len=args.max_len, image_tokens=args.image_tokens, name="s1", shared_prefix=True)
    with open(args.data) as rows, open(args.output, "w") as out, torch.inference_mode():
        for line in rows:
            if not line.strip():
                continue
            row = json.loads(line)
            if "write" in row:
                result = {"text": model.write(processor, row["write"], args.max_len) or None}
            else:
                result = answer(processor, model, row, serving)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
