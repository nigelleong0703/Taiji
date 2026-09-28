"""LoRA (default) or full fine-tuning (--full) for S1's two tasks (choice rows and text rows, mixed in one JSONL).

--full trains every language-model weight except the embeddings/LM head (the same layers LoRA adapts, plus norms);
the vision encoder stays frozen either way. Weights are fp32 masters under bf16 autocast: ~64 GB for 4B (80 GB GPU).

Single GPU: python train.py ...   Multi-GPU: torchrun --nproc_per_node=8 train.py ...

Checkpoints overwrite <out>/checkpoint every --save-every optimizer steps; --resume continues from it,
so an interrupted Vast instance loses at most that many steps.
"""

import argparse
import collections
import contextlib
import json
import math
import multiprocessing
import os
import sys
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from s1 import LORA_TARGETS, S1, encode, encode_text, kind, read_rows, row_at, yes_no_ids
from safetensors.torch import load_file
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

# Encoded rows come back from the worker processes as tensors; the default strategy passes one file descriptor per
# tensor, and a few hundred validation rows at once exceed a low open-file limit ("received 0 items of ancdata").
torch.multiprocessing.set_sharing_strategy("file_system")


_processor = None


def _init_worker(base):
    global _processor
    torch.set_num_threads(1)
    _processor = AutoProcessor.from_pretrained(base)


def _encode(job):
    """One row -> encoded tensors in a worker process, so tokenising overlaps the GPU. None if the row is unusable."""
    row, max_len, image_tokens, seed = job
    random.seed(seed)  # the option shuffle, fixed per row and epoch
    try:
        encoded = (encode_text(_processor, row["context"], row["target"], max_len) if row.get("task") == "text"
                   else encode(_processor, row, max_len, image_tokens, shuffle=True))
    except (ValueError, KeyError, OSError):
        return None
    encoded["source"], encoded["kind"] = row.get("source", "all"), kind(row)
    return encoded


def pack(rows, budget):
    """Rows sorted by length into padded batches of at most `budget` tokens (padding included)."""
    batches = []
    for e in sorted(rows, key=lambda e: len(e["input_ids"])):
        if batches and (len(batches[-1]) + 1) * len(e["input_ids"]) <= budget:
            batches[-1].append(e)
        else:
            batches.append([e])
    return batches


def row_loss(e, out, args):
    if "prompt_len" in e:
        return out * args.text_weight
    if "target" in e:  # soft labels: cross-entropy against the whole distribution
        target = e["target"].to(out.device)
        loss = -(target * torch.log_softmax(out, -1)).sum()
    else:
        label = torch.tensor([e["label"]], device=out.device)
        loss = torch.nn.functional.cross_entropy(out[None], label, label_smoothing=args.label_smoothing)
        target = torch.nn.functional.one_hot(label[0], len(out))
    if args.brier:  # squared error of the probabilities: rewards calibrated confidence, not only the right argmax
        loss = loss + args.brier * ((torch.softmax(out.float(), -1) - target) ** 2).sum()
    return loss


def tally(stats, e, out):
    """Correct/seen per source/kind, plus recall per gold operation (how often WAIT, DONE, ... are chosen)."""
    right = int(out.argmax().item() == e["label"])
    for key in (f"{e['source']}/{e['kind']}",) + ((f"op/{e['keys'][e['label']]}",) if e["kind"] == "operation" else ()):
        stats[key][0] += right
        stats[key][1] += 1
    return right


@torch.inference_mode()
def evaluate(model, rows, args):
    model.eval()
    stats, text = collections.defaultdict(lambda: [0, 0]), []
    for batch in pack(rows, args.batch_tokens):
        for e, out in zip(batch, model(batch), strict=True):
            if "prompt_len" in e:
                text.append(out.item())
            else:
                tally(stats, e, out)
    model.train()
    right = sum(v[0] for k, v in stats.items() if not k.startswith("op/"))
    seen = sum(v[1] for k, v in stats.items() if not k.startswith("op/"))
    return {"acc": round(right / max(seen, 1), 4), "text_loss": round(sum(text) / max(len(text), 1), 4),
            "by": {k: round(v[0] / v[1], 3) for k, v in sorted(stats.items())}}


def save(model, optimizer, scheduler, step, micro, path, args):
    tmp = Path(str(path) + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.lm.save_pretrained(tmp)
    torch.save(model.head.state_dict(), tmp / "head.pt")
    # ponytail: --full skips the optimizer state (32 GB for 4B); a resumed full run restarts Adam's moments.
    torch.save({"optimizer": None if args.full else optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "step": step, "micro": micro, "rng": random.getstate()}, tmp / "trainer.pt")
    (tmp / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3.5-4B")
    p.add_argument("--train", required=True, help="JSONL: choice rows and text rows (see prepare.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--full", action="store_true", help="full fine-tuning instead of LoRA")
    p.add_argument("--lr", type=float, default=None, help="default 1e-4 for LoRA, 1e-5 for --full")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--accum", type=int, default=16, help="rows per optimizer step, per GPU")
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--batch-tokens", type=int, default=32768,
                   help="padded tokens per forward pass; rows of one optimizer step are packed up to this")
    p.add_argument("--image-tokens", type=int, default=400)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--text-weight", type=float, default=1.0, help="multiplies text-row losses (text rows are few)")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="for hard-label choice rows")
    p.add_argument("--head", choices=["candidate", "yesno"], default="candidate",
                   help="yesno: option scores start from the base model's Yes-minus-No logit (Open-Jev's readout)")
    p.add_argument("--brier", type=float, default=0.0, help="adds this weight x Brier score to hard-label choice rows")
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="keep activations instead of recomputing them: faster, more memory (lower --batch-tokens)")
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (speed probes)")
    p.add_argument("--val", help="JSONL scored every --eval-every steps (per source/kind accuracy)")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-rows", type=int, default=300)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="row-encoding processes")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    distributed = "WORLD_SIZE" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if distributed else (0, 1)
    if distributed:
        dist.init_process_group("nccl")
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)
    random.seed(args.seed + rank)
    run_dir = Path(args.out)  # not `out`: loop variables below must not shadow the output folder

    pool = ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_init_worker, initargs=(args.base,))
    args.lr = args.lr or (1e-5 if args.full else 1e-4)
    checkpoint = run_dir / "checkpoint"
    resume_full = args.full and args.resume and checkpoint.exists()
    lm = Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint if resume_full else args.base,
                                                         dtype=torch.float32 if args.full else torch.bfloat16)
    if not args.no_grad_ckpt:  # recomputes every layer's forward in backward: ~1/3 slower, far less activation memory
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.full:
        for name, param in lm.named_parameters():
            param.requires_grad = "language_model" in name and "embed_tokens" not in name
    else:
        lm = get_peft_model(lm, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, target_modules=LORA_TARGETS))
    model = S1(lm, args.head).to(device)
    if args.head == "yesno" and not args.resume:  # a resumed run loads its trained head below
        from transformers import AutoTokenizer

        model.head.init_from_lm(model.generator().lm_head.weight, *yes_no_ids(AutoTokenizer.from_pretrained(args.base)))
    if rank == 0:
        count = sum(q.numel() for q in lm.parameters() if q.requires_grad)
        print(f"trainable language-model parameters: {count:,} of {sum(q.numel() for q in lm.parameters()):,}")

    offsets = read_rows(args.train)
    per_epoch = len(offsets) // (args.accum * world)
    total_steps = max(1, math.ceil(per_epoch * args.epochs))
    lm_params = [q for q in lm.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW([{"params": lm_params, "lr": args.lr},
                                   {"params": model.head.parameters(), "lr": args.head_lr}], weight_decay=0.0)
    warmup = max(1, int(total_steps * args.warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(
        0.1, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))

    step, micro = 0, 0
    if args.resume and checkpoint.exists():
        if not args.full:
            set_peft_model_state_dict(lm, load_file(checkpoint / "adapter_model.safetensors"))
        model.head.load_state_dict(torch.load(checkpoint / "head.pt", map_location=device))
        state = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=False)
        if state["optimizer"]:
            optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, micro = state["step"], state["micro"]
        random.setstate(state["rng"])
        if rank == 0:
            print(f"resumed at optimizer step {step}", flush=True)

    eval_rows = []
    if args.val and rank == 0:
        val_offsets = read_rows(args.val)
        random.Random(0).shuffle(val_offsets)
        jobs = [(row_at(args.val, o), args.max_len, args.image_tokens, o) for o in val_offsets[: args.eval_rows]]
        eval_rows = [e for e in pool.map(_encode, jobs) if e is not None]
        print(f"{len(eval_rows)} validation rows for --eval-every {args.eval_every}", flush=True)

    # Text rows never touch the decision head, so DDP must tolerate unused parameters.
    trainable = (torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
                 if distributed else model)
    model.train()
    if rank == 0:
        rows_per_step = args.accum * world
        print(f"{len(offsets)} rows, {world} GPU(s), {total_steps} optimizer steps of {rows_per_step} rows", flush=True)

    loss_sum = correct = seen = skipped = tokens = text_loss_sum = text_seen = 0
    stats = collections.defaultdict(lambda: [0, 0])
    started, start_step, fallback = time.time(), step, None
    stop = min(total_steps, start_step + args.max_steps) if args.max_steps else total_steps
    last_log = (started, 0)  # (time, tokens) at the previous log line, for the recent speed
    while step < stop:
        epoch = micro // (per_epoch * args.accum)
        order = list(range(len(offsets)))
        random.Random(args.seed + epoch).shuffle(order)  # same order on every rank; each takes its slice
        order = order[rank::world][: per_epoch * args.accum]
        todo = order[micro % len(order):]
        windows = [todo[i:i + args.accum] for i in range(0, len(todo), args.accum)]  # one optimizer step each

        def submit(window, epoch=epoch):  # the next step's rows encode in the workers while this step trains
            return [pool.submit(_encode, (row_at(args.train, offsets[i]), args.max_len, args.image_tokens,
                                          hash((args.seed, epoch, i)))) for i in window]

        pending = submit(windows[0]) if windows else []
        for number_window, window in enumerate(windows):
            futures, pending = pending, (submit(windows[number_window + 1]) if number_window + 1 < len(windows)
                                         else [])
            rows = [f.result() for f in futures]
            skipped += sum(e is None for e in rows)
            rows = [e for e in rows if e is not None]
            micro += len(window)
            if rows:
                fallback = rows[-1]
            elif fallback is not None:  # every DDP rank must still run a synced backward: a zero-weight row
                rows = [{**fallback, "weight": 0.0}]
            batches = pack(rows, args.batch_tokens)
            for number, batch in enumerate(batches):
                last = number == len(batches) - 1
                sync = trainable.no_sync() if distributed and not last else contextlib.nullcontext()
                with sync, torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.full):
                    outputs = trainable(batch)
                    losses = [row_loss(e, out, args) for e, out in zip(batch, outputs, strict=True)]
                    (sum(loss * e.get("weight", 1.0) for loss, e in zip(losses, batch, strict=True))
                     / args.accum).backward()
                for e, out, loss in zip(batch, outputs, losses, strict=True):
                    if not e.get("weight", 1.0):
                        continue
                    tokens += e["input_ids"].shape[0]
                    if "prompt_len" in e:
                        text_loss_sum += out.item()  # unweighted, comparable across --text-weight
                        text_seen += 1
                    else:
                        loss_sum += loss.item()
                        correct += tally(stats, e, out)
                        seen += 1
            torch.nn.utils.clip_grad_norm_(lm_params + list(model.head.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and step % args.log_every == 0:
                elapsed = time.time() - started
                print(json.dumps({"step": step, "of": total_steps,
                                  "choice_loss": round(loss_sum / max(seen, 1), 4),
                                  "choice_acc": round(correct / max(seen, 1), 4),
                                  "text_loss": round(text_loss_sum / max(text_seen, 1), 4), "skipped": skipped,
                                  "tokens_per_s_per_gpu": round(tokens / elapsed),
                                  "recent_tokens_per_s": round((tokens - last_log[1]) / (time.time() - last_log[0])),
                                  "lr": scheduler.get_last_lr()[0],
                                  "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                                  "eta_h": round(elapsed / (step - start_step) * (total_steps - step) / 3600, 2),
                                  "acc_by": {k: round(v[0] / v[1], 2) for k, v in sorted(stats.items())}}),
                      flush=True)
                loss_sum = correct = seen = text_loss_sum = text_seen = 0
                stats.clear()
                last_log = (time.time(), tokens)
            if rank == 0 and eval_rows and (step % args.eval_every == 0 or step == total_steps):
                print(json.dumps({"eval_step": step, **evaluate(model, eval_rows, args)}), flush=True)
            if rank == 0 and step % args.save_every == 0:
                save(model, optimizer, scheduler, step, micro, checkpoint, args)
            if step >= stop:
                break
        if distributed:
            dist.barrier()

    if rank == 0:
        save(model, optimizer, scheduler, step, micro, run_dir / "final", args)
        print(f"saved {run_dir / 'final'}; next: python calibrate.py --adapter {run_dir / 'final'} --val <val.jsonl>",
              flush=True)
    if distributed:
        dist.destroy_process_group()
    # The next window's rows may still be queued in the encoding pool when the last step ends mid-epoch; waiting for
    # the pool at interpreter exit hung the v3 run for over an hour after its final save. Drop them and leave at once.
    pool.shutdown(wait=False, cancel_futures=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
