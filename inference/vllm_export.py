"""Prepare a Taiji checkpoint for vLLM: merge the LoRA, select the Taiji class, package the head.

vLLM loads the merged weights through the stock `Qwen3_5ForConditionalGeneration` backbone, but
the `YesNoHead` is not part of that model. This script:

  1. merges the LoRA into the base weights,
  2. loads `head.pt` and writes an fp32 head bundle (`taiji_head.pt`) for the Taiji scorer,
  3. sets `architectures = ["TaijiForVLLM"]` in the exported `config.json` so vLLM selects the
     Taiji class from the bundled `vllm_model.py` plugin,
  4. writes `taiji_config.json` with the marker token id, marker token, fitted temperatures and
     shard metadata, and validates the saved checkpoint.

    python inference/vllm_export.py --base Qwen/Qwen3.5-2B --adapter Taiji-2B \
        --out taiji-vllm --dtype bfloat16

The merged directory is a normal Hugging Face model plus `taiji_config.json` and
`taiji_head.pt`; point the server at it with `--model taiji-vllm`.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm_head import DEFAULT_HEAD, build_head
from vllm_prompt import MARKER

TAIJI_ARCH = "TaijiForVLLM"


def _set_architectures(model_dir, arch=TAIJI_ARCH):
    """Point the exported config at the Taiji class so vLLM loads the trained head."""
    config_path = Path(model_dir) / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = [arch]
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config


def _validate(model_dir, expected_arch=TAIJI_ARCH):
    """Sanity-check the saved checkpoint: config, single shard index, and readable weights."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    shards = []
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = sorted(p.name for p in model_dir.glob("*.safetensors"))
    missing = [s for s in shards if not (model_dir / s).exists()]
    if missing:
        raise SystemExit(f"Checkpoint is incomplete; missing weight shards: {missing}")
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("architectures") != [expected_arch]:
        raise SystemExit(
            f"config.json architectures is {config.get('architectures')}, expected "
            f"['{expected_arch}']; vLLM would not pick the Taiji class."
        )
    return {"architectures": config["architectures"], "shards": shards}


def export(base, adapter, out, dtype="bfloat16"):
    adapter = Path(adapter)
    head_state = adapter / "head.pt"
    if not head_state.is_file():
        raise SystemExit(f"No head.pt in {adapter}; cannot package the trained decision head.")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"output directory must be empty: {out}")
    processor = AutoProcessor.from_pretrained(base)
    tokenizer = processor.tokenizer

    torch_dtype = getattr(torch, dtype)
    is_lora = (Path(adapter) / "adapter_config.json").exists()
    if is_lora:
        lm = Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch_dtype)
        from peft import PeftModel

        lm = PeftModel.from_pretrained(lm, adapter)
        lm = lm.merge_and_unload()
        print("Merged LoRA into the base weights for vLLM.")
    else:
        # Full fine-tunes must load their own weights; exporting the base here would be wrong.
        lm = Qwen3_5ForConditionalGeneration.from_pretrained(adapter, dtype=torch_dtype)
        print("Loaded full fine-tune weights for vLLM.")

    lm.config.architectures = [TAIJI_ARCH]  # so vLLM selects TaijiForVLLM
    lm.save_pretrained(out)
    processor.save_pretrained(out)
    _set_architectures(out)  # save_pretrained may rewrite architectures
    info = _validate(out)
    print(f"Saved merged model to {out} ({len(info['shards'])} shard(s), arch {TAIJI_ARCH}).")

    train_args = adapter / "train_args.json"
    head_kind = DEFAULT_HEAD
    if train_args.exists():
        head_kind = json.loads(train_args.read_text()).get("head", DEFAULT_HEAD)
    hidden_size = int(lm.config.text_config.hidden_size)

    head = build_head(head_kind, hidden_size)
    head.load_state_dict(torch.load(head_state, map_location="cpu", weights_only=True))
    head.eval()
    head.to(dtype=torch.float32)  # bf16 rounding at the read-out flips near-ties
    torch.save(head.state_dict(), out / "taiji_head.pt")

    temperature = {"temperature": 1.0, "by_kind": {}}
    fitted = Path(adapter) / "temperature.json"
    if fitted.exists():
        temperature = json.loads(fitted.read_text())

    config = {
        "base_model": base,
        "source_adapter": str(adapter),
        "head": head_kind,
        "hidden_size": hidden_size,
        "architectures": TAIJI_ARCH,
        "head_file": "taiji_head.pt",
        "marker_token": MARKER,
        "marker_token_id": tokenizer.convert_tokens_to_ids(MARKER),
        "yes_token_id": tokenizer.convert_tokens_to_ids("Yes"),
        "no_token_id": tokenizer.convert_tokens_to_ids("No"),
        "temperature": temperature.get("temperature", 1.0),
        "temperature_by_kind": temperature.get("by_kind", {}),
        "merged": True,
        "shards": info.get("shards", []),
        "multimodal": False,  # the Taiji vLLM backend scores text states only
    }
    (out / "taiji_config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(f"Wrote head bundle to {out} (head={head_kind}, hidden={hidden_size})")
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="Qwen/Qwen3.5-2B")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()
    export(args.base, args.adapter, args.out, args.dtype)


if __name__ == "__main__":
    main()
