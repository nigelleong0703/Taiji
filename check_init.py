"""train.py --init loads a trained LoRA run cleanly: every adapter tensor lands on a model parameter (CPU is enough).
python check_init.py Qwen/Qwen3.5-2B <run folder>"""
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from s1 import S1
from train import LORA_TARGETS

base, run = sys.argv[1], Path(sys.argv[2])
lm = get_peft_model(Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16),
                    LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS))
weights = load_file(run / "adapter_model.safetensors")
before = {n: p.detach().clone() for n, p in lm.named_parameters() if "lora_B" in n}
result = set_peft_model_state_dict(lm, weights)
changed = sum(not torch.equal(before[n], p) for n, p in lm.named_parameters() if n in before)
model = S1(lm, "yesno")
model.head.load_state_dict(torch.load(run / "head.pt", map_location="cpu"))
print(f"adapter tensors {len(weights)}, unexpected keys {len(result.unexpected_keys)}, lora_B tensors changed "
      f"{changed}/{len(before)}; head loaded")
assert not result.unexpected_keys and changed == len(before), "adapter did not load cleanly"
print("INIT_OK")
