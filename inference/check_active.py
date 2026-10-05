"""Report which fast paths an S1 host will ACTUALLY use, not merely which packages are installed.

    python3 check_active.py                                  # versions and import availability
    python3 check_active.py --base <dir> --adapter <dir>      # also loads the model and prints the
                                                              # attention backend transformers resolved

The distinction matters: transformers auto-detects flash-linear-attention and causal_conv1d for
Qwen3.5, but flash-attn is only used when attn_implementation="flash_attention_2" is requested at
load time. Installing a package is not the same as activating it.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys


def report_versions():
    print("python      ", platform.python_version())
    try:
        import torch
        print("torch       ", torch.__version__, "| built for cuda", torch.version.cuda)
        print("cuda        ", "available" if torch.cuda.is_available() else "not available",
              "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
        print("mps         ", "available" if getattr(torch.backends, "mps", None) and
              torch.backends.mps.is_available() else "not available")
    except ImportError:
        print("torch        not installed")
        return
    try:
        import transformers
        print("transformers", transformers.__version__)
    except ImportError:
        print("transformers not installed")


def report_kernels():
    """Installed is not the same as active; state the rule for each."""
    rules = {
        "flash_attn": "used only when attn_implementation='flash_attention_2' is passed at load",
        "causal_conv1d": "auto-detected by the Qwen3.5 implementation when importable",
        "fla": "auto-detected by the Qwen3.5 implementation when importable",
    }
    for name, rule in rules.items():
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "?")
            print(f"{name:<14} installed {version:<10} -> {rule}")
        except Exception as error:  # noqa: BLE001 - report any import failure verbatim
            print(f"{name:<14} NOT installed      -> {rule}")
            print(f"{'':<14}   ({type(error).__name__}: {str(error)[:70]})")


def report_resolved(base, adapter, device):
    """Load exactly as serve.py does and print the attention backend that was resolved."""
    sys.path.insert(0, ".")
    from s1 import load
    processor, model = load(base, adapter, device=device)
    # load() returns the S1 wrapper; the Hugging Face model hangs off .lm (PEFT already merged).
    inner = getattr(model, "lm", None) or model
    config = getattr(inner, "config", None)
    resolved = getattr(config, "_attn_implementation", None)
    print()
    print("loaded adapter/base :", adapter or base)
    print("attn implementation :", resolved,
          "  <- 'flash_attention_2' means flash-attn is actually in use" if resolved == "flash_attention_2"
          else "  <- flash-attn is NOT in use on this host")
    return processor, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    args = parser.parse_args()

    report_versions()
    print()
    report_kernels()
    if args.base:
        report_resolved(args.base, args.adapter, args.device)
    else:
        print()
        print("pass --base <dir> [--adapter <dir>] to also load the model and print the resolved backend")


if __name__ == "__main__":
    main()
