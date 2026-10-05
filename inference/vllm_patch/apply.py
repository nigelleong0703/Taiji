"""Apply or restore the pinned vLLM 0.29.0 source changes.

Run with the Python executable belonging to the vLLM environment, while its engine
is stopped. The installer checks every patch hunk before changing any source file.
"""

import argparse
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    package = distribution("vllm")
    if package.version != "0.29.0":
        raise SystemExit(f"Expected vLLM 0.29.0, found {package.version}")
    root = Path(package.locate_file("vllm")).resolve()
    manifest_path = root / ".taiji-source-patch.json"
    backup = root / ".taiji-source-patch-backup"
    directory = Path(__file__).resolve().parent
    patch_path = directory / "patches" / "shared-engine.patch"
    helper_relative = "model_executor/layers/taiji_readout.py"
    helper_target = root / helper_relative

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        altered = [name for name, value in manifest["installed"].items()
                   if not (root / name).is_file() or digest(root / name) != value]
        if altered:
            raise SystemExit(f"Installed source changed since patch: {altered}")
        if not args.restore:
            print("Taiji source patch v1 is already installed and verified")
            return
        for name in manifest["original"]:
            shutil.copy2(backup / name, root / name)
        helper_target.unlink()
        manifest_path.unlink()
        shutil.rmtree(backup)
        print("Restored original vLLM source")
        return
    if args.restore:
        raise SystemExit("No Taiji source patch is installed")
    if backup.exists() or helper_target.exists():
        raise SystemExit("Untracked Taiji patch files exist; inspect before installing")
    patch_binary = shutil.which("patch")
    if patch_binary is None:
        raise SystemExit("The system patch command is required")
    command = [patch_binary, "--batch", "--forward", "-p1", "-i", str(patch_path)]
    subprocess.run([*command, "--dry-run"], cwd=root.parent, check=True)
    names = [line.removeprefix("--- a/vllm/").split("\t")[0]
             for line in patch_path.read_text().splitlines() if line.startswith("--- a/vllm/")]
    original = {}
    for name in names:
        source = root / name
        target = backup / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original[name] = digest(source)
        shutil.copy2(source, target)
    try:
        subprocess.run(command, cwd=root.parent, check=True)
        shutil.copy2(directory / "taiji_readout.py", helper_target)
        installed = {name: digest(root / name) for name in [*names, helper_relative]}
        manifest_path.write_text(json.dumps({"version": 1, "original": original,
                                             "installed": installed}, indent=2) + "\n")
    except BaseException:
        for name in names:
            shutil.copy2(backup / name, root / name)
        helper_target.unlink(missing_ok=True)
        shutil.rmtree(backup)
        raise
    print("Installed Taiji source patch v1: one generate engine, decision readout, shared cache")


if __name__ == "__main__":
    main()
