"""Valen UI screens (Valen-Team datasets) -> S1 rows, soft labels kept.

Train rows from Valen-Training-General-100k, val rows from Valen-Eval-General-5k (Valen's own split). Only the
screen sources by default: RICO-ScreenQA (mobile apps, CC-BY-4.0) and ShowUI desktop (upstream ShowUI/OmniAct
terms). One row per question: state = the message text, image = the extracted screenshot, target = Valen's
probabilities. Only the images these rows use are extracted from assets.zip.

python prepare_valen.py --out $S1_HOME/data --images $S1_HOME/images/valen --cache $SCRATCH/valen
Writes valen_ui_{train,val}.jsonl.
"""

import argparse
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPOS = {"train": "Valen-Team/Valen-Training-General-100k", "val": "Valen-Team/Valen-Eval-General-5k"}


def convert(record, source, images):
    content = record["request"]["state"]["messages"][0]["content"]
    shots = [c["image_url"]["url"] for c in content if c["type"] == "image_url"]
    if len(shots) != 1:  # one screenshot per row, like every other image row
        return
    text = "\n".join(c["text"] for c in content if c["type"] == "text")
    for name, question in record["request"]["questions"].items():
        probabilities = record.get("targets", {}).get(name, {}).get("probabilities")
        if not probabilities:
            continue
        if question["type"] == "noul" and not question.get("criteria"):
            question = {**question, "criteria": {"true": "yes", "false": "no"}}
        yield {"state": text, "question": question, "target": probabilities, "image": str(images / shots[0]),
               "source": f"valen_{source}", "host": record["group_id"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--sources", nargs="+", default=["rico_screenqa", "showui_desktop"])
    args = p.parse_args()
    out, images = Path(args.out), Path(args.images)
    out.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    for split, repo in REPOS.items():
        rows, needed = [], set()
        for source in args.sources:
            path = hf_hub_download(repo, f"{source}.jsonl", repo_type="dataset", local_dir=Path(args.cache) / split)
            with open(path) as f:
                for line in f:
                    for row in convert(json.loads(line), source, images):
                        rows.append(row)
                        needed.add(str(Path(row["image"]).relative_to(images)))
        archive = hf_hub_download(repo, "assets.zip", repo_type="dataset", local_dir=Path(args.cache) / split)
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.namelist() if m in needed and not (images / m).exists()]
            z.extractall(images, members)
        missing = [r for r in rows if not Path(r["image"]).exists()]
        rows = [r for r in rows if Path(r["image"]).exists()]
        with open(out / f"valen_ui_{split}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"split": split, "rows": len(rows), "images": len(needed), "missing_images": len(missing),
                          "by_source": {s: sum(r["source"] == f"valen_{s}" for r in rows) for s in args.sources}}),
              flush=True)


if __name__ == "__main__":
    main()
