"""Public human-labelled decision data -> S1 rows (choice, noul, score), for general ability and replay.

The same sources Decision-1.0 used, plus Yelp for ordered scores. Research use: check each license before
any other use (MultiNLI, SNLI and SQuAD 2.0 are CC BY-SA; Yelp's terms are non-commercial).

  banking77, clinc150   choice: 5-20 intent names sampled per question (labels defined at request time)
  multinli, snli        choice entailment/neutral/contradiction; some MultiNLI rows as noul "does it follow"
  squad_v2              noul: can the context answer the question
  cosmos_qa             choice over four answers
  yelp                  score: 1-5 stars
Downloads are cached under --cache; official train/validation splits are kept.
"""

import argparse
import csv
import io
import json
import random
import time
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets/{}/resolve/refs%2Fconvert%2Fparquet/{}"
SOURCES = {
    "banking77": {"train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
                  "val": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"},
    "clinc150": {"train": HF.format("clinc/clinc_oos", "plus/train/0000.parquet"),
                 "val": HF.format("clinc/clinc_oos", "plus/validation/0000.parquet"),
                 "info": "https://datasets-server.huggingface.co/info?dataset=clinc/clinc_oos&config=plus"},
    "multinli": {"train": HF.format("nyu-mll/multi_nli", "default/train/0000.parquet"),
                 "val": HF.format("nyu-mll/multi_nli", "default/validation_matched/0000.parquet")},
    "snli": {"train": HF.format("stanfordnlp/snli", "plain_text/train/0000.parquet"),
             "val": HF.format("stanfordnlp/snli", "plain_text/validation/0000.parquet")},
    "squad_v2": {"train": HF.format("rajpurkar/squad_v2", "squad_v2/train/0000.parquet"),
                 "val": HF.format("rajpurkar/squad_v2", "squad_v2/validation/0000.parquet")},
    "cosmos_qa": {"train": "https://raw.githubusercontent.com/wilburOne/cosmosqa/master/data/train.csv",
                  "val": "https://raw.githubusercontent.com/wilburOne/cosmosqa/master/data/valid.csv"},
    "yelp": {"train": HF.format("Yelp/yelp_review_full", "yelp_review_full/train/0000.parquet"),
             "val": HF.format("Yelp/yelp_review_full", "yelp_review_full/test/0000.parquet")},
}
NLI = {"entailment": "The hypothesis must be true given the context.",
       "neutral": "The hypothesis might or might not be true given the context.",
       "contradiction": "The hypothesis cannot be true given the context."}
STARS = ["1 star: very negative", "2 stars: negative", "3 stars: mixed or neutral", "4 stars: positive",
         "5 stars: very positive"]


def fetch(url, cache):
    path = cache / url.split("//", 1)[1].replace("/", "_")
    if not path.exists():
        for attempt in range(5):
            try:
                path.write_bytes(urllib.request.urlopen(url, timeout=300).read())
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    return path


def table(url, cache):
    path = fetch(url, cache)
    if url.endswith(".csv"):
        return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def humanize(label):
    return label.replace("_", " ")


def intent_row(rnd, text, gold, names, fixed=None):
    others = [n for n in names if n != gold and n != fixed]
    options = rnd.sample(others, min(len(others), rnd.randint(4, 19))) + [gold]
    if fixed and fixed not in options:
        options.append(fixed)
    rnd.shuffle(options)
    criteria = {n: ("None of the other options; the request is out of scope." if n == fixed else humanize(n))
                for n in options}
    return {"state": text, "question": {"type": "choice", "instructions": "Which intent does the message express?",
                                        "criteria": criteria}, "label": gold}


def rows(name, records, rnd, info):
    for r in records:
        if name == "banking77":
            yield intent_row(rnd, r["text"], r["category"], info["names"])
        elif name == "clinc150":
            label = info["names"][r["intent"]]
            yield intent_row(rnd, r["text"], "out_of_scope" if label == "oos" else label,
                             [n for n in info["names"] if n != "oos"] + ["out_of_scope"], fixed="out_of_scope")
        elif name in ("multinli", "snli"):
            if r["label"] not in (0, 1, 2):
                continue
            gold = ["entailment", "neutral", "contradiction"][r["label"]]
            if name == "multinli" and gold != "neutral" and rnd.random() < 0.5:
                yield {"state": r["premise"], "question": {"type": "noul", "instructions": r["hypothesis"],
                       "criteria": {"true": "The statement follows from the context.",
                                    "false": "The statement contradicts the context."}},
                       "label": str(gold == "entailment").lower()}
            else:
                yield {"state": r["premise"], "question": {"type": "choice", "criteria": NLI,
                       "instructions": f"Hypothesis: {r['hypothesis']}\nHow does the context relate to it?"},
                       "label": gold}
        elif name == "squad_v2":
            yield {"state": f"{r['title']}\n{r['context']}", "question": {"type": "noul", "instructions": r["question"],
                   "criteria": {"true": "The context contains the answer to the question.",
                                "false": "The context does not answer the question."}},
                   "label": str(bool(r["answers"]["text"])).lower()}
        elif name == "cosmos_qa":
            yield {"state": r["context"], "question": {"type": "choice", "instructions": r["question"],
                   "criteria": {str(i + 1): r[f"answer{i}"] for i in range(4)}}, "label": str(int(r["label"]) + 1)}
        elif name == "yelp":
            yield {"state": r["text"][:4000], "question": {"type": "score", "criteria": STARS,
                   "instructions": "How positive is this review?"}, "label": str(r["label"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default="cache")
    p.add_argument("--per-source", type=int, default=3000, help="training rows per source")
    p.add_argument("--val-per-source", type=int, default=300)
    p.add_argument("--sources", nargs="+", default=list(SOURCES))
    args = p.parse_args()
    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    with open(out / "general_train.jsonl", "w") as train, open(out / "general_val.jsonl", "w") as val:
        for name in args.sources:
            source, rnd = SOURCES[name], random.Random(name)
            info = {}
            if name == "banking77":
                info["names"] = sorted({r["category"] for r in table(source["train"], cache)})
            if "info" in source:
                features = json.loads(fetch(source["info"], cache).read_text())["dataset_info"]["features"]
                info["names"] = features["intent"]["names"]
            counts = {}
            for split, sink, limit in (("train", train, args.per_source), ("val", val, args.val_per_source)):
                records = table(source[split], cache)
                rnd.shuffle(records)
                n = 0
                for row in rows(name, records, rnd, info):
                    if n >= limit:
                        break
                    row["host"] = name
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
                counts[split] = n
            print(json.dumps({"source": name, **counts}), flush=True)


if __name__ == "__main__":
    main()
