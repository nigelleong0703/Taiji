"""Synthetic rows that prove the pipeline learns, not training data.

choice: pick the option named in the goal (tests the decision head and option shuffling)
vision: name the colour of the screenshot (only solvable if the image reaches the model)
noul:   is "the screen is <colour>" true? (noul type, also needs the image)
score:  rate the priority stated in the text on low/medium/high (score type, ordered levels)
text:   type the origin city from the goal (tests the LM head)
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image

WORDS = "Search Submit Cancel Login Next Back Filters Reset Apply Save Delete Share Menu Help Close Open".split()
COLOURS = {"red": (220, 30, 30), "green": (30, 180, 60), "blue": (30, 60, 220)}
CITIES = "Zurich London Paris Berlin Madrid Rome Vienna Prague Lisbon Oslo Tokyo Seoul".split()


def row(rnd, images):
    kind = rnd.choice(["choice", "vision", "noul", "score", "text"])
    if kind == "text":
        origin, destination = rnd.sample(CITIES, 2)
        goal = f"Find flights from {origin} to {destination}."
        context = {"goal": goal, "field": {"label": "Where from?", "role": "combobox", "value": ""},
                   "page": {"title": "Flights", "text": "Where from? Where to?"}, "recent_actions": []}
        return {"task": "text", "context": context, "target": origin, "host": rnd.choice(["a.test", "b.test"])}
    if kind in ("noul", "score"):
        base = {"page": {"url": "https://a.test/", "title": "Test", "text": ""}, "elements": [], "recent_actions": []}
        if kind == "noul":
            actual, claimed = rnd.choice(list(COLOURS)), rnd.choice(list(COLOURS))
            question = {"type": "noul", "instructions": f"The screen is {claimed}.",
                        "criteria": {"true": "The statement matches the screen", "false": "It does not"}}
            return {"state": base, "question": question, "label": str(actual == claimed).lower(),
                    "image": str(images / f"{actual}.png"), "host": rnd.choice(["a.test", "b.test"])}
        level = rnd.randrange(3)
        base["page"]["text"] = f"Ticket priority: {['low', 'medium', 'high'][level]}."
        question = {"type": "score", "instructions": "How urgent is this ticket?",
                    "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"]}
        return {"state": base, "question": question, "label": str(level), "host": rnd.choice(["a.test", "b.test"])}
    if kind == "vision":
        colour = rnd.choice(list(COLOURS))
        path = images / f"{colour}.png"
        criteria = {c: {"element": f"The screen is {c}"} for c in COLOURS}
        goal = "Report the colour of the screen."
    else:
        options = rnd.sample(WORDS, rnd.randint(3, 8))
        colour, path = rnd.choice(options), None
        criteria = {str(i + 1): {"element": f"[{i + 1}] {w}", "role": "button"} for i, w in enumerate(options)}
        goal = f"Click the {colour} button."
        colour = str(options.index(colour) + 1)
    row = {"state": {"page": {"url": "https://a.test/", "title": "Test", "text": goal}, "elements": [],
                     "recent_actions": []},
           "question": {"type": "choice", "criteria": criteria, "instructions": {"goal": goal}},
           "label": colour, "host": rnd.choice(["a.test", "b.test"])}
    if path:
        row["image"] = str(path)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="smoke")
    p.add_argument("--rows", type=int, default=600)
    args = p.parse_args()
    out = Path(args.out)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    for name, rgb in COLOURS.items():
        Image.new("RGB", (1120, 780), rgb).save(images / f"{name}.png")
    rnd = random.Random(0)
    for split, n in (("train", args.rows), ("val", args.rows // 5)):
        with open(out / f"{split}.jsonl", "w") as f:
            for _ in range(n):
                f.write(json.dumps(row(rnd, images.resolve())) + "\n")
    print(f"wrote {out}/train.jsonl and {out}/val.jsonl")


if __name__ == "__main__":
    main()
