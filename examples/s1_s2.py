"""S1 decides and writes; an unsure decision goes to S2. Run serve.py first.

S1_URL=http://localhost:8000 S1_API_KEY=... [S2_BASE_URL=... S2_API_KEY=... S2_MODEL=...] python examples/s1_s2.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from client import S1, S2  # noqa: E402

s1 = S1(os.environ.get("S1_URL", "http://localhost:8000"), os.environ["S1_API_KEY"])
s2 = S2(os.environ["S2_BASE_URL"], os.environ["S2_API_KEY"], os.environ["S2_MODEL"]) if os.environ.get("S2_MODEL") else None

ticket = "My card was charged twice for the same coffee this morning."
answers = s1.decide(ticket, {
    "intent": {"type": "choice", "instructions": "Which support queue should handle this message?",
               "criteria": {"billing": "Billing and charges", "card": "Lost or blocked card", "other": "Anything else"}},
    "urgent": {"type": "noul", "instructions": "Does the customer need an answer today?",
               "criteria": {"true": "Yes, today", "false": "No"}},
}, s2=s2, threshold=0.5)
print("intent:", answers["intent"]["choice"], f"(confidence {answers['intent']['confidence']:.2f})",
      "via S2" if answers["intent"].get("escalated") else "")
print("urgent:", round(answers["urgent"]["noul"], 2))
print("reply subject:", s1.write({"goal": "Reply to the customer about the double charge.",
                                  "field": {"label": "Subject", "role": "textbox", "value": ""}}))
