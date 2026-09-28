"""client.py without a GPU or paid API: both endpoints are stubbed."""

import json

import httpx
from client import S1, S2

QUESTION = {"type": "choice", "instructions": "Which light is shown?", "criteria": {"red": "Red", "green": "Green"}}


def stub(s1_confidence, s2_reply):
    def handle(request):
        if request.url.host == "s2":
            return httpx.Response(200, json={"choices": [{"message": {"content": s2_reply}}]})
        if request.url.path == "/v1/systemone":
            return httpx.Response(200, json={"answers": {"q": {"type": "choice", "choice": "green",
                                                                "confidence": s1_confidence}}})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"text": "Singapore"})}}]})
        return httpx.Response(404)
    return httpx.MockTransport(handle)


def pair(s1_confidence, s2_reply):
    s1, s2 = S1("http://s1", "k"), S2("http://s2/v1", "k", "m")
    s1.client = s2.client = httpx.Client(transport=stub(s1_confidence, s2_reply))
    return s1, s2


def test_confident_s1_is_not_escalated():
    s1, s2 = pair(0.9, '{"choice": "red"}')
    assert s1.decide("The light is red.", {"q": QUESTION}, s2=s2)["q"]["choice"] == "green"


def test_unsure_s1_takes_s2_choice():
    s1, s2 = pair(0.1, '{"choice": "red"}')
    answer = s1.decide("The light is red.", {"q": QUESTION}, s2=s2)["q"]
    assert answer["choice"] == "red" and answer["escalated"] and answer["s1_choice"] == "green"


def test_invalid_s2_answer_keeps_s1():
    for reply in ('{"choice": "blue"}', "not json", '{"answer": "red"}'):
        s1, s2 = pair(0.1, reply)
        assert s1.decide("The light is red.", {"q": QUESTION}, s2=s2)["q"]["choice"] == "green"


def test_write_returns_the_text():
    s1, _ = pair(0.9, "")
    assert s1.write({"goal": "Fly from Singapore", "field": {"label": "Where from?"}}) == "Singapore"
