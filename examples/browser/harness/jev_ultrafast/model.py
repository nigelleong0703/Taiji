"""TypeSafe makes choices; an OpenAI-compatible model writes field values and, when stuck, a subgoal."""

import base64
import io
import json
import math
import os
import time

import httpx
from PIL import Image

from .questions import NEXT_ACTION, REFLECT, TARGET, TEXT_VALUE, TIE

CLIENT = httpx.Client(http2=True, timeout=25)
# A self-hosted S1 (training/serve.py) speaks the same protocol; point TYPESAFE_URL at it.
TYPESAFE_URL = os.environ.get("TYPESAFE_URL", "https://api.typesafe.ai/v1/systemone")
# ponytail: calibrated on 1 wrong step (0.19) vs 118 correct (min 0.24); recalibrate as traces grow.
TIE_MARGIN = 0.25
# A self-hosted S1 often ranks DONE second on finished pages (training/calibrate.py's DONE report: at 0.1, DONE recall
# 0.57 -> 0.81 at precision 0.78 on held-out rows). Set DONE_FLOOR for such a model; unset, the model's choice stands.
DONE_FLOOR = float(os.environ.get("DONE_FLOOR") or 0)


def post_json(url, key, body, headers=None):
    for attempt in range(3):
        try:
            response = CLIENT.post(url, json=body, headers={"Authorization": f"Bearer {key}", **(headers or {})})
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"Model provider returned HTTP {response.status_code}; no action executed.")
        return response.json()
    raise RuntimeError("Model unavailable")


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    return answer


def present(**fields):
    """An attribute the page does not set arrives as an empty string; sending it is pure payload."""
    return {key: value for key, value in fields.items() if value not in (None, "")}


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = present(role=action.get("role"), value=action.get("value"), href=action.get("href"),
                              checked=action.get("checked"), selected=action.get("selected"),
                              expanded=action.get("expanded"))
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


S1_IMAGE_PIXELS = 400 * 32 * 32  # a self-hosted S1 at --image-tokens 400: 16px patches merged 2x2


def shrink(screenshot):
    """The size the S1 server resizes to anyway (qwen-decision s1.load_image), so it uploads a quarter of the bytes."""
    image = Image.open(io.BytesIO(base64.b64decode(screenshot))).convert("RGB")
    scale = math.sqrt(S1_IMAGE_PIXELS / (image.width * image.height))
    if scale >= 1:
        return screenshot
    image = image.resize((max(32, int(image.width * scale)), max(32, int(image.height * scale))))
    out = io.BytesIO()
    image.save(out, "JPEG", quality=72)  # the capture quality (browser.py)
    return base64.b64encode(out.getvalue()).decode()


def build_request(state, goal, history):
    """The S1 request for one observed page. training/prepare_weblinx.py builds rows with this same function."""
    elements, targets, controls = action_space(state["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    **present(current_value=a.get("current_value", a.get("value", "")),
                              role=a.get("role"), href=a.get("href"), checked=a.get("checked"),
                              selected=a.get("selected"), expanded=a.get("expanded")),
                }
                for index, a in candidates.items()
            },
            "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
        }
    if os.environ.get("TYPESAFE_SCREENSHOT") and state.get("screenshot"):
        # Only for a self-hosted vision S1; the hosted Jev API gets structured state only.
        page_state = {"screenshot": shrink(state["screenshot"])}
    else:
        page_state = {}
    body = {
        "model": os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            **page_state,
            # Dropping this copy (every element also appears in its target question) cuts the request by
            # about a third and costs far more than it saves: measured 24 steps / 156 s against 7 / 24 s,
            # because the policy loses the page view and starts scrolling and looking at random.
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    return body, targets, controls, operations


def choose(state, goal, history, cache_session=None, *, relative_tie=False):
    body, targets, controls, operations = build_request(state, goal, history)
    if cache_session:
        # A private namespace lets a self-hosted S1 reuse the stable goal prefix across turns.
        body = {**body, "cache_session": cache_session}
    started = time.perf_counter()
    result = post_json(TYPESAFE_URL, os.environ["TYPESAFE_API_KEY"], body)
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    done_probability = operation_answer["probabilities"].get("DONE", 0)
    done_floor = bool(DONE_FLOOR) and operation != "DONE" and done_probability >= DONE_FLOOR
    if done_floor:
        operation = "DONE"
    target = None
    target_answer = None
    tie_break = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        head = operation.lower() + "_target"
        target_answer = validate_choice(result["answers"].get(head, {}), targets[operation])
        target = target_answer["choice"]
        ranked = sorted(target_answer["probabilities"].items(), key=lambda item: -item[1])
        gap = ranked[0][1] - ranked[1][1] if len(ranked) >= 2 else None
        # The ratio cancels the softmax denominator: adding low-ranked candidates
        # alone cannot trigger a retry. Keep the original harness policy as default.
        if relative_tie and gap is not None:
            gap /= max(ranked[0][1] + ranked[1][1], 1e-12)
        if gap is not None and gap < TIE_MARGIN:
            # Near tie: ask System 1 again with only the two candidates. One extra request, never a loop.
            pair = {ranked[0][0], ranked[1][0]}
            criteria = {index: body["questions"][head]["criteria"][index] for index in pair}
            tie = {"type": "choice", "criteria": criteria, "instructions": {"goal": goal, "rules": [NEXT_ACTION, TIE]}}
            answer = post_json(TYPESAFE_URL, os.environ["TYPESAFE_API_KEY"], {**body, "questions": {"tie": tie}})
            tie_answer = validate_choice(answer["answers"].get("tie", {}), pair)
            tie_break = {"candidates": [ranked[0][0], ranked[1][0]], "first": target, "final": tie_answer["choice"],
                         "probabilities": tie_answer["probabilities"],
                         "initial_probabilities": {k: p for k, p in ranked[:2]},
                         "gap": gap, "relative": relative_tie}
            target = tie_answer["choice"]
        choice = targets[operation][target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"][index] for index, a in targets[operation].items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "tie_break": tie_break,
        "done_floor": done_floor,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "server_ms": result.get("latency_ms"),  # compute time, when the server reports it; the rest is network
        "server_timing": result.get("timing"),  # a self-hosted S1's encode/model split, when reported
        "request": body,
    }


def field_context(goal, action, page, history):
    # Input values are not page text: without the other fields a lone "Departure" beside filled From/To boxes reads
    # like a city, and a small text model copies the last typed city into a date field.
    others = [{"label": a["label"], "value": a.get("value", "")} for a in page.get("actions", [])
              if a.get("kind") == "fill" and a.get("node") != action.get("node")]
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "form_fields": others[:12],
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def text_model(system, context, env="TEXT_MODEL"):
    """One call to the OpenAI-compatible LLM. Returns its raw content; callers validate it.
    env="S2_MODEL" reads S2_MODEL_* settings, falling back to TEXT_MODEL_* for any that are unset."""

    def setting(name, default=None):
        return os.environ.get(f"{env}_{name}" if name else env) or os.environ.get(
            f"TEXT_MODEL_{name}" if name else "TEXT_MODEL", default
        )

    key = setting("API_KEY")
    if not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor.")
    base = setting("BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = setting("", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if setting("REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    if setting("REASONING") == "omit":
        reasoning = {}
    if setting("REASONING_EFFORT"):
        # Some gateways only honour this form, e.g. deepseek-v4-flash on opencode keeps reasoning otherwise.
        reasoning = {"reasoning_effort": setting("REASONING_EFFORT")}
    headers = {
        name: value
        for name, value in (
            ("User-Agent", setting("USER_AGENT")),
            ("x-opencode-session", setting("SESSION")),
        )
        if value
    }
    messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(context)}]
    started = time.perf_counter()
    if setting("API") == "responses":
        body = {"model": model, "max_output_tokens": 2000, "input": messages}
        if setting("REASONING") != "omit":
            body["reasoning"] = {"effort": setting("REASONING") or "minimal"}
        result = post_json(base + "/responses", key, body, headers)
        content = next(
            (
                part["text"]
                for item in result.get("output", [])
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            ),
            None,
        )
    else:
        body = {
            "model": model,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            **reasoning,
            "messages": messages,
        }
        result = post_json(base + "/chat/completions", key, body, headers)
        content = result["choices"][0]["message"]["content"]
    return content, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }


def field_text(context):
    # Fast model first; a non-reasoning model sometimes answers null. One System 2 attempt, then stop.
    for env in ("TEXT_MODEL", "S2_MODEL"):
        content, meta = text_model(TEXT_VALUE, context, env)
        try:
            output = json.loads(content)
            value = output["text"]
            if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
                raise ValueError()
            return value, {**meta, "escalated": env == "S2_MODEL"}
        except (ValueError, KeyError, TypeError):
            continue
    raise ValueError("Text helper returned no valid field value; nothing typed.")


def reflect(goal, page, history, trigger, *, allow_done=True):
    """System 2: called only when System 1 is stuck. Returns a language subgoal or a stop reason, never an action."""
    context = {
        "goal": goal,
        "today": time.strftime("%Y-%m-%d"),
        "trigger": trigger,
        "page": {"url": page["url"], "title": page["title"], "text": page["text"][:6000]},
        "elements": [e["label"] for e in action_space(page["actions"])[0]][:80],
        "recent_actions": [{k: h.get(k) for k in ("action", "text", "page_changed")} for h in history[-10:]],
    }
    prompt = REFLECT
    if not allow_done:
        prompt += ("\nAn independent completion verifier has already failed. Do not return verdict=done; "
                   "give a subgoal that addresses the failed checks or explain why the task is infeasible.")
    content, meta = text_model(prompt, context, env="S2_MODEL")
    try:
        output = json.loads(content)
        if set(output) != {"verdict", "text"} or output["verdict"] not in {"subgoal", "infeasible", "done"}:
            raise ValueError()
        if output["verdict"] == "done" and not allow_done:
            raise ValueError()
        if not isinstance(output["text"], str) or not output["text"].strip() or len(output["text"]) > 500:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("System 2 returned no valid verdict; nothing executed.") from None
    return output["verdict"], output["text"].strip(), meta
