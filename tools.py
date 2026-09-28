"""Tool calling as S1 decisions, through the existing interfaces only: /v1/systemone for decisions and
/v1/chat/completions for written values. Any Jev-compatible endpoint works (Jev, Decision-1.0, this S1).
prepare_tools.py builds training rows from the same functions, so training questions match inference questions.

One tool call = a few small decisions instead of one generated JSON blob:
  which tool (or none)      choice over tool names + NONE
  optional argument used?   noul, per optional parameter
  enum argument             choice over the enum values
  boolean argument          noul
  any other argument        text: the LM head writes the value (JSON for numbers, arrays and objects)
"""

import ast
import json

NONE = "none"
HISTORY_CHARS = 8000  # most recent conversation kept; older turns are dropped first
RESULT_CHARS = 1500  # a long tool result is cut, the rest of the conversation matters more
DESCRIPTION_CHARS = 300


def function(tool):
    """OpenAI tool schema ({"type": "function", "function": {...}}) or a bare function dict."""
    return tool.get("function", tool)


def parse_call(content):
    """Toucan tool_call content is a Python-literal dict whose 'arguments' is a JSON string."""
    call = content if isinstance(content, dict) else ast.literal_eval(content)
    arguments = call.get("arguments") or {}
    return call["name"], json.loads(arguments) if isinstance(arguments, str) else arguments


def render(messages):
    """Conversation -> plain text, newest turns kept. Accepts Toucan roles and OpenAI's tool roles."""
    lines = []
    for m in messages:
        role, content = m.get("role"), m.get("content") or ""
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            if content:
                lines.append(f"Assistant: {content}")
            for call in m.get("tool_calls") or []:  # OpenAI style
                name, arguments = parse_call(function(call))
                lines.append(f"Tool call: {name} {json.dumps(arguments, ensure_ascii=False)}")
        elif role == "tool_call":  # Toucan style
            name, arguments = parse_call(content)
            lines.append(f"Tool call: {name} {json.dumps(arguments, ensure_ascii=False)}")
        elif role in ("tool_response", "tool"):
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            lines.append(f"Tool result: {text[:RESULT_CHARS]}")
        elif role == "system" and content:
            lines.append(f"System: {content}")
    text = "\n".join(lines)
    return text[-HISTORY_CHARS:]


def short(text):
    return " ".join(str(text or "").split())[:DESCRIPTION_CHARS]


def tool_question(tools):
    criteria = {function(t)["name"]: short(function(t).get("description")) for t in tools}
    criteria[NONE] = "Do not call a tool now; reply to the user instead."
    if len(criteria) > 255:
        raise ValueError("More than 254 tools")
    instructions = "Choose the next tool to call for the conversation."
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def argument_intro(tool, name, schema):
    return (f"Tool: {tool['name']} - {short(tool.get('description'))}\n"
            f"Argument: {name} ({schema.get('type', 'any')}) - {short(schema.get('description'))}")


def include_question(tool, name, schema):
    return {"type": "noul", "instructions": argument_intro(tool, name, schema),
            "criteria": {"true": "This optional argument should be set for this call.",
                         "false": "Leave this optional argument out."}}


def enum_question(tool, name, schema):
    return {"type": "choice", "instructions": argument_intro(tool, name, schema) + "\nChoose its value.",
            "criteria": {str(v): str(v) for v in schema["enum"]}}


def bool_question(tool, name, schema):
    return {"type": "noul", "instructions": argument_intro(tool, name, schema),
            "criteria": {"true": "Set this argument to true.", "false": "Set this argument to false."}}


def text_context(conversation, tool, name, schema):
    return {"conversation": conversation, "tool": tool["name"], "tool_description": short(tool.get("description")),
            "argument": name, "argument_description": short(schema.get("description")),
            "type": schema.get("type", "any")}


def value_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def parse_value(text, schema):
    """Written text -> typed argument; strings stay text, anything else must parse as JSON."""
    if schema.get("type") in (None, "string"):
        return text
    return json.loads(text)


def kind(schema):
    if "enum" in schema and schema["enum"]:
        return "enum"
    return "bool" if schema.get("type") == "boolean" else "text"


def rows_for_call(conversation, tools, called, arguments):
    """Training rows for one decision point: the tool choice, then each argument of the chosen tool."""
    rows = [{"state": conversation, "question": tool_question(tools), "label": called}]
    if called == NONE:
        return rows
    tool = next(function(t) for t in tools if function(t)["name"] == called)
    parameters = tool.get("parameters") or {}
    required = set(parameters.get("required") or [])
    for name, schema in (parameters.get("properties") or {}).items():
        present = name in arguments
        if name not in required:
            rows.append({"state": conversation, "question": include_question(tool, name, schema),
                         "label": str(present).lower()})
        if not present:
            continue
        value = arguments[name]
        if kind(schema) == "enum" and str(value) in map(str, schema["enum"]):
            rows.append({"state": conversation, "question": enum_question(tool, name, schema), "label": str(value)})
        elif kind(schema) == "bool" and isinstance(value, bool):
            rows.append({"state": conversation, "question": bool_question(tool, name, schema),
                         "label": str(value).lower()})
        elif kind(schema) == "text":
            rows.append({"task": "text", "context": text_context(conversation, tool, name, schema),
                         "target": value_text(value)})
    return rows


def call_tool(decide, write, conversation, tools):
    """Next tool call for a conversation, or None. Client side; nothing new on the server.

    decide(state, questions) -> answers: POST /v1/systemone and return its "answers".
    write(context) -> text or None: the value your text model returns (e.g. /v1/chat/completions).
    Returns {"name", "arguments", "confidence", "invalid_arguments"}; hand invalid arguments to a larger model.
    """
    state = conversation if isinstance(conversation, str) else render(conversation)
    choice = decide(state, {"tool": tool_question(tools)})["tool"]
    if choice["choice"] == NONE:
        return None
    tool = next(function(t) for t in tools if function(t)["name"] == choice["choice"])
    parameters = tool.get("parameters") or {}
    properties, required = parameters.get("properties") or {}, set(parameters.get("required") or [])
    questions = {}  # every non-text decision about the arguments is independent given the state: one request
    for name, schema in properties.items():
        if name not in required:
            questions[f"use:{name}"] = include_question(tool, name, schema)
        if kind(schema) == "enum":
            questions[f"enum:{name}"] = enum_question(tool, name, schema)
        elif kind(schema) == "bool":
            questions[f"bool:{name}"] = bool_question(tool, name, schema)
    answers = decide(state, questions) if questions else {}
    arguments, invalid = {}, []
    for name, schema in properties.items():
        if name not in required and answers[f"use:{name}"]["noul"] < 0.5:
            continue
        if kind(schema) == "enum":
            value = answers[f"enum:{name}"]["choice"]
            arguments[name] = next((v for v in schema["enum"] if str(v) == value), value)
        elif kind(schema) == "bool":
            arguments[name] = answers[f"bool:{name}"]["noul"] >= 0.5
        else:
            text = write(text_context(state, tool, name, schema)) or ""
            try:
                arguments[name] = parse_value(text, schema)
            except json.JSONDecodeError:
                arguments[name] = text
                invalid.append(name)
    return {"name": tool["name"], "arguments": arguments, "confidence": choice["confidence"],
            "invalid_arguments": invalid}
