import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from jsonschema import ValidationError

from taiji_agent.mcp_registry import MCPRegistry, ToolBinding, load_server_config
from taiji_agent.runtime import S1Client, S2Client, TaijiAgent, _simple_schema


class FakeRegistry(MCPRegistry):
    def __init__(self):
        schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        self.tools = {"catalog__search": ToolBinding("catalog__search", "catalog", "search", "Search catalog",
                                                      schema, {}, self)}
        self.calls = []

    def describe_for_s1(self):
        return {"catalog__search": "Search catalog (arguments: query)"}

    def model_tools(self, names=None):
        names = names or list(self.tools)
        return [self.tools[name].openai_definition() for name in names]

    async def call(self, name, arguments):
        self.tools[name].validate(arguments)
        self.calls.append((name, arguments))
        return SimpleNamespace(structured_content={"matches": [arguments["query"]]}, content=[], is_error=False)

    @staticmethod
    def result_for_model(result, limit=12000):
        return '{"is_error": false, "result": {"matches": ["' + result.structured_content["matches"][0] + '"]}}'


class FakeS1:
    def __init__(self, choices, writes=()):
        self.choices, self.writes = iter(choices), iter(writes)

    async def decide(self, state, criteria):
        return next(self.choices)

    async def write(self, context):
        return next(self.writes)


class FakeS2:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.calls = []

    async def complete(self, messages, tools, only=None):
        self.calls.append((tools, only, list(messages)))
        return next(self.messages)


def choice(name, confidence=0.9):
    return {"choice": name, "confidence": confidence}


def tool_call(name="catalog__search", args='{"query":"Taiji"}'):
    return {"tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": name, "arguments": args}}]}


def finished(text="The goal is complete."):
    return {"content": text, "tool_calls": []}


@pytest.mark.asyncio
async def test_confident_s1_tool_choice_executes_and_returns_to_shared_loop():
    registry = FakeRegistry()
    s1 = FakeS1([choice("catalog__search"), choice("FINISH")], ['{"query":"Taiji"}', "Found Taiji in the catalog."])
    s2 = FakeS2([finished("Found Taiji in the catalog.")])
    result = await TaijiAgent(registry, s1, s2, approve=lambda _tool, _args: True).run(
        "Find Taiji in the catalog")

    assert result.status == "completed"
    assert result.message == "Found Taiji in the catalog."
    assert registry.calls == [("catalog__search", {"query": "Taiji"})]
    assert result.steps[0]["source"] == "s1"
    assert len(s2.calls) == 1  # S2 verifies S1's proposed completion.


@pytest.mark.asyncio
async def test_uncertain_s1_exposes_registered_schema_to_s2_native_tool_call():
    registry = FakeRegistry()
    s1 = FakeS1([choice("catalog__search", confidence=0.2), choice("FINISH")], ["Done"])
    s2 = FakeS2([tool_call(), finished("The catalog contains Taiji.")])
    result = await TaijiAgent(registry, s1, s2, threshold=0.48,
                              approve=lambda _tool, _args: True).run("Find Taiji")

    assert result.status == "completed"
    assert registry.calls == [("catalog__search", {"query": "Taiji"})]
    assert s2.calls[0][0][0]["function"]["name"] == "catalog__search"
    assert result.steps[0]["source"] == "s2"


@pytest.mark.asyncio
async def test_invalid_s1_arguments_escalate_to_s2_restricted_to_selected_tool():
    registry = FakeRegistry()
    s1 = FakeS1([choice("catalog__search"), choice("FINISH")], ['{"unexpected":true}', "Done"])
    s2 = FakeS2([tool_call(), finished("Found it.")])
    result = await TaijiAgent(registry, s1, s2, approve=lambda _tool, _args: True).run("Find Taiji")

    assert result.status == "completed"
    assert s2.calls[0][1] == "catalog__search"
    assert [tool["function"]["name"] for tool in s2.calls[0][0]] == ["catalog__search"]
    assert result.steps[0]["source"] == "s2"


def test_only_flat_primitive_schemas_use_s1_argument_generation():
    assert _simple_schema({"type": "object", "properties": {"query": {"type": "string"}}})
    assert not _simple_schema({"type": "object", "properties": {"filters": {"type": "object"}}})
    assert not _simple_schema({"type": "object", "properties": {"items": {"type": "array"}}})


@pytest.mark.asyncio
async def test_vllm_mode_can_skip_unavailable_s1_text_generation():
    registry = FakeRegistry()
    s1 = FakeS1([], ["should not be used"])
    s1.supports_write = False
    agent = TaijiAgent(registry, s1, FakeS2([]), approve=lambda _tool, _args: True)
    assert await agent._s1_arguments("Find Taiji", {"recent_steps": []}, "catalog__search") is None


def test_mcp_schema_is_validated_before_dispatch():
    binding = FakeRegistry().tools["catalog__search"]
    with pytest.raises(ValidationError):
        binding.validate({"unexpected": True})


def test_mcp_config_expands_secrets_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DEMO_TOKEN", "secret-value")
    config = tmp_path / "mcp.json"
    config.write_text('{"mcpServers":{"remote":{"url":"https://tools.example/mcp",'
                      '"headers":{"Authorization":"Bearer ${DEMO_TOKEN}"}}}}')
    loaded = load_server_config(config)
    assert loaded["remote"]["headers"]["Authorization"] == "Bearer secret-value"


@pytest.mark.asyncio
async def test_s2_client_sends_native_tool_schema_and_restriction():
    seen = {}

    def respond(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "tool_calls": []}}]})

    s2 = S2Client("https://model.example/v1", "key", "planner")
    await s2.http.aclose()
    s2.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        await s2.complete([{"role": "user", "content": "do work"}],
                          [{"type": "function", "function": {"name": "demo__act", "parameters": {}}}],
                          only="demo__act")
    finally:
        await s2.close()
    assert seen["url"] == "https://model.example/v1/chat/completions"
    assert seen["body"]["tools"][0]["function"]["name"] == "demo__act"
    assert seen["body"]["tool_choice"]["function"]["name"] == "demo__act"


@pytest.mark.asyncio
async def test_s1_client_uses_existing_decision_and_text_endpoints():
    paths = []

    def respond(request):
        paths.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/v1/systemone":
            return httpx.Response(200, json={"answers": {"next_action": {"choice": "demo__act",
                                                                             "confidence": 0.9}}})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"text": "{}"})}}]})

    s1 = S1Client("http://s1", "key")
    await s1.http.aclose()
    s1.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        decision = await s1.decide({"goal": "run"}, {"demo__act": "Act", "FINISH": "Done"})
        text = await s1.write({"arguments": "schema"})
    finally:
        await s1.close()
    assert decision["choice"] == "demo__act"
    assert text == "{}"
    assert paths[0][0] == "/v1/systemone"
    assert paths[0][1]["questions"]["next_action"]["criteria"]["demo__act"] == "Act"
    assert paths[1][0] == "/v1/chat/completions"


@pytest.mark.asyncio
async def test_real_stdio_mcp_server_is_discovered_and_called():
    server = Path(__file__).parents[1] / "examples" / "demo_server.py"
    registry = await MCPRegistry.connect({"demo": {"command": sys.executable, "args": [str(server)]}})
    try:
        assert set(registry.tools) == {"demo__add_note", "demo__list_notes"}
        result = await registry.call("demo__add_note", {"text": "runtime smoke test"})
        assert not result.is_error
        notes = await registry.call("demo__list_notes", {})
        assert notes.structured_content["result"] == ["runtime smoke test"]
    finally:
        await registry.close()
