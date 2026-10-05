import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from jsonschema import ValidationError

from taiji_agent.mcp_registry import MCPRegistry, ToolBinding, load_server_config
from taiji_agent.runtime import S1Client, S2Client, TaijiAgent, _compact_state_for_s1, _simple_schema


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
        self.cache_sessions = []
        self.decision_states = []

    async def decide(self, state, criteria, cache_session=None):
        self.cache_sessions.append(cache_session)
        self.decision_states.append(state)
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
    assert len(s1.cache_sessions) == 2
    assert s1.cache_sessions[0] == s1.cache_sessions[1]
    assert s1.cache_sessions[0]
    assert result.elapsed_ms >= 0
    assert any(event["event"] == "s1_decision" and event["duration_ms"] >= 0 for event in result.trace)
    assert any(event["event"] == "mcp_tool" and event["tool"] == "catalog__search"
               for event in result.trace)


@pytest.mark.asyncio
async def test_s1_cache_session_is_isolated_per_agent_run():
    s1 = FakeS1([choice("FINISH"), choice("FINISH")])
    agent = TaijiAgent(FakeRegistry(), s1, FakeS2([finished(), finished()]))
    await agent.run("Task one")
    await agent.run("Task two")
    assert len(s1.cache_sessions) == 2
    assert s1.cache_sessions[0] != s1.cache_sessions[1]


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


def test_s1_state_compacts_old_page_results_but_preserves_full_s2_history():
    state = {"goal": "compare pages", "recent_steps": [
        {"step": i, "result": "x" * 10000} for i in range(6)
    ]}
    compact = _compact_state_for_s1(state)
    assert len(compact["recent_steps"]) == 4
    assert len(compact["recent_steps"][0]["result"]) <= 1000
    assert len(compact["recent_steps"][-1]["result"]) <= 2300
    assert compact["recent_steps"][-1]["result_truncated_chars"] > 0
    assert len(state["recent_steps"]) == 6
    assert len(state["recent_steps"][-1]["result"]) == 10000


@pytest.mark.asyncio
async def test_s1_compaction_is_opt_in_and_trace_records_full_and_sent_inputs():
    state = {"goal": "compare pages", "recent_steps": [{"result": "x" * 10000}]}
    criteria = {"browser__observe": "Observe page", "FINISH": "Done"}

    full_s1 = FakeS1([choice("browser__observe")])
    full_agent = TaijiAgent(FakeRegistry(), full_s1, FakeS2([]))
    full_agent._run_started = __import__("time").perf_counter()
    full_agent._step_number = 1
    await full_agent._safe_s1_decide(state, criteria, "trace-session")
    full_event = full_agent.trace[0]
    assert full_s1.decision_states[0] == state
    assert full_event["full_input"]["state"] == state
    assert full_event["sent_input"]["state"] == state
    assert full_event["request_body"]["state"] == state
    assert full_event["output"]["choice"] == "browser__observe"

    compact_s1 = FakeS1([choice("browser__observe")])
    compact_agent = TaijiAgent(FakeRegistry(), compact_s1, FakeS2([]), compact_s1_context=True)
    compact_agent._run_started = __import__("time").perf_counter()
    compact_agent._step_number = 1
    await compact_agent._safe_s1_decide(state, criteria, "trace-session")
    compact_event = compact_agent.trace[0]
    assert compact_s1.decision_states[0] == compact_event["sent_input"]["state"]
    assert compact_event["full_input"]["state"] == state
    assert len(compact_event["sent_input"]["state"]["recent_steps"][0]["result"]) < 10000


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
        seen["headers"] = request.headers
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "tool_calls": []}}]})

    s2 = S2Client("https://model.example/v1", "key", "planner", originator="opencode")
    headers = dict(s2.http.headers)
    await s2.http.aclose()
    s2.http = httpx.AsyncClient(transport=httpx.MockTransport(respond), headers=headers)
    try:
        await s2.complete([{"role": "user", "content": "do work"}],
                          [{"type": "function", "function": {"name": "demo__act", "parameters": {}}}],
                          only="demo__act")
    finally:
        await s2.close()
    assert seen["url"] == "https://model.example/v1/chat/completions"
    assert seen["headers"]["originator"] == "opencode"
    assert seen["body"]["tools"][0]["function"]["name"] == "demo__act"
    assert seen["body"]["tool_choice"]["function"]["name"] == "demo__act"


@pytest.mark.asyncio
async def test_s2_client_retries_with_auto_for_auto_only_providers():
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(400, json={"error": {"message":
                'only `"auto"` is supported for `tool_choice`'}})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "demo__act", "arguments": "{}"}}]}}]})

    s2 = S2Client("https://model.example/v1", "key", "planner")
    await s2.http.aclose()
    s2.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        message = await s2.complete([{"role": "user", "content": "inspect"}],
                                    [{"type": "function", "function": {"name": "demo__act",
                                      "parameters": {"type": "object", "properties": {}}}}],
                                    only="demo__act")
    finally:
        await s2.close()
    assert message["tool_calls"][0]["function"]["name"] == "demo__act"
    assert bodies[0]["tool_choice"]["function"]["name"] == "demo__act"
    assert bodies[1]["tool_choice"] == "auto"
    assert s2.tool_choice_fallback is True


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
        decision = await s1.decide({"goal": "run"}, {"demo__act": "Act", "FINISH": "Done"},
                                   cache_session="task-session")
        text = await s1.write({"arguments": "schema"})
    finally:
        await s1.close()
    assert decision["choice"] == "demo__act"
    assert text == "{}"
    assert paths[0][0] == "/v1/systemone"
    assert paths[0][1]["questions"]["next_action"]["criteria"]["demo__act"] == "Act"
    assert paths[0][1]["cache_session"] == "task-session"
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


class FakeBrowserRegistry(FakeRegistry):
    """A browser adapter: open/observe return a numbered action table, act consumes one action id."""

    def __init__(self):
        def obj(**properties):
            return {"type": "object", "properties": properties}

        self.tools = {
            "browser__open": ToolBinding("browser__open", "browser", "open", "Open a page",
                                         obj(url={"type": "string"}), {}, self),
            "browser__observe": ToolBinding("browser__observe", "browser", "observe", "Read the page",
                                            obj(), {}, self),
            "browser__act": ToolBinding("browser__act", "browser", "act", "Perform one action",
                                        obj(action_id={"type": "string"}, text={"type": "string"}), {}, self),
            "browser__screenshot": ToolBinding("browser__screenshot", "browser", "screenshot",
                                               "Screenshot", obj(), {}, self),
        }
        self.calls = []

    def describe_for_s1(self):
        return {name: tool.description for name, tool in self.tools.items()}

    def model_tools(self, names=None):
        names = names or list(self.tools)
        return [self.tools[name].openai_definition() for name in names]

    async def call(self, name, arguments):
        self.tools[name].validate(arguments)
        self.calls.append((name, arguments))
        if name == "browser__screenshot":
            return SimpleNamespace(structured_content={"screenshot": "AAAA"}, content=[], is_error=False)
        if name == "browser__act":
            page = {"url": "https://x/", "title": "Results", "text": "flights listed", "actions": [
                {"id": "e3", "kind": "click", "label": "Select flight", "node": 3},
                {"id": "look", "kind": "look", "label": "Look at the screen", "node": None}]}
        else:
            page = {"url": "https://x/", "title": "Search", "text": "form", "actions": [
                {"id": "e1", "kind": "click", "label": "Search", "node": 1},
                {"id": "e2", "kind": "fill", "label": "Where to?", "node": 2, "value": ""},
                {"id": "look", "kind": "look", "label": "Look at the screen", "node": None}]}
        return SimpleNamespace(structured_content=None,
                               content=[SimpleNamespace(text=json.dumps(page))], is_error=False)

    @staticmethod
    def result_for_model(result, limit=12000):
        return json.dumps({"is_error": False, "result": [block.text for block in result.content]})


@pytest.mark.asyncio
async def test_browser_actions_drive_s1_choices_and_execute_over_mcp(monkeypatch):
    import taiji_agent.runtime as runtime

    decisions = [{"choice": "e1", "confidence": 0.8, "operation": "CLICK",
                  "probabilities": {"e1": 0.8}, "model": "s1", "latency_ms": 12, "request": {}},
                 {"choice": "DONE", "confidence": 0.9, "operation": "DONE",
                  "probabilities": {"DONE": 0.9}, "model": "s1", "latency_ms": 9, "request": {}}]
    seen = []

    def fake_choose(page, goal, history, cache_session=None):
        decision = decisions[min(len(seen), len(decisions) - 1)]
        seen.append({**decision, "page": page, "history": list(history),
                     "cache_session": cache_session})
        return decision

    monkeypatch.setattr(runtime, "BROWSER_EXAMPLE", True)
    monkeypatch.setattr(runtime, "browser_choose", fake_choose)
    registry = FakeBrowserRegistry()
    s1 = FakeS1([])
    s2 = FakeS2([finished("The cheapest result is visible.")])
    agent = TaijiAgent(registry, s1, s2, approve=lambda _tool, _args: True)
    result = await agent.run("Find the cheapest flight", url="https://x/")
    assert result.status == "completed"
    assert registry.calls[0] == ("browser__open", {"url": "https://x/"})
    assert registry.calls[1] == ("browser__act", {"action_id": "e1"})
    assert [item["choice"] for item in seen] == ["e1", "DONE"]
    assert seen[0]["page"]["actions"][0]["id"] == "e1"
    assert seen[0]["history"] == []
    assert seen[0]["cache_session"] == seen[1]["cache_session"]
    assert any(event["event"] == "s1_decision" and event.get("source") == "browser"
               and event.get("choice") == "e1" for event in agent.trace)
    assert result.steps[-1]["browser_action"]["id"] == "e1"


class FakePlannerS2(FakeS2):
    def __init__(self, plans=(), messages=()):
        super().__init__(messages)
        self.plans = iter(plans)
        self.plan_calls = []

    async def plan(self, system, context):
        self.plan_calls.append({"system": system, "context": context})
        return next(self.plans)


@pytest.mark.asyncio
async def test_repeated_actions_escalate_to_a_planner_subgoal(monkeypatch):
    import taiji_agent.runtime as runtime

    class StaticPage(FakeBrowserRegistry):
        async def call(self, name, arguments):
            self.tools[name].validate(arguments)
            self.calls.append((name, arguments))
            page = {"url": "https://x/", "title": "Search", "text": "unchanged", "actions": [
                {"id": "e1", "kind": "click", "label": "Search", "node": 1}]}
            return SimpleNamespace(structured_content=None,
                                   content=[SimpleNamespace(text=json.dumps(page))], is_error=False)

    goals = []

    def fake_choose(page, goal, history, cache_session=None):
        goals.append(goal)
        return {"choice": "e1", "confidence": 0.8}

    monkeypatch.setattr(runtime, "BROWSER_EXAMPLE", True)
    monkeypatch.setattr(runtime, "browser_choose", fake_choose)
    s2 = FakePlannerS2(plans=[{"verdict": "subgoal", "text": "open the departure field first"}])
    agent = TaijiAgent(StaticPage(), FakeS1([]), s2, max_steps=5, approve=lambda _tool, _args: True)
    result = await agent.run("book a flight", url="https://x/")
    assert len(s2.plan_calls) == 1
    assert s2.plan_calls[0]["context"]["elements"][0].startswith("e1 click Search")
    assert s2.plan_calls[0]["context"]["trigger"] == "Three actions did not change the page."
    assert any("Planner subgoal: open the departure field first" in goal for goal in goals)
    assert any(event["event"] == "s2_reflection" and event.get("verdict") == "subgoal"
               for event in agent.trace)
    assert result.status == "step_limit"


@pytest.mark.asyncio
async def test_a_planner_done_finishes_the_run_without_a_verifier(monkeypatch):
    """The model's own completion claim ends the run; no separate check has to confirm it."""
    import taiji_agent.runtime as runtime

    class StaticPage(FakeBrowserRegistry):
        async def call(self, name, arguments):
            self.tools[name].validate(arguments)
            self.calls.append((name, arguments))
            page = {"url": "https://x/", "title": "Search", "text": "unchanged", "actions": [
                {"id": "e1", "kind": "click", "label": "Search", "node": 1}]}
            return SimpleNamespace(structured_content=None,
                                   content=[SimpleNamespace(text=json.dumps(page))], is_error=False)

    def fake_choose(page, goal, history, cache_session=None):
        return {"choice": "e1", "confidence": 0.8}

    monkeypatch.setattr(runtime, "BROWSER_EXAMPLE", True)
    monkeypatch.setattr(runtime, "browser_choose", fake_choose)
    s2 = FakePlannerS2(plans=[{"verdict": "done",
                               "text": "Singapore to Denpasar flights are listed, cheapest from 376 dollars."}])
    agent = TaijiAgent(StaticPage(), FakeS1([]), s2, max_steps=5, approve=lambda _tool, _args: True)
    result = await agent.run("book a flight", url="https://x/")
    assert result.status == "completed"
    assert "cheapest from 376 dollars" in result.message
    assert any(event["event"] == "s2_reflection" and event.get("verdict") == "done" for event in agent.trace)


@pytest.mark.asyncio
async def test_audit_trail_is_written_as_the_run_happens(monkeypatch, tmp_path):
    """One flushed JSON line per event, with a header naming who ran what against which endpoints."""
    import taiji_agent.runtime as runtime

    def fake_choose(page, goal, history, cache_session=None):
        return {"choice": "DONE", "confidence": 0.9, "operation": "DONE", "probabilities": {"DONE": 0.9}}

    monkeypatch.setattr(runtime, "BROWSER_EXAMPLE", True)
    monkeypatch.setattr(runtime, "browser_choose", fake_choose)
    s2 = FakeS2([finished("The cheapest flight is visible.")])
    agent = TaijiAgent(FakeBrowserRegistry(), FakeS1([]), s2, approve=lambda _tool, _args: True,
                       audit_dir=tmp_path)
    result = await agent.run("find a flight", url="https://x/")
    assert result.status == "completed"
    assert agent.audit_path is not None and agent.audit_path.parent == tmp_path

    lines = [json.loads(line) for line in agent.audit_path.read_text().splitlines()]
    assert lines[0]["event"] == "run_started"
    assert lines[0]["goal"] == "find a flight" and lines[0]["url"] == "https://x/"
    assert lines[0]["run"] and lines[0]["ts"] and lines[0]["host"]
    assert any(line["event"] == "mcp_tool" for line in lines)
    assert any(line["event"] == "s1_decision" for line in lines)
    assert lines[-1]["event"] == "run_finished" and lines[-1]["status"] == "completed"
    assert all(line["run"] == lines[0]["run"] for line in lines)


@pytest.mark.asyncio
async def test_a_second_run_continues_the_conversation(monkeypatch, tmp_path):
    """A follow-up keeps the page, the transcript and the tool history: nothing is reconnected."""
    import taiji_agent.runtime as runtime

    seen = []

    def fake_choose(page, goal, history, cache_session=None):
        seen.append({"goal": goal, "history": list(history)})
        return {"choice": "DONE", "confidence": 0.9, "operation": "DONE", "probabilities": {"DONE": 0.9}}

    monkeypatch.setattr(runtime, "BROWSER_EXAMPLE", True)
    monkeypatch.setattr(runtime, "browser_choose", fake_choose)
    s2 = FakeS2([finished("first turn done"), finished("second turn done")])
    registry = FakeBrowserRegistry()
    agent = TaijiAgent(registry, FakeS1([]), s2, approve=lambda _tool, _args: True, audit_dir=tmp_path)

    first = await agent.run("open the travel category", url="https://x/")
    opens = sum(1 for name, _args in registry.calls if name == "browser__open")
    second = await agent.run("now open the first book")

    assert first.status == second.status == "completed"
    assert sum(1 for name, _args in registry.calls if name == "browser__open") == opens  # not reopened
    assert [item["goal"] for item in seen] == ["open the travel category", "now open the first book"]
    assert agent._messages[0]["role"] == "system"
    transcript = [message.get("content", "") for message in agent._messages]
    assert any("open the travel category" in text for text in transcript)
    assert any("now open the first book" in text for text in transcript)
    assert agent._turn == 2


@pytest.mark.asyncio
async def test_a_scroll_and_reclick_cycle_counts_as_stuck(monkeypatch):
    """Scrolling changes the page every step, so only the repeated control reveals the loop."""

    s2 = FakePlannerS2(plans=[{"verdict": "subgoal", "text": "stop scrolling and pick the book"}])
    agent = TaijiAgent(FakeBrowserRegistry(), FakeS1([]), s2, max_steps=20,
                       approve=lambda _tool, _args: True)
    agent._browser_history = [
        {"action": name, "kind": "scroll" if name.startswith("scroll") else "click",
         "before": f"page-{index}", "page_changed": True,
         "url": "https://books.toscrape.com/", "title": "Books"}
        for index, name in enumerate(["e30", "scroll_down", "scroll_up", "e3"] * 3)
    ]
    agent._reflected_at = 0
    agent._reflections = []

    assert await agent._check_progress("goal", [], [], 1) is None
    assert len(s2.plan_calls) == 1, "the 4-step scroll cycle never escalated"
    assert agent._subgoal == "stop scrolling and pick the book"


@pytest.mark.asyncio
async def test_paging_through_a_list_is_not_a_stall(monkeypatch):
    """Clicking the same control is progress when each click lands on a different page."""

    s2 = FakePlannerS2(plans=[{"verdict": "subgoal", "text": "should not be needed"}])
    agent = TaijiAgent(FakeBrowserRegistry(), FakeS1([]), s2, max_steps=20,
                       approve=lambda _tool, _args: True)
    agent._browser_history = [
        {"action": "Next", "kind": "click", "before": f"page-{index}", "page_changed": True,
         "url": f"https://quotes.toscrape.com/page/{index}/", "title": f"Quotes {index}"}
        for index in range(1, 9)
    ]
    agent._reflected_at = 0
    agent._reflections = []

    assert await agent._check_progress("goal", [], [], 1) is None
    assert s2.plan_calls == [], "paging through a list was mistaken for a stall"


@pytest.mark.asyncio
async def test_the_model_chooses_when_to_look(monkeypatch):
    """The picture is sent on the decision after the model picks the look action, not by a heuristic."""
    import taiji_agent.runtime as runtime

    monkeypatch.setenv("TAIJI_S1_SCREENSHOT", "auto")
    seen = []
    decisions = iter([
        {"choice": "look", "confidence": 0.9, "operation": "LOOK", "probabilities": {"LOOK": 0.9}},
        {"choice": "DONE", "confidence": 0.9, "operation": "DONE", "probabilities": {"DONE": 0.9}},
    ])

    def fake_choose(page, goal, history, cache_session=None):
        seen.append({"has_image": "screenshot" in page,
                     "offers_look": any(a.get("id") == "look" for a in page.get("actions", []))})
        return next(decisions)

    monkeypatch.setattr(runtime, "BROWSER_EXAMPLE", True)
    monkeypatch.setattr(runtime, "browser_choose", fake_choose)
    s2 = FakeS2([finished("done")])
    agent = TaijiAgent(FakeBrowserRegistry(), FakeS1([]), s2, max_steps=6,
                       approve=lambda _tool, _args: True)
    result = await agent.run("look at this page", url="https://x/")

    assert result.status == "completed"
    assert [item["has_image"] for item in seen] == [False, True]
    assert all(item["offers_look"] for item in seen)
    assert any(event["event"] == "s1_look" for event in agent.trace)


@pytest.mark.asyncio
async def test_asking_to_look_repeatedly_is_a_stall(monkeypatch):
    """A look changes nothing, so three in a row is the same non-progress as three identical clicks."""
    import taiji_agent.runtime as runtime

    monkeypatch.setenv("TAIJI_S1_SCREENSHOT", "auto")
    look = {"choice": "look", "confidence": 0.9, "operation": "LOOK", "probabilities": {"LOOK": 0.9}}
    decisions = iter([look] * 4 + [{"choice": "DONE", "confidence": 0.9, "operation": "DONE",
                                    "probabilities": {"DONE": 0.9}}])

    monkeypatch.setattr(runtime, "BROWSER_EXAMPLE", True)
    monkeypatch.setattr(runtime, "browser_choose",
                        lambda page, goal, history, cache_session=None: next(decisions))
    s2 = FakePlannerS2(plans=[{"verdict": "subgoal", "text": "stop looking and click the link"}],
                       messages=[finished("done")])
    agent = TaijiAgent(FakeBrowserRegistry(), FakeS1([]), s2, max_steps=8,
                       approve=lambda _tool, _args: True)
    result = await agent.run("look at this page", url="https://x/")

    assert result.status == "completed"
    assert len(s2.plan_calls) == 1, "repeated looks never escalated"
    assert agent._subgoal == "stop looking and click the link"


def test_page_is_captured_from_the_raw_result_not_the_capped_rendered_text():
    """The model-facing text is size-capped; a capped observation must not be kept as the current page."""
    agent = TaijiAgent(FakeRegistry(), FakeS1([]), FakeS2([]))
    page = {"url": "https://x/", "title": "T", "text": "y" * 20000,
            "actions": [{"id": "e7", "kind": "click", "label": "Next", "node": 7}]}
    raw = SimpleNamespace(structured_content=None,
                          content=[SimpleNamespace(text=json.dumps(page))], is_error=False)
    agent._capture_page(raw)
    assert list(agent._page_actions) == ["e7"]

    agent._page_actions, agent._page = {}, {}
    agent._capture_page('{"is_error": false, "result": ["{\\"actions\\": [ … [truncated]')
    assert agent._page_actions == {}


def test_a_model_acting_over_mcp_gets_the_same_action_list_with_ids():
    """S2 must see the list it has to choose from. The capped tool text lost it entirely on big pages."""
    agent = TaijiAgent(FakeRegistry(), FakeS1([]), FakeS2([]))
    actions = [{"id": f"e{i}", "kind": "click", "label": f"item {i}", "node": i} for i in range(1, 121)]
    page = {"url": "https://x/", "title": "T", "text": "z" * 30000, "actions": actions}
    raw = SimpleNamespace(structured_content=None,
                          content=[SimpleNamespace(text=json.dumps(page))], is_error=False)
    assert agent._capture_page(raw) is True
    view = json.loads(agent._browser_view())
    assert view["action_count"] == 120
    assert [item["id"] for item in view["actions"]][:3] == ["e1", "e2", "e3"]
    assert view["actions"][-1]["id"] == "e120"
    assert view["text"]
    assert '"e120"' not in json.dumps({"result": [json.dumps(page)]})[:12000]


@pytest.mark.asyncio
async def test_second_stall_hands_the_turn_to_s2_as_an_actor(monkeypatch):
    import taiji_agent.runtime as runtime

    class StaticPage(FakeBrowserRegistry):
        async def call(self, name, arguments):
            self.tools[name].validate(arguments)
            self.calls.append((name, arguments))
            page = {"url": "https://x/", "title": "Search", "text": "unchanged", "actions": [
                {"id": "e1", "kind": "click", "label": "Search", "node": 1}]}
            return SimpleNamespace(structured_content=None,
                                   content=[SimpleNamespace(text=json.dumps(page))], is_error=False)

    monkeypatch.setattr(runtime, "BROWSER_EXAMPLE", True)
    monkeypatch.setattr(runtime, "browser_choose",
                        lambda page, goal, history, cache_session=None: {"choice": "e1", "confidence": 0.8})
    s2 = FakePlannerS2(plans=[{"verdict": "subgoal", "text": "try the other button"}],
                       messages=[finished("S2 took the turn and finished.")])
    agent = TaijiAgent(StaticPage(), FakeS1([]), s2, max_steps=10, approve=lambda _tool, _args: True)
    result = await agent.run("book a flight", url="https://x/")
    assert result.status == "completed"
    notes = [message["content"] for _tools, _only, messages in s2.calls
             for message in messages if isinstance(message, dict) and message.get("role") == "user"]
    assert any("still stuck after its last subgoal" in note for note in notes)
    assert any("try the other button" in note for note in notes)


@pytest.mark.asyncio
async def test_browser_policy_can_return_to_other_registered_tools(monkeypatch):
    import taiji_agent.runtime as runtime

    class MixedRegistry(FakeBrowserRegistry):
        def __init__(self):
            super().__init__()
            self.tools.update(FakeRegistry().tools)

        async def call(self, name, arguments):
            if name == "catalog__search":
                self.calls.append((name, arguments))
                return SimpleNamespace(structured_content=None,
                                       content=[SimpleNamespace(text='{"matches":["Taiji"]}')], is_error=False)
            return await super().call(name, arguments)

    decisions = iter([{"choice": "use_tools", "confidence": 0.9},
                      {"choice": "DONE", "confidence": 0.9}])
    seen = []

    def decide(page, *args):
        seen.append(page)
        return next(decisions)

    monkeypatch.setattr(runtime, "BROWSER_EXAMPLE", True)
    monkeypatch.setattr(runtime, "browser_choose", decide)
    registry = MixedRegistry()
    s1 = FakeS1([choice("catalog__search")], ['{"query":"Taiji"}'])
    agent = TaijiAgent(registry, s1, FakeS2([finished("Done")]), approve=lambda *args: True)
    result = await agent.run("Read the page and search the catalog", url="https://x/")
    assert result.status == "completed"
    assert registry.calls[1] == ("catalog__search", {"query": "Taiji"})
    assert s1.decision_states[0]["page"]["title"] == "Search"
    assert any(a["id"] == "use_tools" for a in seen[0]["actions"])
    assert any(e["event"] == "capability_switch" for e in result.trace)


@pytest.mark.asyncio
async def test_s2_recovery_retains_control_until_explicit_delegation(monkeypatch):
    import taiji_agent.runtime as runtime

    decisions = []

    def decide(*args):
        decisions.append(True)
        return {"choice": "e1", "confidence": 0.8}

    class RecoveryAgent(TaijiAgent):
        async def _check_progress(self, *args):
            self._planner_active = True
            return None

    monkeypatch.setattr(runtime, "BROWSER_EXAMPLE", True)
    monkeypatch.setattr(runtime, "browser_choose", decide)
    s2 = FakeS2([{"tool_calls": [{"id": "recover", "type": "function", "function": {
        "name": "browser__observe", "arguments": "{}"}}]}, finished("Recovered")])
    agent = RecoveryAgent(FakeBrowserRegistry(), FakeS1([]), s2, approve=lambda *args: True)
    result = await agent.run("task", url="https://x/")
    assert result.status == "completed"
    assert len(decisions) == 1
    assert len(s2.calls) == 2


@pytest.mark.asyncio
async def test_s2_can_delegate_a_concrete_subgoal_to_s1():
    s2 = FakeS2([{"tool_calls": [{"id": "delegate", "type": "function", "function": {
        "name": "agent__delegate_s1", "arguments": '{"subgoal":"Search Taiji","capability":"tools"}'}}]}])
    agent = TaijiAgent(FakeRegistry(), FakeS1([]), s2)
    agent._planner_active = True
    messages = []
    result = await agent._s2_turn(messages, [], 1, only=None)
    assert result["kind"] == "tool" and result["entry"] is None
    assert not agent._planner_active and agent._use_tools
    assert agent._subgoal == "Search Taiji"
    assert messages[-1]["content"] == "Delegated to S1."


@pytest.mark.asyncio
async def test_screenshot_tool_result_is_attached_as_an_image():
    agent = TaijiAgent(FakeBrowserRegistry(), FakeS1([]), FakeS2([]), approve=lambda *args: True)
    agent._browser = agent._browser_tools()
    result = await agent._execute("browser__screenshot", {}, [], 1, "s2")
    messages = []
    agent._record_s2_tool(messages, "browser__screenshot", {}, result, 1)
    assert "AAAA" not in messages[-2]["content"]
    assert messages[-1]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAAA"
