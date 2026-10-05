import { resolve } from "node:path";
import { appendFileSync } from "node:fs";
import {
  createAgentSessionRuntime, createAgentSessionServices, createAgentSessionFromServices,
  createMcpExtension, createToolSearchExtension, createCodemodeExtension,
  ModelRuntime, SessionManager, SettingsManager,
  createBashToolDefinition, defineTool,
  type CreateAgentSessionRuntimeFactory, type ExtensionFactory,
} from "@earendil-works/pi-coding-agent";
import type { Config } from "./config.js";
import { browserExecutionGate, controlTools, controller, envContext, names, s1Provider, zeroCost } from "./bridge.js";
import { splitTool } from "./environments.js";
import { PrefixAudit } from "./cache-audit.js";
import { modelLimits } from "./model-limits.js";
import { compactObservation, inlinePage } from "./observations.js";
import { normalizeContext, type Message } from "@earendil-works/pi-ai";
import { streamSimple as streamS2 } from "@earendil-works/pi-ai/api/openai-completions";
import { LivePlanner } from "./live-planner.js";
import { terminalExecutionGate } from "./execution-policy.js";

// Weights trained on layout v5 read an ordered step list (state.task.steps); older weights only
// handle short phases, so S2 is told to keep each phase to one to three operations.
function phaseGuidance() {
  if (process.env.TAIJI_LAYOUT === "v5") return "Give each environment phase an ordered steps list (up to six operations, each directly visible in the page or window, for example ['press All Clear', 'press 1', 'press 2']) and a done_when that is visible when the last step is complete. S1 executes the steps in order, recovers from failed attempts, reports DONE when done_when holds and asks S2 when a step cannot be grounded. After S1 reports DONE, inspect the state and delegate the next phase.";
  return "Keep environment phases short: one to three operations whose combined effect is visible in the page or window (for example 'press All Clear', done when the display shows 0; then 'enter 12', done when the display shows 12). S1 reliably chooses the next control for a short concrete step; it is not reliable at long ordered sequences, at clearing leftover state on its own, or at deciding when to ask for help. After S1 reports DONE, inspect the state and delegate the next short phase.";
}

export interface RuntimeOptions { fresh?: boolean; sessionFile?: string; mcp?: boolean }

export async function createRuntime(config: Config, options: RuntimeOptions = {}) {
  const factory: CreateAgentSessionRuntimeFactory = async ({ cwd, sessionManager, sessionStartEvent }) => {
    process.env.TAIJI_PILOT_TAB_STATE = resolve(config.stateDir, "browser-tabs", `${sessionManager.getSessionId()}.json`);
    const settings = SettingsManager.create(cwd, config.agentDir);
    settings.setProjectTrusted(true);
    settings.applyOverrides({ defaultTools: ["+tool_search"],
      retry: { enabled: true, maxRetries: 2 },
      compaction: { enabled: true } });
    let s1Decided = false;
    const audit = (event: Record<string, unknown>) => {
      if (["s1_decision", "s1_tool_selection"].includes(String(event.type))) s1Decided = true;
      appendFileSync(resolve(config.stateDir, "audit.jsonl"),
        JSON.stringify({ ...event, session: sessionManager.getSessionId(), at: Date.now() }) + "\n");
    };
    const prefix = new PrefixAudit(resolve(config.stateDir, `prefix-${sessionManager.getSessionId()}.json`));
    let s2Started = 0;
    let planner: LivePlanner | undefined;
    const limits = await modelLimits(config);
    audit({ type: "model_limits", model: config.model, ...limits });
    const models = await ModelRuntime.create({ authPath: resolve(config.agentDir, "auth.json"),
      modelsPath: null, modelsStorePath: resolve(config.agentDir, "models-cache.json"), refreshOnCreate: false });
    models.registerProvider("taiji-proxy", {
      streamSimple: (model, context, opts) => streamS2(model as any, context, opts),
      baseUrl: config.baseUrl, api: "openai-completions", apiKey: "$TAIJI_S2_API_KEY",
      headers: { Originator: config.originator },
      models: [{ id: config.model, name: config.model, reasoning: true,
        input: ["text", "image"], contextWindow: limits.contextWindow, maxTokens: limits.maxTokens,
        cost: zeroCost,
        compat: { supportsDeveloperRole: false, supportsStore: false } }],
    });
    await models.setRuntimeApiKey("taiji-proxy", config.apiKey);
    const s2 = models.getModel("taiji-proxy", config.model)!;
    planner = new LivePlanner(async (messages, signal) => {
      const started = performance.now();
      const response = await streamS2(s2 as any, normalizeContext({ messages: [...messages] }), {
        apiKey: config.apiKey, headers: { Originator: config.originator }, signal, reasoning: "low",
        onPayload: payload => {
          const value = { ...(payload as any), reasoning_effort: limits.reasoning };
          audit({ type: "s2_request", mode: "background" });
          audit(prefix.inspect(value)); return value;
        },
      }).result();
      audit({ type: "s2_background_usage", usage: response.usage, elapsed_ms: Math.round(performance.now() - started),
        stop_reason: response.stopReason, error: response.errorMessage });
      return response;
    }, audit);
    if (config.enableS1) {
      models.registerProvider("taiji-s1", { api: "openai-completions", apiKey: "$TAIJI_S1_API_KEY",
        baseUrl: process.env.TAIJI_S1_URL, streamSimple: s1Provider(audit),
        models: [{ id: "browser-policy", name: "Taiji S1 execution policy", reasoning: false,
          // Pi accounts for the full session here; the provider sends only S1's bounded projection.
          input: ["text", "image"], cost: zeroCost, contextWindow: limits.contextWindow, maxTokens: 1024 }] });
      await models.setRuntimeApiKey("taiji-s1", process.env.TAIJI_S1_API_KEY!);
      models.registerVirtualModel({ provider: "taiji-proxy", id: "session",
        name: "Taiji S1 + S2 session", thinkingLevels: ["off", "minimal", "low"],
        contextWindow: limits.contextWindow, maxTokens: limits.maxTokens,
        async route(request) {
          let owner = request.reason === "direct" ? "s2" : controller(request.messages) ?? (request.state as { owner?: string } | undefined)?.owner ?? "s2";
          if (request.reason === "user") { s1Decided = false; planner!.reset(); owner = "s2"; }
          if (request.reason === "retry") { planner!.reset(); owner = "s2"; }
          if (owner === "s1") {
            const last = request.messages.findLast(m => m.role === "toolResult");
            if (last?.role === "toolResult" && last.toolName === names.delegate) s1Decided = false;
            const assistant = request.messages.findLast(m => m.role === "assistant");
            const phase = request.messages.findLast(m => m.role === "toolResult" && m.toolName === names.delegate && !m.isError);
            if (s1Decided && phase?.role === "toolResult" && last?.role === "toolResult" && assistant?.role === "assistant" && assistant.provider === "taiji-s1") {
              planner!.poll(phase.toolCallId, request.messages, request.signal);
            }
          } else planner!.reset();
          audit({ type: "route", owner, reason: request.reason });
          return { model: owner === "s1" ? models.getModel("taiji-s1", "browser-policy")! : s2,
            thinkingLevel: owner === "s1" ? "off" : "low",
            state: { owner } };
        },
      });
    }
    const requestCompat: ExtensionFactory = pi => {
      const executors = new Map<string, string>();
      pi.on("tool_call", (event, ctx) => {
        const messages = ctx.sessionManager.getBranch().flatMap(entry => entry.type === "message" ? [entry.message] : [])
          .filter((m): m is Message => ["user", "assistant", "toolResult"].includes(m.role));
        const caller = messages.findLast(message => message.role === "assistant");
        const executor = caller?.role === "assistant" ? caller.provider : "unknown";
        executors.set(event.toolCallId, executor);
        audit({ type: "tool_dispatch", tool: event.toolName, tool_call_id: event.toolCallId,
          executor, arguments: event.input });
        if (config.enableS1) {
          const rejected = terminalExecutionGate(messages, event.toolName, event.toolCallId)
            ?? browserExecutionGate(messages, event.toolName, event.toolCallId);
          if (rejected) { audit({ type: "execution_contract", ...rejected, tool: event.toolName }); return rejected; }
        }
      });
      pi.on("tool_result", (event, ctx) => {
        const result = compactObservation(event);
        const executor = executors.get(event.toolCallId) ?? "unknown";
        executors.delete(event.toolCallId);
        audit({ type: "tool_execution_result", tool: event.toolName, executor,
          tool_call_id: event.toolCallId, is_error: event.isError });
        if (event.toolName === names.handoff && executor === "taiji-s1") {
          const messages = ctx.sessionManager.getBranch().flatMap(entry => entry.type === "message" ? [entry.message] : [])
            .filter((m): m is Message => ["user", "assistant", "toolResult"].includes(m.role));
          const phase = messages.findLast(m => m.role === "toolResult" && m.toolName === names.delegate && !m.isError);
          const prepared = phase?.role === "toolResult" ? planner?.takeForHandoff(phase.toolCallId) : undefined;
          if (prepared && phase?.role === "toolResult") {
            const suggestion = prepared.response.content.filter(block => block.type !== "thinking").map(block => block.type === "toolCall"
              ? { type: "tool_suggestion", name: block.name, arguments: block.arguments } : block);
            audit({ type: "s2_background_consulted", phase: phase.toolCallId, only_after_s1_handoff: true });
            return { content: [...event.content, { type: "text" as const, text:
              "Background S2 suggestion from an EARLIER execution snapshot (" + prepared.snapshot_messages +
              " messages). Advisory only; no proposed action was executed. S1 has now handed back control. " +
              "Use the latest handoff reason, final state and subsequent action history to decide anew; " +
              "discard any outdated suggestion.\n" + JSON.stringify(suggestion) }], details: event.details };
          }
        }
        if (splitTool(event.toolName)?.tool === "act") {
          const messages = ctx.sessionManager.getBranch().flatMap(entry => entry.type === "message" ? [entry.message] : [])
            .filter((m): m is Message => ["user", "assistant", "toolResult"].includes(m.role));
          const previous = envContext(messages).page;
          let next: any;
          if (result) next = inlinePage(result);
          const state = (page: any) => JSON.stringify([page?.url, page?.text, page?.actions]);
          audit({ type: "browser_action_result", env: splitTool(event.toolName)?.server, executor, tool_call_id: event.toolCallId,
            is_error: event.isError, page_changed: previous && next ? state(previous) !== state(next) : null,
            url_changed: previous && next ? previous.url !== next.url : null });
        }
        if (result) {
          const { measurement, ...replacement } = result;
          audit({ type: "observation_size", tool: event.toolName, ...measurement });
          return replacement;
        }
      });
      pi.on("before_provider_request", event => {
        if (event.payload && typeof event.payload === "object" && "model" in event.payload
            && event.payload.model === config.model) {
          const payload = { ...event.payload, reasoning_effort: limits.reasoning };
          s2Started = performance.now();
          audit({ type: "s2_request", mode: "foreground" });
          audit(prefix.inspect(payload));
          return payload;
        }
      });
      pi.on("session_shutdown", () => { planner?.reset(); });
      pi.on("message_end", event => {
        const m = event.message;
        if (m.role === "assistant" && m.provider === "taiji-proxy" ) audit({ type: "s2_usage", usage: m.usage,
          elapsed_ms: s2Started ? Math.round(performance.now() - s2Started) : null });
      });
      pi.on("provider_stream_event", event => {
        const data = event.data as { usage?: unknown } | undefined;
        if (event.model === config.model && data?.usage) audit({ type: "s2_raw_usage", usage: data.usage });
      });
    };
    const services = await createAgentSessionServices({ cwd, agentDir: config.agentDir,
      modelRuntime: models, settingsManager: settings,
      resourceLoaderOptions: {
        noExtensions: true,
        additionalSkillPaths: config.skillPaths,
        appendSystemPrompt: [
          "Handle the user's task using available capabilities and skills. A URL is an optional tool argument, not a required task entry point. Tools and skill catalog remain available across S1/S2 handoffs; do not repeatedly search or replace them. Read a skill only when relevant. Screenshots are available as native image tool results: decide whether one helps. Environment observations (browser pages, desktop windows) and completed S1 tool calls are in this same history. Finish when handled or user input is needed; the session remains available for conversation. Do not repeat a mutation after an unknown execution outcome: inspect state first.",
          "Treat the user's requested entities, route, dates, filters and constraints as authoritative. Site defaults, prior assistant assumptions and delegated subgoals cannot replace them. Ground a delegated subgoal in the original request and current observation; delegate a concrete next step when useful, rather than asking the small policy to invent a task plan. Environment tool results expose inline current page/window text and control values. S1 retains the full observed action table in session metadata. Handoff carries recent operations, final state and exit reason; no observation file reads are needed. Recent failed actions are evidence for choosing a different approach or requesting more observation, not evidence of completion.",
          ...(config.enableS1 ? ["You are S2 and own the task lifecycle, planning and recovery. You can receive execution updates while S1 continues working. Make the next task-level decision from the observed results: revise/delegate a phase, use taiji_continue to retain the current phase when local execution should continue, use another capability or answer when handled. Background reasoning is advisory and cannot interrupt an active S1 phase. S1 retains execution ownership until it reports phase DONE or asks for help (or a user interruption/protocol failure requires recovery). A prepared background suggestion is only shown after S1 hands back control; decide anew from the latest handoff and history, discard stale suggestions, and inspect fresh state before state-sensitive mutations. Completing one tool call is not completing the phase. Never repeat work just because S1 has not reported DONE. S1 selects local actions; it is not the sole task termination switch. Terminal commands, scripts and code edits are assigned entirely to you (S2). Decide whether they are needed, generate them and execute them directly. This is execution reasoning as well as planning; do not delegate terminal work to S1. S1 selects environment operations (browser, desktop) and non-terminal MCP tools, writes short MCP arguments and reads their results. Delegate concrete non-terminal MCP phases with handler=tools and a done_when condition. Keep tools and skills stable during handoffs. Delegate according to the handler capability contract, not the task topic. A phase must have a visible done_when condition and only require supported local operations. " + phaseGuidance() + " An environment handler (handler=browser, handler=desktop, or another server exposing observe/act) selects operations on the current page or window, not URL navigation, app launching or tool discovery. Prepare navigation yourself (browser navigate tool, desktop focus_app tool), then delegate a concrete observed operation. General non-terminal MCP selection uses handler=tools. For browser or desktop tasks, frame constraints and the next execution phase; you may inspect tabs/windows/pages/images and prepare navigation using existing tabs or windows. As soon as the page or window supports local execution, call taiji_delegate with handler set to that environment, a concrete subgoal, visible done_when and target_id from an observed tab or window (tab_id is accepted for the browser). Pass the target as a structured parameter, not just prose. An environment handler executes grounded page or window actions; the tools handler can choose navigation or other tools and generate their arguments. Rather than doing clicks, typing, keys, scrolling or wait actions yourself, let S1 choose and execute those actions, including whether it needs READ or LOOK. After S1 asks for help, diagnose from its retained action history, revise the phase, then delegate again. S1 DONE reports completion of that phase; plan/delegate the next phase or answer when the task is handled. On handoff, distinguish model_help and phase_done from input_limit or tool_error: input_limit means a harness guard blocked inference, not that S1 judged the page difficult. Continue from retained native history and handoff.state; use its recent_actions, final page state and observed results directly. Complete actionable snapshots stay inline for S1; no observation documents or files are used. Do not delegate the identical oversized input repeatedly. Keep screenshots model-selected. There is no separate verifier. Do not delegate already completed work solely to increase S1 counts. If S1 failed or lacks an essential capability, you may explicitly choose executor=s2 with a reason through taiji_delegate; this fallback is audited. Do not silently carry on as the environment executor after planning."] : []),
        ],
        extensionFactories: [requestCompat, createToolSearchExtension(), createCodemodeExtension({ mode: "on" }),
          createMcpExtension(options.mcp === false ? { loadConfig: () => ({ servers: [], errors: [] }) } : {})],
      },
    });
    const model = models.getModel("taiji-proxy", config.enableS1 ? "session" : config.model);
    if (!model) throw new Error(`Model not registered: ${config.model}`);
    return { ...await createAgentSessionFromServices({ services, sessionManager, sessionStartEvent,
      model, thinkingLevel: "low",
      customTools: [...(config.enableS1 ? controlTools() : []), defineTool({
        ...createBashToolDefinition(cwd), name: "run_command", label: "Run command",
        description: "Execute a shell command in the session working directory. Choose whether a command is useful and write it yourself. Returns stdout/stderr and execution status; use the results to decide the next action. timeout is in seconds.",
      })] }), services, diagnostics: services.diagnostics };
  };
  const manager = options.sessionFile
    ? SessionManager.open(resolve(options.sessionFile), config.sessionDir, config.cwd)
    : options.fresh ? SessionManager.create(config.cwd, config.sessionDir)
      : SessionManager.continueRecent(config.cwd, config.sessionDir);
  return createAgentSessionRuntime(factory, { cwd: config.cwd, agentDir: config.agentDir, sessionManager: manager });
}
