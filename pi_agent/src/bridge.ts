import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { inlinePage, pageState } from "./observations.js";
import { Type } from "typebox";
import { defineTool } from "@earendil-works/pi-coding-agent";
import { createAssistantMessageEventStream, getCurrentTools,
  type AssistantMessage, type Message, type StreamFunction, type ToolCall } from "@earendil-works/pi-ai";
import { packageDir, repoDir } from "./config.js";
import { rankCandidates } from "./s1.js";
import { toolDecision } from "./tool-policy.js";
import { s2ExecutionTool } from "./execution-policy.js";
import { envNames, environments, environmentsIn, envOf, splitTool, targetKey } from "./environments.js";

export type Audit = (event: Record<string, unknown>) => void;
/** Control tools plus the browser environment's names (the default environment). */
export const names = { ...envNames("browser"), handoff: "taiji_handoff", resume: "taiji_continue", delegate: "taiji_delegate" };
export const zeroCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

export function controller(messages: readonly Message[]) {
  const start = messages.findLastIndex(m => m.role === "user");
  for (let i = messages.length - 1; i > start; i--) {
    const m = messages[i];
    if (m.role === "assistant" && m.provider === "taiji-proxy") return "s2";
    if (m.role !== "toolResult" || m.isError) continue;
    if (m.toolName === names.handoff) return "s2";
    if (m.toolName === names.resume) return messages.slice(start + 1, i).some(previous =>
      previous.role === "toolResult" && previous.toolName === names.delegate && !previous.isError) ? "s1" : "s2";
    if (m.toolName === names.delegate) {
      const call = messages.slice(0, i).flatMap(message => message.role === "assistant" ? message.content : [])
        .find(block => block.type === "toolCall" && block.id === m.toolCallId);
      return call?.type === "toolCall" && call.arguments.executor === "s2" ? "s2" : "s1";
    }
  }
  return undefined;
}

export function controlTools() {
  const result = (text: string) => ({ content: [{ type: "text" as const, text }], details: {} });
  return [defineTool({ name: names.handoff, label: "Hand off to S2",
    description: "Transfer control to S2 with operations, final environment state, exit kind and reason. input_limit is a harness guard, not model-selected help. S2 continues from inline native history and final state; no observation files are used. phase_done ends the delegated phase; it is not a verification gate.",
    parameters: Type.Object({ reason: Type.String(),
      kind: Type.Optional(Type.Union([Type.Literal("model_help"), Type.Literal("phase_done"),
        Type.Literal("input_limit"), Type.Literal("tool_error"), Type.Literal("unsupported")])),
      state: Type.Optional(Type.Object({
        env: Type.Optional(Type.String()), target_id: Type.Optional(Type.Union([Type.Integer(), Type.String()])),
        tab_id: Type.Optional(Type.Integer()), url: Type.Optional(Type.String()),
        subgoal: Type.String(), done_when: Type.String(), steps: Type.Optional(Type.Array(Type.String())),
        recent_actions: Type.Array(Type.Unknown()),
        last_tool: Type.Optional(Type.Object({ name: Type.String(), is_error: Type.Boolean() })),
        page: Type.Optional(Type.Unknown()),
        input_limit: Type.Optional(Type.Unknown()),
      })),
    }),
    async execute(_id, input) { return result(JSON.stringify(input)); } }),
    defineTool({ name: names.resume, label: "Continue S1 execution",
      description: "S2 keeps the existing phase and allows S1 to continue local execution. Use when observed progress is useful but more local execution is needed. It does not replace the goal or erase execution history; S2 can reason again after further results.",
      parameters: Type.Object({ reason: Type.String({ minLength: 1 }) }),
      async execute(_id, input) { return result(JSON.stringify(input)); } }),
    defineTool({ name: names.delegate, label: "Delegate to S1",
      description: "Delegate a concrete environment or non-terminal MCP phase to S1 with a stop condition. Use handler=tools for non-terminal MCP tool selection/arguments. An environment handler (an MCP server exposing observe/act, e.g. handler=browser or handler=desktop) can only select operations on the CURRENT observed page or window: click/type/keys/scroll/read/image. It cannot navigate to URLs, open apps or discover tools. Prepare the required environment via the appropriate tools before delegating; supply an observed target_id (browser tab or desktop window; tab_id is accepted for the browser), one local phase and its visible done_when condition. Terminal commands, scripts and code edits belong directly to S2 and must not be delegated to S1. Original user constraints stay authoritative. Explicit executor=s2 browser fallback requires a reason.",
      parameters: Type.Object({ subgoal: Type.String({ minLength: 1 }),
        handler: Type.Optional(Type.String({ minLength: 1, description: "For operations on an observed page or window: the environment's server name, i.e. the prefix of its observe/act tools (mcp__<env>__observe), e.g. browser, desktop, sandbox. Use tools only for other non-terminal MCP tools." })),
        target_id: Type.Optional(Type.Union([Type.Integer({ minimum: 1 }), Type.String({ minLength: 1 })])),
        tab_id: Type.Optional(Type.Integer({ minimum: 1 })),
        done_when: Type.String({ minLength: 1 }),
        steps: Type.Optional(Type.Array(Type.String({ minLength: 1 }), { maxItems: 8,
          description: "Optional ordered operations of this phase, each directly visible in the page or window. S1 runs them until done_when holds or it asks for help." })),
        executor: Type.Optional(Type.Union([Type.Literal("s1"), Type.Literal("s2")])),
        reason: Type.Optional(Type.String({ minLength: 1 })) }),
      async execute(_id, input) {
        if (input.executor === "s2" && !input.reason?.trim()) {
          return { ...result("Explicit S2 execution requires a reason describing failed or unsupported S1 execution."), isError: true };
        }
        return result(JSON.stringify({ ...input, executor: input.executor ?? "s1" }));
      } })];
}

/** A fixed execution contract; no tool-schema swapping or guessed browser actions. */
export function browserExecutionGate(messages: readonly Message[], toolName: string, toolCallId: string) {
  if (splitTool(toolName)?.tool !== "act") return undefined;
  const caller = messages.findLast(m => m.role === "assistant" && m.content.some(
    b => b.type === "toolCall" && (b.id === toolCallId || toolCallId.startsWith(b.id + "/"))));
  if (caller?.role !== "assistant" || caller.provider !== "taiji-proxy") return undefined;
  const start = messages.findLastIndex(m => m.role === "user");
  for (let i = messages.length - 1; i > start; i--) {
    const message = messages[i];
    if (message.role !== "toolResult" || message.isError) continue;
    if (message.toolName === names.handoff) break;
    if (message.toolName !== names.delegate) continue;
    const call = messages.slice(start + 1, i).flatMap(m => m.role === "assistant" ? m.content : [])
      .find(b => b.type === "toolCall" && b.id === message.toolCallId);
    if (call?.type === "toolCall" && call.arguments.executor === "s2" && String(call.arguments.reason ?? "").trim()) return undefined;
    break;
  }
  return { block: true, reason: "Local environment actions belong to S1. Call taiji_delegate with a concrete subgoal and done_when, then let S1 execute. If S1 execution failed or is unsupported, explicitly select executor=s2 with a reason before acting." };
}

type Action = { id: string; kind: string; label: string; node?: number };
type Target = number | string;
type Page = { url: string; title: string; text: string; actions: Action[]; screenshot?: string; tab_id?: number; target_id?: Target };
const pageTarget = (page?: Page) => page?.target_id ?? page?.tab_id;

/** The active environment's page, task goal and full operation history, for any observe/act environment. */
export function envContext(messages: readonly Message[], known: Iterable<string> = []) {
  const envs = environmentsIn(messages);
  for (const env of known) envs.add(env);
  let page: Page | undefined;
  let env: string | undefined;
  let fresh = false;
  let subgoal = "";
  let doneWhen = "";
  let steps: string[] | undefined;
  let handler: string | undefined;
  let targetId: Target | undefined;
  const history: Record<string, unknown>[] = [];
  const calls = new Map<string, ToolCall>();
  const users: string[] = [];
  let priorAnswer = "";
  for (const m of messages) {
    if (m.role === "user") {
      users.push(typeof m.content === "string" ? m.content : m.content.filter(b => b.type === "text").map(b => b.text).join("\n"));
      subgoal = "";
      doneWhen = "";
      steps = undefined;
      handler = undefined;
      targetId = undefined;
      fresh = false;
      history.length = 0;
    }
    if (m.role === "assistant") {
      for (const b of m.content) if (b.type === "toolCall") calls.set(b.id, b);
      if (!m.content.some(b => b.type === "toolCall")) {
        const answer = m.content.filter(b => b.type === "text").map(b => b.text).join("\n");
        if (answer) priorAnswer = answer;
      }
    }
    if (m.role !== "toolResult") continue;
    const call = calls.get(m.toolCallId);
    let text = m.content.filter(b => b.type === "text").map(b => b.text).join("\n");
    if (m.toolName === names.delegate && !m.isError) {
      subgoal = String(call?.arguments.subgoal ?? "");
      doneWhen = String(call?.arguments.done_when ?? "");
      steps = Array.isArray(call?.arguments.steps) ? call.arguments.steps.map(String) : undefined;
      handler = typeof call?.arguments.handler === "string" ? call.arguments.handler : undefined;
      const target = call?.arguments.target_id ?? call?.arguments.tab_id;
      targetId = typeof target === "number" || typeof target === "string" ? target : undefined;
      // Keep failed-action evidence across replanning instead of giving S1 amnesia.
    }
    const source = envOf(m.toolName, envs);
    if (!source) continue;
    const tool = splitTool(m.toolName)!.tool;
    if (tool === "screenshot_image") {
      if (page && !m.isError && source === env) {
        const image = m.content.find(b => b.type === "image");
        if (image) page = { ...page, screenshot: image.data };
        history.push({ action: "look", kind: "look", page_changed: false });
      }
      continue;
    }
    if (tool.startsWith("list")) continue;
    if (tool.startsWith("close")) { page = undefined; continue; }
    if (m.isError) { history.push({ tool: m.toolName, arguments: call?.arguments, action: call?.arguments.action_id, kind: "error", error: text }); continue; }
    let next: Page;
    try { next = inlinePage(m as any); } catch { page = undefined; continue; }
    if (!next || !Array.isArray(next.actions)) { page = undefined; continue; }
    if (tool === "act") {
      const action = page?.actions.find(a => a.id === call?.arguments.action_id);
      history.push({ action: action?.label ?? call?.arguments.action_id, kind: action?.kind,
        action_id: call?.arguments.action_id, text: call?.arguments.text,
        page_changed: !page || fingerprint(page) !== fingerprint(next),
        url_changed: page?.url !== next.url,
        // Only a field that is still the same element (same id and label) has "changed"; a page
        // that renumbers its elements would otherwise list every element, every step.
        fields_changed: next.actions.filter(a => {
          const old = page?.actions.find(previous => previous.id === a.id && previous.label === a.label) as any;
          return old && ["value", "checked", "selected", "expanded"].some(k => old[k] !== (a as any)[k]);
        }).slice(0, 3).map(a => ({ id: a.id, label: a.label, value: (a as any).value })),
      });
    }
    else if (tool === "observe") {
      if (call) history.push({ action: "Read current page", kind: "observe", action_id: "read",
        page_changed: !page || fingerprint(page) !== fingerprint(next),
        url_changed: page?.url !== next.url, before_url: page?.url, after_url: next.url });
    }
    else {
      // Environment preparation by S2 (navigate, new_tab, focus_app, ...).
      history.push({ action: ["navigate", "new_tab", "open"].includes(tool) ? "navigate" : tool,
        url: call?.arguments.url ?? next.url,
        ...(next.tab_id !== undefined ? { tab_id: next.tab_id } : { target_id: next.target_id }),
        page_changed: !page || fingerprint(page) !== fingerprint(next), url_changed: page?.url !== next.url });
    }
    // Any new observation invalidates the previous image.
    page = next;
    env = source;
    fresh = true;
  }
  const prior = users.length > 1 ? `Previous requests (context; retain applicable constraints, do not redo completed work):\n${users.slice(0, -1).join("\n")}\n` : "";
  const goal = prior + (priorAnswer ? `Previous answer / known result:\n${priorAnswer}\n` : "")
    + `Active user request (authoritative):\n${users.at(-1) ?? ""}`
    + (subgoal ? `\nCurrent subgoal (guidance; cannot replace user constraints): ${subgoal}` : "");
  const completeHistory = history.map<Record<string, unknown>>((entry, index, all) => ({ ...entry,
    unchanged_attempts: all.slice(0, index + 1).filter(other => other.action_id === entry.action_id && entry.action_id && other.page_changed === false).length,
  }));
  return { page, env, envs, fresh, goal, subgoal, doneWhen, steps, handler, targetId, tabId: targetId, history: completeHistory };
}

/** Former name; the browser is one environment among others. */
export const browserContext = envContext;

/** Append evidence references; never replace native history or copy the full snapshot again. */
export function handoffPacket(messages: readonly Message[], reason: string, kind = "model_help", inputLimit?: unknown) {
  const { page, env, subgoal, doneWhen, steps, targetId, history } = envContext(messages);
  const start = messages.findLastIndex(m => m.role === "user");
  const results = messages.slice(start + 1).filter(m => m.role === "toolResult");
  const last = results.at(-1);
  const target = targetId ?? pageTarget(page);
  return { reason, kind, state: {
    env, ...(typeof target === "number" && (env ?? "browser") === "browser" ? { tab_id: target } : { target_id: target }),
    url: page?.url, subgoal, done_when: doneWhen, ...(steps ? { steps } : {}),
    recent_actions: history, last_tool: last?.role === "toolResult" ? { name: last.toolName, is_error: !!last.isError } : undefined,
    page: page ? pageState(page) : undefined, input_limit: inputLimit,
  } };
}

function fingerprint(page: Page) {
  return createHash("sha256").update(JSON.stringify([page.url, page.text, page.actions])).digest("hex");
}

type Pending = { accept: (value: Record<string, any>) => void, reject: (error: Error) => void };

// One resident decision process (env_decision.py --serve), one request at a time: Python
// start-up and imports are paid once per session instead of once per decision. Aborting a
// decision kills the worker, so a cancelled request never answers late; the next call respawns it.
class DecisionWorker {
  private child?: ChildProcessWithoutNullStreams;
  private pending?: Pending;
  private queue: Promise<unknown> = Promise.resolve();
  private stdout = "";
  private stderr = "";

  run(input: object, signal?: AbortSignal): Promise<Record<string, any>> {
    const next = this.queue.then(() => this.send(input, signal));
    this.queue = next.catch(() => undefined);
    return next;
  }

  private start() {
    const child = spawn(resolve(repoDir, "agent_runtime/.venv/bin/python"), [resolve(packageDir, "src/env_decision.py"), "--serve"], {
      cwd: repoDir, env: process.env, stdio: ["pipe", "pipe", "pipe"],
    });
    child.stdout.on("data", data => {
      this.stdout += data;
      const end = this.stdout.indexOf("\n");
      if (end < 0 || !this.pending) return;
      const line = this.stdout.slice(0, end);
      this.stdout = this.stdout.slice(end + 1);
      const { accept, reject } = this.pending;
      this.pending = undefined;
      hold(child, false);
      try {
        const reply = JSON.parse(line);
        if (reply.ok) accept(reply.result);
        else reject(Object.assign(new Error(reply.error), { model_calls: reply.model_calls }));
      } catch (error) { reject(error as Error); }
    });
    child.stderr.on("data", data => { this.stderr = (this.stderr + data).slice(-4000); });
    child.on("close", code => {
      if (this.child === child) this.child = undefined;
      this.pending?.reject(new Error(this.stderr.trim().split("\n").at(-1) || `S1 handler exited (${code})`));
      this.pending = undefined;
    });
    child.stdin.on("error", error => this.pending?.reject(error));
    this.child = child;
    this.stdout = "";
    this.stderr = "";
    return child;
  }

  private send(input: object, signal?: AbortSignal) {
    return new Promise<Record<string, any>>((accept, reject) => {
      if (signal?.aborted) return reject(signal.reason ?? new Error("aborted"));
      const child = this.child ?? this.start();
      hold(child, true);
      const abort = () => { child.kill(); };
      signal?.addEventListener("abort", abort, { once: true });
      this.pending = {
        accept: value => { signal?.removeEventListener("abort", abort); accept(value); },
        reject: error => { signal?.removeEventListener("abort", abort); reject(signal?.aborted ? signal.reason ?? error : error); },
      };
      child.stdin.write(JSON.stringify(input) + "\n");
    });
  }

  stop() { this.child?.kill(); }
}

// An idle worker must not keep the CLI alive; a busy one must (its reply is awaited).
function hold(child: ChildProcessWithoutNullStreams, busy: boolean) {
  for (const handle of [child, child.stdin, child.stdout, child.stderr] as { ref(): void, unref(): void }[]) {
    if (busy) handle.ref(); else handle.unref();
  }
}

const worker = new DecisionWorker();
process.once("exit", () => worker.stop());

export function decision(input: object, signal?: AbortSignal): Promise<Record<string, any>> {
  return worker.run(input, signal);
}

export function s1Provider(audit: Audit, compute = decision, computeTools = toolDecision): StreamFunction {
  const cacheSession = randomUUID();
  return (model, context, options) => {
    const stream = createAssistantMessageEventStream();
    const output: AssistantMessage = { role: "assistant", content: [], api: model.api, provider: model.provider,
      model: model.id, timestamp: Date.now(), stopReason: "pending",
      usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { ...zeroCost, total: 0 } } };
    stream.push({ type: "start", partial: output });
    void (async () => {
      const started = performance.now();
      let name = names.handoff, args: Record<string, any> = { kind: "unsupported", reason: "Unsupported S1 input; use S2." };
      try {
        options?.signal?.throwIfAborted();
        const tools = getCurrentTools(context.messages);
        const available = new Set(tools.map(t => t.name));
        const { page, env: observedEnv, envs, fresh, goal, subgoal, doneWhen, steps, handler, targetId, history } = envContext(context.messages, environments(tools));
        const lastResult = context.messages.findLast(m => m.role === "toolResult");
        const control = (n: string) => [names.delegate, names.handoff].includes(n);
        const isEnv = (n: string) => envOf(n, envs) !== undefined;
        // A handler names an environment only if its observe/act tools are present; otherwise (e.g.
        // handler=browser while the web page lives in the sandbox environment) the observed one owns it.
        const live = new Set(environments(tools));
        const handlerEnv = handler && live.has(handler) ? handler : undefined;
        const env = handlerEnv ?? (observedEnv && live.has(observedEnv) ? observedEnv : undefined)
          ?? (live.has("browser") ? "browser" : [...live][0] ?? observedEnv ?? "browser");
        const n = envNames(env);
        // handler=tools with a target that an observed environment owns is still that environment's phase.
        const ownedTarget = targetId !== undefined && observedEnv !== undefined && live.has(observedEnv);
        const general = (handler === "tools" && !ownedTarget) || (handlerEnv === undefined && targetId === undefined
          && ((!page && [...available].some(name => !isEnv(name) && !control(name)))
            || (lastResult?.role === "toolResult" && !isEnv(lastResult.toolName) && !control(lastResult.toolName))));
        if (general) {
          const selected = await computeTools(context.messages, audit, options?.signal);
          if (selected.answer !== undefined) {
            output.content.push({ type: "text", text: selected.answer });
            output.stopReason = "stop";
            stream.push({ type: "text_start", contentIndex: 0, partial: output });
            stream.push({ type: "text_delta", contentIndex: 0, delta: selected.answer, partial: output });
            stream.push({ type: "text_end", contentIndex: 0, content: selected.answer, partial: output });
            audit({ type: "s1_finish", elapsed_ms: Math.round(performance.now() - started) });
            stream.push({ type: "done", reason: "stop", message: output }); stream.end(output); return;
          }
          name = selected.name!; args = selected.arguments!;
        }
        else if (lastResult?.role === "toolResult" && lastResult.isError) args = { kind: "tool_error", reason: "Last tool failed or its execution outcome is unknown. S2 must inspect state before retrying." };
        else if (goal.length > 12000) args = { kind: "input_limit", reason: "Harness goal guard blocked S1 before inference; use S2 with native history.",
          input_limit: { source: "browser_goal", unit: "characters", actual: goal.length, limit: 12000 } };
        else if (!page && lastResult?.role === "toolResult" && isEnv(lastResult.toolName)) {
          args = { kind: "tool_error", reason: `${envOf(lastResult.toolName, envs)} result could not be parsed. Inspect the complete tool output with S2.` };
        }
        else if (!page || !fresh || observedEnv !== env || (targetId !== undefined && pageTarget(page) !== targetId)) {
          name = n.observe; args = {};
        }
        else {
          const selected = await compute({ env, page, goal, subgoal, done_when: doneWhen, ...(steps ? { steps } : {}), history, cache_session: cacheSession,
            capabilities: [...available].filter(name => [n.act, n.observe, n.screenshot].includes(name)).sort() }, options?.signal);
          audit({ type: "s1_decision", env, ...selected, elapsed_ms: Math.round(performance.now() - started), image: !!page.screenshot });
          if (selected.operation === "ASK_S2") {
            args = { reason: "S1 selected ASK_S2: use the original user request and current observations to plan, recover, or reach another capability." };
          } else if (selected.operation === "READ") { name = n.observe; args = {}; }
          else if (selected.operation === "DONE" || selected.operation === "BLOCKED") {
            args = { kind: selected.operation === "DONE" ? "phase_done" : "model_help", reason: `S1 ${selected.operation}${subgoal ? " for the delegated phase" : ""}. Stop local actions. Plan the next phase and delegate it to S1 if more local execution is needed, or respond when the user task is handled.` };
          } else if (selected.operation === "LOOK") { name = n.screenshot; args = {}; }
          else {
            const action = page.actions.find(a => a.id === selected.choice);
            if (!action) throw new Error("S1 chose an action outside the observed table");
            if (selected.operation === "SELECT") throw new Error("Select adapter is not implemented; use S2");
            name = n.act; args = { action_id: action.id, ...(selected.text ? { text: selected.text } : {}) };
          }
        }
        if (name === n.act && !page?.actions.some(action => action.id === args.action_id)) {
          throw new Error("S1 action is not grounded in the current observed table; inspect state before acting");
        }
        const target = targetId ?? pageTarget(page);
        if ([n.act, n.observe, n.screenshot].includes(name) && target !== undefined) {
          args[targetKey(tools.find(t => t.name === name), env)] = target;
        }
        if (s2ExecutionTool(name, tools.find(t => t.name === name))) {
          name = names.handoff; args = { kind: "unsupported", reason: "Terminal commands, scripts and code edits belong entirely to S2 by user instruction." };
        }
        if (!available.has(name)) { name = names.handoff; args = { kind: "unsupported", reason: "Requested capability unavailable; use S2." }; }
      } catch (error) {
        if (options?.signal?.aborted) {
          output.stopReason = "aborted"; output.errorMessage = "S1 request cancelled";
          stream.push({ type: "error", reason: "aborted", error: output }); stream.end(output); return;
        }
        name = names.handoff; args = { reason: String(error),
          kind: (error as any)?.input_limit ? "input_limit" : "tool_error", input_limit: (error as any)?.input_limit };
        audit({ type: "s1_error", error: String(error), input_limit: (error as any)?.input_limit, model_calls: (error as { model_calls?: unknown })?.model_calls ?? [] });
      }
      if (name === names.handoff) {
        args = handoffPacket(context.messages, args.reason, args.kind ?? "model_help", args.input_limit);
        audit({ type: "s1_handoff", ...args });
      }
      const call: ToolCall = { type: "toolCall", id: randomUUID(), name, arguments: args };
      output.content.push(call);
      stream.push({ type: "toolcall_start", contentIndex: 0, partial: output });
      stream.push({ type: "toolcall_end", contentIndex: 0, toolCall: call, partial: output });
      output.stopReason = "toolUse";
      audit({ type: "s1_tool", name, arguments: args, elapsed_ms: Math.round(performance.now() - started) });
      stream.push({ type: "done", reason: "toolUse", message: output }); stream.end(output);
    })();
    return stream;
  };
}

export async function intake(messages: readonly Message[], signal?: AbortSignal) {
  const user = messages.findLast(m => m.role === "user");
  const text = user && (typeof user.content === "string" ? user.content : user.content.filter(b => b.type === "text").map(b => b.text).join("\n"));
  return rankCandidates({ question: "Choose the model needed for the next decision. Terminal commands, shell operations, scripts and code edits are assigned entirely to S2 by user instruction. S1 can execute browser/desktop operations and non-terminal MCP tools, inspect outputs and give short answers. Use S2 for terminal work, ambiguous objectives, complex planning/reasoning or longer answers.",
    observation: String(text), options: [{ id: "s1", label: "Browser or desktop execution, non-terminal MCP arguments/inspection, or simple short answer" },
      { id: "s2", label: "Terminal commands/scripts/code edits, clarification, complex planning/reasoning or longer answers" }] }, signal);
}
