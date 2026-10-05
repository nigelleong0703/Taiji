/**
 * Environments S1 can execute in: any MCP server exposing `observe` and `act` that return the
 * Page shape (browser via Pilot, desktop via Cua, ...). Pi names MCP tools `mcp__<server>__<tool>`,
 * so an environment is identified by its server name; nothing here is browser-specific.
 */
import type { Message } from "@earendil-works/pi-ai";

export type EnvNames = { env: string; observe: string; act: string; screenshot: string };

const MCP = /^mcp__(.+?)__(.+)$/;

export function envNames(env: string): EnvNames {
  const prefix = `mcp__${env}__`;
  return { env, observe: `${prefix}observe`, act: `${prefix}act`, screenshot: `${prefix}screenshot_image` };
}

/** `mcp__desktop__act` -> { server: "desktop", tool: "act" }. */
export function splitTool(name: string) {
  const match = MCP.exec(name);
  return match ? { server: match[1], tool: match[2] } : undefined;
}

/** Servers that expose both observe and act among the offered tools. */
export function environments(tools: readonly { name: string }[]): string[] {
  const byServer = new Map<string, Set<string>>();
  for (const { name } of tools) {
    const parts = splitTool(name);
    if (parts) byServer.set(parts.server, (byServer.get(parts.server) ?? new Set()).add(parts.tool));
  }
  return [...byServer].filter(([, tools]) => tools.has("observe") && tools.has("act")).map(([server]) => server).sort();
}

/** Environments evidenced by a transcript (an observe/act call was made), plus the browser default. */
export function environmentsIn(messages: readonly Message[]): Set<string> {
  const found = new Set(["browser"]);
  for (const m of messages) {
    if (m.role !== "toolResult") continue;
    const parts = splitTool(m.toolName);
    if (parts && (parts.tool === "observe" || parts.tool === "act")) found.add(parts.server);
  }
  return found;
}

export function envOf(toolName: string, envs: ReadonlySet<string>): string | undefined {
  const server = splitTool(toolName)?.server;
  return server && envs.has(server) ? server : undefined;
}

/** The argument name an environment tool uses for its target (Pilot: tab_id; others: target_id). */
export function targetKey(tool: { parameters?: unknown } | undefined, env: string): "tab_id" | "target_id" {
  const properties = (tool?.parameters as { properties?: Record<string, unknown> } | undefined)?.properties ?? {};
  if ("target_id" in properties) return "target_id";
  if ("tab_id" in properties) return "tab_id";
  return env === "browser" ? "tab_id" : "target_id";
}
