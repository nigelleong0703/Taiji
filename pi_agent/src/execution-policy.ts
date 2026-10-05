import { getCurrentTools, type Message, type Tool } from "@earendil-works/pi-ai";

/** User-selected ownership: command/script execution and code edits belong to S2. */
export function s2ExecutionTool(name: string, tool?: Tool) {
  if (["bash", "powershell", "run_command", "write", "edit", "execute_code", "codemode"].includes(name)) return true;
  if (/^mcp__(terminal|shell|exec|command)__/i.test(name)) return true;
  const properties = (tool?.parameters as any)?.properties;
  return !!properties && ["command", "commands", "script", "code"].some(key => key in properties);
}

export function terminalExecutionGate(messages: readonly Message[], name: string, id: string) {
  const caller = messages.findLast(m => m.role === "assistant" && m.content.some(
    b => b.type === "toolCall" && (b.id === id || id.startsWith(b.id + "/"))));
  if (caller?.role !== "assistant" || caller.provider !== "taiji-s1") return undefined;
  const tool = getCurrentTools(messages).find(t => t.name === name);
  if (s2ExecutionTool(name, tool)) return { block: true,
    reason: "Terminal commands, scripts and code edits are assigned to S2 by the user. Hand off to S2; S1 must not generate or execute them." };
}
