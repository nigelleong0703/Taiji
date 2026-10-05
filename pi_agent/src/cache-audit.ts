import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

function hash(value: unknown) { return createHash("sha256").update(JSON.stringify(value) ?? "null").digest("hex"); }

/** Compare serialized provider requests, rather than inferring cache reuse from model switches. */
export class PrefixAudit {
  private previous?: { messages: string[]; tools: string; system: string };
  constructor(private path?: string) {
    if (path) try { this.previous = JSON.parse(readFileSync(path, "utf8")); } catch { /* First request. */ }
  }
  inspect(payload: Record<string, unknown>) {
    const messages = Array.isArray(payload.messages) ? payload.messages : [];
    const system = messages.filter((m: any) => m.role === "system");
    const tools = payload.tools;
    const previous = this.previous;
    const hashes = messages.map(hash);
    const prefixPreserved = !!previous && hashes.length >= previous.messages.length && previous.messages.every((h, i) => h === hashes[i]);
    this.previous = { messages: hashes, tools: hash(tools), system: hash(system) };
    if (this.path) writeFileSync(this.path, JSON.stringify(this.previous));
    return { type: "s2_prefix", messages: messages.length, tools_hash: hash(tools), system_hash: hash(system),
      tools_stable: previous ? hash(tools) === previous.tools : null,
      system_stable: previous ? hash(system) === previous.system : null,
      previous_messages_preserved: previous ? prefixPreserved : null };
  }
}
