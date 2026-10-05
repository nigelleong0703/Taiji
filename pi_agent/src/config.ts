import { existsSync, mkdirSync, copyFileSync, readFileSync } from "node:fs";
import { parseEnv } from "node:util";
import { homedir } from "node:os";
import { dirname, resolve, delimiter } from "node:path";
import { fileURLToPath } from "node:url";

export const packageDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const repoDir = resolve(packageDir, "..");

export function configure(env: NodeJS.ProcessEnv = process.env) {
  function positiveInteger(name: string) {
    if (!env[name]) return undefined;
    const value = Number(env[name]);
    if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`);
    return value;
  }
  const cwd = resolve(env.TAIJI_PI_CWD || repoDir);
  env.TAIJI_PI_CWD = cwd;
  const stateDir = resolve(env.TAIJI_PI_STATE_DIR || resolve(packageDir, ".state"));
  const agentDir = resolve(stateDir, "agent");
  const sessionDir = resolve(stateDir, "sessions");
  mkdirSync(agentDir, { recursive: true });
  mkdirSync(sessionDir, { recursive: true });
  const s1Config = resolve(stateDir, "s1.env");
  if (existsSync(s1Config)) {
    const saved = parseEnv(readFileSync(s1Config, "utf8"));
    for (const name of ["TAIJI_S1_URL", "TAIJI_S1_API_KEY"]) env[name] ||= saved[name];
  }
  const mcpPath = resolve(agentDir, "mcp.json");
  if (!existsSync(mcpPath)) copyFileSync(resolve(packageDir, "mcp.example.json"), mcpPath);
  env.PI_CODING_AGENT_DIR = agentDir;
  env.PILOT_MCP_SCRIPT ||= resolve(homedir(), "Desktop/pilot/mcp-server/dist/index.js");
  env.TAIJI_PI_INLINE_STATE = "1";
  env.TAIJI_PILOT_STDIO_WRAPPER = resolve(packageDir, "src/pilot_stdio.mjs");
  env.TAIJI_S2_API_KEY ||= "opencodex";
  env.TAIJI_S1_URL ||= "http://127.0.0.1:8010";
  env.TAIJI_S1_API_KEY ||= "taiji-local-run";
  const model = env.TAIJI_S2_MODEL || "opencode-go/deepseek-v4-flash-vision-exp";
  return {
    cwd, stateDir, agentDir, sessionDir, mcpPath, model,
    baseUrl: env.TAIJI_S2_BASE_URL || "http://127.0.0.1:10100/v1",
    apiKey: env.TAIJI_S2_API_KEY,
    originator: env.TAIJI_S2_ORIGINATOR || "opencode",
    reasoning: env.TAIJI_S2_REASONING_EFFORT || undefined,
    contextWindow: positiveInteger("TAIJI_S2_CONTEXT_WINDOW"),
    maxTokens: positiveInteger("TAIJI_S2_MAX_TOKENS"),
    enableS1: env.TAIJI_PI_ENABLE_S1 !== "0",
    skillPaths: [resolve(homedir(), ".pi/agent/skills"),
      ...(env.TAIJI_PI_SKILLS || "").split(delimiter).filter(Boolean).map(p => resolve(p))]
      .filter(p => existsSync(p)),
  };
}
export type Config = ReturnType<typeof configure>;
