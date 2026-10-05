import type { Config } from "./config.js";

type Metadata = { context_window?: number; context_length?: number; max_output_tokens?: number;
  reasoning_efforts?: { value: string }[];
  capabilities?: { context_length?: number; max_output_tokens?: number; reasoning_effort?: string[] } };

export async function modelLimits(config: Config) {
  let metadata: Metadata | undefined;
  let fetchError: unknown;
  try {
    const response = await fetch(`${config.baseUrl.replace(/\/$/, "")}/models`, {
      headers: { Authorization: `Bearer ${config.apiKey}`, Originator: config.originator },
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error(`Model metadata HTTP ${response.status}`);
    const body = await response.json() as { data?: (Metadata & { id: string })[] };
    metadata = body.data?.find(model => model.id === config.model);
  } catch (error) { fetchError = error; }
  const contextWindow = config.contextWindow ?? metadata?.context_window ?? metadata?.context_length ?? metadata?.capabilities?.context_length;
  if (!Number.isSafeInteger(contextWindow) || !contextWindow || contextWindow < 32768) {
    throw new Error(`No valid session context capacity for ${config.model}. Set TAIJI_S2_CONTEXT_WINDOW to the provider's actual capacity. ${fetchError ?? "Model metadata missing"}`);
  }
  const supported = metadata?.reasoning_efforts?.map(e => e.value) ?? metadata?.capabilities?.reasoning_effort;
  const reasoning = config.reasoning ?? supported?.[0] ?? "low";
  if (supported?.length && !supported.includes(reasoning)) {
    throw new Error(`Unsupported reasoning effort ${reasoning} for ${config.model}; supported: ${supported.join(", ")}`);
  }
  const providerOutput = metadata?.max_output_tokens ?? metadata?.capabilities?.max_output_tokens;
  const maxTokens = config.maxTokens ?? Math.min(providerOutput ?? 8192, 8192);
  if (providerOutput && maxTokens > providerOutput) throw new Error("TAIJI_S2_MAX_TOKENS exceeds provider output capacity");
  if (maxTokens >= contextWindow) throw new Error("Output token budget must be smaller than context capacity");
  return { contextWindow, maxTokens, reasoning, source: config.contextWindow ? "explicit override" : "provider metadata" };
}
