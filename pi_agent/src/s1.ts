
export async function rankCandidates(
  input: { question: string; observation?: string; state?: object; options: { id: string; label: string }[] },
  signal?: AbortSignal,
  env: NodeJS.ProcessEnv = process.env,
) {
  const ids = input.options.map(o => o.id);
  if (!ids.length || new Set(ids).size !== ids.length) throw new Error("Candidate IDs must be nonempty and unique.");
  const base = env.TAIJI_S1_URL;
  const key = env.TAIJI_S1_API_KEY;
  if (!base || !key) throw new Error("S1 is not configured. Continue with the main model.");
  const started = performance.now();
  const response = await fetch(`${base.replace(/\/$/, "")}/v1/systemone`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    signal: AbortSignal.any([...(signal ? [signal] : []), AbortSignal.timeout(45000)]),
    body: JSON.stringify({ state: input.state ?? { observation: input.observation }, questions: {
      rank: { type: "choice", instructions: input.question,
        criteria: Object.fromEntries(input.options.map(o => [o.id, o.label])) },
    } }),
  });
  if (!response.ok) throw new Error(`S1 returned HTTP ${response.status}. Continue with the main model.`);
  const result = await response.json() as { answers: { rank: { choice: string; probabilities: Record<string, number>; confidence: number } }; latency_ms?: number };
  const answer = result.answers?.rank;
  const probabilities = answer?.probabilities;
  if (!answer || !ids.includes(answer.choice) || !probabilities
      || Object.keys(probabilities).length !== ids.length
      || ids.some(id => !Number.isFinite(probabilities[id]) || probabilities[id] < 0 || probabilities[id] > 1)
      || Math.abs(ids.reduce((sum, id) => sum + probabilities[id], 0) - 1) > 0.02
      || probabilities[answer.choice] < Math.max(...Object.values(probabilities)) - 1e-6) {
    throw new Error("Invalid S1 ranking. Continue with the main model.");
  }
  return { ...answer, elapsed_ms: Math.round(performance.now() - started), server_ms: result.latency_ms };
}

/** The existing S1 field writer returns a text envelope, not OpenAI tool calls. */
export async function writeText(context: object, signal?: AbortSignal, env: NodeJS.ProcessEnv = process.env) {
  if (!env.TAIJI_S1_URL || !env.TAIJI_S1_API_KEY) throw new Error("S1 writer is not configured");
  const response = await fetch(`${env.TAIJI_S1_URL.replace(/\/$/, "")}/v1/chat/completions`, {
    method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${env.TAIJI_S1_API_KEY}` },
    signal: AbortSignal.any([...(signal ? [signal] : []), AbortSignal.timeout(45000)]),
    body: JSON.stringify({ model: "s1", messages: [{ role: "user", content: JSON.stringify(context) }] }),
  });
  if (!response.ok) throw new Error(`S1 writer returned HTTP ${response.status}`);
  const body = await response.json() as any;
  const text = JSON.parse(body.choices?.[0]?.message?.content).text;
  if (typeof text !== "string" || !text.trim()) throw new Error("S1 writer returned no value; ask S2");
  return text;
}
