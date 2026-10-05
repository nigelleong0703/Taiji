import { createAssistantMessageEventStream, type AssistantMessage, type Message } from '@earendil-works/pi-ai';

/** S2 reasons concurrently; only the native agent loop executes its eventual response. */
export class LivePlanner {
  private phase?: string;
  private abort?: AbortController;
  private pending = false;
  private reply?: AssistantMessage;
  private snapshotMessages = 0;
  constructor(private run: (messages: readonly Message[], signal: AbortSignal) => Promise<AssistantMessage>,
    private audit: (event: Record<string, unknown>) => void) {}

  reset() {
    this.abort?.abort(); this.abort = undefined;
    this.phase = undefined; this.pending = false; this.reply = undefined; this.snapshotMessages = 0;
  }

  poll(phase: string, messages: readonly Message[], signal?: AbortSignal) {
    if (phase !== this.phase) { this.reset(); this.phase = phase; }
    if (!this.pending) {
      this.pending = true;
      const abort = this.abort = new AbortController();
      const snapshot = [...messages];
      this.snapshotMessages = snapshot.length;
      const joined = signal ? AbortSignal.any([signal, abort.signal]) : abort.signal;
      this.audit({ type: 's2_background_start', phase, messages: snapshot.length });
      void this.run(snapshot, joined).then(reply => {
        if (!joined.aborted && this.abort === abort) this.reply = reply;
      }).catch(error => {
        if (!joined.aborted && this.abort === abort) {
          this.audit({ type: 's2_background_error', error: String(error) });
          // Background failures are advisory too; they must never interrupt S1.
          this.reply = { role: 'assistant', provider: 'taiji-proxy', api: 'openai-completions', model: '',
            timestamp: Date.now(), content: [], stopReason: 'error', errorMessage: String(error),
            usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
        }
      });
    }
    return undefined;
  }

  /** Only an explicit S1 handoff may consume a prepared suggestion. Never execute it. */
  takeForHandoff(phase: string) {
    const prepared = phase === this.phase && this.reply && !['error', 'aborted'].includes(this.reply.stopReason)
      ? { response: this.reply, snapshot_messages: this.snapshotMessages } : undefined;
    this.reset();
    return prepared;
  }
}

export function replayResponse(message: AssistantMessage) {
  const stream = createAssistantMessageEventStream();
  stream.push({ type: 'start', partial: message });
  if (message.stopReason === 'error' || message.stopReason === 'aborted') {
    stream.push({ type: 'error', reason: message.stopReason, error: message });
  } else {
    for (const [contentIndex, block] of message.content.entries()) {
      if (block.type === 'toolCall') {
        stream.push({ type: 'toolcall_start', contentIndex, partial: message });
        stream.push({ type: 'toolcall_end', contentIndex, toolCall: block, partial: message });
      } else if (block.type === 'text') {
        stream.push({ type: 'text_start', contentIndex, partial: message });
        stream.push({ type: 'text_delta', contentIndex, delta: block.text, partial: message });
        stream.push({ type: 'text_end', contentIndex, content: block.text, partial: message });
      } else if (block.type === 'thinking') {
        stream.push({ type: 'thinking_start', contentIndex, partial: message });
        stream.push({ type: 'thinking_delta', contentIndex, delta: block.thinking, partial: message });
        stream.push({ type: 'thinking_end', contentIndex, content: block.thinking, partial: message });
      }
    }
    stream.push({ type: 'done', reason: message.stopReason as 'stop' | 'length' | 'toolUse', message });
  }
  stream.end(message);
  return stream;
}
