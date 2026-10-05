import { splitTool } from './environments.js';

/** Decode the inline MCP payload; observations never require a file read. */
export function inlinePage(event: { content: { type: string; text?: string }[]; structuredContent?: any; details?: any }) {
  const raw = event.details?.inline_page ?? event.structuredContent?.structuredContent ?? event.structuredContent;
  if (raw && Array.isArray(raw.actions)) return raw;
  const content = event.structuredContent?.content ?? event.content;
  try {
    const page = JSON.parse(content.filter((b: any) => b.type === 'text').map((b: any) => b.text).join('\n'));
    if (!Array.isArray(page.actions)) return undefined;
    if (page.action_columns) {
      const { action_columns, ...rest } = page;
      return { ...rest, actions: page.actions.map((row: unknown[]) => Object.fromEntries(action_columns.map((k: string, i: number) => [k, row[i]]))) };
    }
    return page;
  } catch { return undefined; }
}

/** S2 sees page state; S1 retains the complete actionable snapshot inline. */
export function pageState(page: any) {
  return { tab_id: page.tab_id, ...(page.target_id !== undefined ? { target_id: page.target_id } : {}), url: page.url, title: page.title, text: page.text,
    controls: page.actions.filter((a: any) => ['value', 'checked', 'selected', 'expanded'].some(k => k in a))
      .map((a: any) => Object.fromEntries(['id', 'label', 'role', 'value', 'checked', 'selected', 'expanded'].filter(k => k in a).map(k => [k, a[k]]))),
    links: page.actions.filter((a: any) => a.href || a.role === "link").map((a: any) => ({ id: a.id, label: a.label, href: a.href })),
    action_count: page.actions.length };
}

export function compactObservation(event: { toolName: string; toolCallId: string; isError: boolean;
  content: { type: string; text?: string }[]; details?: unknown; structuredContent?: any }) {
  // Any environment server (browser, desktop, ...) whose result is a page with an action table.
  const tool = splitTool(event.toolName)?.tool;
  if (event.isError || !tool || tool === 'screenshot_image' || tool.startsWith('list')) return undefined;
  const page = inlinePage(event);
  if (!page) return undefined;
  const compact = JSON.stringify(pageState(page));
  const { fullOutputPath: _discard, ...details } = (event.details ?? {}) as any;
  return { content: [{ type: 'text' as const, text: compact }], details: { ...details, inline_page: page },
    structuredContent: { content: [], structuredContent: page },
    measurement: { original_bytes: Buffer.byteLength(JSON.stringify(page)), compact_bytes: Buffer.byteLength(compact), actions: page.actions.length } };
}
