import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { compactObservation } from "../src/observations.js";
import { browserContext, names } from "../src/bridge.js";

test("new result compacts metadata without losing original page state, prices or field values", () => {
  const directory = mkdtempSync(resolve(tmpdir(), "taiji-observation-"));
  const page = { url: "https://example.test", title: "Test", text: "Total SGD 376", actions: Array.from({ length: 20 }, (_, i) => ({ id: `r${i}`, node: i,
    kind: "fill", label: "Departure", role: "textbox", value: "2026-12-27", expanded: false })) };
  const event = { toolName: names.observe, toolCallId: "observation", isError: false, details: { server: "browser" },
    content: [{ type: "text", text: JSON.stringify(page, null, 2) }] };
  const result = compactObservation(event)!;
  assert.ok(result.measurement.compact_bytes < result.measurement.original_bytes);
  const shown = JSON.parse(result.content[0].text);
  assert.equal(shown.text, page.text);
  assert.equal(shown.controls[0].value, "2026-12-27");
  assert.equal(shown.controls[0].expanded, false);
  assert.deepEqual(readdirSync(directory), []);
  assert.equal(result.details.fullOutputPath, undefined);
  const context = browserContext([{ role: "toolResult", toolName: names.observe, toolCallId: "observation",
    content: result.content, details: result.details, structuredContent: result.structuredContent, timestamp: 1, isError: false } as any]);
  assert.deepEqual(context.page, page);
});

test("non-page and failed results are left unchanged", () => {
  const event = { toolName: names.observe, toolCallId: "bad", isError: false, content: [{ type: "text", text: "not JSON" }] };
  assert.equal(compactObservation(event), undefined);
  assert.equal(compactObservation({ ...event, isError: true }), undefined);
});
