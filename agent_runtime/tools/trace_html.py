"""Render a runtime trace as a step-by-step HTML replay, including the exact request sent to S1.

The request recorded on every decision is the single source of truth: it is what the model actually
saw, and unlike the model-facing tool text it is never truncated.

Usage: python -m tools.trace_html <trace.json> <out.html>
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")


def panels_of(trace):
    """Group trace events into one panel per model decision, plus the initial page open."""
    steps = trace.get("steps", [])
    panels = [{"kind": "setup", "step": step} for step in steps if step.get("source") == "setup"]
    current = None
    for event in trace.get("trace", []):
        if event["event"] == "s1_decision":
            current = {"kind": "decision", "decision": event, "tools": [], "notes": []}
            panels.append(current)
        elif event["event"] == "mcp_tool" and current is not None:
            current["tools"].append(event)
        elif event["event"] == "s2_reflection" and current is not None:
            current["notes"].append(event)
    by_step = {}
    for step in steps:
        by_step.setdefault(step.get("step"), []).append(step)
    for panel in panels:
        if panel["kind"] == "decision":
            number = panel["decision"].get("step")
            panel["executed"] = [s for s in by_step.get(number, []) if s.get("browser_action")]
    return panels


def chosen_of(panel):
    executed = panel.get("executed") or []
    return (executed[0].get("browser_action") or {}) if executed else {}


def screenshot_block(image):
    """The picture the model was given, rendered as a picture.

    The recorded request carries it as bare base64; dumping that into the JSON panel buried the
    request under ~60 KB of image data, so it is shown here and only referenced there.
    """
    if not image:
        return "<p class='muted'>这一步没有附带截图（纯文本决策）。</p>"
    return ("<img class='shot' alt='viewport the model saw' src='data:image/jpeg;base64,%s'>"
            "<p class='muted'>原始截图 %d 字符（base64 JPEG）；已从下面的请求体里摘出。</p>"
            % (image, len(str(image))))


def element_rows(elements):
    """The element table the model saw, from the request itself."""
    rows = []
    for element in elements or []:
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                html.escape(str(element.get("index", ""))),
                html.escape(str(element.get("label", ""))[:80]),
                html.escape(", ".join(element.get("operations") or [])),
                html.escape(str(element.get("role", ""))),
                html.escape(str(element.get("value", element.get("current_value", "")))[:40])))
    if not rows:
        return "<p class='muted'>请求里没有元素表。</p>"
    return ("<table class='grid'><thead><tr><th>#</th><th>元素</th><th>可用操作</th><th>role</th>"
            "<th>当前值</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def matches(chosen_label, element_text):
    """Best-effort join from the executed action back to the option text the model chose."""
    text = element_text.split("]", 1)[-1].strip()
    if chosen_label and chosen_label in text:
        return True
    base = chosen_label[5:] if chosen_label.startswith("Open ") else chosen_label
    return bool(base) and base in text


def option_rows(question_key, question, chosen_label, chosen_operation):
    criteria = question.get("criteria") or {}
    if not isinstance(criteria, dict):
        return ""
    rows = []
    for key, value in criteria.items():
        if isinstance(value, dict):
            label = str(value.get("element", ""))
            extra = " / ".join(
                "%s=%s" % (k, v) for k, v in value.items() if k != "element" and v not in ("", None))
        else:
            label, extra = str(value), ""
        hit = (question_key == "operation" and str(key) == str(chosen_operation)) \
            or (question_key != "operation" and matches(chosen_label, label))
        rows.append("<tr%s><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            " class='chosen'" if hit else "", html.escape(str(key)), html.escape(label[:90]),
            html.escape(extra[:80])))
    return ("<table class='grid'><thead><tr><th>选项</th><th>内容（模型看到的原文）</th><th>附加上下文</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def observed_actions(panel):
    """Only newer traces carry the raw observation; show it when present, including duplicate nodes."""
    for tool in panel.get("tools") or []:
        actions = tool.get("observed_actions")
        if not actions:
            continue
        counts = {}
        for action in actions:
            counts[action.get("node")] = counts.get(action.get("node"), 0) + 1
        rows = []
        for action in actions:
            duplicate = counts.get(action.get("node"), 0) > 1 and action.get("node") is not None
            rows.append("<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                " class='duprow'" if duplicate else "", html.escape(str(action.get("id"))),
                html.escape(str(action.get("kind"))), html.escape(str(action.get("label"))[:80]),
                html.escape(str(action.get("node")))))
        return ("<details><summary>本步的真实观测动作（含 id 与 node；同节点多动作为黄色）</summary>"
                "<table class='grid'><thead><tr><th>id</th><th>kind</th><th>label</th><th>node</th></tr>"
                "</thead><tbody>" + "".join(rows) + "</tbody></table></details>")
    return ""


def render_panel(panel, index):
    if panel["kind"] == "setup":
        step = panel["step"]
        return ("<section id='p%d' class='panel'><h2>第 0 步 · 打开页面</h2>"
                "<p class='muted'>运行时先按 <code>--url</code> 打开起始页；动作表由这次观测产生。</p>"
                "<p>工具 <code>%s</code>，参数 <code>%s</code>，耗时 %s ms，状态 %s</p>"
                "</section>" % (index, html.escape(step.get("tool", "")),
                                html.escape(json.dumps(step.get("arguments"), ensure_ascii=False)),
                                step.get("duration_ms"), html.escape(str(step.get("status")))))

    decision = panel["decision"]
    request = decision.get("request") or {}
    state = request.get("state") or {}
    page = state.get("page") or {}
    image = state.get("screenshot")
    request_text = json.dumps(
        {**request, "state": {**state, "screenshot": "omitted here; the picture is above"}},
        ensure_ascii=False, indent=2) if image else json.dumps(request, ensure_ascii=False, indent=2)
    questions = request.get("questions") or {}
    chosen = chosen_of(panel)
    probabilities = decision.get("probabilities") or {}
    bars = "".join(
        "<div class='bar'><span>%s</span><i style='width:%d%%'></i><b>%.3f</b></div>"
        % (html.escape(str(key)), int(float(value) * 100), float(value))
        for key, value in sorted(probabilities.items(), key=lambda item: -item[1])[:8])
    target_questions = "".join(
        "<h3>target 选项：%s</h3>%s" % (html.escape(key), option_rows(key, question,
                                                                      chosen.get("label") or "",
                                                                      decision.get("operation")))
        for key, question in questions.items() if key != "operation")
    executed = "".join(
        "<p>执行 <code>%s</code>，参数 <code>%s</code>，状态 <b>%s</b>，耗时 %s ms</p>"
        % (html.escape(str(item.get("browser_action", {}).get("id"))),
           html.escape(json.dumps(item.get("arguments"), ensure_ascii=False)),
           html.escape(str(item.get("status"))), item.get("duration_ms"))
        + ("<p>由 S1 写出的字段文本：<b>%s</b></p>" % html.escape(str((item.get("arguments") or {}).get("text")))
           if (item.get("arguments") or {}).get("text") else "")
        for item in panel.get("executed") or [])
    notes = "".join(
        "<div class='note'><b>S2 反思</b>（触发：%s）→ <i>%s</i>：%s</div>"
        % (html.escape(str(note.get("trigger"))), html.escape(str(note.get("verdict"))),
           html.escape(str(note.get("text"))[:400]))
        for note in panel["notes"])

    return ("<section id='p%d' class='panel'><h2>第 %s 步 · S1 决策</h2>"
            "<p class='muted'>operation=<b>%s</b> · 选择 <b>%s</b>（%s） · confidence %.3f · 决策耗时 %s ms</p>"
            "<p>页面：%s &nbsp;|&nbsp; 标题：%s</p>"
            "<h3>模型看到的画面</h3>%s"
            "<h3>元素表（模型在此编号上选择）</h3>%s"
            "<h3>operation 选项</h3>%s"
            "%s"
            "<h3>模型给出的概率</h3>%s"
            "<h3>执行</h3>%s%s%s"
            "<details><summary>发给 S1 的完整请求体（原样 JSON）</summary><pre>%s</pre></details>"
            "<details><summary>页面文本</summary><pre>%s</pre></details>"
            "</section>"
            % (index, html.escape(str(decision.get("step"))), html.escape(str(decision.get("operation"))),
               html.escape(str(decision.get("choice"))), html.escape(str(chosen.get("kind"))),
               float(decision.get("confidence") or 0), decision.get("latency_ms"),
               html.escape(str(page.get("url", ""))), html.escape(str(page.get("title", ""))),
               screenshot_block(image),
               element_rows(state.get("elements")),
               option_rows("operation", questions.get("operation") or {}, chosen.get("label") or "",
                           decision.get("operation")),
               target_questions, bars, executed, notes, observed_actions(panel),
               html.escape(request_text),
               html.escape((page.get("text") or "")[:8000])))


CSS = """
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0f1115;color:#e6e8ee}
header{padding:14px 24px;border-bottom:1px solid #262a33;background:#151922;position:sticky;top:0;z-index:5}
h1{font-size:16px;margin:0 0 6px}h2{font-size:15px;margin:0 0 8px}
h3{font-size:13px;margin:18px 0 6px;color:#9aa4b2}
.wrap{display:flex;align-items:flex-start}
img.shot{max-width:720px;border:1px solid #262a33;border-radius:6px;display:block}
nav{width:250px;padding:14px;border-right:1px solid #262a33;overflow:auto;
  height:calc(100vh - 96px);position:sticky;top:96px}
nav button{display:block;width:100%;text-align:left;margin:0 0 4px;padding:7px 9px;
  border:1px solid #262a33;border-radius:6px;background:#171b24;color:#cfd6e4;
  cursor:pointer;font:12px/1.35 inherit}
nav button:hover{border-color:#3b4a63}
nav button.on{background:#243044;border-color:#3f6ad8;color:#fff}
main{flex:1;padding:18px 24px 80px;max-width:1150px}
.panel{display:none}.panel.on{display:block}
.muted{color:#8b95a5}
table.grid{width:100%;border-collapse:collapse;font-size:12.5px;margin:4px 0}
table.grid th,table.grid td{border:1px solid #262a33;padding:4px 7px;text-align:left;vertical-align:top}
table.grid th{background:#171b24;color:#9aa4b2}
tr.chosen{background:#1d3a24}
tr.chosen td:first-child{border-left:3px solid #46c46b}
tr.duprow{color:#e0b341}
pre{background:#12161e;border:1px solid #262a33;border-radius:6px;padding:10px;overflow:auto;
  max-height:460px;font-size:11.5px;white-space:pre-wrap}
code{background:#1b2029;padding:1px 5px;border-radius:4px}
.bar{display:flex;align-items:center;gap:8px;margin:2px 0;font-size:12px}
.bar span{width:170px;color:#9aa4b2}
.bar i{height:10px;background:#3f6ad8;display:inline-block;border-radius:3px;min-width:2px}
.note{background:#2a2418;border:1px solid #5a4a20;border-radius:6px;padding:8px 10px;margin:8px 0}
details{margin:8px 0}summary{cursor:pointer;color:#9aa4b2}
"""

JS = """
const panels=[...document.querySelectorAll('.panel')];
const buttons=[...document.querySelectorAll('nav button')];
let at=0;
function show(i){at=Math.max(0,Math.min(panels.length-1,i));
 panels.forEach((p,k)=>p.classList.toggle('on',k===at));
 buttons.forEach((b,k)=>b.classList.toggle('on',k===at));
 buttons[at].scrollIntoView({block:'nearest'});}
buttons.forEach((b,k)=>b.onclick=()=>show(k));
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')show(at+1);if(e.key==='ArrowLeft')show(at-1);});
show(0);
"""


def build_html(trace, panels):
    goal = ""
    for panel in panels:
        if panel["kind"] != "decision":
            continue
        questions = (panel["decision"].get("request") or {}).get("questions") or {}
        goal = (questions.get("operation") or {}).get("instructions", {}).get("goal", "") or goal
        if goal:
            break
    latencies = sorted(d["latency_ms"] for d in (p["decision"] for p in panels if p["kind"] == "decision")
                       if d.get("latency_ms"))
    nav = "".join(
        "<button>%s</button>" % html.escape(
            "第 0 步 · 打开页面" if panel["kind"] == "setup" else
            "第 %s 步 · %s" % (panel["decision"].get("step"),
                               (chosen_of(panel).get("label") or chosen_of(panel).get("kind") or "")[:24]))
        for panel in panels)
    body = "".join(render_panel(panel, index) for index, panel in enumerate(panels))
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>Taiji trajectory</title>"
            "<style>" + CSS + "</style></head><body>"
            "<header><h1>Taiji agent trajectory</h1>"
            "<div class='muted'>__GOAL__</div>"
            "<div class='muted'>状态 __STATUS__ · 总耗时 __ELAPSED__ ms · __COUNT__ 步 · "
            "S1 决策耗时中位数 __MEDIAN__ ms · ← → 或点左侧切换</div></header>"
            "<div class='wrap'><nav>__NAV__</nav><main>__BODY__</main></div>"
            "<script>" + JS + "</script></body></html>")
    for token, value in (("__GOAL__", html.escape(goal)), ("__STATUS__", html.escape(str(trace.get("status")))),
                         ("__ELAPSED__", str(trace.get("elapsed_ms"))), ("__COUNT__", str(len(panels))),
                         ("__MEDIAN__", str(latencies[len(latencies) // 2] if latencies else "?")),
                         ("__NAV__", nav), ("__BODY__", body)):
        page = page.replace(token, value)
    return page


def main():
    trace_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else trace_path.with_suffix(".html")
    trace = json.loads(trace_path.read_text())
    panels = panels_of(trace)
    out_path.write_text(build_html(trace, panels))
    print("wrote", out_path, len(panels), "panels")


if __name__ == "__main__":
    main()
