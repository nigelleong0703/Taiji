# Browser agent example

A browser agent driven by Taiji: the model picks each operation and target on the page, writes the text for
fields it chose to type into, and a System Two LLM is asked for a subgoal only when the agent is stuck (BLOCKED, three
actions without a page change, the same action on the same page three times, or a page that keeps changing).

`harness/` is [Browser Use's jev-ultrafast](https://github.com/browser-use/jev-ultrafast) (MIT) with that escalation
and a configurable decision endpoint. It is one integration; the model itself does not depend on it.

## Run

Start the server (repository root, see the main README), then in `harness/` create an env file:

```bash
TYPESAFE_URL=http://<host>:8000/v1/systemone      # Taiji decides
TYPESAFE_API_KEY=<S1_API_KEY>
TYPESAFE_SCREENSHOT=1                             # this model uses the screenshot (the date pickers need it)
TEXT_MODEL=s1                                     # the same model writes field values
TEXT_MODEL_BASE_URL=http://<host>:8000/v1
TEXT_MODEL_API_KEY=<S1_API_KEY>
S2_MODEL=<any OpenAI-compatible model>            # System Two, only when stuck
S2_MODEL_BASE_URL=<its base URL>
S2_MODEL_API_KEY=<its key>
```

```bash
cd harness && uv sync
uv run --env-file my.env python ../agent_eval.py --tasks wikipedia_godel,books_travel,flights_bali --out report.json
```

`agent_eval.py` passes its independent checker into the agent. A `DONE` choice only stops the run if the checker
confirms the final page; otherwise System Two gets the failed check and can give the fast policy a corrective
subgoal. A failed check cannot be overridden by another model saying “done.”

### Run System Two with vLLM

Use Taiji's S1 server for its trained decision head and vLLM as the larger System Two planner. On a GPU host with
vLLM installed, start a supported Qwen3.5 instruct model:

```bash
export S2_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
vllm serve Qwen/Qwen3.5-9B-Instruct --host 127.0.0.1 --port 8001 \
  --served-model-name taiji-s2 --api-key "$S2_API_KEY"
```

Set these in the browser-agent env file:

```bash
S2_MODEL=taiji-s2
S2_MODEL_BASE_URL=http://127.0.0.1:8001/v1
S2_MODEL_API_KEY=<the same S2_API_KEY>
S2_MODEL_REASONING=omit
```

This assumes the browser agent runs on the same host. For a remote agent, keep vLLM bound to loopback and forward
port 8001 over SSH. System Two is called only when S1 is stuck or the independent verifier rejects `DONE`, so the
larger model handles planning and recovery while Taiji handles frequent page actions. Qwen3.5 is supported by
current vLLM releases; the exact model size must fit the GPU and context budget. See [vLLM's supported-model list](https://docs.vllm.ai/en/latest/models/supported_models/).

The agent attaches to your local Chrome (remote debugging on) and opens its own window. To run it hidden, start a
separate headless Chrome and point the harness at it:

```bash
google-chrome --headless=new --remote-debugging-port=9333 --user-data-dir=/tmp/taiji-chrome about:blank &
BU_NAME=hidden BU_CDP_URL=http://127.0.0.1:9333 uv run --env-file my.env python ../agent_eval.py --tasks books_travel
```

Results so far are in the main README (simple sites pass; Google Flights reaches the right results but does not yet
declare completion).
