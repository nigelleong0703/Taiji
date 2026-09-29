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

`agent_eval.py` checks every run's final page itself (URL, form values, text); a DONE choice is not trusted.

The agent attaches to your local Chrome (remote debugging on) and opens its own window. To run it hidden, start a
separate headless Chrome and point the harness at it:

```bash
google-chrome --headless=new --remote-debugging-port=9333 --user-data-dir=/tmp/taiji-chrome about:blank &
BU_NAME=hidden BU_CDP_URL=http://127.0.0.1:9333 uv run --env-file my.env python ../agent_eval.py --tasks books_travel
```

Results so far are in the main README (simple sites pass; Google Flights reaches the right results but does not yet
declare completion).

`teacher.py` / `teacher_round.sh`: an experiment that used a hosted Jev-compatible model as a teacher on varied sites
(checked outcomes only). It was stopped: about 9% of runs passed, so it is kept for reference, not used for training.
