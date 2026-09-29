# Evaluation and measurement notes

The figures below are project-reported measurements for the published Taiji-2B weights. They are not independently audited. The repository currently distributes the inference engine and demos; data conversion and model-training code are maintained separately and are not included in this public release.

## Decision benchmarks

**Jev-format comparison:** 490 held-out questions across intent classification, yes/no, scoring, and tool choice, evaluated with the same grading script and hosted endpoint:

| Questions | n | Jev (hosted) | Taiji-2B |
|---|---:|---:|---:|
| All | 490 | 0.859 | **0.882** |
| General choice | 248 | 0.891 | 0.899 |
| General yes/no | 33 | 0.970 | 0.970 |
| General score | 19 | 0.632 | 0.684 |
| Tool choice | 116 | 0.871 | 0.914 |
| Tool yes/no | 74 | 0.743 | 0.784 |

The systems disagreed on 57 questions (Taiji correct on 34, Jev on 23; McNemar p ≈ 0.19), which supports “on par” rather than a proven win. The questions come from public dataset domains represented in training, so this comparison does not establish generalization to unrelated domains.

On a separate 2,945-row held-out decision set, the calibrated model reached 0.86 accuracy and 0.02 expected calibration error. Field text exact match was 0.76.

## Speed and memory

Server time on one H100 for a real Google Flights page request (~6,500 input tokens, screenshot, and three questions) was 219 ms median: 171 ms model time and 59 ms input preparation. Network latency is additional. Serving used about 6 GB of VRAM. These figures depend on hardware, request length, and software versions; measure your own workload before sizing a deployment.

## Browser example

The included browser integration passed 6/6 simple-site tasks (two Wikipedia tasks and books.toscrape). On Google Flights, it filled the route, selected dates, and sorted by price, but did not reliably declare completion; `p(DONE)` stayed below 0.1 on the finished results page. Without a screenshot it could not operate the date picker.

## Known limitations

- Held-out accuracy does not guarantee correct real-world actions. A `DONE` decision is not proof of task completion; verify outcomes independently.
- Completion detection is weak (held-out completion recall 0.44 on WebLINX-derived rows).
- Browser requests may include every page element and question instruction; measured examples exceeded 6,000 tokens.
- Training data has mixed licenses and terms. Review each original dataset card before reuse. WebLINX is CC BY-NC-SA 4.0, which is why the released weights are non-commercial.
