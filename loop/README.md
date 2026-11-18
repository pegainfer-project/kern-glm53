# The GLM-5.3 PP4 kernel loop

Agents optimize the PP4 prefill's kernels overnight against one number, the AgentX corpus's mean time to
first token on an idle PP4 (`score`), on a sub-model small enough to measure in a minute (layers 0-10 and
the head, `gen`). The orchestrator merges what passes into `main` and logs each merge in
[`HISTORY.md`](HISTORY.md), the climb. The agents' brief is [`TASK.md`](TASK.md); each run's target is
under [`rounds/`](rounds/), and each run keeps its memory in `notes/<run>.md`.

## The score

[`agentx.py`](agentx.py) replays the corpus (`semianalysis cc-traces-weka-062126-256k`, 68k requests)
through an infinite prefix cache, cuts each request's new tokens into 16384-row chunks and prices the
chunks on the bench grid ([`workload.toml`](workload.toml)): 128 / 512 / 2048 / 8192 / 16384 rows over
0-256k of context, at the serving manifest's bounds (1M). [`weights.json`](weights.json) is the result,
one weight a grid point; [`score`](score) projects every point's time from the 11 layers to 78 and sums.

## Gates

| | what | GPUs |
|---|---|---|
| [`check`](check) | `kern test` against main: every changed span, the end-to-end logits, the noise floor (4096 rows, 64k bounds) | 1 |
| [`longcheck`](longcheck) | the 1M bounds, teacher-forced over 262144 tokens of real text in 16 chunks, against main | 1 |
| [`fullcheck`](fullcheck) | the four serving stages, six chat prompts, against SGLang and main (orchestrator only) | 4 |

## GPUs

[`withgpu`](withgpu) leases GPUs with one flock per GPU and skips GPUs that run anything outside a lease;
`touch <locks>/drain` on a host holds new partial leases there until a whole-host lease gets it.

## Running it

Copy [`local.env.example`](local.env.example) to `local.env` and fill in the paths. Then:

```sh
loop/build && loop/gen                       # kernels, manifests, launch counts
loop/bench loop/out/l11.json work/main       # a report and its score
loop/agents up idx 10 loop/rounds/r1/idx.md
loop/agents status
```
