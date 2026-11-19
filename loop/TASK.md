# Task: GLM-5.3's PP4 prefill, shortest time to first token

GLM-5.3 (78 layers: DeepSeek V3.2's sparse attention, an indexer on layers 0-2 and every fourth from 6,
the other layers reusing the last indexer's picks; 256-expert top-8 MoE from layer 3; FP8 128x128 block
weights) is served on GB300 (`sm_103a`, 152 SMs, ~8 TB/s HBM) by the kern runtime as a packed prefill cut
into four pipeline stages, one GPU each: up to 16 sequences and 16384 rows a call, contexts up to 1M. The
forward pass is a manifest (`scripts/gen.py`, its kernels' launches in `scripts/glm_ops.py`); kern runs
it blindly. `README.md` describes the forward and its kernels. `ROUND.md` names your target; this file
covers what applies to every round.

## The objective

**The AgentX corpus's mean time to first token on an idle PP4, `loop/score`'s `all`.** Every request of
the corpus (68k, prefixes cached as an infinite cache would) is a run of 16384-row chunks over its
session's context; `loop/weights.json` prices them on the bench grid (`loop/agentx.py` says how). Two
scenarios make it up, and both count:

- `short`: a request of one chunk, a turn over its session's context (96% of requests; median 1.6k new
  tokens over 85k).
- `long`: a request of several chunks, a long fresh prompt (4% of requests, 44% of the new tokens).

The bench grid is 128 / 512 / 2048 / 8192 / 16384 rows over 0 / 64k / 128k / 192k / 256k of context;
the corpus stops at 256k. Serving runs to 1M, and the 1M bounds are what every bench and the 1M gate run.

`main` at the loop's start, projected to the 78-layer model from the 11-layer sub-model:

| | short | long | all |
|---|---|---|---|
| mean TTFT | 271.6 ms | 1229.6 ms | 309.2 ms |

Where it goes (`loop/score <report> --by-op`, share of `all`): the routed MoE GEMMs 44% (fp8.moe.gu 29,
fp8.moe.dn 15), FlashMLA's sparse attention 17.5%, fp8.o 5.5%, silu_moe 4.3%, the indexer ~13% at
2048 rows over 128k and 23% at 16384 over 128k, then glue (combine, norms and quantization, gather,
q_post, kv_dequant, uk / uv) about 12%. At 128-512 rows the MoE reads all of its weights and sits at
HBM's roofline; at 16384 rows it is near fp8's.

## The rules

- **Fixed**: PP4 (four stages, one GPU each; a cut only at an indexer layer), the latent cache's format
  (SGLang's fp8 DSA row, 656 bytes a token) and the indexer cache's (e4m3 keys, f32 scale, 132 bytes a
  token), the weights' precision (e4m3 with UE8M0 block scales), the kern runtime (`$GLM_KERN`: read it,
  never change it; a needed runtime feature goes to `notes/<run>.md` as a proposal).
- **Open**: every kernel and every launch, the GEMMs included (another tile, another kernel, a fused
  epilogue); the generator; how the indexer computes. Its logits may be bf16 (as vLLM runs DSA on SM100);
  any other new rounding is a numerics change that must be measured and explained (below).
- Program lanes (`"lane"` / `"after"` on a call, `$GLM_KERN/docs/manifest.md`) and PDL are available.
  A call on a side lane must launch at every var value: one whose every launch a `when` gates off, joined by
  a PDL call, can hang a captured replay (seen twice at 16384 rows).

## What you edit

- `source/`: kernel CUDA. A new kernel or variant gets an entry in `kernels.toml` (`source`, `flags`,
  `includes`, `defines`); `loop/build --update` compiles everything with the pinned nvcc and repins
  whatever changed (byte-identical in every worktree). Vendored upstream code keeps its license and a
  provenance note beside it (see `source/k31/README.md`).
- `scripts/gen.py`, `scripts/glm_ops.py`: the ops and the calls.
- `notes/<run>.md`: your memory (below). Your own scratch harnesses under `work/`.

Never edit `loop/`. Never commit `build/`, `loop/out/` or anything generated.

## Tools

All take paths relative to the repository root and lease GPUs themselves through `loop/withgpu`; never
run GPU work outside it. Outputs must be on the shared filesystem (the GPUs are on another host): use
`work/`.

| Command | What it does | GPUs | Time |
|---|---|---|---|
| `loop/build [--update]` | compile kernels, check (or repin) their hashes | none | ~1 min |
| `loop/gen [OUT]` | the sub-model, layers 0-10 and the head: `l11.json` (1M bounds, bench and 1M gate), `l11c.json` (64k bounds, span check); launches per layer | none | 5 s |
| `loop/count M...` | launches, per layer of each kind | none | 0 s |
| `loop/bench M OUT` | kern bench of `l11.json` over the grid; prints each shape and `loop/score` | 1 | ~1 min |
| `loop/ab A B OUT [N]` | A B B A ... on one GPU; per-shape medians and the score's change. The number to quote | 1 | ~2 min a pair |
| `loop/pp A B OUT [N]` | the PP4 serving stages (scripts/deploy.sh's `p`), A against B, all four GPUs; `pp` is stage 0 (layers 0-17, real inputs), the orchestrator's gate; `pp1-3` runs on synthetic activations, printed only | 4 | ~4.5 min a pair |
| `loop/graph ABDIR` | the same A/B's sub-model graph p50s over the corpus, unprojected: a win shows here too | 0 | seconds |
| `loop/score REPORT [--by-op] [--grid]` | mean TTFT short / long / all; by op; per shape | none | 0 s |
| `loop/longcheck M OUT` | **the numerics gate**: `l11.json` (1M bounds) teacher-forced against main's recordings, `long` (262144 tokens of real text in sixteen 16384-row chunks) and `short` (six chat prompts, 300-12k tokens, three chunks each); top-1 agreement, \|dlp\|, flips and confident flips | 1 | ~3 min |
| `loop/check M OUT` | optional, to localize: `kern test` of `l11c.json` against main's, span by span (which call's outputs moved, by how much) | 1 | ~25 min |

`kern bench` checks that a replay emits the tokens the one before it did: a kernel that is not
deterministic run to run fails the bench. Keep every kernel deterministic (no atomics deciding an order
or a set that reaches the output).

Main's manifests and reports are in `$GLM_LOOP_DATA/base/` (`l11.json`, `l11c.json`, `bench-*.json`):
do not re-measure main, compare against it with `loop/ab`.

## Numerics

Not bit-exactness, but every difference explained. A change passes `loop/longcheck`, and its commit
message says where its numerical difference from main comes from (a summation order, an f32
intermediate now bf16, another exp, ...) and its measured size (`loop/longcheck`'s two lines).
"Bit-identical" is a fine explanation when the gate says so. A FAIL means the change is wrong, never the
tolerance. A near-tie that flips is noise; a confident token that flips is a bug.

`loop/longcheck`'s floor: the 11-layer model's distributions are flat, and any change of rounding or
summation order moves near-tied indexer logits across the top-k's cut, which moves the picks. A run of
vLLM's top-k against another of itself (its picks' order varies run to run):

| pass | top-1 | confident flips | \|dlp\| mean |
|---|---|---|---|
| long | 91.35-91.41% | 2-3 | 0.069 |
| short | 94.84-95.04% | 0 | 0.041 |

A change inside it (long: top-1 >= 90.8%, confident <= 4, |dlp| <= 0.075; short: top-1 >= 94.3%,
confident <= 2, |dlp| <= 0.045) passes as order noise; `next_token DIFFERS` alone is expected there. One
outside it is a numerics change: say why in the commit; the orchestrator runs the 78-layer model against
SGLang before merging it.

## Commits and wins

- One optimization a commit (`git commit -s`), subject `glm53: <what is true after the commit, lowercase,
  no period>`. The message has: the score before -> after from `loop/ab` (short / long / all), the shapes
  it moved, launches per layer before -> after when they changed, the numerics line with `loop/longcheck`'s
  two lines.
- A commit is a win when `loop/longcheck` passes and `loop/ab` shows `all` lower (by 0.1%) and neither `short`
  nor `long` higher beyond noise (0.3%), and `loop/graph` on the same A/B shows `all` not higher; fewer launches
  at equal time is a win too. The score projects 11 layers to 78 by each call's share of its time, and a PDL
  call's time holds its wait on the call before, so splitting an op into more calls can move the score alone:
  the graph doesn't move that way.
  The orchestrator gates on `loop/pp`'s stage 0 (all lower by 0.1%, short and long not up by 0.3%). Stages 1-3
  run on synthetic activations, so a data-dependent kernel (the top-k, the MoE's routing) is measured on real
  inputs only in stage 0 and the sub-model; for deep layers use real dumps (idx's notes: chain the stages). Commit each win as soon as it
  passes. Keep every commit building and passing on its own.
- The orchestrator merges wins into `main`, re-measures there and logs each in `loop/HISTORY.md`; when
  `main` moves it tells you in `ROUND.md`: rebase onto it then.
- This is a public repository. No machine names, user names, absolute paths or internal hostnames in
  anything you commit (use `$VARS` or `<placeholders>`).

## References (read-only)

- kern runtime and manifest format: `$GLM_KERN` (`docs/manifest.md`, `docs/runtime.md`, `docs/bench.md`,
  `crates/kern-manifest/src/types.rs`). A launch can carry `when: {var, min, max}` to route an op to
  different kernels by a var's value.
- `$GLM_K31`: kern-k31, Kimi-K3.1 on kern, whose MLA layers carry the same DSA indexer (32 heads x 128,
  top 2048). Its `git log` is a record of P-side optimizations with measurements: bf16 MQA logits on
  DeepGEMM 2.8 (e1c8cf0), the top-k's bf16 pair compares (69d5583), the tcgen05 absorb and v-up
  (d421893, 51b4aed), lanes (df94212, 4c7c7f8). `source/k31/k31_topk.cu` here is its top-k at c4a820a.
- `$GLM_ATLAS`: kern-atlas, an index of kernels by op class (ours and upstream's): look there before
  writing one.
- `$GLM_REF/`: upstream sources: `sglang`, `vllm`, `flashinfer`, `trtllm`, `DeepGEMM`; `insights.md`,
  fusion ideas collected from them for K3.
- `third_party/`: DeepGEMM and FlashMLA at the pinned commits.
- Skills: `KernelWiki` (Blackwell kernel techniques) and `ncu-report-skill` (`ncu` is installed). The GPU
  is GB300, not B200: adjust.

## Boundaries

No network (no curl, pip, git fetch). Read only your worktree and the paths above. Never search home
directories or `find /`. Never touch processes or containers you did not start.

## Time and sessions

`ROUND.md` ends with your deadline; the session is killed then, so commit each win as soon as it passes.
Keep working until the deadline: an edit -> build -> gen -> longcheck -> ab cycle is 5-15 minutes. If your
target is closed (at its roofline, or fused as far as it sensibly goes), write why in `notes/<run>.md`,
then take the next most expensive op of the score that no other run is on, and keep going.

This runs as a loop of fresh sessions. Your only memory is the repository:

- **`notes/<run>.md`**: read it first, update it before you stop: the current score, what dominates your
  target, every idea tried with its measured result and why it was kept or reverted.
- **git**: commit each passing improvement on its own. Leave the tree clean.
