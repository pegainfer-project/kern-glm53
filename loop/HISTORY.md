# The climb

Every optimization merged into `main`, in order, each measured on `main` right after it landed:
`loop/ab` of main before it against main with it (two ABBA pairs on one GPU), `loop/score`'s mean TTFT
of the AgentX corpus on an idle PP4, projected to 78 layers. `after` is the commit of main it landed on
(the row's own commit is the next one on main: `git log --ancestry-path <after>..main | tail -1`). `Δ all`
is against the row above. Numerics are `loop/longcheck`'s long pass against the row above.

| # | after | what | short ms | long ms | all ms | Δ all | launches | numerics |
|---|---|---|---|---|---|---|---|---|
| 0 | 8374b60 | baseline: deterministic top-k (k31_topk), reproducible builds | 271.59 | 1229.57 | 309.17 | — | 436 | 78 layers vs SGLang fp8 KV: top-1 95.22%, 10 confident |
| 1 | 619e0d1 | silu_moe over the routed rows only, not the padding | 265.90 | 1227.21 | 303.61 | -1.80% | 436 | top-1 100.000% · flips 0 (confident 0) |
