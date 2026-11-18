# The climb

Every optimization merged into `main`, in order, each measured on `main` right after it landed:
`loop/score`'s mean TTFT of the AgentX corpus on an idle PP4, projected to 78 layers. Row 0 is a bench of
main; every later row is the row above times `loop/ab`'s ratio of main with it to main before it (two
ABBA pairs on one GPU, `loop/chain.py`), since GPUs differ by more than most changes. `after` is the commit of main it landed on
(the row's own commit is the next one on main: `git log --ancestry-path <after>..main | tail -1`). `Δ all`
is that A/B's. Numerics are `loop/longcheck`'s long pass against the row above.

| # | after | what | short ms | long ms | all ms | Δ all | launches | numerics |
|---|---|---|---|---|---|---|---|---|
| 0 | 8374b60 | baseline: deterministic top-k (k31_topk), reproducible builds | 271.59 | 1229.57 | 309.17 | — | 436 | 78 layers vs SGLang fp8 KV: top-1 95.22%, 10 confident |
| 1 | 619e0d1 | silu_moe over the routed rows only, not the padding | 265.30 | 1222.78 | 302.86 | -2.04% | 436 | top-1 100.000% · flips 0 (confident 0) |
| 2 | 6d7ec7c | routed GEMMs on DeepGEMM's m-grouped config: swap-AB, BN 128, 2-CTA cluster | 253.32 | 1181.50 | 289.73 | -4.34% | 436 | top-1 100.000% · flips 0 (confident 0) |
| 3 | 868bab6 | MoE glue (silu, gather, combine) moves 16 bytes a thread, a block per row | 248.20 | 1149.30 | 283.54 | -2.13% | 436 | top-1 100.000% · flips 0 (confident 0) |
| 4 | ae3b376 | indexer logits deal q blocks and key splits evenly over the SMs | 245.68 | 1138.05 | 280.68 | -1.01% | 516 | top-1 100.000% · flips 0 (confident 0) |
| 5 | e6be11b | q_post ropes q_pe in 16-byte vectors | 243.34 | 1125.07 | 277.92 | -0.98% | 516 | top-1 100.000% · flips 0 (confident 0) |
| 6 | 7de1aa5 | bf16 indexer logits, k31_topk (f4ddc91) over bf16 rows | 240.23 | 1105.13 | 274.15 | -1.36% | 516 | top-1 91.120% · flips 23279 (confident 3) |
| 7 | e1982ff | shared expert's GEMMs take BN 32 / 64 tiles for few rows | 239.82 | 1104.83 | 273.74 | -0.15% | 540 | top-1 100.000% · flips 0 (confident 0) |
