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
| 8 | 25aca61 | v up-projection writes o_proj's e4m3 input, quant_o gone | 236.63 | 1087.03 | 269.97 | -1.38% | 529 | top-1 100.000% · flips 0 (confident 0) |
| 9 | 1cb05d3 | kv_dequant lifts only the rows some top-k picked, up to 2048 rows | 235.96 | 1090.20 | 269.45 | -0.19% | 545 | top-1 100.000% · flips 0 (confident 0) |
| 10 | 434bbdd | indexer head weights bf16, 32-head sum in bf16 | 234.06 | 1080.17 | 267.23 | -0.82% | 545 | top-1 91.117% · flips 23285 (confident 2); 78 layers vs SGLang 95.28% / 10 confident (main 95.20% / 11) |
| 11 | d333279 | routing tables in two launches: a bitmap tally per 256-token slice, each routed row's place | 233.77 | 1081.14 | 266.99 | -0.09% | 529 | top-1 100.000% · flips 0 (confident 0) |
| 12 | 6c72463 | norms move 16 bytes a thread, skip the unread bf16 row | 232.27 | 1070.73 | 265.14 | -0.69% | 529 | top-1 100.000% · flips 0 (confident 0) |
| 13 | 083031f | latent lift on lane 1 beside the query chain; up to 2048 rows the shared expert beside the routed experts | 231.02 | 1070.40 | 263.92 | -0.46% | 553 | top-1 100.000% · flips 0 (confident 0) |
| 14 | 5c8f061 | flash_mla's softmax on two warpgroups, sixteen keys a thread | 229.36 | 1060.69 | 261.95 | -0.75% | 553 | top-1 100.000% · flips 0 (confident 0) |
| 15 | bec98eb | flash_mla's exponentials ex2.approx.ftz, as FlashMLA's own build | 228.21 | 1053.94 | 260.58 | -0.52% | 553 | top-1 100.000% · flips 0 (confident 0) |
| 16 | 0d6b950 | flash_mla's li bookkeeping after so_ready, first pair barrier gone | 227.40 | 1050.26 | 259.66 | -0.35% | 553 | top-1 100.000% · flips 0 (confident 0) |
| 17 | 4dbc60c | projections at few rows take swap-AB or narrow tiles | 226.69 | 1049.71 | 258.96 | -0.27% | 584 | top-1 100.000% · flips 0 (confident 0) |
