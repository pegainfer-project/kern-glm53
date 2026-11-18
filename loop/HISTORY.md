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
| 18 | d3f90b5 | from 257 rows fp8.o, q_b and the dense MLP run BN 256 on a 2-CTA cluster | 223.98 | 1035.29 | 255.79 | -1.22% | 592 | top-1 100.000% · flips 0 (confident 0) |
| 19 | efc2414 | indexer top-k reads a long row once against its sample's threshold | 222.06 | 1023.80 | 253.49 | -0.90% | 592 | top-1 100.000% · flips 0 (confident 0) |
| 20 | 39dcf15 | indexer logits on DeepGEMM 2.8's split, 384 keys on three math warpgroups | 221.84 | 1023.32 | 253.26 | -0.09% | 592 | top-1 100.000% · flips 0 (confident 0) |
| 21 | e9b21ad | routed gate-up epilogue applies silu and quantizes, silu_moe gone | 218.37 | 1006.49 | 249.26 | -1.58% | 584 | top-1 100.000% · flips 0 (confident 0) |
| 22 | dfd2f45 | MoE combine takes the next layer's input norm when that layer shares the picks | 218.04 | 1004.71 | 248.88 | -0.15% | 579 | top-1 100.000% · flips 0 (confident 0) |
| 23 | 1a76317 | past 1024 rows the shared expert's GEMMs take the multicast tile | 217.53 | 1004.22 | 248.37 | -0.20% | 587 | top-1 100.000% · flips 0 (confident 0) |
| 24 | 4f3c271 | o_proj adds into the residual stream, norm_post reads res alone | 217.51 | 1003.12 | 248.31 | -0.03% | 587 | top-1 100.000% · flips 0 (confident 0) |

## 78-layer checkpoints

`loop/fullcheck`: the PP4 serving stages teacher-forced over six chat prompts (24300 positions) against
SGLang's prompt logprobs (its fp8-KV run), and against the checkpoint before.

| main | vs SGLang: top-1 / confident flips / \|dlp\| mean | vs the checkpoint before |
|---|---|---|
| row 0 | 95.222% / 10 / 0.0555 | |
| row 7 (`ff9ecc7`) | 95.198% / 11 / 0.0555 | 98.374% / 1 |
| row 7 + idx's bf16 head sum (row 10's commit) | 95.276% / 10 / 0.0556 | 98.309% / 1 |
| 25 | 46ed875 | past 256 rows the v up-projection on a 2-CTA cluster sharing W_UV | 217.11 | 1001.01 | 247.85 | -0.19% | 598 | top-1 100.000% · flips 0 (confident 0) |
| 26 | 572de46 | flash_mla's row-maximum barrier joins the four warps of a 32-head half | 216.75 | 999.46 | 247.44 | -0.16% | 598 | top-1 100.000% · flips 0 (confident 0) |
| 27 | 299650f | indexer top-k writes its picks as stream rows itself, glm_topk_stream gone | 216.49 | 999.85 | 247.20 | -0.10% | 598 | top-1 100.000% · flips 0 (confident 0) |
| 28 | b060a52 | above 4096 rows the routed GEMMs take block M 256 | 215.77 | 991.13 | 246.17 | -0.42% | 622 | top-1 100.000% · flips 0 (confident 0) |
| 29 | 85d5f37 | routed gate-up gathers its rows itself, moe_gather gone | 213.74 | 981.18 | 243.83 | -0.95% | 614 | top-1 100.000% · flips 0 (confident 0) |
| 30 | 7dd2298 | one metadata launch schedules every indexer sub-chunk of a call | 213.64 | 979.93 | 243.68 | -0.06% | 539 | top-1 100.000% · flips 0 (confident 0) |
| 31 | a11f967 | shared expert is the grouped GEMMs' group 0, its own GEMMs and silu_shared gone | 213.50 | 978.37 | 243.48 | -0.08% | 459 | top-1 100.000% · flips 0 (confident 0) |
