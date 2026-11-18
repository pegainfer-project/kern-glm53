Target: the MoE layer around the routed GEMMs: `silu_moe`, `moe_gather`, `moe_combine`, `routing` / `moe_psum`, the shared expert (`fp8.shared.*`, `silu_shared`), and the GEMMs' own configuration (`fp8.moe.gu`, `fp8.moe.dn`, DeepGEMM's psum-grouped fp8 1d1d).

- `silu_moe` is 4.3% of the score: its grid covers the bound of the padded routed rows (TOPK x rows + 256 x 128), most of whose blocks return at once. Fold it into the gate-up GEMM's epilogue, or size it by the real count.
- The routed GEMMs are 44% of the score. At 128-2048 rows they read every expert's weights once and sit near HBM's roofline; at 16384 rows near fp8's. Check that claim per shape before taking it (bytes or flops over time, `ncu`), then look for what is left: tile and block_m for few rows per expert, the psum layout's padding, multicast, the epilogue writing e4m3 for the next GEMM directly.
- The shared expert is independent of the routed path from the norm on: a program lane can run it beside the routed GEMMs (`$GLM_KERN/docs/manifest.md`, lanes).
- `moe_gather` and `moe_combine` are glue that can fold into their neighbours (the gather into the quantization, the combine into the next layer's norm).
