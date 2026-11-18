Target: the sparse attention and what feeds it: `flash_mla` (FlashMLA's bf16 sparse prefill), `kv_dequant` (the fp8 latent cache lifted to bf16 for the whole context, every layer), `uk` / `uv` (the bf16 batched absorb and v-up), `q_post_*`, `quant_o`.

- `flash_mla` is 17.5% of the score: about 4 ms a layer at 16384 rows against a ~2.9 ms bf16 roofline, 0.5 ms at 2048 rows. `kv_dequant` is O(context) a layer whatever the rows: at 128 rows over 1M it was 10% of a stage.
- Ideas, measure each: dequantize only the rows the top-k picked (a short chunk picks far fewer rows than its context holds), or attend the fp8 rows directly (an fp8 sparse MLA kernel reading the 656-byte row: FlashMLA's sparse decode path reads that format; TRT-LLM's e4m3 sparse MLA is in `$GLM_ATLAS/attention/`). Attention in e4m3 is a numerics change: allowed only with both gates and an explanation.
- kern-k31 moved its absorb and v-up onto tcgen05 (d421893, 51b4aed: -40..-55% on those ops) and fused the v-up's epilogue; `quant_o` could fold into the v-up's epilogue the same way.
- The cache format (SGLang's fp8 DSA row) is fixed.
