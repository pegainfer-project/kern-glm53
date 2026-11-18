Target: launches and the glue between the big kernels, per layer: norms and quantization (`add_rms_quant`, `rms_quant`), `mla_prep_*`, the projections' small GEMMs at few rows (`fp8.a.*`, `fp8.b.*`, `fp8.o`, `weights_proj`, the router's `gemm_f32`), the head; and overlap: program lanes and PDL across the layer.

- A layer launches 26 kernels (MoE sharing picks) to 61 (MoE with an indexer, 16 indexer sub-calls). `loop/count` prints them.
- `fp8.o` is 5.5% of the score; at 128 rows it runs about 4x off its weight-read roofline (100 MB a layer). The same holds for the other projections at few rows: DeepGEMM's tile and swizzle choice by `when` on `tokens` is the lever.
- The indexer chain (weights_proj, the indexer's query, logits, top-k) and the latent chain (q_b, absorb, rope, kv) are independent until the attention: lanes can overlap them, as kern-k31 does on D (4c7c7f8, df94212). Coordinate in `notes/glue.md` what you put on lanes; the indexer's and MoE's kernels belong to the `idx` and `moe` runs.
- Fold norms into the GEMMs' epilogues or the next kernel's prologue where it saves a pass over the rows.
