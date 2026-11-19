# From DeepGEMM

Copied from [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) at `ab69f76` (third_party/DeepGEMM), MIT
(`LICENSE`, beside it).

- `sm100_mqa_logits.cuh`: `deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh`, below 32k keys of context. The
  math warpgroups' register budget is what the register file leaves them (224 at two, 152 at three); a contiguous
  split of more than 256 keys loads as 128-key TMA tiles; a block of more than 128 q x heads (an MMA of N 256) takes two
  TMEM stages. Its math warpgroups run GLM's indexer only: a split's every row comes out of TMEM as relu'd bf16 pairs
  before the head sums, which read their weights from shared memory, the same operations in the same order as
  upstream's; a row's logits go out at a 32-bit element offset, and the KV stage is released with no proxy fence.
- `glm_mqa_logits.cuh`: `deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh` reduced to GLM's one configuration
  (fp8, per-key scales, bf16 head weights and sums, compressed bf16 logits, scheduled) and retiled: a 16-row q block's
  two 8-row halves each an MMA of N 256 over one shared-memory copy of a 256-key split, two math warpgroups of 128 keys,
  two TMEM stages; a split's rows come out of TMEM as relu'd bf16 pairs before the head sums, the same operations in
  the same order as upstream's, the weights read from shared memory where they sum.
- `sm100_mqa_logits_metadata.cuh`: `deep_gemm/include/deep_gemm/scheduler/sm100_mqa_logits_metadata.cuh`, the
  kernel's body a device function of one block, and `sm100_mqa_logits_metadata_batched`, which schedules every
  sub-chunk of a call in one launch, block j the sub-chunk's rows into its own slice of the schedule.
