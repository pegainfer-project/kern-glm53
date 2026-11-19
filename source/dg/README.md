# From DeepGEMM

Copied from [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) at `ab69f76` (third_party/DeepGEMM), MIT
(`LICENSE`, beside it).

- `sm100_mqa_logits.cuh`: `deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh`. The math warpgroups'
  register budget is what the register file leaves them (224 at two, 152 at three); a contiguous split of
  more than 256 keys loads as 128-key TMA tiles (a box is at most 256 rows); a block of more than 128 q x
  heads (an MMA of N 256) takes two TMEM stages. Its math warpgroups run GLM's indexer only (bf16 head
  weights and sum, compressed bf16 logits): a split's every row comes out of TMEM as relu'd bf16 pairs before
  the head sums, which read their weights from shared memory, the same operations in the same order as
  upstream's; a row's logits go out at a 32-bit element offset advanced a split at a time, and the KV stage
  is released with no proxy fence (the scales were read into registers before).
- `sm100_mqa_logits_metadata.cuh`: `deep_gemm/include/deep_gemm/scheduler/sm100_mqa_logits_metadata.cuh`, the
  kernel's body a device function of one block, and `sm100_mqa_logits_metadata_batched`, which schedules every
  sub-chunk of a call in one launch, block j the sub-chunk's rows into its own slice of the schedule.
