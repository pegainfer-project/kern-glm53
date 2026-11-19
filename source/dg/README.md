# From DeepGEMM

Copied from [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) at `ab69f76` (third_party/DeepGEMM), MIT
(`LICENSE`, beside it).

- `sm100_mqa_logits.cuh`: `deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh`, two changes so it runs
  as DeepGEMM 2.8 (`5081f85`) does, SPLIT_KV 384 on three math warpgroups: the math warpgroups' register
  budget is what the register file leaves them (224 at two, 152 at three), and a contiguous split of more
  than 256 keys loads as 128-key TMA tiles (a box is at most 256 rows).
- `sm100_mqa_logits_metadata.cuh`: `deep_gemm/include/deep_gemm/scheduler/sm100_mqa_logits_metadata.cuh`, the
  kernel's body a device function of one block, and `sm100_mqa_logits_metadata_batched`, which schedules every
  sub-chunk of a call in one launch, block j the sub-chunk's rows into its own slice of the schedule.
