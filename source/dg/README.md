# From DeepGEMM

Copied from [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) at `ab69f76` (third_party/DeepGEMM), MIT
(`LICENSE`, beside it).

- `sm100_mqa_logits.cuh`: `deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh`, two changes so it runs
  as DeepGEMM 2.8 (`5081f85`) does, SPLIT_KV 384 on three math warpgroups: the math warpgroups' register
  budget is what the register file leaves them (224 at two, 152 at three), and a contiguous split of more
  than 256 keys loads as 128-key TMA tiles (a box is at most 256 rows).
