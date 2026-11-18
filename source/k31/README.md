# From kern-k31

Copied from [kern-k31](https://github.com/pegainfer-project/kern-k31) (Apache-2.0, as this repository).

- `k31_topk.cu`: `source/k31_topk.cu` at `c4a820a`, unmodified. Each row's 2048 largest f32 logits as
  ascending positions, -1 past min(len, 2048): vLLM's `top_k_per_row_prefill` set, equal logits at the
  cut taken by smaller position, the same bytes on every run. K3.1's indexer is GLM-5.3's (32 heads x 128).
