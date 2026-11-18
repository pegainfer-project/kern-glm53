# FlashMLA head64 sparse prefill, two softmax warpgroups

Copied from [FlashMLA](https://github.com/deepseek-ai/FlashMLA) at `4f38f29e` (MIT, `LICENSE` here):
`csrc/kernels/sm100/prefill/sparse/fwd/head64/{config.h,phase1.cuh}`, namespace `head64` renamed
`head64x2`. `source/glm_flashmla2.cu` instantiates it (`Prefill`, d_qk 576).

Upstream runs one softmax warpgroup: a thread holds one head and 32 keys of a 64-key block, and the MMA
warp waits on it before every S V GEMM (its chain of ~370 dependent instructions a block, one warp an
SM sub-partition, is longer than the block's QK and SV MMAs). Here warpgroup 3 is a second softmax
warpgroup (512 threads): thread (lane quadrant q, group g) holds head (q & 1) * 32 + lane and keys
(q / 2) * 32 + 16 g + [0, 16). What changed:

- `config.h`: 512 threads, `NUM_SM_THREADS` 256 arrive on `bar_p_free`, `bar_so_ready`,
  `bar_k_valid_free` and `bar_clc_empty`; the dual GEMM's exchange buffer has 8 slots of 16 keys; the
  row maximum is exchanged over four threads through two alternating buffers (`rowwise_max_buf`), which
  also carry a pair's `li` at the end of a row (`rowwise_li_buf` gone); TMEM columns 464-467 (`LI`) carry
  group 0's partial sums; named barriers: one per warp pair and group, `sm_sync` over both groups.
- `phase1.cuh`: the softmax loop for both groups (the dual GEMM's halves exchanged by warps q and q ^ 2
  of a group, as upstream's `retrieve_mask_and_reduce_p` does for 32 keys); each group rescales its half
  of O's columns. `li` is upstream's chain: a pair's 32 keys summed as two float2 lanes in key order,
  group 0 summing its first 16 and leaving the float2 in TMEM, group 1 continuing it with its 16 held
  exponentials after the next `sm_sync` (the row's last block after a closing one), then
  `fma(li, scale, x + y)`; group 1 adds its pair's `li` and writes the epilogue. `setmaxnreg`: softmax
  groups 200, the others 56.

The products, the exponentials, S, `li` and the outputs are upstream's, bit for bit.
