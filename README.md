# kern-glm53-pp-prefill

GLM-5.3 (78 layers, DeepSeek V3.2's sparse attention, 256-expert MoE, FP8 128x128 block weights) as a
kern manifest: a packed prefill of up to 16 sequences and 16384 rows a call over contexts up to 1M,
cut into four pipeline stages, one a GPU. The MTP layer is not part of it.

```
git submodule update --init --recursive            # DeepGEMM, FlashMLA (and their CUTLASS) at the pinned commits
NVCC=<nvcc 13.1.115> python3 scripts/build.py      # build/*.cubin, checked against kernels.toml, into kern's blob cache
PATH=<cuda>/bin:$PATH scripts/deploy.sh <kern checkout> out   # out/glm53.json, out/p/stage{0..3}.json
```

`scripts/gen.py`'s module doc describes the forward. Stages start at layers 0, 18, 38 and 58, each an
indexer (`full`) layer, so a shared layer's top-k never crosses a cut and only the residual stream
(`h`, `res`) moves between stages. Each stage is a single-GPU manifest over the GLM-5.3 FP8
checkpoint (GLM-5.2-FP8 has the same tensors).

## Choices

- **FP8 scales.** The checkpoint's f32 128x128 block scales are re-quantized at load to UE8M0 (power
  of two) block scales, DeepGEMM's packed layout, the way vLLM's `requant_weight_ue8m0` prepares a
  128x128-block checkpoint for DeepGEMM on Blackwell; activations are quantized per 128 columns with
  UE8M0 scales as vLLM's `per_token_group_quant_fp8(..., use_ue8m0=True)` does.
- **Latent cache in bf16** (576 values a token: kv_norm | roped k_pe). vLLM stores GLM-5's MLA cache
  as `fp8_ds_mla` (656 bytes a token); FlashMLA's bf16 sparse prefill reads this one directly. The
  indexer cache is vLLM's: e4m3 keys with one f32 scale, page-blocked (132 bytes a token).
- **Sparse attention at every length.** A row attends the top 2048 keys its indexer picked, and every
  key when its sequence has at most 2048, which is dense attention through the sparse kernel.
- **MoE.** FlashInfer's routing tables over the router's top-8, then DeepGEMM's m-grouped fp8 GEMM in
  its psum layout, so only each expert's own row blocks run.

## Kernels

| module | what | provenance |
|---|---|---|
| `glm_deepgemm` | fp8 1d1d GEMMs (dense, m-grouped psum), bf16 batched GEMM (W_UK, W_UV), fp8 MQA logits | new upstream: DeepGEMM `ab69f76b`, MIT |
| `glm_flashmla` | sparse MLA prefill, 64 heads, d_qk 576 | new upstream: FlashMLA `4f38f29e`, MIT |
| `vllm_topk` | the indexer's top-2048 per row | new upstream: vLLM `f4edf9f0` `csrc/sampler.cu`, Apache-2.0 (`source/vllm-topk/`) |
| `glm_glue` | norms, quantization, rope, cache writes, row plan, MoE gather / combine | ours, after vLLM's GLM-5 math |
| `glm_prep` | load-time UE8M0 re-quantization, kv_b split | ours, after vLLM's `requant_weight_ue8m0` |
| `k3_router_argmax+EXPERTS=256+TOPK=8` | router top-8, vocab argmax | rebuilt from K3 with new defines (`source/k3/README.md`) |
| `copy_rows` | each sequence's last row | reused from K3 |
| `flashinfer_moe_routing` | MoE routing tables | reused from K3 (prebuilt, Apache-2.0, `prebuilt/LICENSE.flashinfer`) |
| `embedding`, `weight_prep`, `k3_weight_prep` | embedding gather, rope table, fills | reused from kern's kernel index |

## Licenses

Our sources (`source/glm_*.cu`, `scripts/`) are Apache-2.0 (`LICENSE`), as are those copied from
kern-k3-pp-prefill (`source/k3/`). DeepGEMM and FlashMLA are git submodules under `third_party/` at
the commits `kernels.toml` pins, MIT-licensed by DeepSeek (their own `LICENSE`); `source/glm_deepgemm.cu`
and `source/glm_flashmla.cu` only instantiate their templates. vLLM's top-k is vendored with its
Apache-2.0 license (`source/vllm-topk/`), FlashInfer's routing cubin with its own
(`prebuilt/LICENSE.flashinfer`).

`scripts/flashmla_probe.cu` records FlashMLA's launch parameter blocks without a GPU (upstream's host
code run against stand-ins for the CUDA calls); `scripts/glm_ops.py` lays them out from that record.
