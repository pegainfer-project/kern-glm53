// DeepGEMM's sm100 kernels at GLM-5.3's prefill shapes, unmodified but for the routed GEMMs, whose changed
// copies read tile-major weights: dg/glm_moe_gu_silu.cuh (ending in silu and the e4m3 quantization) and
// dg/glm_moe_dn.cuh. The generator names an
// instance by its mangled template arguments (scripts/glm_deepgemm.py) and passes exactly the
// shared memory asserted here.
//   fp8: block-scaled e4m3 x e4m3 -> bf16, UE8M0 scales a 128-element K block for both operands
//        (vLLM's DeepGEMM recipe for 128x128-block checkpoints), Normal and m-grouped contiguous
//        over the 256 routed experts in the psum layout (only an expert's own row blocks run), the
//        gate-up one writing act = e4m3(bf16(silu(gate) * up)) with its UE8M0 scales instead of bf16;
//   bf16: batched bhr,hdr->bhd over the 64 heads, MLA's q absorption and v up-projection;
//   mqa logits: the lightning indexer's fp8 scores, 32 heads of 128, bf16 head weights and the heads' relu sum
//        in bf16 into compressed bf16 logits (vLLM's bf16 indexer logits on SM100), its (16-row q block, 256-key
//        split) work spread evenly over the SMs by DeepGEMM's metadata kernel (every sub-chunk of a call in one
//        launch, a block each), a split's keys read once for MMAs of N 256 over both 8-row halves, two math
//        warpgroups (source/dg/glm_mqa_logits.cuh); below 32k keys of context the vendored kernel's 8-row block.
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>
#include "dg/sm100_mqa_logits.cuh"
#include "dg/glm_mqa_logits.cuh"
#include "dg/sm100_mqa_logits_metadata.cuh"
#include "dg/glm_moe_gu_silu.cuh"
#include "dg/glm_moe_dn.cuh"

using namespace deep_gemm;
using e4m3 = cutlass::float_e4m3_t;
using bf16 = cutlass::bfloat16_t;

#define GLM_FP8(BM, BN, STAGES, TYPE, GROUPS, SMEM)                                                     \
  static_assert(sizeof(layout::SM100FP8FP4GemmSharedStorage<STAGES, 2, 2, BM, BN, 128, BM, 64, BM, BN, \
                                                            1, e4m3, e4m3, bf16>) == SMEM);             \
  { auto p = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<                                     \
        cute::UMMA::Major::K, cute::UMMA::Major::K, 128, 128, 128, 0, 0, 0, BM, BN, 128, GROUPS,       \
        128, 128, 128, STAGES, 2, 128, 128, 1, true, 152, false, false, TYPE, false, e4m3, e4m3, bf16,  \
        epilogue::transform::EpilogueIdentity>);                                                        \
    asm volatile("" ::"l"(p)); }

// Narrow BN for few rows: more CTAs over the weight. SWZ is the C/D swizzle (BN bf16 bytes, at most 128).
#define GLM_FP8_NARROW(BN, STAGES, SWZ, SMEM)                                                             \
  static_assert(sizeof(layout::SM100FP8FP4GemmSharedStorage<STAGES, 2, 2, 128, BN, 128, 128, SWZ / 2, 128, \
                                                            128, 1, e4m3, e4m3, bf16>) == SMEM);         \
  { auto p = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<                                       \
        cute::UMMA::Major::K, cute::UMMA::Major::K, 128, 128, 128, 0, 0, 0, 128, BN, 128, 1,             \
        128, 128, SWZ, STAGES, 2, 128, 128, 1, true, 152, false, false, GemmType::Normal, false, e4m3,   \
        e4m3, bf16, epilogue::transform::EpilogueIdentity>);                                              \
    asm volatile("" ::"l"(p)); }

// Swap-AB (UMMA M over the weight rows, N over the tokens), BLOCK_N 128, the token block split over a
// 2-CTA cluster: DeepGEMM's own choice for its m-grouped GEMMs, BLOCK_M the psum layout's alignment.
#define GLM_FP8_SWAP(BM, STAGES, MC, GROUPS, SMEM)                                                       \
  static_assert(sizeof(layout::SM100FP8FP4GemmSharedStorage<STAGES, 2, 2, BM / MC, 128, 128, 16, 128,  \
                                                            (BM + 127) / 128 * 128, 128, 1, e4m3, e4m3, \
                                                            bf16>) == SMEM);                            \
  { auto p = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<                                     \
        cute::UMMA::Major::K, cute::UMMA::Major::K, 128, 128, 128, 0, 0, 0, BM, 128, 128, GROUPS,       \
        128, 128, 128, STAGES, 2, 128, 128, MC, (MC > 1), 152, true, false,                             \
        GemmType::MGroupedContiguousWithPsumLayout, false, e4m3, e4m3, bf16,                            \
        epilogue::transform::EpilogueIdentity>);                                                        \
    asm volatile("" ::"l"(p)); }

// The routed gate-up GEMM at GLM_FP8_SWAP(BM, STAGES, 2, 256) with silu and e4m3 out (dg/glm_moe_gu_silu.cuh), its
// glm::SiluSmem past the tile's 1024-aligned shared storage; over 257 groups (the MoE) or one (the dense MLP).
static_assert(sizeof(glm::SiluSmem) == 576);
#define GLM_FP8_GU_SILU(BM, GROUPS, STAGES, SMEM)                                                        \
  static_assert(SMEM % 1024 == 0);                                                                       \
  { auto p = reinterpret_cast<void*>(&glm_moe_gu_silu_impl<                                             \
        cute::UMMA::Major::K, cute::UMMA::Major::K, 128, 128, 128, 0, 0, 0, BM, 128, 128, GROUPS,       \
        128, 128, 128, STAGES, 2, 128, 128, 2, true, 152, true, false,                                   \
        GemmType::MGroupedContiguousWithPsumLayout, false, e4m3, e4m3, bf16,                            \
        epilogue::transform::EpilogueIdentity>);                                                        \
    asm volatile("" ::"l"(p)); }

// The routed down GEMM at GLM_FP8_SWAP(BM, STAGES, 2, 257) over tile-major weights (dg/glm_moe_dn.cuh).
#define GLM_FP8_DN(BM, STAGES, SMEM)                                                                     \
  static_assert(sizeof(layout::SM100FP8FP4GemmSharedStorage<STAGES, 2, 2, BM / 2, 128, 128, 16, 128,   \
                                                            (BM + 127) / 128 * 128, 128, 1, e4m3, e4m3, \
                                                            bf16>) == SMEM);                            \
  { auto p = reinterpret_cast<void*>(&glm_moe_dn_impl<                                                  \
        cute::UMMA::Major::K, cute::UMMA::Major::K, 128, 128, 128, 0, 0, 0, BM, 128, 128, 257,          \
        128, 128, 128, STAGES, 2, 128, 128, 2, true, 152, true, false,                                   \
        GemmType::MGroupedContiguousWithPsumLayout, false, e4m3, e4m3, bf16,                            \
        epilogue::transform::EpilogueIdentity>);                                                        \
    asm volatile("" ::"l"(p)); }

#define GLM_BF16(BM, BN, STAGES, SMEM)                                                                 \
  static_assert(sizeof(layout::SM100BF16GemmSharedStorage<STAGES, 2, 2, BM, BN, 64, BM, 64, bf16>) == \
                SMEM);                                                                                \
  { auto p = reinterpret_cast<void*>(&sm100_bf16_gemm_impl<                                           \
        cute::UMMA::Major::K, cute::UMMA::Major::K, 0, 0, 0, BM, BN, 64, 64, 128, 128, 128, STAGES,   \
        128, 128, 1, true, 152, 128, false, false, GemmType::Batched, false, bf16,                    \
        epilogue::transform::EpilogueIdentity, 100>);                                                 \
    asm volatile("" ::"l"(p)); }

static_assert(sizeof(layout::MQALogitsSharedStorage<32, 128, false, 4, 256, 3, 5, 3, e4m3, float>) == 220160);

void glm_instantiate() {
  GLM_FP8(128, 128, 5, GemmType::Normal, 1, 202752)
  GLM_FP8(128, 256, 3, GemmType::Normal, 1, 185344)
  GLM_FP8(128, 128, 5, GemmType::MGroupedContiguousWithPsumLayout, 256, 202752)
  GLM_FP8(128, 256, 3, GemmType::MGroupedContiguousWithPsumLayout, 256, 185344)
  GLM_FP8_NARROW(32, 10, 64, 232448)
  GLM_FP8_NARROW(64, 7, 128, 212992)
  GLM_FP8_SWAP(32, 11, 2, 256, 223232)
  GLM_FP8_SWAP(64, 10, 2, 256, 224256)
  GLM_FP8_DN(128, 8, 214016)
  GLM_FP8_DN(256, 6, 215040)
  GLM_FP8_GU_SILU(128, 257, 8, 214016)
  GLM_FP8_GU_SILU(256, 257, 6, 215040)
  GLM_FP8_GU_SILU(128, 1, 8, 214016)
  GLM_FP8_GU_SILU(256, 1, 6, 215040)
  GLM_BF16(128, 128, 6, 230400)
  GLM_BF16(128, 256, 4, 230400)
  static_assert(sizeof(layout::MQALogitsSharedStorage<32, 128, false, 8, 256, 2, 4, 2, e4m3, __nv_bfloat16>) == 202752);
  { auto p = reinterpret_cast<void*>(&sm100_mqa_logits<32, 128, false, true, false, true, 8, 256, 2, 4, 152,
                                                       128, 256, e4m3, __nv_bfloat16, __nv_bfloat16>);
    asm volatile("" ::"l"(p)); }
  { auto p = reinterpret_cast<void*>(&sched::sm100_mqa_logits_metadata_batched<8, 256, 152, 1, 1, 15>);
    asm volatile("" ::"l"(p)); }
  static_assert(sizeof(layout::MQALogitsSharedStorage<32, 128, false, 16, 256, 1, 4, 2, e4m3, __nv_bfloat16>) == 202752);
  { auto p = reinterpret_cast<void*>(&glm_mqa_logits<16, 256, 4, 152>);
    asm volatile("" ::"l"(p)); }
  { auto p = reinterpret_cast<void*>(&sched::sm100_mqa_logits_metadata_batched<16, 256, 152, 1, 1, 15>);
    asm volatile("" ::"l"(p)); }
}
