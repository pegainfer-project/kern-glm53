// DeepGEMM's sm100 kernels at GLM-5.3's prefill shapes, unmodified. The generator names an
// instance by its mangled template arguments (scripts/glm_deepgemm.py) and passes exactly the
// shared memory asserted here.
//   fp8: block-scaled e4m3 x e4m3 -> bf16, UE8M0 scales a 128-element K block for both operands
//        (vLLM's DeepGEMM recipe for 128x128-block checkpoints), Normal and m-grouped contiguous
//        over the 256 routed experts in the psum layout (only an expert's own row blocks run);
//   bf16: batched bhr,hdr->bhd over the 64 heads, MLA's q absorption and v up-projection;
//   mqa logits: the lightning indexer's fp8 scores, 32 heads of 128, compressed logits.
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>
#include <deep_gemm/impls/sm100_mqa_logits.cuh>

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
  GLM_BF16(128, 128, 6, 230400)
  GLM_BF16(128, 256, 4, 230400)
  { auto p = reinterpret_cast<void*>(&sm100_mqa_logits<32, 128, false, true, false, false, 4, 256, 3, 5, 152,
                                                       128, 256, e4m3, float, float>);
    asm volatile("" ::"l"(p)); }
}
