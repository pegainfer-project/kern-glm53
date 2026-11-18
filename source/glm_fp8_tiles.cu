// DeepGEMM's sm100 fp8 1d1d Normal GEMM, unmodified, in two layouts DeepGEMM's heuristics pick for the
// projections. At few rows swap-AB: the weight rows on UMMA M (BN 128 of them a CTA), the tokens on N in a
// BM-row block split over a 2-CTA cluster that multicasts it, more CTAs over the weight than BM 128 x BN
// 256 leaves at 128-512 rows (BM 48, 96 and 128 hang in this layout and are not built). From a few hundred
// rows BM 128 x BN 256 on a 2-CTA cluster along M, each CTA loading half the weight tile and multicasting
// it (an even count of row blocks, see scripts/glm_ops.py). The generator names an instance by its
// template arguments.
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;
using e4m3 = cutlass::float_e4m3_t;
using bf16 = cutlass::bfloat16_t;

#define GLM_FP8_SWAP(BM, STAGES, SMEM)                                                                   \
  static_assert(sizeof(layout::SM100FP8FP4GemmSharedStorage<STAGES, 2, 2, BM / 2, 128, 128, 16, 128,   \
                                                            128, 128, 1, e4m3, e4m3, bf16>) == SMEM);   \
  { auto p = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<                                     \
        cute::UMMA::Major::K, cute::UMMA::Major::K, 128, 128, 128, 0, 0, 0, BM, 128, 128, 1, 128, 128, \
        128, STAGES, 2, 128, 128, 2, true, 152, true, false, GemmType::Normal, false, e4m3, e4m3, bf16, \
        epilogue::transform::EpilogueIdentity>);                                                        \
    asm volatile("" ::"l"(p)); }

#define GLM_FP8_MCB(STAGES, SMEM)                                                                         \
  static_assert(sizeof(layout::SM100FP8FP4GemmSharedStorage<STAGES, 2, 2, 128, 128, 128, 128, 64, 128, 256, \
                                                            1, e4m3, e4m3, bf16>) == SMEM);              \
  { auto p = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<                                       \
        cute::UMMA::Major::K, cute::UMMA::Major::K, 128, 128, 128, 0, 0, 0, 128, 256, 128, 1, 128, 128,   \
        128, STAGES, 2, 128, 128, 2, false, 152, false, false, GemmType::Normal, false, e4m3, e4m3, bf16, \
        epilogue::transform::EpilogueIdentity>);                                                          \
    asm volatile("" ::"l"(p)); }

void glm_instantiate() {
  GLM_FP8_MCB(5, 204800)
  GLM_FP8_SWAP(32, 11, 223232)
  GLM_FP8_SWAP(64, 10, 224256)
}
