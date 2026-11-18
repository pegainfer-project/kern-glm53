// MLA's v up-projection with o_proj's input quantization in its epilogue: DeepGEMM's sm100 bf16 batched
// GEMM over the 64 heads, unmodified, instantiated with an epilogue operator of ours. Its overload of
// sm100_store_cd lands each row's 128-column group of D as e4m3 with a UE8M0 scale in DeepGEMM's packed
// layout, the numbers glm_glue's glm_quant makes of the bf16 D: the accumulator rounded to bf16, the
// group's amax, scale = 2^ceil(log2(max(amax, 1e-10) / 448)), each value times the scale's inverse.
#include <deep_gemm/epilogue/sm100_store_cd.cuh>

namespace deep_gemm::epilogue {

// EpilogueArgs as o_proj's scales: sfd [K / 512][sfd_stride] words, shape_m live rows, shape_n a head's columns.
struct GlmQuantO : EpilogueArgs {
  template <uint32_t STORE_BLOCK_N>
  CUTLASS_DEVICE static uint32_t apply_index_n(const uint32_t& n_idx) { return n_idx; }
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N, uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages, uint32_t kNumUMMAStoreThreads, uint32_t kNumOverlappedTmemCols,
          GemmType kGemmType, bool kWithAccumulation, typename smem_t>
CUTLASS_DEVICE void sm100_store_cd(smem_t& smem, uint32_t& tma_stage_idx, const uint32_t& tmem_base_addr,
                                   const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx,
                                   const bool&, const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                                   const GlmQuantO& epilogue_op, const bool&,
                                   const cutlass::arch::ClusterTransactionBarrier*,
                                   const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
                                   const cute::TmaDescriptor& tensor_map_cd) {
  constexpr uint32_t kGroups = BLOCK_N / 128;
  static_assert(BLOCK_M == 128 and STORE_BLOCK_M == 128 and kSwizzleCDMode == 128 and kGemmType == GemmType::Batched and
                not kWithAccumulation and kNumOverlappedTmemCols == 0 and BLOCK_N % 128 == 0 and kGroups <= 4);
  static_assert(sizeof(typename smem_t::cd_dtype) * STORE_BLOCK_M * STORE_BLOCK_N == 128 * 128, "a stage holds 128 x 128 e4m3");
  const uint32_t row_in_block = epilogue_warp_idx * 32 + lane_idx;
  uint32_t sf_word = 0;
#pragma unroll
  for (uint32_t g = 0; g < kGroups; ++g) {
    auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem.cd[tma_stage_idx]);
    if (epilogue_warp_idx == 0) cute::tma_store_wait<kNumTMAStoreStages - 1>();
    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

    uint32_t values[128];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
      [&]<size_t... Is>(cute::index_sequence<Is...>) {
        cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_base_addr + g * 128 + i * 32, values[i * 32 + Is]...);
      }(cute::make_index_sequence<32>{});
    cutlass::arch::fence_view_async_tmem_load();
    if (g == kGroups - 1) {
      ptx::tcgen05_before_thread_sync();
      tmem_empty_barrier->arrive(0u);
    }

    nv_bfloat162 v[64];
#pragma unroll
    for (uint32_t p = 0; p < 64; ++p) v[p] = math::cast_into_bf16x2(values[2 * p], values[2 * p + 1]);
    const uint32_t b = __float_as_uint(fmaxf(__bfloat162float(math::get_packed_bf16_amax(v)), 1e-10f) / 448.f);
    const uint32_t e = ((b >> 23) & 0xff) + ((b & 0x7fffff) != 0);
    const auto inv = __bfloat162bfloat162(__float2bfloat16(1.f / __uint_as_float(e << 23)));
    sf_word |= e << (g * 8);

#pragma unroll
    for (uint32_t c = 0; c < 8; ++c) {
      const uint32_t r = row_in_block, col = c ^ (r % 8);
      ptx::st_shared(smem_base_ptr + r * 128 + col * 16, math::scale_bf16x2_into_fp8x4(v[8 * c + 0], v[8 * c + 1], inv, inv),
                     math::scale_bf16x2_into_fp8x4(v[8 * c + 2], v[8 * c + 3], inv, inv),
                     math::scale_bf16x2_into_fp8x4(v[8 * c + 4], v[8 * c + 5], inv, inv),
                     math::scale_bf16x2_into_fp8x4(v[8 * c + 6], v[8 * c + 7], inv, inv));
    }

    cute::tma_store_fence();
    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
    if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
      cute::SM90_TMA_STORE_3D::copy(&tensor_map_cd, smem_base_ptr, base_n_idx + g * 128, base_m_idx, batch_idx);
      cute::tma_store_arrive();
    }
    __syncwarp();
    tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
  }

  const uint32_t row = base_m_idx + row_in_block;
  if (row >= epilogue_op.shape_m) return;
  const uint32_t kb = (batch_idx * epilogue_op.shape_n + base_n_idx) / 128;
  uint8_t* word = reinterpret_cast<uint8_t*>(epilogue_op.sfd + (kb / 4) * epilogue_op.sfd_stride + row);
  if constexpr (kGroups == 4) {
    *reinterpret_cast<uint32_t*>(word) = sf_word;
  } else if constexpr (kGroups == 2) {
    *reinterpret_cast<uint16_t*>(word + kb % 4) = (uint16_t)sf_word;
  } else {
    word[kb % 4] = (uint8_t)sf_word;
  }
}

}  // namespace deep_gemm::epilogue

#include <deep_gemm/impls/sm100_bf16_gemm.cuh>

using namespace deep_gemm;
using bf16 = cutlass::bfloat16_t;

static_assert(sizeof(layout::SM100BF16GemmSharedStorage<4, 2, 2, 128, 256, 64, 128, 64, bf16>) == 230400);
// a 2-CTA cluster on M sharing B: each CTA loads half of a head's 256 columns
static_assert(sizeof(layout::SM100BF16GemmSharedStorage<6, 2, 2, 128, 128, 64, 128, 64, bf16>) == 230400);

#define GLM_BF16_BATCHED(MC, STAGES, EPILOGUE)                                                                        \
  { auto p = reinterpret_cast<void*>(&sm100_bf16_gemm_impl<                                                         \
        cute::UMMA::Major::K, cute::UMMA::Major::K, 0, 0, 0, 128, 256, 64, 64, 128, 128, 128, STAGES, 128, 128, MC, \
        MC == 1, 152, 128, false, false, GemmType::Batched, false, bf16, EPILOGUE, 100>);                          \
    asm volatile("" ::"l"(p)); }

void glm_instantiate() {
  GLM_BF16_BATCHED(1, 4, epilogue::GlmQuantO)
  GLM_BF16_BATCHED(2, 6, epilogue::GlmQuantO)
}
