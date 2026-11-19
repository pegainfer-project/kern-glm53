// q_b's projection with q_pe's rope in its epilogue: DeepGEMM's sm100 fp8 1d1d Normal GEMM, unmodified,
// instantiated with an epilogue operator of ours in the layouts the generator runs it in (BN 128 or 256 up to
// 256 rows, BN 256 on a 2-CTA cluster multicasting the weight tile past). Its overload of sm100_store_cd is
// DeepGEMM's bf16 store but for the last 64 columns of each of the 64 heads' 256 (192 nope | 64 pe; the
// indexer's query columns past them are left as they are), which land rotated by the row's position as glm_glue's glm_q_post rotates them: the
// accumulator rounded to bf16, each pair (x, y) -> bf16(x cos - y sin), bf16(y cos + x sin) in f32.
#include <deep_gemm/epilogue/sm100_store_cd.cuh>

namespace deep_gemm::epilogue {

// EpilogueArgs as the rope's: sfd the rows' positions, sfd_stride the live rows, shape_m | shape_n << 32 the
// rope table [position][cos 32 | sin 32] bf16.
struct GlmRopeQ : EpilogueArgs {
  template <uint32_t STORE_BLOCK_N>
  CUTLASS_DEVICE static uint32_t apply_index_n(const uint32_t& n_idx) { return n_idx; }
};

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N, uint32_t kSwizzleCDMode,
          uint32_t kNumTMAStoreStages, uint32_t kNumUMMAStoreThreads, uint32_t kNumOverlappedTmemCols,
          GemmType kGemmType, bool kWithAccumulation, typename smem_t>
CUTLASS_DEVICE void sm100_store_cd(smem_t& smem, uint32_t& tma_stage_idx, const uint32_t& tmem_base_addr,
                                   const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t&,
                                   const bool&, const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                                   const GlmRopeQ& epilogue_op, const bool& reverse_store_order,
                                   const cutlass::arch::ClusterTransactionBarrier* tmem_overlap_barrier,
                                   const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier,
                                   const cute::TmaDescriptor& tensor_map_cd) {
  constexpr uint32_t kHead = 256, kNope = 192, kHalf = 32, kHeads = 64;
  constexpr uint32_t kNumElemsPerBankGroup = 8;
  static_assert(cute::is_same_v<typename smem_t::cd_dtype, cutlass::bfloat16_t> and kGemmType == GemmType::Normal and
                not kWithAccumulation and kSwizzleCDMode == 128 and BLOCK_M == STORE_BLOCK_M and
                BLOCK_N % STORE_BLOCK_N == 0 and kHead % BLOCK_N == 0 and kHead - BLOCK_N <= kNope);
  // the rope columns are the last 64 of a head's last tile: the row's 32 cos and 32 sin, bf16 pairs, in registers
  constexpr uint32_t kRope0 = kNope - (kHead - BLOCK_N);
  const bool rope_tile = base_n_idx % kHead == kHead - BLOCK_N and base_n_idx < kHeads * kHead;
  const auto row_idx = base_m_idx + epilogue_warp_idx * 32 + lane_idx;
  uint32_t cs[2 * kHalf / 2] = {};
  if (rope_tile and row_idx < epilogue_op.sfd_stride) {
    const auto* table = reinterpret_cast<const uint4*>(
        static_cast<uint64_t>(epilogue_op.shape_m) | static_cast<uint64_t>(epilogue_op.shape_n) << 32);
    const auto* row = table + reinterpret_cast<const int*>(epilogue_op.sfd)[row_idx] * (2 * kHalf * 2 / 16);
#pragma unroll
    for (uint32_t w = 0; w < 2 * kHalf * 2 / 16; ++w) {
      const uint4 x = __ldg(row + w);
      cs[4 * w] = x.x, cs[4 * w + 1] = x.y, cs[4 * w + 2] = x.z, cs[4 * w + 3] = x.w;
    }
  }
  const auto bf = [&](const uint32_t& e) { return __uint_as_float(e % 2 ? cs[e / 2] & 0xffff0000u : cs[e / 2] << 16); };

  const auto store = [&]<bool kReverse>() {
    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
#pragma unroll
    for (uint32_t s = 0; s < kNumStores; ++s, tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages) {
      const auto store_idx = kReverse ? kNumStores - 1 - s : s;
      auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem.cd[tma_stage_idx]);
      const auto swizzled = [&](const uint32_t& bank_group_idx) {
        const auto row = bank_group_idx / 8 + lane_idx, col = (bank_group_idx % 8) ^ (row % 8);
        return smem_base_ptr + epilogue_warp_idx * 32 * kSwizzleCDMode + row * 128 + col * 16;
      };
      if (epilogue_warp_idx == 0) cute::tma_store_wait<kNumTMAStoreStages - 1>();
      cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

      constexpr uint32_t kNumLoads = STORE_BLOCK_N / kNumElemsPerBankGroup;
      constexpr uint32_t kNumOverlapLoads = math::constexpr_ceil_div(kNumOverlappedTmemCols, kNumElemsPerBankGroup);
      static_assert(kNumOverlapLoads <= kNumLoads);
#pragma unroll
      for (uint32_t i = 0; i < kNumLoads; ++i) {
        const auto load_idx = kReverse ? kNumLoads - 1 - i : i;
        const auto off = store_idx * STORE_BLOCK_N + load_idx * kNumElemsPerBankGroup;
        uint32_t values[kNumElemsPerBankGroup];
        [&]<size_t... Is>(cute::index_sequence<Is...>) {
          cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_base_addr + off, values[Is]...);
        }(cute::make_index_sequence<kNumElemsPerBankGroup>{});
        cutlass::arch::fence_view_async_tmem_load();
        if constexpr (kNumOverlapLoads > 0) {
          if (s == 0 and i + 1 == kNumOverlapLoads) {
            ptx::tcgen05_before_thread_sync();
            tmem_overlap_barrier->arrive(0u);
          }
        }
        if (s == kNumStores - 1 and i == kNumLoads - 1) {
          ptx::tcgen05_before_thread_sync();
          tmem_empty_barrier->arrive(0u);
        }

        nv_bfloat162 v[4];
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) v[j] = math::cast_into_bf16x2(values[2 * j], values[2 * j + 1]);
        if (off >= kRope0 and rope_tile) {
          const auto pr = (off - kRope0) / 2;
#pragma unroll
          for (uint32_t j = 0; j < 4; ++j) {
            const float2 x = __bfloat1622float2(v[j]);
            const float c = bf(pr + j), sn = bf(kHalf + pr + j);
            v[j] = __floats2bfloat162_rn(x.x * c - x.y * sn, x.y * c + x.x * sn);
          }
        }
        const auto* u = reinterpret_cast<const uint32_t*>(v);
        ptx::st_shared(swizzled(load_idx), u[0], u[1], u[2], u[3]);
      }

      cute::tma_store_fence();
      cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
      if (epilogue_warp_idx == 0 and cute::elect_one_sync()) {
        cute::SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_base_ptr, base_n_idx + store_idx * STORE_BLOCK_N, base_m_idx);
        cute::tma_store_arrive();
      }
      __syncwarp();
    }
  };
  if (reverse_store_order)
    store.template operator()<true>();
  else
    store.template operator()<false>();
}

}  // namespace deep_gemm::epilogue

#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;
using e4m3 = cutlass::float_e4m3_t;
using bf16 = cutlass::bfloat16_t;

void glm_instantiate() {
  static_assert(sizeof(layout::SM100FP8FP4GemmSharedStorage<5, 2, 2, 128, 128, 128, 128, 64, 128, 128, 1, e4m3, e4m3,
                                                            bf16>) == 202752);
  { auto p = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<
        cute::UMMA::Major::K, cute::UMMA::Major::K, 128, 128, 128, 0, 0, 0, 128, 128, 128, 1, 128, 128, 128, 5, 2, 128,
        128, 1, true, 152, false, false, GemmType::Normal, false, e4m3, e4m3, bf16, epilogue::GlmRopeQ>);
    asm volatile("" ::"l"(p)); }
  static_assert(sizeof(layout::SM100FP8FP4GemmSharedStorage<3, 2, 2, 128, 256, 128, 128, 64, 128, 256, 1, e4m3, e4m3,
                                                            bf16>) == 185344);
  { auto p = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<
        cute::UMMA::Major::K, cute::UMMA::Major::K, 128, 128, 128, 0, 0, 0, 128, 256, 128, 1, 128, 128, 128, 3, 2, 128,
        128, 1, true, 152, false, false, GemmType::Normal, false, e4m3, e4m3, bf16, epilogue::GlmRopeQ>);
    asm volatile("" ::"l"(p)); }
  static_assert(sizeof(layout::SM100FP8FP4GemmSharedStorage<5, 2, 2, 128, 128, 128, 128, 64, 128, 256, 1, e4m3, e4m3,
                                                            bf16>) == 204800);
  { auto p = reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<
        cute::UMMA::Major::K, cute::UMMA::Major::K, 128, 128, 128, 0, 0, 0, 128, 256, 128, 1, 128, 128, 128, 5, 2, 128,
        128, 2, false, 152, false, false, GemmType::Normal, false, e4m3, e4m3, bf16, epilogue::GlmRopeQ>);
    asm volatile("" ::"l"(p)); }
}
