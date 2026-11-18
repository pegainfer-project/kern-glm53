// GLM-5.3's routed gate-up GEMM with silu and the e4m3 quantization of its output in the epilogue:
// DeepGEMM's sm100_fp8_fp4_gemm_1d1d_impl (third_party/DeepGEMM at the commit kernels.toml pins,
// deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh, MIT, Copyright (c) 2025 DeepSeek),
// copied and changed: the A tile is gathered from the tokens' rows by an otherwise idle warp (cp.async by the
// routing's row -> token table), the B and SFB loads take an n-block's 64 gate and 64 up rows, the swap-AB store
// becomes glm::silu_store (upstream's bf16 staging, then silu, the quantization and global stores), and the
// kernel takes the row table's, the tokens', the activation's and its scales' pointers. Everything else is upstream's.
#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/packing.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/epilogue/sm100_store_cd.cuh>
#include <deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <deep_gemm/layout/gemm.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

namespace deep_gemm {

// ---- GLM-5.3: the gate-up GEMM's silu-and-quantize epilogue (swap-AB, 2-CTA cluster) ----
// The CTA pair of an n-block pair j holds act columns [128j, 128j + 128): CTA r its 64 gate rows
// (TMEM lanes 0-63) and the matching 64 up rows (lanes 64-127) of columns 128j + 64r + [0, 64).
// act = bf16(silu(g) * u) over g, u rounded to bf16 (glm_silu_rows over a bf16 D), its UE8M0 scale
// from the pair's 128-column amax: each CTA's 64-column amax of a chunk of 16 tokens goes to its peer
// through DSMEM, and the chunk is quantized and stored once the next one is computed, so the swap's
// latency hides behind it. The scale byte goes into DeepGEMM's packed layout from CTA 0.
namespace glm {

constexpr uint32_t kSlots = 4;

// A token's 64-column amax from the peer CTA, tagged with its chunk + 1 in the high word; per stage, the
// barrier the gather warp's copies arrive on
struct SiluSmem {
    uint64_t peer[kSlots][16];
    uint64_t a_full[8];
};

// A chunk computed and sent, not yet stored: this thread's 8 act values and its token's amax
struct SiluPending {
    float a[8];
    float m;
    uint32_t chunk, row, col;
    bool live = false;
};

CUTLASS_DEVICE uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

CUTLASS_DEVICE uint32_t peer_addr(uint32_t a, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
    return r;
}

CUTLASS_DEVICE uint32_t fp8x4(float a, float b, float c, float d) {
    const __nv_fp8x4_e4m3 q(make_float4(a, b, c, d));
    return *reinterpret_cast<const uint32_t*>(&q);
}

// s[k] = g[k] / (1 + __expf(-g[k])), each the correctly rounded quotient `/` gives: div.rn's fast path
// (a refined reciprocal, then one remainder correction) on all eight at once, exact while neither operand
// is zero, subnormal or past 2^64, and `/` itself for a thread holding one that is.
CUTLASS_DEVICE void silu8(const float (&g)[8], float (&s)[8]) {
    float y[8];
    bool slow = false;
    #pragma unroll
    for (uint32_t k = 0; k < 8; ++ k) {
        y[k] = 1.f + __expf(-g[k]);
        float r;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(y[k]));
        r = fmaf(r, fmaf(-y[k], r, 1.f), r);
        const float q = __fmul_rn(g[k], r);
        s[k] = fmaf(r, fmaf(-y[k], q, g[k]), q);
        const float x = fabsf(g[k]);
        slow |= not (y[k] < 0x1p64f) or not (x < 0x1p64f) or not (x >= 0x1p-64f);
    }
    if (slow) {
        #pragma unroll
        for (uint32_t k = 0; k < 8; ++ k)
            s[k] = g[k] / y[k];
    }
}

CUTLASS_DEVICE void wait_parity(uint64_t* bar, const uint32_t& parity) {
    uint32_t done = 0;
    while (not done)
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
                     : "=r"(done) : "r"(smem_u32(bar)), "r"(parity) : "memory");
}

CUTLASS_DEVICE uint64_t tagged(uint32_t chunk, float m) {
    return static_cast<uint64_t>(chunk + 1) << 32 | __float_as_uint(m);
}

// Waits for the peer's amax of p's chunk, then stores p's e4m3 values and (CTA 0) its scale byte
CUTLASS_DEVICE void silu_finish(SiluPending& p, SiluSmem& s, const uint32_t& lane_idx, const uint32_t& j,
                                const uint32_t& q, uint8_t* act_out, const uint32_t& act_ld, uint8_t* sf_out,
                                const uint32_t& sf_rows) {
    if (not p.live)
        return;
    float inv = 0.f;
    uint32_t e = 0;
    if (q == 0) {
        const auto at = smem_u32(&s.peer[p.chunk % kSlots][j]);
        uint64_t got;
        do
            asm volatile("ld.relaxed.cluster.shared::cta.b64 %0, [%1];" : "=l"(got) : "r"(at) : "memory");
        while ((got >> 32) != p.chunk + 1);
        const uint32_t b = __float_as_uint(fmaxf(fmaxf(p.m, __uint_as_float(static_cast<uint32_t>(got))), 1e-10f) / 448.f);
        e = ((b >> 23) & 0xff) + ((b & 0x7fffff) != 0);
        inv = 1.f / __uint_as_float(e << 23);
    }
    inv = __shfl_sync(0xffffffffu, inv, lane_idx & ~7u);
    *reinterpret_cast<uint2*>(act_out + static_cast<size_t>(p.row) * act_ld + p.col * 64 + q * 8) =
        make_uint2(fp8x4(p.a[0] * inv, p.a[1] * inv, p.a[2] * inv, p.a[3] * inv),
                   fp8x4(p.a[4] * inv, p.a[5] * inv, p.a[6] * inv, p.a[7] * inv));
    if (cute::block_rank_in_cluster() == 0 and q == 0) {
        const uint32_t kb = p.col / 2;
        sf_out[(static_cast<size_t>(kb >> 2) * sf_rows + p.row) * 4 + (kb & 3)] = static_cast<uint8_t>(e);
    }
    p.live = false;
}

// Per 16-token chunk: the accumulators land in smem as upstream's swap-AB store lays out its bf16 D
// (two 128B-swizzled atoms of 64 columns, the gate's then the up's), then thread t takes token t / 8 and
// act columns 8 (t % 8) + [0, 8).
// With overlapped TMEM columns (block M 256: the SF columns borrow the accumulators' tail) the chunks of
// accumulator stage 0 run backwards, and the overlap barrier is arrived once the overlapped columns are read,
// as upstream's swap-AB store does.
template <uint32_t kNumTMAStoreStages, uint32_t kNumOverlappedTmemCols, typename smem_t>
CUTLASS_DEVICE void silu_store(smem_t& smem, SiluSmem& s, SiluPending& p, uint32_t& chunk, uint32_t& stage,
                               const uint32_t& tmem_base_addr, const uint32_t& base_m_idx,
                               const uint32_t& n_block_idx, const uint32_t& effective_m,
                               const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                               uint8_t* act_out, const uint32_t& act_ld, uint8_t* sf_out, const uint32_t& sf_rows,
                               const bool& reverse,
                               const cutlass::arch::ClusterTransactionBarrier* tmem_overlap_barrier,
                               const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier) {
    constexpr uint32_t kSwizzle = 128, kAtomBytes = 16 * kSwizzle;
    constexpr uint32_t kNumOverlapLoads = math::constexpr_ceil_div(kNumOverlappedTmemCols, 8u);
    const uint32_t rank = cute::block_rank_in_cluster();
    const uint32_t t = epilogue_warp_idx * 32 + lane_idx, j = t / 8, q = t % 8;
    const auto num_stores = effective_m / 16;
    for (uint32_t it = 0; it < num_stores; ++ it, ++ chunk, stage = (stage + 1) % kNumTMAStoreStages) {
        const uint32_t st = reverse ? num_stores - 1 - it : it;
        cutlass::arch::NamedBarrier::sync(128, 0);
        auto cd = reinterpret_cast<uint8_t*>(smem.cd[stage]);
        #pragma unroll
        for (uint32_t i = 0; i < 2; ++ i) {
            uint32_t v[8];
            const uint32_t tmem_addr = tmem_base_addr + st * 16 + (reverse ? 1 - i : i) * 8;
            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr, v[0], v[1], v[2], v[3]);
            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000, v[4], v[5], v[6], v[7]);
            cutlass::arch::fence_view_async_tmem_load();
            if constexpr (kNumOverlapLoads > 0) {
                if (it * 2 + i + 1 == cute::min<uint32_t>(kNumOverlapLoads, num_stores * 2)) {
                    ptx::tcgen05_before_thread_sync();
                    tmem_overlap_barrier->arrive(0u);
                }
            }
            if (it == num_stores - 1 and i == 1) {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }
            const uint32_t row = lane_idx % 8, col = (epilogue_warp_idx % 2) * 4 + lane_idx / 8;
            ptx::SM90_U32x4_STSM_T<int>::copy(math::cast_into_bf16_and_pack(v[0], v[1]),
                                              math::cast_into_bf16_and_pack(v[2], v[3]),
                                              math::cast_into_bf16_and_pack(v[4], v[5]),
                                              math::cast_into_bf16_and_pack(v[6], v[7]),
                                              cd + (epilogue_warp_idx / 2) * kAtomBytes + (reverse ? 1 - i : i) * 8 * kSwizzle +
                                                  row * kSwizzle + (col ^ row) * 16);
        }
        cutlass::arch::NamedBarrier::sync(128, 0);

        const uint32_t at = j * kSwizzle + ((q ^ (j % 8)) * 16);
        const uint4 gw = *reinterpret_cast<const uint4*>(cd + at);
        const uint4 uw = *reinterpret_cast<const uint4*>(cd + kAtomBytes + at);
        const auto g2 = reinterpret_cast<const nv_bfloat162*>(&gw);
        const auto u2 = reinterpret_cast<const nv_bfloat162*>(&uw);
        float g[8], u[8], a[8], m = 0.f;
        #pragma unroll
        for (uint32_t k = 0; k < 4; ++ k) {
            const float2 gk = __bfloat1622float2(g2[k]), uk = __bfloat1622float2(u2[k]);
            g[2 * k] = gk.x, g[2 * k + 1] = gk.y, u[2 * k] = uk.x, u[2 * k + 1] = uk.y;
        }
        silu8(g, a);
        #pragma unroll
        for (uint32_t k = 0; k < 8; ++ k) {
            a[k] = __bfloat162float(__float2bfloat16_rn(a[k] * u[k]));
            m = fmaxf(m, fabsf(a[k]));
        }
        #pragma unroll
        for (uint32_t o = 4; o; o >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (q == 0)
            asm volatile("st.relaxed.cluster.shared::cluster.b64 [%0], %1;"
                         :: "r"(peer_addr(smem_u32(&s.peer[chunk % kSlots][j]), rank ^ 1)), "l"(tagged(chunk, m)) : "memory");

        silu_finish(p, s, lane_idx, j, q, act_out, act_ld, sf_out, sf_rows);
        #pragma unroll
        for (uint32_t k = 0; k < 8; ++ k)
            p.a[k] = a[k];
        p.m = m, p.chunk = chunk, p.row = base_m_idx + st * 16 + j, p.col = n_block_idx, p.live = true;
    }
}

} // namespace glm


template <cute::UMMA::Major kMajorA, cute::UMMA::Major kMajorB,
          uint32_t kGranKA, uint32_t kGranKB, uint32_t kKAlignment,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleCDMode,
          uint32_t kNumStages, uint32_t kNumTMAStoreStages,
          uint32_t kNumNonEpilogueThreads, uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast, bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kSwapAB, bool kEnsureZeroPadding,
          GemmType kGemmType, bool kWithAccumulation,
          typename a_dtype_t, typename b_dtype_t, typename cd_dtype_t,
          typename epilogue_op_t>
CUTLASS_GLOBAL void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)
glm_moe_gu_silu_impl(int* grouped_layout, const int* route, const uint8_t* a_rows, uint8_t* act_out, uint8_t* sf_out,
                             uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                             const __grid_constant__ epilogue_op_t epilogue_op,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                             const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    using Allocator = cute::conditional_t<kNumMulticast == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

    // NOTES: an FP4xFP4 pair uses the dedicated MXF4 MMA (packed 4-bit operands, K-major only, UMMA_K=64);
    //        all other combinations use the MXF8F6F4 MMA (unpacked operands, UMMA_K=32)
    constexpr bool kIsMXF4 = cute::is_same_v<a_dtype_t, cutlass::float_e2m1_t> and
                             cute::is_same_v<b_dtype_t, cutlass::float_e2m1_t>;

    // The host launches the epilogue operator directly as a kernel argument
    DG_STATIC_ASSERT(sizeof(epilogue_op_t) == sizeof(EpilogueArgs),
                     "Epilogue operators must not add state to `EpilogueArgs`");

    // C/D type: BF16 and FP32 are supported, with or without accumulation; FP8 C/D requires the
    // dynamically-scaled epilogue with per-32 UE8M0 SFD output, batched only, without accumulation
    constexpr bool kWithOutputSF = cute::is_same_v<epilogue_op_t, epilogue::transform::EpilogueDynamicScaledFP8>;
    DG_STATIC_ASSERT(kWithOutputSF ? (cute::is_same_v<cd_dtype_t, cutlass::float_e4m3_t> and
                                      kGemmType == GemmType::Batched and not kWithAccumulation) :
                                     (cute::is_same_v<cd_dtype_t, float> or cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>),
                     "Invalid C/D data dtype");

    DG_STATIC_ASSERT(kSwapAB and kNumMulticast == 2 and BLOCK_N == 128 and
                     kGemmType == GemmType::MGroupedContiguousWithPsumLayout and kMajorB == cute::UMMA::Major::K,
                     "the silu epilogue runs the swap-AB m-grouped psum GEMM on a 2-CTA cluster");

    // MMA Configs
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;
    constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;
    constexpr uint32_t UMMA_K = kIsMXF4 ? 64 : 32;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / (kIsMulticastOnA ? kNumMulticast: 1);
    constexpr uint32_t LOAD_BLOCK_N = BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t kPackFactor = get_smem_pack_factor<a_dtype_t>();
    DG_STATIC_ASSERT(kPackFactor == get_smem_pack_factor<b_dtype_t>(), "A/B SMEM pack factor must match");
    constexpr bool kIsFP4A = cute::is_same_v<a_dtype_t, cutlass::float_e2m1_t> or
                             cute::is_same_v<a_dtype_t, cutlass::detail::float_e2m1_unpacksmem_t>;
    constexpr bool kIsFP4B = cute::is_same_v<b_dtype_t, cutlass::float_e2m1_t> or
                             cute::is_same_v<b_dtype_t, cutlass::detail::float_e2m1_unpacksmem_t>;
    DG_STATIC_ASSERT((kIsMXF4 and BLOCK_K == 256) or (not kIsMXF4 and BLOCK_K == 128), "Invalid block K");
    DG_STATIC_ASSERT(BLOCK_K % UMMA_K == 0, "Block K must be divisible by UMMA K");
    DG_STATIC_ASSERT(kNumMulticast == 1 or kNumMulticast == 2, "Only support 1/2 multicast");
    DG_STATIC_ASSERT((kSwapAB and BLOCK_N == LAYOUT_AD_M) or
                     (not kSwapAB and (BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == LAYOUT_AD_M)), "Invalid block size");

    // MXF4 only supports K-major operands and 32-element SF granularity
    DG_STATIC_ASSERT(not kIsMXF4 or (kMajorA == cute::UMMA::Major::K and kMajorB == cute::UMMA::Major::K), "MXF4 only supports K-major");
    DG_STATIC_ASSERT(not kIsMXF4 or (kGranKA == 32 and kGranKB == 32), "MXF4 only supports kGranK=32");

    // SF configs
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M = math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N = math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_K = BLOCK_K / 128;
    constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
    constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
    DG_STATIC_ASSERT(kGranKA == 32 or kGranKA == 128, "Invalid granularity K for A");
    DG_STATIC_ASSERT(kGranKB == 32 or kGranKB == 128, "Invalid granularity K for B");
    DG_STATIC_ASSERT(not is_k_grouped_contiguous(kGemmType) or
                     (kGranKA == kGranKB and (kGranKA == 32 or kGranKA == 128)),
                     "K-grouped SF requires matching granularity K 32/128");
    DG_STATIC_ASSERT(not is_k_grouped_contiguous(kGemmType) or kKAlignment % BLOCK_K == 0,
                     "K alignment must be divisible by block K");

    // Epilogue configs
    // Always enable pipeline for better performance
    constexpr uint32_t kNumEpilogueStages = 2;
    DG_STATIC_ASSERT(kNumTMAStoreStages == 1 or kNumTMAStoreStages == 2, "Invalid number of TMA store stages");
    // NOTES: To maximize epilogue threads utilization, process an entire BLOCK_N
    //        per store stage for swap-AB cases, and an entire BLOCK_M for non-swap cases
    constexpr uint32_t STORE_BLOCK_M =        kSwapAB ? 16      : cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N =        kSwapAB ? BLOCK_N : kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = kSwapAB ? kNumEpilogueThreads: STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

    // NOTES: Make sure we have enough shared memory for UMMA padding
    constexpr uint32_t UMMA_A_SIZE_PER_STAGE = math::constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(a_dtype_t) / kPackFactor;

    // Tensor memory size and offsets
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M * SF_BLOCK_K / 32;
    constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N * SF_BLOCK_K / 32;
    constexpr uint32_t kNumSFTmemCols = kNumSFATmemCols + kNumSFBTmemCols;
    constexpr uint32_t kNumOverlappedTmemCols = cute::max<uint32_t>(kNumAccumTmemCols + kNumSFTmemCols, 512) - 512;
    constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFTmemCols - kNumOverlappedTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols - kNumOverlappedTmemCols;
    constexpr uint32_t kTmemStartColOfSFB = kTmemStartColOfSFA + kNumSFATmemCols;
    DG_STATIC_ASSERT(kNumOverlappedTmemCols <= UMMA_N, "Invalid overlapped tensor memory columns");
    DG_STATIC_ASSERT(kSwapAB or kNumOverlappedTmemCols == 0 or kNumOverlappedTmemCols <= STORE_BLOCK_N,
                     "Non-swap overlapped tensor memory columns must fit in the first epilogue store");
    DG_STATIC_ASSERT(32 <= kNumTmemCols and kNumTmemCols <= 512, "Invalid tensor memory columns");

    // Synchronize the cluster before 2-CTA TMEM allocation
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : void();

    // Utils
    const bool is_leader_cta = cute::block_rank_in_cluster() == 0;
    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == 0) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const auto shape_sfa_k = math::ceil_div(shape_k, kGranKA * 4);
    const auto shape_sfb_k = math::ceil_div(shape_k, kGranKB * 4);

    using SharedStorage = layout::SM100FP8FP4GemmSharedStorage<
        kNumStages, kNumEpilogueStages, kNumTMAStoreStages, LOAD_BLOCK_M, LOAD_BLOCK_N, BLOCK_K,
        STORE_BLOCK_M, STORE_BLOCK_N, SF_BLOCK_M, SF_BLOCK_N, SF_BLOCK_K,
        a_dtype_t, b_dtype_t, cd_dtype_t>;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
    auto& silu = *reinterpret_cast<glm::SiluSmem*>(smem_buffer + math::constexpr_align<uint32_t>(sizeof(SharedStorage), 1024));
    DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <= sizeof(smem.a[0]) + sizeof(smem.b), "Memory out of bound for UMMA");

    // Initialize barriers
    if (warp_idx == 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // Arrive at all CTAs
            smem.sf_full_barriers[i].init(1);
            smem.empty_barriers[i].init(1);
            // Arrive only at the leader CTA
            smem.full_barriers[i].init(kNumMulticast * (1 + 32 * SF_BLOCK_K));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" :: "r"(glm::smem_u32(&silu.a_full[i])));
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++ i) {
            // Arrive at all CTAs
            smem.tmem_full_barriers[i].init(1);
            // Arrive only at the leader CTA
            smem.tmem_empty_barriers[i].init(kNumMulticast * kNumUMMAStoreThreads);
            smem.tmem_overlap_barriers[i].init(kNumMulticast * kNumUMMAStoreThreads);
        }

        #pragma unroll
        for (uint32_t i = 0; i < glm::kSlots * 16; ++ i)
            (&silu.peer[0][0])[i] = 0;

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == 2) {
        // Allocate tensor memory
        Allocator().allocate(kNumTmemCols, &smem.tmem_ptr);
    }
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs, kEnsureZeroPadding, kKAlignment, kGranKA * 4>(
        shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline and TMA phases
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;

        // Flip phases only if reach the next first stage
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // Dispatch warps into different roles
    if (warp_idx == 0 and cute::elect_one_sync()) {
        // TMA load warp
        DG_STATIC_ASSERT(not kIsFP4A and not kIsFP4B and kMajorA == cute::UMMA::Major::K, "e4m3 K-major operands");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Use dynamic load block M, when swap-AB is enabled
            const auto load_block_m = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast : LOAD_BLOCK_M;

            // For k-grouped layout, the number of block K is variable
            const auto num_total_k_blocks = cute::max(1u, math::ceil_div(scheduler.current_shape_k, BLOCK_K));
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait consumer release
                smem.empty_barriers[stage_idx].wait(phase ^ 1);

                // Compute offsets
                // M-grouped masked A stores groups along M; contiguous A uses global M.
                constexpr bool kAWithGroupMNOffset = kGemmType == GemmType::MGroupedMasked;
                uint32_t m_idx = scheduler.template get_global_idx<kAWithGroupMNOffset, sched::IndexType::MN> (
                    shape_m, BLOCK_M, m_block_idx);
                // M-grouped K-major B stores groups along N.
                constexpr bool kBWithGroupMNOffset = (kMajorB == cute::UMMA::Major::K) and
                                                     (is_m_grouped_contiguous(kGemmType) or kGemmType == GemmType::MGroupedMasked);
                uint32_t n_idx = scheduler.template get_global_idx<kBWithGroupMNOffset, sched::IndexType::MN> (
                    shape_n, BLOCK_N, n_block_idx, m_block_idx);

                DG_STATIC_ASSERT(kGemmType == GemmType::Normal or is_k_grouped_contiguous(kGemmType) or kGemmType == GemmType::Batched or
                                 kMajorA == cute::UMMA::Major::K, "Invalid major");
                // K-grouped A uses a shared K arena; other MN-major A groups along K.
                constexpr bool kAWithGroupKOffset = is_k_grouped_contiguous(kGemmType) or (kMajorA == cute::UMMA::Major::MN);
                // K-grouped B uses a shared K arena; M-grouped MN-major B groups along K.
                constexpr bool kBWithGroupKOffset = is_k_grouped_contiguous(kGemmType) or (kMajorB == cute::UMMA::Major::MN);
                uint32_t k_a_idx = scheduler.template get_global_idx<kAWithGroupKOffset, sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);
                uint32_t k_b_idx = scheduler.template get_global_idx<kBWithGroupKOffset, sched::IndexType::K> (
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

                // Add 2 CTA offsets
                if constexpr (kNumMulticast > 1) {
                    m_idx += kIsMulticastOnA ? (cute::block_rank_in_cluster() * load_block_m) : 0;
                    n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N);
                }

                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);

                // Issue SFA and SFB TMAs first, so that the transpose can overlap the A/B transfer.
                // No swizzling, so one TMA for one SF block is enough. SF loads land on their own barrier.
                uint32_t sf_arrival_bytes = 0;
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    uint32_t sfa_m_idx = m_block_idx * BLOCK_M;
                    uint32_t sfa_k_idx = scheduler.template get_global_idx<(not is_m_grouped_contiguous(kGemmType)), sched::IndexType::SF_K>(
                        shape_sfa_k, SF_BLOCK_K, k_block_idx / kNumSFAStagesPerLoad);
                    tma::copy<SF_BLOCK_M, SF_BLOCK_K, 0>(&tensor_map_sfa, &smem.sf_full_barriers[stage_idx], smem.sfa[stage_idx], sfa_m_idx, sfa_k_idx);
                    sf_arrival_bytes += sizeof(smem.sfa[0]);
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    uint32_t sfb_n_idx = n_block_idx * (BLOCK_N / 2);
                    uint32_t sfb_k_idx = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(
                        shape_sfb_k, SF_BLOCK_K, k_block_idx / kNumSFBStagesPerLoad, m_block_idx);
                    tma::copy<SF_BLOCK_N / 2, SF_BLOCK_K, 0>(&tensor_map_sfb, &smem.sf_full_barriers[stage_idx], smem.sfb[stage_idx], sfb_n_idx, sfb_k_idx);
                    tma::copy<SF_BLOCK_N / 2, SF_BLOCK_K, 0>(&tensor_map_sfb, &smem.sf_full_barriers[stage_idx], smem.sfb[stage_idx] + SF_BLOCK_N / 2,
                                                             sfb_n_idx + shape_n / 2, sfb_k_idx);
                    sf_arrival_bytes += sizeof(smem.sfb[0]);
                }
                smem.sf_full_barriers[stage_idx].arrive_and_expect_tx(sf_arrival_bytes);

                // The n-block's 64 gate rows, then its 64 up rows (shape_n / 2 further)
                if constexpr (kMajorB == cute::UMMA::Major::K) {
                    const uint32_t gate_idx = n_idx - n_block_idx * BLOCK_N + n_block_idx * (BLOCK_N / 2);
                    tma::copy<BLOCK_K, LOAD_BLOCK_N / 2, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, &smem.full_barriers[stage_idx], smem.b[stage_idx], k_b_idx, gate_idx, kNumMulticast, batch_idx);
                    tma::copy<BLOCK_K, LOAD_BLOCK_N / 2, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, &smem.full_barriers[stage_idx], smem.b[stage_idx] + LOAD_BLOCK_N / 2 * BLOCK_K,
                        k_b_idx, gate_idx + shape_n / 2, kNumMulticast, batch_idx);
                }
                if constexpr (kMajorB == cute::UMMA::Major::MN)
                    tma::copy<LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode, b_dtype_t, kIsBatchedMM>(
                        &tensor_map_b, &smem.full_barriers[stage_idx], smem.b[stage_idx], n_idx, k_b_idx, kNumMulticast, batch_idx);

                // Arrive at full barriers
                if (is_leader_cta) {
                    smem.full_barriers[stage_idx].arrive_and_expect_tx(LOAD_BLOCK_N * BLOCK_K * kNumMulticast);
                } else {
                    smem.full_barriers[stage_idx].arrive(0u);
                }
            }
        }
    } else if (warp_idx == 1 and is_leader_cta) {
        // MMA issue warp
        // NOTES: only the leader CTA will do this
        // Make instruction descriptor
        auto instr_desc = kSwapAB ? cute::UMMA::make_instr_desc_block_scaled<b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t,
                                                                             UMMA_M, UMMA_N, kMajorB, kMajorA>()
                                  : cute::UMMA::make_instr_desc_block_scaled<a_dtype_t, b_dtype_t, float, cutlass::float_ue8m0_t,
                                                                             UMMA_M, UMMA_N, kMajorA, kMajorB>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a_desc = mma::sm100::make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(smem.a[0], 0, 0);
        auto b_desc = mma::sm100::make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(smem.b[0], 0, 0);
        uint32_t a_desc_lo = lane_idx < kNumStages ? a_desc.lo + lane_idx * sizeof(smem.a[0]) / 16 : 0u;
        uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * sizeof(smem.b[0]) / 16 : 0u;

        // Checks for MMA instructions
        // NOTES: CUTLASS does not have such checks except the MMA traits, but we are not using these traits
        DG_STATIC_ASSERT((UMMA_M == 64  and UMMA_N %  8 == 0 and  8 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 128 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256) or
                         (UMMA_M == 256 and UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256),
                         "Invalid MMA instruction shape");

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Wait until this accumulator stage is fully reusable.
            // The overlap barrier is deferred to the first K block so that
            // the SF UTCCP can be issued while the preceding epilogue drains.
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            smem.tmem_empty_barriers[accum_stage_idx].wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
                auto umma_arrive = [](const uint64_t* barrier) {
                    if constexpr (kNumMulticast == 1) {
                        cutlass::arch::umma_arrive(barrier);
                    } else {
                        constexpr uint16_t kCTAMask = (1 << kNumMulticast) - 1;
                        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, kCTAMask);
                    }
                };
                umma_arrive(reinterpret_cast<uint64_t*>(&smem.empty_barriers[stage_idx]));

                // NOTES: the tensor memory accumulator pipeline has nothing to do with multicasting
                if (do_tmem_full_arrive)
                    umma_arrive(reinterpret_cast<uint64_t*>(&smem.tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            };

            // Dynamic update of UMMA N based on effective M, when swap-AB is enabled
            if constexpr (kSwapAB) {
                uint32_t umma_n = scheduler.get_aligned_effective_m_in_block(m_block_idx);
                mma::sm100::update_instr_desc_with_umma_n(instr_desc, umma_n);
            }

            // Launch MMAs
            const auto num_total_k_blocks = cute::max(1u, math::ceil_div(scheduler.current_shape_k, BLOCK_K));
            #pragma unroll 4
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                const auto a_desc_base_lo = ptx::exchange(a_desc_lo, stage_idx);
                const auto b_desc_base_lo = ptx::exchange(b_desc_lo, stage_idx);

                // Wait A/B TMA and SF-transpose arrival
                smem.full_barriers[stage_idx].wait(phase);
                ptx::tcgen05_after_thread_sync();

                const uint32_t sfa_stage_in_group_idx = k_block_idx % kNumSFAStagesPerLoad;
                const uint32_t sfb_stage_in_group_idx = k_block_idx % kNumSFBStagesPerLoad;
                if (cute::elect_one_sync()) {
                    // Do SF copy at certain stages
                    // TODO: process shared memory descriptor by addition
                    using cute_utccp_t = cute::conditional_t<kNumMulticast == 1,
                        cute::SM100_UTCCP_4x32dp128bit_1cta, cute::SM100_UTCCP_4x32dp128bit_2cta>;
                    if (sfa_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_K * SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem.sfa[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
                        }
                    }
                    if (sfb_stage_in_group_idx == 0) {
                        #pragma unroll
                        for (uint32_t i = 0; i < SF_BLOCK_K * SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i) {
                            auto smem_ptr = smem.sfb[stage_idx] + i * kNumUTCCPAlignedElems;
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
                        }
                    }

                }
                __syncwarp();

                // The overlap barrier is waited after SF UTCCP but before the
                // first UMMA: UTCCP can overlap the preceding epilogue, and
                // subsequent K blocks share the same accumulator columns.
                if constexpr (kNumOverlappedTmemCols > 0) {
                    if (k_block_idx == 0 and scheduler.current_iter > 0) {
                        ptx::tcgen05_before_thread_sync();
                        const auto preceding_iter_idx = scheduler.current_iter - 1;
                        const auto preceding_stage_idx = preceding_iter_idx % kNumEpilogueStages;
                        const auto preceding_phase_idx = (preceding_iter_idx / kNumEpilogueStages) & 1;
                        smem.tmem_overlap_barriers[preceding_stage_idx].wait(preceding_phase_idx);
                        ptx::tcgen05_after_thread_sync();
                    }
                }

                if (cute::elect_one_sync()) {
                    // Issue UMMA
                    using mma_t = cute::conditional_t<kIsMXF4,
                        cute::conditional_t<kNumMulticast == 1, ptx::SM100_MMA_MXF4_SS, ptx::SM100_MMA_MXF4_2x1SM_SS>,
                        cute::conditional_t<kNumMulticast == 1, ptx::SM100_MMA_MXF8F6F4_SS, ptx::SM100_MMA_MXF8F6F4_2x1SM_SS>>;
                    #pragma unroll
                    for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / UMMA_K; ++ umma_k_idx) {
                        const uint32_t offset = umma_k_idx * UMMA_K;
                        // Which 128-K SF sub-block this UMMA K step belongs to
                        const uint32_t subblock_idx = offset / kNumUTCCPAlignedElems;
                        // SF id (in units of 32 K-elements within the 128-K sub-block):
                        //   gran-32  selects the 32-element SF at this UMMA K offset (MXF4's `scale_vec::2X`
                        //            consumes two 32-element SFs per UMMA K=64 step, hence a step of 2);
                        //   gran-128 uses the per-load stage id (whole 512-K SF span).
                        const uint32_t sf_id_in_subblock = (offset % kNumUTCCPAlignedElems) / 32;
                        const uint32_t tmem_col_sfa = kTmemStartColOfSFA + subblock_idx * SF_BLOCK_M / 32;
                        const uint32_t tmem_col_sfb = kTmemStartColOfSFB + subblock_idx * SF_BLOCK_N / 32;
                        const uint32_t sfa_id = (kGranKA == 32 ? sf_id_in_subblock : sfa_stage_in_group_idx);
                        const uint32_t sfb_id = (kGranKB == 32 ? sf_id_in_subblock : sfb_stage_in_group_idx);
                        const auto runtime_instr_desc = kSwapAB ?
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfb_id, sfa_id):
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);

                        a_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorA, LOAD_BLOCK_M, kSwizzleAMode, a_dtype_t>(a_desc_base_lo, 0, offset);
                        b_desc.lo = mma::sm100::advance_umma_desc_lo<kMajorB, LOAD_BLOCK_N, kSwizzleBMode, b_dtype_t>(b_desc_base_lo, 0, offset);
                        if constexpr (kSwapAB) {
                            mma_t::fma(b_desc, a_desc, accum_stage_idx * (UMMA_N - kNumOverlappedTmemCols),
                                       umma_k_idx > 0 or k_block_idx > 0, runtime_instr_desc,
                                       tmem_col_sfb, tmem_col_sfa);
                        } else {
                            mma_t::fma(a_desc, b_desc, accum_stage_idx * (UMMA_N - kNumOverlappedTmemCols),
                                       umma_k_idx > 0 or k_block_idx > 0, runtime_instr_desc,
                                       tmem_col_sfa, tmem_col_sfb);
                        }
                    }
                }
                __syncwarp();

                // Commit to the mbarrier object
                // No explicit `tcgen05.fence::before_thread_sync` is needed, as this is implicitly performed by `tcgen05.commit`
                empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
            }
        }

        // To safely deconstruct barriers, we need another round of waits
        const auto iter_idx = scheduler.current_iter - 1;
        if (kNumMulticast > 1 and iter_idx >= 0) {
            const auto accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
            smem.tmem_empty_barriers[iter_idx % kNumEpilogueStages].wait(accum_phase_idx);
        }
    } else if (warp_idx == 2 or (SF_BLOCK_K == 2 and warp_idx == 3)) {
        // UTCCP transposer
        // NOTES: use up to 2 warps to transpose, one per SF sub-block on K (only warp 3 when SF_BLOCK_K == 2)
        const uint32_t sf_k_subblock_idx = warp_idx - 2;
        auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
            DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
            uint32_t values[4];
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++ i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto num_total_k_blocks = cute::max(1u, math::ceil_div(scheduler.current_shape_k, BLOCK_K));
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                // Wait SF TMA arrival
                smem.sf_full_barriers[stage_idx].wait(phase);

                // Transpose for UTCCP at certain stages
                if (k_block_idx % kNumSFAStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem.sfa[stage_idx] + sf_k_subblock_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems);
                }
                if (k_block_idx % kNumSFBStagesPerLoad == 0) {
                    #pragma unroll
                    for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++ i)
                        utccp_required_smem_warp_transpose(smem.sfb[stage_idx] + sf_k_subblock_idx * SF_BLOCK_N + i * kNumUTCCPAlignedElems);
                }
                // The gather warp's A rows of this stage landed
                glm::wait_parity(&silu.a_full[stage_idx], phase);

                // TODO: figure out whether the proxy fence is valid for 2-CTA cases
                cutlass::arch::fence_view_async_shared();

                smem.full_barriers[stage_idx].arrive(0u);
            }
        }
    } else if (warp_idx == 3) {
        // A gather warp: this CTA's half of the block's token rows, straight from the tokens by the routing's row
        // table, one 128 B row of a K block per 8 lanes (16 B a cp.async, the 128B swizzle by hand); each lane's
        // copies of a stage arrive on silu.a_full, which the UTCCP warp waits before its full-barrier arrival.
        DG_STATIC_ASSERT(SF_BLOCK_K == 1 and kMajorA == cute::UMMA::Major::K and BLOCK_K == 128 and kNumStages <= 8,
                         "one SF warp, 128 B rows, a_full per stage");
        constexpr uint32_t kRowsPerLane = LOAD_BLOCK_M / 4;
        const uint32_t c = lane_idx % 8, r0 = lane_idx / 8;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            const auto load_block_m = scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast;
            const uint32_t p0 = m_block_idx * BLOCK_M + cute::block_rank_in_cluster() * load_block_m;
            const uint32_t end = cute::min(scheduler.current_psum_m, p0 + load_block_m);
            const uint8_t* src[kRowsPerLane];
            #pragma unroll
            for (uint32_t i = 0; i < kRowsPerLane; ++ i) {
                const uint32_t p = p0 + r0 + i * 4;
                src[i] = p < end ? a_rows + static_cast<size_t>(__ldg(route + p)) * shape_k + c * 16 : nullptr;
            }
            const auto num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                smem.empty_barriers[stage_idx].wait(phase ^ 1);
                const uint32_t dst = glm::smem_u32(smem.a[stage_idx]);
                #pragma unroll
                for (uint32_t i = 0; i < kRowsPerLane; ++ i) {
                    const uint32_t r = r0 + i * 4;
                    if (src[i] != nullptr)
                        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                                     :: "r"(dst + r * 128 + ((c ^ (r % 8)) * 16)), "l"(src[i] + k_block_idx * BLOCK_K) : "memory");
                }
                asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"
                             :: "r"(glm::smem_u32(&silu.a_full[stage_idx])) : "memory");
            }
        }
    } else if (warp_idx >= kNumNonEpilogueThreads / 32 and warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
        // Epilogue warp groups
        const auto epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);

        // NOTES: tensor memory addresses are simplified, as the hardware will ignore the warp index bits,
        // i.e., no need for `tmem_ptr |= (epilogue_warp_idx * 32) << 16`.
        // NOTES: we also forbid two CTAs to share the same SM and its tensor memory
        DG_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(&smem.tmem_ptr) == 0);

        // Share store pipeline between blocks
        uint32_t tma_stage_idx = 0, silu_chunk = 0;
        glm::SiluPending silu_pending;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            // Wait UMMA arrival
            smem.tmem_full_barriers[accum_stage_idx].wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const auto tmem_base_addr = accum_stage_idx * (UMMA_N - kNumOverlappedTmemCols);
            const auto base_m_idx = scheduler.template get_global_idx<false, sched::IndexType::MN>(shape_m, BLOCK_M, m_block_idx);

            DG_STATIC_ASSERT(kNumEpilogueStages == 2, "the reverse order depends on exactly 2 accumulator stages");
            glm::silu_store<kNumTMAStoreStages, kNumOverlappedTmemCols>(
                smem, silu, silu_pending, silu_chunk, tma_stage_idx, tmem_base_addr, base_m_idx, n_block_idx,
                scheduler.get_aligned_effective_m_in_block(m_block_idx), epilogue_warp_idx, lane_idx, act_out,
                shape_n / 2, sf_out, shape_m, kNumOverlappedTmemCols > 0 and accum_stage_idx == 0,
                &smem.tmem_overlap_barriers[accum_stage_idx], &smem.tmem_empty_barriers[accum_stage_idx]);
        }
        const uint32_t t = epilogue_warp_idx * 32 + lane_idx;
        glm::silu_finish(silu_pending, silu, lane_idx, t / 8, t % 8, act_out, shape_n / 2, sf_out, shape_m);
    }

    // TODO: Remove redundant synchronization
    kNumMulticast > 1 ? comm::cluster_sync_with_relaxed_arrive() : __syncthreads();

    // Deallocate tensor memory
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_100f");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
