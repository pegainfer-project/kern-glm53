#pragma once

#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/layout/mqa_logits.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/sm100_mqa_logits.cuh>

namespace deep_gemm {

// GLM's indexer logits, DeepGEMM's sm100_mqa_logits (ab69f76, MIT) cut down to its one configuration: fp8 q and keys,
// per-key f32 scales, bf16 head weights and head sums, compressed bf16 logits, the metadata kernel's schedule. A CTA
// scores a q block of BLOCK_Q rows, kHalves MMAs of N 256 (8 rows x 32 heads) over each 128-key half of a split:
// every half of the block reads the one shared-memory copy of the split's keys, the fill the power cap pays for.
// Warpgroup g takes the keys [128 g, 128 g + 128) of every split through its TMEM stage, the block's halves in turn,
// a half's 8 rows out of TMEM as relu'd bf16 pairs before the head sums (their weights read from shared memory).
template <uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t kNumKVStages, uint32_t kNumSMs>
CUTLASS_GLOBAL __launch_bounds__(384, 1)
void glm_mqa_logits(const uint32_t num_q_tokens, const uint32_t num_kv_tokens,
                    const uint32_t logits_stride,
                    const uint32_t* cu_seq_len_k_start,
                    const uint32_t* cu_seq_len_k_end,
                    const uint32_t* schedule_meta,
                    __nv_bfloat16* logits,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_kv,
                    const __grid_constant__ cute::TmaDescriptor tensor_map_weights) {
    using e4m3 = cutlass::float_e4m3_t;
    using bf16 = __nv_bfloat16;
    constexpr uint32_t kNumHeads = 32, kHeadDim = 128, kNumQStages = 1;
    constexpr uint32_t kNumMathWarpGroups = 2, kNumMathThreads = 256, kSpecWarpStart = 8;
    constexpr uint32_t kNumTmemStages = kNumMathWarpGroups;
    constexpr uint32_t kHalfRows = 8, kHalves = BLOCK_Q / kHalfRows;
    constexpr uint32_t UMMA_M = 128, UMMA_N = kHalfRows * kNumHeads, UMMA_K = 32;
    DG_STATIC_ASSERT(SPLIT_KV == kNumMathWarpGroups * UMMA_M and BLOCK_Q % kHalfRows == 0, "GLM's tile");

    using SharedStorage = layout::MQALogitsSharedStorage<kNumHeads, kHeadDim, false, BLOCK_Q, SPLIT_KV,
                                                         kNumQStages, kNumKVStages, kNumTmemStages, e4m3, bf16>;
    extern __shared__ __align__(SharedStorage::kSwizzleAlignment) uint8_t smem_buffer[];
    auto& smem = *reinterpret_cast<SharedStorage*>(smem_buffer);
    constexpr uint32_t kQBytes = BLOCK_Q * kNumHeads * kHeadDim, kWeightBytes = BLOCK_Q * kNumHeads * 2;
    constexpr uint32_t kKVBytes = SPLIT_KV * kHeadDim, kSFBytes = SPLIT_KV * 4;

    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();
    const auto sm_idx = blockIdx.x;

    if (warp_idx == kSpecWarpStart) {
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_weights);
        cute::prefetch_tma_descriptor(&tensor_map_kv);
        cute::prefetch_tma_descriptor(&tensor_map_sf_kv);
    }
    if (warp_idx == kSpecWarpStart + 1 and cute::elect_one_sync()) {
        smem.full_q_barriers[0].init(1);
        smem.empty_q_barriers[0].init(kNumMathThreads + 32);
        #pragma unroll
        for (uint32_t i = 0; i < kNumKVStages; ++ i) {
            smem.full_kv_barriers[i].init(1);
            smem.empty_kv_barriers[i].init(kNumMathThreads);
        }
        #pragma unroll
        for (uint32_t i = 0; i < kNumTmemStages; ++ i) {
            smem.full_tmem_barriers[i].init(1);
            smem.empty_tmem_barriers[i].init(128);
        }
        cutlass::arch::fence_barrier_init();
    } else if (warp_idx == kSpecWarpStart + 2) {
        cute::TMEM::Allocator1Sm().allocate(512, &smem.tmem_ptr_in_smem);
    }
    __syncthreads();

    RingPipeline<kNumQStages> q_pipeline;
    RingPipeline<kNumKVStages> kv_pipeline;
    RingPipeline<kNumTmemStages> tmem_pipeline;
    const auto make_scheduler = [&]() {
        return sched::SM100MQALogitsScheduler<BLOCK_Q, SPLIT_KV, kNumSMs, true>(
            sm_idx, num_q_tokens, num_kv_tokens, cu_seq_len_k_start, cu_seq_len_k_end, schedule_meta);
    };

    cudaGridDependencySynchronize();

    if (warp_idx == kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_dealloc<56>();
        if (cute::elect_one_sync()) {
            auto scheduler = make_scheduler();
            uint32_t q_block_idx, kv_base, num_kv_splits;
            while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
                CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
                smem.empty_q_barriers[q_stage_idx].wait(q_phase ^ 1);
                const uint32_t q_token_base = q_block_idx * BLOCK_Q;
                #pragma unroll
                for (uint32_t h = 0; h < kHalves; ++ h)
                    tma::copy<kHeadDim, UMMA_N, 0>(&tensor_map_q, &smem.full_q_barriers[q_stage_idx],
                                                   smem.smem_q[q_stage_idx] + h * UMMA_N * kHeadDim, 0,
                                                   (q_token_base + h * kHalfRows) * kNumHeads);
                tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights, &smem.full_q_barriers[q_stage_idx],
                                                 smem.smem_weights[q_stage_idx], 0, q_token_base);
                smem.full_q_barriers[q_stage_idx].arrive_and_expect_tx(kQBytes + kWeightBytes);
            }
        }
        __syncwarp();
    } else if (warp_idx == kSpecWarpStart + 1) {
        cutlass::arch::warpgroup_reg_dealloc<56>();
        if (cute::elect_one_sync()) {
            auto scheduler = make_scheduler();
            uint32_t q_block_idx, kv_base, num_kv_splits;
            while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
                #pragma unroll 1
                for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    smem.empty_kv_barriers[kv_stage_idx].wait(kv_phase ^ 1);
                    const uint32_t offset = kv_base + kv_split_idx * SPLIT_KV;
                    tma::copy<kHeadDim, SPLIT_KV, 0>(&tensor_map_kv, &smem.full_kv_barriers[kv_stage_idx],
                                                     smem.smem_kv[kv_stage_idx], 0, offset);
                    tma::copy<SPLIT_KV, 1, 0>(&tensor_map_sf_kv, &smem.full_kv_barriers[kv_stage_idx],
                                              smem.smem_sf_kv[kv_stage_idx], offset, 0);
                    smem.full_kv_barriers[kv_stage_idx].arrive_and_expect_tx(kKVBytes + kSFBytes);
                }
            }
        }
        __syncwarp();
    } else if (warp_idx == kSpecWarpStart + 2) {
        cutlass::arch::warpgroup_reg_dealloc<56>();
        auto instr_desc = cute::UMMA::make_instr_desc<e4m3, e4m3, float, UMMA_M, UMMA_N, cute::UMMA::Major::K, cute::UMMA::Major::K>();
        auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
        auto scheduler = make_scheduler();
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.next_q_block(q_block_idx, kv_base, num_kv_splits)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);
            for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
                if (cute::elect_one_sync()) {
                    #pragma unroll
                    for (uint32_t h = 0; h < kHalves; ++ h) {
                        #pragma unroll
                        for (uint32_t i = 0; i < kNumMathWarpGroups; ++ i) {
                            CUTE_TIE_DECL(tmem_pipeline.advance(), tmem_stage_idx, tmem_phase);
                            smem.empty_tmem_barriers[tmem_stage_idx].wait(tmem_phase ^ 1);
                            ptx::tcgen05_after_thread_sync();
                            #pragma unroll
                            for (uint32_t k = 0; k < kHeadDim / UMMA_K; ++ k) {
                                auto a_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                                    smem.smem_kv[kv_stage_idx], i * UMMA_M, k * UMMA_K);
                                auto b_desc = mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, kHeadDim>(
                                    smem.smem_q[q_stage_idx], h * UMMA_N, k * UMMA_K);
                                ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_stage_idx * UMMA_N, k, runtime_instr_desc);
                            }
                            ptx::umma_arrive_no_elect(smem.full_tmem_barriers[tmem_stage_idx]);
                        }
                    }
                }
                __syncwarp();
            }
            smem.empty_q_barriers[q_stage_idx].arrive();
        }
    } else if (warp_idx == kSpecWarpStart + 3) {
        cutlass::arch::warpgroup_reg_dealloc<56>();
    } else if (warp_idx < kSpecWarpStart) {
        cutlass::arch::warpgroup_reg_alloc<224>();
        const auto math_warpgroup_idx = warp_idx / 4;
        const auto math_thread_idx = warp_idx * 32 + lane_idx;
        tmem_pipeline.advance(math_warpgroup_idx);
        uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
        float accum[kNumHeads];

        auto scheduler = make_scheduler();
        uint32_t q_block_idx, kv_base, num_kv_splits;
        while (scheduler.template next_q_block<true>(q_block_idx, kv_base, num_kv_splits, seq_k_start, seq_k_end)) {
            CUTE_TIE_DECL(q_pipeline.advance(), q_stage_idx, q_phase);
            smem.full_q_barriers[q_stage_idx].wait(q_phase);

            // Row i's logit for key k at element base_i + k (rows x stride < 2^32); a block whose rows start at one
            // key checks only each row's end
            uint32_t base[BLOCK_Q], len[BLOCK_Q];
            bool same = true;
            #pragma unroll
            for (uint32_t i = 0; i < BLOCK_Q; ++ i) {
                base[i] = (q_block_idx * BLOCK_Q + i) * logits_stride - seq_k_start[i];
                len[i] = seq_k_end[i] - seq_k_start[i];
                same &= seq_k_start[i] == seq_k_start[0];
            }

            const auto run_splits = [&](auto same_t) {
                constexpr bool kSame = decltype(same_t)::value;
                for (uint32_t kv_split_idx = 0; kv_split_idx < num_kv_splits; ++ kv_split_idx) {
                    const uint32_t key = kv_base + kv_split_idx * SPLIT_KV + math_thread_idx;
                    CUTE_TIE_DECL(kv_pipeline.advance(), kv_stage_idx, kv_phase);
                    smem.full_kv_barriers[kv_stage_idx].wait(kv_phase);
                    const auto scale_kv = static_cast<bf16>(ptx::ld_shared(smem.smem_sf_kv[kv_stage_idx] + math_thread_idx));

                    #pragma unroll
                    for (uint32_t h = 0; h < kHalves; ++ h) {
                        CUTE_TIE_DECL(tmem_pipeline.advance(kNumMathWarpGroups), tmem_stage_idx, tmem_phase);
                        smem.full_tmem_barriers[tmem_stage_idx].wait(tmem_phase);
                        ptx::tcgen05_after_thread_sync();
                        if (h == kHalves - 1)
                            smem.empty_kv_barriers[kv_stage_idx].arrive();

                        // The half's relu'd bf16 pairs out of TMEM first: the stage goes back to the MMA before the head sums
                        nv_bfloat162 relu[kHalfRows][kNumHeads / 2];
                        #pragma unroll
                        for (uint32_t i = 0; i < kHalfRows; ++ i) {
                            ptx::tmem_load_32dp32b<kNumHeads>(tmem_stage_idx * UMMA_N + i * kNumHeads, reinterpret_cast<uint32_t*>(accum));
                            cutlass::arch::fence_view_async_tmem_load();
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads / 2; ++ j)
                                relu[i][j] = ptx::cvt_relu_bf16x2_f32(make_float2(accum[2 * j], accum[2 * j + 1]));
                        }
                        ptx::tcgen05_before_thread_sync();
                        smem.empty_tmem_barriers[tmem_stage_idx].arrive();

                        #pragma unroll
                        for (uint32_t i = 0; i < kHalfRows; ++ i) {
                            const auto r = h * kHalfRows + i;
                            const auto w = reinterpret_cast<const uint4*>(smem.smem_weights[q_stage_idx] + r * kNumHeads);
                            nv_bfloat162 wr[kNumHeads / 2];
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads / 8; ++ j)
                                *reinterpret_cast<uint4*>(wr + 4 * j) = w[j];
                            auto sum_0 = __floats2bfloat162_rn(0.0f, 0.0f);
                            auto sum_1 = __floats2bfloat162_rn(0.0f, 0.0f);
                            #pragma unroll
                            for (uint32_t j = 0; j < kNumHeads / 2; j += 2) {
                                sum_0 = __hfma2(relu[i][j], wr[j], sum_0);
                                sum_1 = __hfma2(relu[i][j + 1], wr[j + 1], sum_1);
                            }
                            const auto sum = __hadd2_rn(sum_0, sum_1);
                            const auto result = static_cast<bf16>(__hadd_rn(sum.x, sum.y) * scale_kv);
                            if (key - seq_k_start[kSame ? 0 : r] < len[r])
                                logits[base[r] + key] = result;
                        }
                    }
                }
            };
            if (same)
                run_splits(cute::true_type{});
            else
                run_splits(cute::false_type{});

            smem.empty_q_barriers[q_stage_idx].arrive();
        }
        cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
        if (warp_idx == 0)
            cute::TMEM::Allocator1Sm().free(0, 512);
    }
}

} // namespace deep_gemm
