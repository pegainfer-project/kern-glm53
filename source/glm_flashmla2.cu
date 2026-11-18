// FlashMLA's sm100 sparse prefill (head64, d_qk 576, d_v 512) with two softmax warpgroups:
// source/flashmla_h64x2/ (README.md there says what changed). scripts/flashmla_probe.cu records the launch's
// parameter blocks that scripts/glm_ops.py lays out.
#include "flashmla_h64x2/phase1.cuh"

namespace sm100::prefill::sparse_fwd::head64x2 {

static_assert(sizeof(KernelTemplate<SparseAttnFwdMode::Prefill, 576>::SharedMemoryPlan) == GLM_SMEM);
template void run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, 576>(const SparseAttnFwdParams& params);

}
