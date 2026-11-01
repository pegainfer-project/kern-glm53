// The parameter blocks of FlashMLA's sparse prefill launch, recorded without a GPU: upstream's own
// host `run()` builds them while this executable stands in for the CUDA calls it makes. Every
// tensor-map encode is logged and its 128 bytes filled with a marker, so the dump of the launch's
// two arguments says where each descriptor sits and what the rest of the bytes are.
//   nvcc -gencode arch=compute_103a,code=sm_103a -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -cudart shared \
//     -I<FlashMLA>/csrc -I<FlashMLA>/csrc/kerutils/include -I<FlashMLA>/csrc/cutlass/include \
//     -o flashmla_probe scripts/flashmla_probe.cu scripts/flashmla_probe_hooks.cpp -lcuda
//   ./flashmla_probe <s_q> <stride_kv_s_kv> <paint byte> > layout.json
#include "kernels/sm100/prefill/sparse/fwd/head64/instantiations/phase1_h64_k576.cu"

// Bytes the launcher never writes keep whatever the stack held; painting it first makes them visible.
__attribute__((noinline)) static void paint(unsigned char v) {
  volatile unsigned char stack[1 << 16];
  for (unsigned i = 0; i < sizeof(stack); ++i) stack[i] = v;
}

int main(int argc, char** argv) {
  paint(argc > 3 ? atoi(argv[3]) : 0);
  SparseAttnFwdParams p{};
  p.s_q = argc > 1 ? atoi(argv[1]) : 16384;
  p.s_kv = 0x7fffffff;
  p.h_q = 64, p.h_kv = 1, p.d_qk = 576, p.d_v = 512, p.topk = 2048;
  p.sm_scale = 0.0625f, p.sm_scale_div_log2 = 0.0625f * 1.4426950408889634f;
  p.q = reinterpret_cast<cutlass::bfloat16_t*>(0x100000000ull);
  p.kv = reinterpret_cast<cutlass::bfloat16_t*>(0x200000000ull);
  p.indices = reinterpret_cast<int*>(0x300000000ull);
  p.out = reinterpret_cast<cutlass::bfloat16_t*>(0x400000000ull);
  p.max_logits = reinterpret_cast<float*>(0x500000000ull);
  p.lse = reinterpret_cast<float*>(0x600000000ull);
  p.stride_q_s_q = 64 * 576, p.stride_q_h_q = 576;
  p.stride_kv_s_kv = argc > 2 ? atoi(argv[2]) : 576, p.stride_kv_h_kv = 576;
  p.stride_indices_s_q = 2048, p.stride_indices_h_kv = 2048;
  p.num_sm = 152;
  sm100::prefill::sparse_fwd::head64::run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill, 576>(p);
  return 0;
}
