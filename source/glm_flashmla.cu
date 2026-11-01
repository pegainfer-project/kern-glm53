// FlashMLA's sm100 sparse prefill (head64, d_qk 576, d_v 512), unmodified: upstream's own
// instantiation of the kernel template, built to a cubin. scripts/flashmla_probe.cu records the
// launch's parameter blocks that scripts/glm_flashmla.py lays out.
#include "kernels/sm100/prefill/sparse/fwd/head64/instantiations/phase1_h64_k576.cu"
