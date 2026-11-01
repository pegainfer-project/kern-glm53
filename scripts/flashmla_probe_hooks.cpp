// The CUDA entry points FlashMLA's host launcher calls, stood in for by recorders (flashmla_probe.cu).
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <vector>

struct Encode {
  int dtype, rank, swizzle, l2, interleave, oob;
  unsigned long long base;
  unsigned long long dims[5], strides[4];
  unsigned box[5], elem[5];
};
static std::vector<Encode> encodes;

extern "C" CUresult cuTensorMapEncodeTiled(CUtensorMap* map, CUtensorMapDataType dtype, cuuint32_t rank, void* base,
                                           const cuuint64_t* dims, const cuuint64_t* strides, const cuuint32_t* box,
                                           const cuuint32_t* elem, CUtensorMapInterleave interleave,
                                           CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2,
                                           CUtensorMapFloatOOBfill oob) {
  Encode e{};
  e.dtype = dtype, e.rank = rank, e.swizzle = swizzle, e.l2 = l2, e.interleave = interleave, e.oob = oob;
  e.base = reinterpret_cast<unsigned long long>(base);
  for (unsigned i = 0; i < rank; ++i) e.dims[i] = dims[i], e.box[i] = box[i], e.elem[i] = elem[i];
  for (unsigned i = 0; i + 1 < rank; ++i) e.strides[i] = strides[i];
  unsigned char* p = reinterpret_cast<unsigned char*>(map);
  for (int i = 0; i < 128; ++i) p[i] = 0xA0 + encodes.size();
  encodes.push_back(e);
  return CUDA_SUCCESS;
}

extern "C" cudaError_t cudaGetDriverEntryPoint(const char*, void** fn, unsigned long long,
                                               cudaDriverEntryPointQueryResult* status) {
  *fn = reinterpret_cast<void*>(&cuTensorMapEncodeTiled);
  if (status) *status = cudaDriverEntryPointSuccess;
  return cudaSuccess;
}
extern "C" cudaError_t cudaGetDriverEntryPointByVersion(const char*, void** fn, unsigned int, unsigned long long,
                                                        cudaDriverEntryPointQueryResult* status) {
  *fn = reinterpret_cast<void*>(&cuTensorMapEncodeTiled);
  if (status) *status = cudaDriverEntryPointSuccess;
  return cudaSuccess;
}
extern "C" cudaError_t cudaFuncSetAttribute(const void*, cudaFuncAttribute, int) { return cudaSuccess; }
extern "C" cudaError_t cudaGetLastError() { return cudaSuccess; }
extern "C" cudaError_t cudaPeekAtLastError() { return cudaSuccess; }

static void dump(const char* name, const void* p, size_t n) {
  const unsigned char* b = static_cast<const unsigned char*>(p);
  printf("  \"%s\": \"", name);
  for (size_t i = 0; i < n; ++i) printf("%02x", b[i]);
  printf("\",\n");
}

extern "C" cudaError_t __cudaGetKernel(cudaKernel_t* k, const void*) { *k = nullptr; return cudaSuccess; }
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t, dim3 grid, dim3 block, void** args, size_t smem, cudaStream_t) {
  printf("{\n  \"grid\": [%u, %u, %u], \"block\": [%u, %u, %u], \"smem\": %zu,\n", grid.x, grid.y, grid.z, block.x,
         block.y, block.z, smem);
  dump("params", args[0], 0x90);
  dump("tma", args[1], 0x500);
  printf("  \"encodes\": [\n");
  for (size_t i = 0; i < encodes.size(); ++i) {
    const Encode& e = encodes[i];
    printf("    {\"marker\": %zu, \"dtype\": %d, \"rank\": %d, \"swizzle\": %d, \"l2\": %d, \"oob\": %d, \"base\": %llu, "
           "\"dims\": [", 0xA0 + i, e.dtype, e.rank, e.swizzle, e.l2, e.oob, e.base);
    for (int j = 0; j < e.rank; ++j) printf("%s%llu", j ? ", " : "", e.dims[j]);
    printf("], \"strides\": [");
    for (int j = 0; j + 1 < e.rank; ++j) printf("%s%llu", j ? ", " : "", e.strides[j]);
    printf("], \"box\": [");
    for (int j = 0; j < e.rank; ++j) printf("%s%u", j ? ", " : "", e.box[j]);
    printf("], \"elem\": [");
    for (int j = 0; j < e.rank; ++j) printf("%s%u", j ? ", " : "", e.elem[j]);
    printf("]}%s\n", i + 1 < encodes.size() ? "," : "");
  }
  printf("  ]\n}\n");
  return cudaSuccess;
}

