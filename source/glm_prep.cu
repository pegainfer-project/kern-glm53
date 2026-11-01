// GLM-5.3's derived weights, run once inside weight loading over the checkpoint's own tensors.
//
// glm_requant: the checkpoint's e4m3 weights with f32 128x128 block scales, re-quantized to UE8M0
// block scales the way vLLM prepares them for DeepGEMM on Blackwell (requant_weight_ue8m0): a block
// dequantized in f32, sf = 2^ceil(log2(max(amax, 1e-4) / 448)), w = e4m3(x * (1 / sf)). Source rows
// come in groups of `group_rows` (one expert's matrix, or the whole tensor); a group lands at
// destination row g * dst_group_rows + dst_row0, so [gate; up] of every expert interleave by expert.
// Scales are DeepGEMM's per-row packed words: row n of group g, column block kb at byte (kb & 3) of
// word g * sf_group + (kb >> 2) * sf_ld + dst_row0 + n.
//   grid (ceil(rows / 128), K / 128), block 256
//
// glm_kvb: kv_b_proj dequantized to bf16 [64 * 448, 512] (W_UV is a view of it), and W_UK
// transposed per head into [64, 512, 192] for the q absorption's batched GEMM.
//   grid (64 * 448 * 512 / 256), block 256
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

typedef unsigned char u8;

__device__ __forceinline__ float e4m3(u8 b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return float(v);
}

extern "C" __global__ void __launch_bounds__(256) glm_requant(const u8* __restrict__ w, const float* __restrict__ s,
                                                              u8* __restrict__ wq, u8* __restrict__ sf, int rows,
                                                              int K, int group_rows, int dst_group_rows,
                                                              int dst_row0, int sf_ld, long long sf_group) {
  __shared__ float sm[8];
  const int rb = blockIdx.x, kb = blockIdx.y, t = threadIdx.x;
  const int i = t >> 1, c0 = kb * 128 + (t & 1) * 64, sr = rb * 128 + i;
  const float scale = s[(size_t)rb * (K / 128) + kb];
  float v[64], amax = 0.f;
  const bool live = sr < rows;
#pragma unroll
  for (int j = 0; j < 64; j += 16) {
    uint4 u = live ? *reinterpret_cast<const uint4*>(w + (size_t)sr * K + c0 + j) : make_uint4(0, 0, 0, 0);
    const u8* b = reinterpret_cast<const u8*>(&u);
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      v[j + k] = e4m3(b[k]) * scale;
      amax = fmaxf(amax, fabsf(v[j + k]));
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if ((t & 31) == 0) sm[t >> 5] = amax;
  __syncthreads();
  amax = sm[0];
#pragma unroll
  for (int k = 1; k < 8; ++k) amax = fmaxf(amax, sm[k]);
  const unsigned bits = __float_as_uint(fmaxf(amax, 1e-4f) / 448.f);
  unsigned e = (bits >> 23) & 0xff;
  e += (bits & 0x7fffff) != 0;
  const float inv = 1.f / __uint_as_float(e << 23);
  if (!live) return;
  const int g = sr / group_rows, ri = sr % group_rows;
  u8* d = wq + ((size_t)g * dst_group_rows + dst_row0 + ri) * K + c0;
#pragma unroll
  for (int j = 0; j < 64; j += 16) {
    uint4 u;
    u8* b = reinterpret_cast<u8*>(&u);
#pragma unroll
    for (int k = 0; k < 16; ++k) b[k] = __nv_cvt_float_to_fp8(v[j + k] * inv, __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint4*>(d + j) = u;
  }
  if ((t & 1) == 0)
    sf[((size_t)g * sf_group + (size_t)(kb >> 2) * sf_ld + dst_row0 + ri) * 4 + (kb & 3)] = (u8)e;
}

extern "C" __global__ void glm_kvb(const u8* __restrict__ w, const float* __restrict__ s,
                                   __nv_bfloat16* __restrict__ kvb, __nv_bfloat16* __restrict__ wuk) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 64 * 448 * 512) return;
  const int row = idx >> 9, c = idx & 511;
  const __nv_bfloat16 v = __float2bfloat16(e4m3(w[idx]) * s[(row >> 7) * 4 + (c >> 7)]);
  kvb[idx] = v;
  const int h = row / 448, r = row % 448;
  if (r < 192) wuk[((size_t)h * 512 + c) * 192 + r] = v;
}
