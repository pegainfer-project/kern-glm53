// GLM-5.3's routing tables for the MoE's grouped GEMMs' psum layout, from the router's top-8 ids, in three
// launches: glm_moe_count tallies each slice of SLICE tokens' rows per expert, glm_moe_place lays the groups
// out, the shared expert's T rows first (psum[0] = T), routed expert e's from its `tile`-aligned offset past
// them (psum[e + 1] its end), and puts row (t, k) at p = exp2perm[t * 8 + k], an expert's rows in token
// order; glm_moe_scatter writes rows[p] = t, the token the gate-up GEMM gathers, and its activation's scale
// words at p (the shared expert's row t is token t), or past 4096 rows glm_moe_place itself. Every count is
// a popcount of a token bitmap: the same tables every run.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define EXPERTS 256
#define TOPK 8
#define SLICE 256
#define WORDS (SLICE / 32)
#define SF_WORDS 12  // a 6144-column row's UE8M0 scales, four a word
#define H 6144

// Bit i of mask[e][w] is token 32 w + i of the slice routed to expert e.
__device__ void slice_masks(const int* __restrict__ ids, int t0, int rows, unsigned (*mask)[WORDS]) {
  const int i = threadIdx.x;
  for (int k = i; k < EXPERTS * WORDS; k += SLICE) (&mask[0][0])[k] = 0;
  __syncthreads();
  if (i < rows) {
    const int4* p = reinterpret_cast<const int4*>(ids + (size_t)(t0 + i) * TOPK);
    const int4 a = p[0], b = p[1];
    const int e[TOPK] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int k = 0; k < TOPK; ++k) atomicOr(&mask[e[k]][i / 32], 1u << (i % 32));
  }
  __syncthreads();
}

__device__ int popc_masks(const unsigned* m) {
  int c = 0;
#pragma unroll
  for (int w = 0; w < WORDS; ++w) c += __popc(m[w]);
  return c;
}

//   grid (ceil(T / SLICE)), block SLICE; hist [slices][EXPERTS]
extern "C" __global__ void __launch_bounds__(SLICE) glm_moe_count(const int* __restrict__ ids, int T,
                                                                   int* __restrict__ hist) {
  __shared__ unsigned mask[EXPERTS][WORDS];
  const int t0 = blockIdx.x * SLICE;
  slice_masks(ids, t0, min(SLICE, T - t0), mask);
  hist[blockIdx.x * EXPERTS + threadIdx.x] = popc_masks(mask[threadIdx.x]);
}

// With rows_out, also glm_moe_scatter's writes, a token a thread (where a few CTAs keep up with them).
//   grid (ceil(T / SLICE)), block SLICE; xsf [SF_WORDS][x_rows] words, asf [SF_WORDS][a_rows]
extern "C" __global__ void __launch_bounds__(SLICE) glm_moe_place(const int* __restrict__ ids, int T,
                                                                   const int* __restrict__ hist, int tile,
                                                                   int* __restrict__ psum,
                                                                   int* __restrict__ exp2perm,
                                                                   int* __restrict__ rows_out,
                                                                   const unsigned* __restrict__ xsf, int x_rows,
                                                                   unsigned* __restrict__ asf, int a_rows) {
  __shared__ unsigned mask[EXPERTS][WORDS];
  __shared__ int base[EXPERTS];
  const int e = threadIdx.x, s = blockIdx.x, slices = gridDim.x, t0 = s * SLICE;
  int before = 0, count = 0;
  for (int b = 0; b < slices; ++b) {
    const int h = hist[b * EXPERTS + e];
    before += b < s ? h : 0;
    count += h;
  }
  const int blocks = (count + tile - 1) / tile;
  base[e] = blocks;
  __syncthreads();
  for (int o = 1; o < EXPERTS; o <<= 1) {
    const int v = e >= o ? base[e - o] : 0;
    __syncthreads();
    base[e] += v;
    __syncthreads();
  }
  const int start = ((T + tile - 1) / tile + base[e] - blocks) * tile;
  if (s == 0) psum[e + 1] = start + count;
  if (s == 0 and e == 0) psum[0] = T;
  __syncthreads();
  base[e] = start + before;

  const int n = min(SLICE, T - t0);
  slice_masks(ids, t0, n, mask);
  const int i = threadIdx.x;
  if (i >= n) return;
  const int4* p = reinterpret_cast<const int4*>(ids + (size_t)(t0 + i) * TOPK);
  const int4 a = p[0], b = p[1];
  const int x[TOPK] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  int out[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    const unsigned* m = mask[x[k]];
    int r = __popc(m[i / 32] & ((1u << (i % 32)) - 1));
    for (int w = 0; w < i / 32; ++w) r += __popc(m[w]);
    out[k] = base[x[k]] + r;
  }
  const int t = t0 + i;
  int4* q = reinterpret_cast<int4*>(exp2perm + (size_t)t * TOPK);
  q[0] = make_int4(out[0], out[1], out[2], out[3]);
  q[1] = make_int4(out[4], out[5], out[6], out[7]);
  if (rows_out == nullptr) return;
  unsigned w[SF_WORDS];
#pragma unroll
  for (int j = 0; j < SF_WORDS; ++j) w[j] = xsf[(size_t)j * x_rows + t];
  rows_out[t] = t;
#pragma unroll
  for (int j = 0; j < SF_WORDS; ++j) asf[(size_t)j * a_rows + t] = w[j];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    rows_out[out[k]] = t;
#pragma unroll
    for (int j = 0; j < SF_WORDS; ++j) asf[(size_t)j * a_rows + out[k]] = w[j];
  }
}

// Token t's grouped rows (the shared expert's t, its routed exp2perm[t * 8 + k]): word y of its scales at each
// of them (y < SF_WORDS), or (y = SF_WORDS) t in the row table.
//   grid (ceil(T / 256), SF_WORDS + 1), block 256; xsf [SF_WORDS][x_rows] words, asf [SF_WORDS][a_rows]
extern "C" __global__ void __launch_bounds__(256) glm_moe_scatter(const int* __restrict__ exp2perm, int T,
                                                                  int* __restrict__ rows_out,
                                                                  const unsigned* __restrict__ xsf, int x_rows,
                                                                  unsigned* __restrict__ asf, int a_rows) {
  const int t = blockIdx.x * 256 + threadIdx.x, y = blockIdx.y;
  if (t >= T) return;
  const int4* q = reinterpret_cast<const int4*>(exp2perm + (size_t)t * TOPK);
  const int4 a = q[0], b = q[1];
  const int p[TOPK + 1] = {t, a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  if (y == SF_WORDS) {
#pragma unroll
    for (int k = 0; k <= TOPK; ++k) rows_out[p[k]] = t;
    return;
  }
  const unsigned w = xsf[(size_t)y * x_rows + t];
#pragma unroll
  for (int k = 0; k <= TOPK; ++k) asf[(size_t)y * a_rows + p[k]] = w;
}

// out[t] = bf16(shared[t] + sum_k wts[t, k] * fc2[exp2perm[t * 8 + k]]), the sum in f32 in that order (glm_glue's
// glm_moe_combine's), a thread's 27 row loads (24 columns of the shared row and the 8 routed) issued before its sums.
//   grid (T), block 256: 24 columns a thread, eight at a time
extern "C" __global__ void __launch_bounds__(256) glm_moe_combine(const __nv_bfloat16* __restrict__ fc2,
                                                                  const int* __restrict__ exp2perm,
                                                                  const float* __restrict__ wts,
                                                                  const __nv_bfloat16* __restrict__ shared,
                                                                  __nv_bfloat16* __restrict__ out, int T) {
  const int t = blockIdx.x;
  const int4* q = reinterpret_cast<const int4*>(exp2perm + (size_t)t * TOPK);
  const float4* w4 = reinterpret_cast<const float4*>(wts + (size_t)t * TOPK);
  const int4 a = q[0], b = q[1];
  const float4 wa = w4[0], wb = w4[1];
  const int p[TOPK] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  const float w[TOPK] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
  uint4 v[3][TOPK + 1];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int c = (i * 256 + threadIdx.x) * 8;
    v[i][0] = *reinterpret_cast<const uint4*>(shared + (size_t)t * H + c);
#pragma unroll
    for (int k = 0; k < TOPK; ++k) v[i][k + 1] = *reinterpret_cast<const uint4*>(fc2 + (size_t)p[k] * H + c);
  }
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int c = (i * 256 + threadIdx.x) * 8;
    float acc[8];
    const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(&v[i][0]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(s2[j]);
      acc[2 * j] = f.x, acc[2 * j + 1] = f.y;
    }
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v[i][k + 1]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h[j]);
        acc[2 * j] = fmaf(w[k], f.x, acc[2 * j]);
        acc[2 * j + 1] = fmaf(w[k], f.y, acc[2 * j + 1]);
      }
    }
    __nv_bfloat162 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
    *reinterpret_cast<uint4*>(out + (size_t)t * H + c) = *reinterpret_cast<const uint4*>(o);
  }
}

// glm_prep's glm_requant (the same re-quantization, the same scales) into the routed gate-up GEMM's tile-major
// weight layout: the 64 rows x 128 bytes of a (64-row block, K block) tile contiguous, the tiles of a row block
// in K order, so each TMA box of the GEMM is one 8 KB read. Destination row R, column block kb sit at byte
// ((R / 64) * (K / 128) + kb) * 8192 + (R % 64) * 128 of wq.
//   grid (ceil(rows / 128), K / 128), block 256
__device__ __forceinline__ float e4m3_float(unsigned char b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return float(v);
}

extern "C" __global__ void __launch_bounds__(256) glm_requant_tiled(const unsigned char* __restrict__ w,
                                                                    const float* __restrict__ s,
                                                                    unsigned char* __restrict__ wq,
                                                                    unsigned char* __restrict__ sf, int rows, int K,
                                                                    int group_rows, int dst_group_rows, int dst_row0,
                                                                    int sf_ld, long long sf_group) {
  __shared__ float sm[8];
  const int rb = blockIdx.x, kb = blockIdx.y, t = threadIdx.x;
  const int i = t >> 1, c0 = kb * 128 + (t & 1) * 64, sr = rb * 128 + i;
  const float scale = s[(size_t)rb * (K / 128) + kb];
  float v[64], amax = 0.f;
  const bool live = sr < rows;
#pragma unroll
  for (int j = 0; j < 64; j += 16) {
    uint4 u = live ? *reinterpret_cast<const uint4*>(w + (size_t)sr * K + c0 + j) : make_uint4(0, 0, 0, 0);
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&u);
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      v[j + k] = e4m3_float(b[k]) * scale;
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
  const size_t R = (size_t)g * dst_group_rows + dst_row0 + ri;
  unsigned char* d = wq + ((R / 64) * (K / 128) + kb) * 8192 + (R % 64) * 128 + (t & 1) * 64;
#pragma unroll
  for (int j = 0; j < 64; j += 16) {
    uint4 u;
    unsigned char* b = reinterpret_cast<unsigned char*>(&u);
#pragma unroll
    for (int k = 0; k < 16; ++k) b[k] = __nv_cvt_float_to_fp8(v[j + k] * inv, __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint4*>(d + j) = u;
  }
  if ((t & 1) == 0)
    sf[((size_t)g * sf_group + (size_t)(kb >> 2) * sf_ld + dst_row0 + ri) * 4 + (kb & 3)] = (unsigned char)e;
}
