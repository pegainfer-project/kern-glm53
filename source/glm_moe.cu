// GLM-5.3's routing tables for the routed GEMMs' psum layout, from the router's top-8 ids, in two launches:
// glm_moe_count tallies each slice of SLICE tokens' rows per expert, glm_moe_place lays the experts out
// (expert e's rows from its `tile`-aligned offset, psum[e] its end) and puts row (t, k) at
// p = exp2perm[t * 8 + k], an expert's rows in token order: rows[p] = t, the token the gate-up GEMM gathers,
// and its activation's scale words at p. Every count is a popcount of a token bitmap: the same tables
// every run.
#include <stdint.h>

#define EXPERTS 256
#define TOPK 8
#define SLICE 256
#define WORDS (SLICE / 32)
#define SF_WORDS 12  // a 6144-column row's UE8M0 scales, four a word

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
  const int start = (base[e] - blocks) * tile;
  if (s == 0) psum[e] = start + count;
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
  unsigned w[SF_WORDS];
#pragma unroll
  for (int j = 0; j < SF_WORDS; ++j) w[j] = xsf[(size_t)j * x_rows + t];
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    rows_out[out[k]] = t;
#pragma unroll
    for (int j = 0; j < SF_WORDS; ++j) asf[(size_t)j * a_rows + out[k]] = w[j];
  }
}
