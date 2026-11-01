// GLM-5.3's prefill glue: everything between the vendored GEMMs, the indexer, the top-k and the
// sparse attention of a packed chunk. Each entry is a plain extern "C" ABI with its grid noted; T is
// the chunk's rows, a row's sequence and position come from glm_plan.
//
// Numerics follow vLLM's GLM-5 path on Blackwell (vllm/models/deepseek_v32, fused_norm_rope /
// fused_q, per_token_group_quant_fp8 with UE8M0 scales):
//   - residual add + RMSNorm as vLLM's fused_add_rms_norm: the sum lands in bf16, then
//     bf16(bf16(x * rrms) * gamma);
//   - q_a / kv_a norms as the Triton fused_norm_rope: bf16((x * rrms) * gamma) in f32;
//   - activations quantized per 128 columns, scale = 2^ceil(log2(max(amax, eps) / 448)), DeepGEMM's
//     packed layout: byte (kb & 3) of word (kb >> 2, row), words row-contiguous ([K/512][sf_rows]);
//   - the indexer key (LayerNorm, interleaved rope on its first 64 dims) and query quantized in f32
//     with eps 1e-4, the key cached as e4m3 bytes and one f32 scale.
// Paged index cache, page-blocked as vLLM's: page p holds 64 rows of 128 e4m3 bytes, then their 64
// f32 scales (132 bytes a token). The latent cache is [slot][576] bf16: kv_norm | roped k_pe.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
typedef unsigned char u8;

#define H 6144
#define QL 2048
#define KVL 512
#define HALF 32
#define HEADS 64
#define QHD 256
#define NOPE 192
#define MLA 576
#define IH 32
#define ID 128
#define TOPK 2048
#define PAGE 64
#define IDX_PAGE (PAGE * (ID + 4))
#define EPS 1e-5f

__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float rnd(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Sum over a 256-thread block, every thread gets the total.
__device__ __forceinline__ float block_sum(float v, float* sm) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  __syncthreads();
  if (l == 0) sm[w] = v;
  __syncthreads();
  float t = l < 8 ? sm[l] : 0.f;
  return warp_sum(t);
}

// The UE8M0 scale of a block: (exponent byte, the power of two itself).
__device__ __forceinline__ float ue8m0(float amax, float eps, u8* e) {
  const unsigned b = __float_as_uint(fmaxf(amax, eps) / 448.f);
  unsigned x = (b >> 23) & 0xff;
  x += (b & 0x7fffff) != 0;
  *e = (u8)x;
  return __uint_as_float(x << 23);
}

__device__ __forceinline__ unsigned fp8x4(float a, float b, float c, float d) {
  const __nv_fp8x4_e4m3 q(make_float4(a, b, c, d));
  return *reinterpret_cast<const unsigned*>(&q);
}

__device__ __forceinline__ void put_sf(u8* sf, int sf_rows, int r, int kb, u8 e) {
  sf[((size_t)(kb >> 2) * sf_rows + r) * 4 + (kb & 3)] = e;
}

// Four values of one lane in a warp's 128-column block quantized: the lane writes its 4 bytes, lane 0 the scale.
__device__ __forceinline__ void quant4(const float v[4], u8* q, u8* sf, int sf_rows, int r, int kb, float eps) {
  const float amax = warp_max(fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
  u8 e;
  const float inv = 1.f / ue8m0(amax, eps, &e);
  *reinterpret_cast<unsigned*>(q) = fp8x4(v[0] * inv, v[1] * inv, v[2] * inv, v[3] * inv);
  if ((threadIdx.x & 31) == 0) put_sf(sf, sf_rows, r, kb, e);
}

__device__ __forceinline__ void ld4(const bf16* p, float v[4]) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
  const float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
  v[0] = a.x, v[1] = a.y, v[2] = b.x, v[3] = b.y;
}
__device__ __forceinline__ void st4(bf16* p, const float v[4]) {
  __nv_bfloat162 h[2] = {__floats2bfloat162_rn(v[0], v[1]), __floats2bfloat162_rn(v[2], v[3])};
  *reinterpret_cast<uint2*>(p) = *reinterpret_cast<const uint2*>(h);
}

// A row's sequence, position and indexer bounds. seq_lens are the sequences' lengths with this
// chunk's rows; the indexer's keys are gathered sequence after sequence (glm_idx_gather), so row r
// scores keys [ks, ke) of that stream and klen = ke - ks of them.
//   grid (ceil(T / 256)), block 256
extern "C" __global__ void glm_plan(const long long* __restrict__ cu, const int* __restrict__ seq_lens, int nseq,
                                    int* __restrict__ pos, unsigned* __restrict__ ks, unsigned* __restrict__ ke,
                                    int* __restrict__ klen, int* __restrict__ row_seq, int* __restrict__ zero,
                                    int T) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= T) return;
  int s = 0;
  unsigned goff = 0;
  while (s + 1 < nseq && cu[s + 1] <= r) goff += seq_lens[s++];
  const int p = seq_lens[s] - (int)(cu[s + 1] - r);
  pos[r] = p;
  ks[r] = goff;
  ke[r] = goff + p + 1;
  klen[r] = p + 1;
  row_seq[r] = s;
  zero[r] = 0;
}

// res = bf16(a + res) (a may be null: res is already the stream), normed = rms(res) * gamma, and
// normed quantized per 128 columns (q may be null).
//   grid (T), block 256: warp w owns column blocks 6w .. 6w + 5
extern "C" __global__ void __launch_bounds__(256) glm_add_rms_quant(
    const bf16* __restrict__ a, bf16* __restrict__ res, const bf16* __restrict__ gamma, bf16* __restrict__ normed,
    u8* __restrict__ q, u8* __restrict__ sf, int sf_rows, int T) {
  __shared__ float sm[8];
  const int r = blockIdx.x, w = threadIdx.x >> 5, l = threadIdx.x & 31;
  bf16* row = res + (size_t)r * H;
  float v[6][4], ss = 0.f;
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const int c = (w * 6 + i) * 128 + l * 4;
    ld4(row + c, v[i]);
    if (a) {
      float x[4];
      ld4(a + (size_t)r * H + c, x);
#pragma unroll
      for (int j = 0; j < 4; ++j) v[i][j] = rnd(v[i][j] + x[j]);
      st4(row + c, v[i]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) ss += v[i][j] * v[i][j];
  }
  const float rr = rsqrtf(block_sum(ss, sm) / H + EPS);
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const int kb = w * 6 + i, c = kb * 128 + l * 4;
    float g[4], y[4];
    ld4(gamma + c, g);
#pragma unroll
    for (int j = 0; j < 4; ++j) y[j] = rnd(rnd(v[i][j] * rr) * g[j]);
    if (normed) st4(normed + (size_t)r * H + c, y);
    if (q) quant4(y, q + (size_t)r * H + c, sf, sf_rows, r, kb, 1e-10f);
  }
}

// x [T, K] bf16 (rows ldx apart) quantized per 128 columns into q [T, K] and the packed scales.
//   grid (ceil(T * K / 1024)), block 256: a warp per (row, column block)
extern "C" __global__ void glm_quant(const bf16* __restrict__ x, int ldx, u8* __restrict__ q, u8* __restrict__ sf,
                                     int sf_rows, int K, int T) {
  const long long gw = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nb = K >> 7;
  if (gw >= (long long)T * nb) return;
  const int r = (int)(gw / nb), kb = (int)(gw % nb), c = kb * 128 + (threadIdx.x & 31) * 4;
  float v[4];
  ld4(x + (size_t)r * ldx + c, v);
  quant4(v, q + (size_t)r * K + c, sf, sf_rows, r, kb, 1e-10f);
}

// act = bf16(silu(g) * u) over gate | up halves of gu [rows, 2I], quantized per 128 columns into
// q [rows, I]. `n` (may be null) holds the live row count; rows past it are skipped.
//   grid (ceil(rows * I / 1024)), block 256
extern "C" __global__ void glm_silu_mul_quant(const bf16* __restrict__ gu, int I, u8* __restrict__ q,
                                              u8* __restrict__ sf, int sf_rows, const int* __restrict__ n,
                                              int rows) {
  const long long gw = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nb = I >> 7;
  if (gw >= (long long)rows * nb) return;
  const int r = (int)(gw / nb), kb = (int)(gw % nb), c = kb * 128 + (threadIdx.x & 31) * 4;
  if (n && r >= *n) return;
  float g[4], u[4], v[4];
  ld4(gu + (size_t)r * 2 * I + c, g);
  ld4(gu + (size_t)r * 2 * I + I + c, u);
#pragma unroll
  for (int j = 0; j < 4; ++j) v[j] = rnd(g[j] / (1.f + __expf(-g[j])) * u[j]);
  quant4(v, q + (size_t)r * I + c, sf, sf_rows, r, kb, 1e-10f);
}

// The projection [q_a 2048 | kv_a 512 | k_pe 64 | wk 128] of a row (bf16, lda apart):
//   q_c = bf16(rms(q_a) * g_q), quantized per 128 into qq / qsf (q_b's input);
//   kv[slot] = bf16(rms(kv_a) * g_kv) | rope(k_pe);
//   idx (full layers, else null): k = rope(layernorm(wk)), e4m3 with one f32 scale.
// rope [pos][64]: 32 cos then 32 sin, interleaved pairs (GLM's rope_interleave).
//   grid (T), block 256
extern "C" __global__ void __launch_bounds__(256) glm_mla_prep(
    const bf16* __restrict__ x, int lda, const bf16* __restrict__ g_q, const bf16* __restrict__ g_kv,
    const bf16* __restrict__ rope, const int* __restrict__ pos, const long long* __restrict__ slot,
    bf16* __restrict__ kv, u8* __restrict__ idx, const bf16* __restrict__ ik_w, const bf16* __restrict__ ik_b,
    u8* __restrict__ qq, u8* __restrict__ qsf, int sf_rows, int T) {
  __shared__ float sm[8];
  const int r = blockIdx.x, t = threadIdx.x, w = t >> 5, l = t & 31;
  const bf16* row = x + (size_t)r * lda;
  float qv[2][4], kv2[2], ssq = 0.f, ssk;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    ld4(row + (w * 2 + i) * 128 + l * 4, qv[i]);
#pragma unroll
    for (int j = 0; j < 4; ++j) ssq += qv[i][j] * qv[i][j];
  }
  kv2[0] = bf(row[QL + 2 * t]), kv2[1] = bf(row[QL + 2 * t + 1]);
  ssk = kv2[0] * kv2[0] + kv2[1] * kv2[1];
  const float rq = rsqrtf(block_sum(ssq, sm) / QL + EPS);
  const float rk = rsqrtf(block_sum(ssk, sm) / KVL + EPS);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int kb = w * 2 + i, c = kb * 128 + l * 4;
    float g[4], y[4];
    ld4(g_q + c, g);
#pragma unroll
    for (int j = 0; j < 4; ++j) y[j] = rnd(qv[i][j] * rq * g[j]);
    quant4(y, qq + (size_t)r * QL + c, qsf, sf_rows, r, kb, 1e-10f);
  }
  const long long s = slot[r];
  if (s < 0) return;
  const int p = pos[r];
  const bf16* cs = rope + (size_t)p * 2 * HALF;
  bf16* dst = kv + s * MLA;
  dst[2 * t] = __float2bfloat16(kv2[0] * rk * bf(g_kv[2 * t]));
  dst[2 * t + 1] = __float2bfloat16(kv2[1] * rk * bf(g_kv[2 * t + 1]));
  if (t < HALF) {
    const float x1 = bf(row[QL + KVL + 2 * t]), x2 = bf(row[QL + KVL + 2 * t + 1]);
    const float c = bf(cs[t]), sn = bf(cs[HALF + t]);
    dst[KVL + 2 * t] = __float2bfloat16(x1 * c - x2 * sn);
    dst[KVL + 2 * t + 1] = __float2bfloat16(x2 * c + x1 * sn);
  }
  if (idx == nullptr || w != 7) return;
  float k[4], kw[4], kb[4];
  ld4(row + QL + KVL + 2 * HALF + l * 4, k);
  ld4(ik_w + l * 4, kw);
  ld4(ik_b + l * 4, kb);
  const float mean = warp_sum(k[0] + k[1] + k[2] + k[3]) / ID;
  float d2 = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) d2 += (k[j] - mean) * (k[j] - mean);
  const float rstd = rsqrtf(warp_sum(d2) / ID + 1e-6f);
  float n[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) n[j] = (k[j] - mean) * rstd * kw[j] + kb[j];
  if (l < 16) {
#pragma unroll
    for (int j = 0; j < 4; j += 2) {
      const int pr = (l * 4 + j) >> 1;
      const float c = bf(cs[pr]), sn = bf(cs[HALF + pr]), e = n[j], o = n[j + 1];
      n[j] = e * c - o * sn;
      n[j + 1] = o * c + e * sn;
    }
  }
  const float amax = warp_max(fmaxf(fmaxf(fabsf(n[0]), fabsf(n[1])), fmaxf(fabsf(n[2]), fabsf(n[3]))));
  u8 e;
  const float sc = ue8m0(amax, 1e-4f, &e);
  u8* page = idx + (size_t)(s / PAGE) * IDX_PAGE;
  const int i = (int)(s % PAGE);
  *reinterpret_cast<unsigned*>(page + i * ID + l * 4) = fp8x4(n[0] / sc, n[1] / sc, n[2] / sc, n[3] / sc);
  if (l == 0) reinterpret_cast<float*>(page + PAGE * ID)[i] = sc;
}

// After q_b: the rope of q_pe into the tail of the absorbed query qmla [T, 64, 576], and on full
// layers the indexer query (rope on its first 64 dims, e4m3 per (row, head)) with its head weight
// w = f32(iw) * scale * 128^-0.5 * 32^-0.5. qb [T, ldb]: 64 heads of nope 192 | pe 64, then the
// indexer's 32 heads of 128 at column 16384.
//   grid (T), block 256
extern "C" __global__ void __launch_bounds__(256) glm_q_post(
    const bf16* __restrict__ qb, int ldb, const bf16* __restrict__ rope, const int* __restrict__ pos,
    bf16* __restrict__ qmla, u8* __restrict__ iq, const bf16* __restrict__ iw, float* __restrict__ wout, int full,
    int T) {
  const int r = blockIdx.x, t = threadIdx.x, w = t >> 5, l = t & 31;
  const bf16* row = qb + (size_t)r * ldb;
  const bf16* cs = rope + (size_t)pos[r] * 2 * HALF;
  const int h = t >> 2;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int pr = (t & 3) * 8 + j;
    const float x1 = bf(row[h * QHD + NOPE + 2 * pr]), x2 = bf(row[h * QHD + NOPE + 2 * pr + 1]);
    const float c = bf(cs[pr]), sn = bf(cs[HALF + pr]);
    bf16* d = qmla + ((size_t)r * HEADS + h) * MLA + KVL + 2 * pr;
    d[0] = __float2bfloat16(x1 * c - x2 * sn);
    d[1] = __float2bfloat16(x2 * c + x1 * sn);
  }
  if (!full) return;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int ih = w * 4 + k;
    float v[4];
    ld4(row + HEADS * QHD + ih * ID + l * 4, v);
    if (l < 16) {
#pragma unroll
      for (int j = 0; j < 4; j += 2) {
        const int pr = (l * 4 + j) >> 1;
        const float c = bf(cs[pr]), sn = bf(cs[HALF + pr]), e = v[j], o = v[j + 1];
        v[j] = e * c - o * sn;
        v[j + 1] = o * c + e * sn;
      }
    }
    const float amax = warp_max(fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
    u8 e;
    const float sc = ue8m0(amax, 1e-4f, &e);
    *reinterpret_cast<unsigned*>(iq + ((size_t)r * IH + ih) * ID + l * 4) =
        fp8x4(v[0] / sc, v[1] / sc, v[2] / sc, v[3] / sc);
    if (l == 0) {
      float x = bf(iw[(size_t)r * IH + ih]);
      x *= sc;
      x *= 0.08838834764831845f;
      x *= 0.17677669529663687f;
      wout[(size_t)r * IH + ih] = x;
    }
  }
}

// The indexer keys of every sequence, back to back: page j of sequence s (its rows
// [64 j, min(64 j + 64, len))) lands at stream rows goff(s) + 64 j, values in kg [., 128] and
// scales in ksf. A block a page; blocks past the last page return.
//   grid (ceil(ctx / 64) + seqs), block 256
extern "C" __global__ void glm_idx_gather(const u8* __restrict__ idx, const int* __restrict__ block_table,
                                          int max_pages, const int* __restrict__ seq_lens, int nseq,
                                          u8* __restrict__ kg, float* __restrict__ ksf) {
  int b = blockIdx.x, s = 0;
  unsigned goff = 0;
  for (; s < nseq; ++s) {
    const int pages = (seq_lens[s] + PAGE - 1) / PAGE;
    if (b < pages) break;
    b -= pages;
    goff += seq_lens[s];
  }
  if (s == nseq) return;
  const int n = min(PAGE, seq_lens[s] - b * PAGE);
  const u8* src = idx + (size_t)block_table[(size_t)s * max_pages + b] * IDX_PAGE;
  const size_t d = goff + (size_t)b * PAGE;
  const uint4* sv = reinterpret_cast<const uint4*>(src);
  uint4* dv = reinterpret_cast<uint4*>(kg + d * ID);
  for (int i = threadIdx.x; i < n * (ID / 16); i += blockDim.x) dv[i] = sv[i];
  if (threadIdx.x < n) ksf[d + threadIdx.x] = reinterpret_cast<const float*>(src + PAGE * ID)[threadIdx.x];
}

// The top-k's sequence-relative key indices (-1 past a short row's keys) as cache slots through the
// row's page table, in place, and the row's live count min(klen, 2048) for the attention.
//   grid (T), block 256
extern "C" __global__ void glm_topk_slots(int* __restrict__ ind, const int* __restrict__ block_table, int max_pages,
                                          const int* __restrict__ row_seq, const int* __restrict__ klen,
                                          int* __restrict__ tlen, int T) {
  const int r = blockIdx.x;
  const int* bt = block_table + (size_t)row_seq[r] * max_pages;
  int* row = ind + (size_t)r * TOPK;
  for (int i = threadIdx.x; i < TOPK; i += blockDim.x) {
    const int k = row[i];
    row[i] = k < 0 ? -1 : bt[k / PAGE] * PAGE + k % PAGE;
  }
  if (threadIdx.x == 0) tlen[r] = min(klen[r], TOPK);
}

// The expert offsets the grouped GEMM's psum layout reads: expert e's rows start at its 128-aligned
// CTA offset (FlashInfer's routing tables) and end count[e] rows later.
//   grid (1), block 256
extern "C" __global__ void glm_moe_psum(const int* __restrict__ counts, int* __restrict__ psum) {
  __shared__ int sm[256];
  const int e = threadIdx.x, c = counts[e];
  sm[e] = (c + 127) >> 7;
  __syncthreads();
  for (int o = 1; o < 256; o <<= 1) {
    const int v = e >= o ? sm[e - o] : 0;
    __syncthreads();
    sm[e] += v;
    __syncthreads();
  }
  psum[e] = (sm[e] - ((c + 127) >> 7)) * 128 + c;
}

// Expanded row (t, k) of the routed activation: token t's e4m3 row and its scale words at permuted
// row exp2perm[t * 8 + k].
//   grid (T * 8), block 256
extern "C" __global__ void glm_moe_gather(const u8* __restrict__ xq, const unsigned* __restrict__ xsf, int x_rows,
                                          const int* __restrict__ exp2perm, u8* __restrict__ aq,
                                          unsigned* __restrict__ asf, int a_rows) {
  const int e = blockIdx.x, t = e >> 3, p = exp2perm[e];
  if (p < 0) return;
  const uint4* s = reinterpret_cast<const uint4*>(xq + (size_t)t * H);
  uint4* d = reinterpret_cast<uint4*>(aq + (size_t)p * H);
  for (int i = threadIdx.x; i < H / 16; i += blockDim.x) d[i] = s[i];
  if (threadIdx.x < H / 512) asf[(size_t)threadIdx.x * a_rows + p] = xsf[(size_t)threadIdx.x * x_rows + t];
}

// out[t] = bf16(sum_k wts[t, k] * fc2[exp2perm[t * 8 + k]] + shared[t]), sums in f32.
//   grid (T), block 256: 24 columns a thread
extern "C" __global__ void __launch_bounds__(256) glm_moe_combine(const bf16* __restrict__ fc2,
                                                                  const int* __restrict__ exp2perm,
                                                                  const float* __restrict__ wts,
                                                                  const bf16* __restrict__ shared,
                                                                  bf16* __restrict__ out, int T) {
  __shared__ int sp[8];
  __shared__ float sw[8];
  const int t = blockIdx.x;
  if (threadIdx.x < 8) sp[threadIdx.x] = exp2perm[t * 8 + threadIdx.x], sw[threadIdx.x] = wts[t * 8 + threadIdx.x];
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const int c = (i * 256 + threadIdx.x) * 4;
    float acc[4], v[4];
    ld4(shared + (size_t)t * H + c, acc);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if (sp[k] < 0) continue;
      ld4(fc2 + (size_t)sp[k] * H + c, v);
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[j] += sw[k] * v[j];
    }
    st4(out + (size_t)t * H + c, acc);
  }
}
