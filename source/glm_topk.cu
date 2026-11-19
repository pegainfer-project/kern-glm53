// The indexer's top-k, k31_topk's set (source/k31/k31_topk.cu): each row's 2048 largest bf16 logits
// among its first `lens[r]`, ties by smaller position, ascending, the same bytes every run.
//
//   extern "C" __global__ void glm_topk(const bf16* logits, int stride, const int* lens, int* out,
//                                       const unsigned* ks, int* tlen, unsigned char* pick);
//   grid (rows), block NT, dynamic smem glm_topk_smem
//
// A row of at most HELD_MAX takes k31_topk_held's route. A longer row's sample (k31_topk's, read in
// 16-byte loads all at once) puts a threshold T, a bf16 key, where 1.125 x 2048 keys are expected at or
// above it (1.75 x in a row shorter than SHORT, whose sample misleads more often). One read against T
// keeps every position at or above it as a candidate (its position and key - T in a word, up to CAND)
// and counts every one of them in a 4096-bin histogram of key - T, the last bin taking the rest. The
// histogram gives the cut key K and how many of K's ties the 2048 take; with more candidates than CAND
// kept, one more read at T = K keeps them. A radix select over the ties' positions takes the first ones;
// the taken positions set bits of a bitmap of the row, which leave in order in equal shares a thread. A
// row past 2^20, fewer than 2048 candidates, a cut in the last bin, or more ties than CAP take k31_topk's
// exact route, so the set never depends on the sample.
#include "k31/k31_topk.cu"

#define CAND (2 * CAP)  // candidates kept, a word each, over Smem's ckey and cpos
#define SHORT 98304     // a long row shorter than this wants 1.75 x 2048 candidates of its sample

extern "C" __device__ const int glm_topk_smem = sizeof(Smem);

__device__ __forceinline__ unsigned key16(unsigned h) { return h ^ (h & 0x8000u ? 0xffffu : 0x8000u); }
__device__ __forceinline__ unsigned unkey16(unsigned k) { return k ^ (k & 0x8000u ? 0x8000u : 0xffffu); }

// Two bf16 of a word as keys, -0 as +0 and every NaN as 0x7fff (above +inf).
__device__ __forceinline__ unsigned keys2(unsigned w) {
  const __nv_bfloat162 x = __hadd2(*reinterpret_cast<const __nv_bfloat162*>(&w), __bfloat162bfloat162(0.0f));
  const unsigned b = *reinterpret_cast<const unsigned*>(&x);
  return key16(b & 0xffffu) | key16(b >> 16) << 16;
}

// The key where the sample (k31_topk's, chunks 8-aligned) expects `want` / 16 keys at or above it: the
// bottom of its 12-bit bin. A warp's chunk shares a weight, and its lanes' equal bins add once.
__device__ __forceinline__ unsigned sample_key(Smem& s, const bf16* row, unsigned n, unsigned want) {
  constexpr unsigned PER = SAMPLE_CHUNK / 8, ENDS = (HEAD + TAIL) / SAMPLE_CHUNK, STEP = NT / PER, B = 8;
  const Sample m = sample_of(n);
  const unsigned t = threadIdx.x;
  for (unsigned i = t; i < BINS; i += NT) s.u.hist[i] = 0;
  __syncthreads();
  const auto at = [&](unsigned c) {
    return c == 0     ? 0
           : c < ENDS ? ((n - TAIL) & ~7u) + (c - 1) * SAMPLE_CHUNK
                      : HEAD + (unsigned)((unsigned long long)(c - ENDS) * (n - HEAD - TAIL - SAMPLE_CHUNK) /
                                          (m.interior - 1)) & ~7u;
  };
  for (unsigned c0 = t / PER; c0 < m.chunks; c0 += B * STEP) {
    uint4 v[B];
#pragma unroll
    for (int u = 0; u < B; ++u) {
      const unsigned c = c0 + u * STEP;
      v[u] = c < m.chunks ? __ldg(reinterpret_cast<const uint4*>(row + at(c) + t % PER * 8)) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < B; ++u) {
      const unsigned c = c0 + u * STEP, w = c < ENDS ? 16u : m.w16;
      if (c >= m.chunks) break;
      const unsigned x[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const unsigned k = keys2(x[q]);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const unsigned bin = h ? k >> 20 : k >> 4 & 0xfff, peers = __match_any_sync(~0u, bin);
          if (__ffs(peers) - 1 == (t & 31)) atomicAdd(&s.u.hist[bin], __popc(peers) * w);
        }
      }
    }
  }
  __syncthreads();
  unsigned c[BPT], sum = 0;
#pragma unroll
  for (int k = 0; k < BPT; ++k) {
    const unsigned bin = BINS - 1 - BPT * t - k;
    c[k] = s.u.hist[bin], sum += c[k];
  }
  unsigned ab;
  Scan(s.scan).ExclusiveSum(sum, ab);
  if (t == 0) s.cut = 0;
  __syncthreads();
#pragma unroll
  for (int k = 0; k < BPT; ++k) {
    if (ab < want && ab + c[k] >= want) s.cut = BINS - 1 - BPT * t - k;
    ab += c[k];
  }
  __syncthreads();
  return s.cut << 4;
}

// The row's positions at or above key T: up to CAND as candidates (position << 12 | min(key - T, 4095)),
// all of them counted by min(key - T, 4095) in s.u.hist; their count in s.ncand.
#define KU 4  // 16-byte loads a thread a keep tile

__device__ __forceinline__ void keep(Smem& s, const bf16* row, unsigned n, unsigned T) {
  const int t = threadIdx.x;
  unsigned* cand = s.ckey;
  for (int i = t; i < BINS; i += NT) s.u.hist[i] = 0;
  if (t == 0) s.ncand = 0;
  __syncthreads();
  const __nv_bfloat162 thr = __bfloat162bfloat162(__ushort_as_bfloat16((unsigned short)unkey16(T)));
  constexpr unsigned TILE = NT * 8 * KU;
  const auto load = [&](unsigned base, uint4* v) {
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      const unsigned i = base + (u * NT + t) * 8;
      v[u] = i < n ? __ldg(reinterpret_cast<const uint4*>(row + i)) : make_uint4(0, 0, 0, 0);
    }
  };
  uint4 cur[KU], nxt[KU];
  load(0, cur);
  for (unsigned base = 0; base < n; base += TILE) {
    if (base + TILE < n) load(base + TILE, nxt);
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      const unsigned w[4] = {cur[u].x, cur[u].y, cur[u].z, cur[u].w}, i = base + (u * NT + t) * 8;
      unsigned below = ~0u;
#pragma unroll
      for (int q = 0; q < 4; ++q) below &= __hlt2_mask(*reinterpret_cast<const __nv_bfloat162*>(&w[q]), thr);
      if (below == ~0u || i >= n) continue;
      unsigned k[4], hit = 0;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        k[q] = keys2(w[q]);
        hit |= ((k[q] & 0xffffu) >= T) << 2 * q | (k[q] >> 16 >= T) << 2 * q + 1;
      }
      hit &= 0xffu >> 8 - min(n - i, 8u);
      if (!hit) continue;
      unsigned c = atomicAdd(&s.ncand, __popc(hit));
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (hit >> j & 1) {
          const unsigned d = min((j & 1 ? k[j / 2] >> 16 : k[j / 2] & 0xffffu) - T, BINS - 1u);
          atomicAdd(&s.u.hist[d], 1u);
          if (c < CAND) cand[c] = (i + j) << 12 | d;
          ++c;
        }
    }
#pragma unroll
    for (int u = 0; u < KU; ++u) cur[u] = nxt[u];
  }
  __syncthreads();
}

// The row's 2048 into o from the candidates at or above T; false (o untouched) for k31_topk's route.
__device__ __forceinline__ bool stream(Smem& s, const bf16* row, unsigned n, unsigned T, int* o) {
  const int t = threadIdx.x;
  const unsigned* cand = s.ckey;
  keep(s, row, n, T);
  if (s.ncand < TK) return false;
  cut_bin(s, TK);
  if (s.cut == BINS - 1) return false;
  if (s.ncand > CAND) {
    T += s.cut;
    __syncthreads();
    keep(s, row, n, T);
    if (s.ncand > CAND) return false;
    cut_bin(s, TK);
  }
  const unsigned m = s.ncand, D = s.cut, ties = s.count, need = TK - s.above;
  if (ties > CAP) return false;
  // the first `need` positions holding K: those whose ~position is at or above pthr
  unsigned pthr = 0;
  if (need < ties) {
    if (t == 0) s.ntie = 0;
    __syncthreads();
    for (unsigned i = t; i < m; i += NT)
      if ((cand[i] & 0xfff) == D) s.u.tie[atomicAdd(&s.ntie, 1u)] = ~(cand[i] >> 12);
    __syncthreads();
    if (ties <= NT) {
      // the tie with need - 1 ties before it by position
      if (t < ties) {
        const unsigned mine = s.u.tie[t];
        unsigned before = 0;
        for (unsigned j = 0; j < ties; ++j) before += s.u.tie[j] > mine;
        if (before == need - 1) s.cut = mine;
      }
      __syncthreads();
      pthr = s.cut;
    } else {
      unsigned pgt;
      select_smem(s, s.u.tie, ties, need, pthr, pgt, 32 - __clz(n - 1), ~0u);
    }
  }
  const auto taken = [&](unsigned c) { return (c & 0xfff) > D || ((c & 0xfff) == D && ~(c >> 12) >= pthr); };
  if (n > BITMAP_MAX) {
    if (t == 0) s.nsel = 0;
    __syncthreads();
    for (unsigned i = t; i < m; i += NT)
      if (taken(cand[i])) s.sel[atomicAdd(&s.nsel, 1u)] = cand[i] >> 12;
    __syncthreads();
    int k[TK / NT];
#pragma unroll
    for (int j = 0; j < TK / NT; ++j) k[j] = s.sel[t * (TK / NT) + j];
    __syncthreads();
    Sort(s.u.sort).Sort(k, 0, 32 - __clz(n - 1));
#pragma unroll
    for (int j = 0; j < TK / NT; ++j) o[t * (TK / NT) + j] = k[j];
    return true;
  }
  const unsigned words = (n + 31) / 32, per = (words + NT - 1) / NT, w0 = min(t * per, words),
                 w1 = min(w0 + per, words);
  unsigned* bits = s.u.bits;
  for (unsigned w = t; w < words; w += NT) bits[w] = 0;
  __syncthreads();
  for (unsigned i = t; i < m; i += NT)
    if (taken(cand[i])) atomicOr(&bits[cand[i] >> 17], 1u << (cand[i] >> 12 & 31));
  __syncthreads();
  // each word's owner marks the slot it starts at and each of its slots a multiple of 4 (k31's held)
  unsigned c = 0, q;
  for (unsigned w = w0; w < w1; ++w) c += __popc(bits[w]);
  Scan(s.scan).ExclusiveSum(c, q);
  unsigned* mark = reinterpret_cast<unsigned*>(s.sel);
  for (unsigned w = w0; w < w1; ++w) {
    const unsigned cw = __popc(bits[w]);
    if (cw) mark[q] = w << 5;
    for (unsigned k = (q + 3) & ~3u; k < q + cw; k += 4) mark[k] = w << 5 | (k - q);
    q += cw;
  }
  __syncthreads();
  unsigned k = 4 * t, mk = mark[k], wd = mk >> 5, x = bits[wd];
  x &= ~0u << nth_bit(x, mk & 31);
  int v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i, ++k) {
    if (!x) mk = mark[k], wd = mk >> 5, x = bits[wd];
    v[i] = wd * 32 + __ffs(x) - 1, x &= x - 1;
  }
  reinterpret_cast<int4*>(o)[t] = make_int4(v[0], v[1], v[2], v[3]);
  return true;
}

// k31_topk's exact route and ascending output.
__device__ __forceinline__ void exact_row(Smem& s, const bf16* row, unsigned n, int* o) {
  const int t = threadIdx.x;
  exact(s, row, n);
  if (n <= BITMAP_MAX) {
    const unsigned words = (n + 31) / 32, per = (words + NT - 1) / NT, w0 = min(t * per, words), w1 = min(w0 + per, words);
    for (unsigned w = t; w < words; w += NT) s.u.bits[w] = 0;
    __syncthreads();
    for (int i = t; i < TK; i += NT) atomicOr(&s.u.bits[s.sel[i] >> 5], 1u << (s.sel[i] & 31));
    __syncthreads();
    unsigned c = 0, at;
    for (unsigned w = w0; w < w1; ++w) c += __popc(s.u.bits[w]);
    Scan(s.scan).ExclusiveSum(c, at);
    for (unsigned w = w0; w < w1; ++w)
      for (unsigned b = s.u.bits[w]; b; b &= b - 1) o[at++] = (int)(w * 32 + __ffs(b) - 1);
    return;
  }
  int k[TK / NT];
#pragma unroll
  for (int j = 0; j < TK / NT; ++j) k[j] = s.sel[t * (TK / NT) + j];
  __syncthreads();
  Sort(s.u.sort).Sort(k, 0, 32 - __clz(n - 1));
#pragma unroll
  for (int j = 0; j < TK / NT; ++j) o[t * (TK / NT) + j] = k[j];
}

// The row's 2048 as positions relative to its first key into o, by the route its length takes.
__device__ __forceinline__ void row_topk(unsigned char* dyn, const bf16* row, unsigned n, int* o) {
  Smem& s = *reinterpret_cast<Smem*>(dyn);
  if (n <= TK) {
    for (int i = threadIdx.x; i < TK; i += NT) o[i] = i < (int)n ? i : -1;
    return;
  }
  if (n <= HELD_MAX) {
    Held& h = *reinterpret_cast<Held*>(dyn);
    if (n <= NT * 8) held<8>(h, row, n, o);
    else if (n <= NT * 16) held<16>(h, row, n, o);
    else if (n <= NT * 32) held<32>(h, row, n, o);
    else held<64>(h, row, n, o);
    return;
  }
  if (n <= 1u << 20 && stream(s, row, n, sample_key(s, row, n, n < SHORT ? 16 * TK * 7 / 4 : 16 * TK * 9 / 8), o)) return;
  __syncthreads();
  exact_row(s, row, n, o);
}

// Then, as rows of glm_kv_dequant's stream (glm_glue's glm_topk_stream): each position offset by the row's
// first key ks, each picked key flagged in `pick` (may be null), the row's live count min(lens, 2048) in tlen.
extern "C" __global__ void __launch_bounds__(NT, 2) glm_topk(const bf16* __restrict__ logits, int stride,
                                                            const int* __restrict__ lens, int* __restrict__ out,
                                                            const unsigned* __restrict__ ks, int* __restrict__ tlen,
                                                            unsigned char* __restrict__ pick) {
  asm volatile("griddepcontrol.launch_dependents;");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  extern __shared__ __align__(16) unsigned char dyn[];
  const int t = threadIdx.x, r = blockIdx.x;
  int* o = out + (size_t)r * TK;
  row_topk(dyn, logits + (size_t)r * stride, (unsigned)max(lens[r], 0), o);
  __syncthreads();
  const int base = (int)ks[r];
  int4 v = reinterpret_cast<int4*>(o)[t];
  int* w = &v.x;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    w[i] = w[i] < 0 ? -1 : base + w[i];
    if (pick && w[i] >= 0) pick[w[i]] = 1;
  }
  reinterpret_cast<int4*>(o)[t] = v;
  if (t == 0) tlen[r] = min(lens[r], TK);
}
