"""The vendored kernels as manifest ops. Each launch's tensor maps and parameter blocks are laid out the
way the upstream host code builds them (DeepGEMM's `make_tma_*_desc`, FlashMLA's sparse prefill
`run()`, FlashInfer's `runPostTopKPipeline`, vLLM's `top_k_per_row_prefill`), so the manifest carries
the whole launch and no host code runs."""
import math

from pinned import module

T = "tokens"
SMS = 152  # every DeepGEMM template is built for 152 SMs (source/glm_deepgemm.cu)
EPILOGUE = {"pack": {"size": 24, "fields": [{"at": 20, "f32": 1.0}]}}  # deep_gemm::EpilogueArgs, alpha 1


def tm(at, param, dtype, dims, strides, box, swizzle=0, l2=256):
    return {"at": at, "tensormap": {"param": param, "dtype": dtype, "dims": dims, "strides": strides, "box": box,
                                    "swizzle": swizzle, "l2_promotion": l2}}


def desc(*fields):
    """One descriptor as a kernel's `__grid_constant__ CUtensorMap` argument."""
    return {"pack": {"size": 128, "fields": list(fields)}}


def launch(mod, entry, grid, block, smem=0, params=None, args=None, **extra):
    l = {**module(mod), "entry": entry, "grid": grid, "block": block, **extra}
    if smem:
        l["shared_mem"] = smem
    if params is not None:
        l["params"] = params
    if args is not None:
        l["args"] = args
    return l


def plain(mod, entry, params, grid, block, smem=0):
    return {"params": params, "impl": {"launches": [launch(mod, entry, grid, block, smem)]}}


def nulled(mod, entry, params, null, grid, block):
    """An entry whose pointer params at `null` are nullptr: the op drops them from its interface."""
    keep = [i for i in range(len(params)) if i not in null]
    pos = {i: k for k, i in enumerate(keep)}
    args = [{"i64": 0} if i in null else {"param": pos[i]} for i in range(len(params))]
    kparams = ["i64" if i in null else p for i, p in enumerate(params)]
    return {"params": [params[i] for i in keep],
            "impl": {"launches": [launch(mod, entry, grid, block, params=kparams, args=args)]}}


# ---- DeepGEMM sm100 fp8 1d1d: e4m3 x e4m3 -> bf16, UE8M0 scales packed four to an i32 (MN-major)
FP8_TILE = {128: (5, 202752), 256: (3, 185344)}  # BN -> (stages, shared memory)
FP8_ENTRY = ("_ZN9deep_gemm28sm100_fp8_fp4_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj128ELj128ELj128ELj0ELj0ELj0"
             "ELj128ELj{bn}ELj128ELj{groups}ELj128ELj128ELj128ELj{stages}ELj2ELj128ELj128ELj1ELb1ELj152ELb0ELb0E"
             "LNS_8GemmTypeE{gemm}ELb0EN7cutlass12float_e4m3_tES6_NS5_10bfloat16_tENS_8epilogue9transform"
             "16EpilogueIdentityEEEvPijjjT28_14CUtensorMap_stSD_SD_SD_SD_")


def fp8_gemm(n, k, rows, lda, ldc, sfa_rows, groups=1, bn=None):
    """C[m, :n] (rows ldc apart) = A[m, :k] (lda apart) . W[n, k]^T with both operands' block scales.
    Interface: a, w, sfa, sfb, c, then m (dense) or the psum offsets of `groups` experts (grouped:
    expert g's rows end at psum[g] and start at the previous end rounded up to 128, its weight rows
    at g * n, its scale words at g * k / 512). The tensor maps span `rows` rows of A and C."""
    bn = bn or (256 if n >= 4096 and n % 256 == 0 else 128)
    stages, smem = FP8_TILE[bn]
    grouped = groups > 1
    kw = k // 512
    entry = FP8_ENTRY.format(bn=bn, groups=groups, stages=stages, gemm=5 if grouped else 0)
    params = ["in buffer<u8>", "in buffer<fp8e4m3>", "in buffer<u8>", "in buffer<u8>", "out buffer<bf16>",
              "in buffer<i32>" if grouped else "i32"]
    maps = [desc(tm(0, 0, "u8", [k, rows], [lda], [128, 128], 128)),
            desc(tm(0, 1, "u8", [k, n * groups], [k], [128, bn], 128)),
            desc(tm(0, 2, "i32", [sfa_rows, kw], [sfa_rows * 4], [128, 1])),
            desc(tm(0, 3, "i32", [n, kw * groups], [n * 4], [bn, 1])),
            desc(tm(0, 4, "bf16", [n, rows], [ldc * 2], [64, 128], 128))]
    head = [{"param": 5}, {"i32": rows}] if grouped else [{"i64": 0}, {"param": 5}]
    kparams = ["in buffer<i32>" if grouped else "i64", "i32", "i32", "i32", "bytes<24>", *["bytes<128>"] * 5]
    return {"params": params, "impl": {"launches": [launch(
        "glm_deepgemm", entry, [SMS, 1, 1], [256, 1, 1], smem, params=kparams,
        args=[*head, {"i32": n}, {"i32": k}, EPILOGUE, *maps], pdl=True)]}}


# ---- DeepGEMM sm100 bf16, batched over the 64 heads: D[h][b][d] = A[h][b][:k] . B[h][d][:k]
BF16_TILE = {128: (6, 230400), 256: (4, 230400)}
BF16_ENTRY = ("_ZN9deep_gemm20sm100_bf16_gemm_implILN4cute4UMMA5MajorE0ELS3_0ELj0ELj0ELj0ELj128ELj{bn}ELj64ELj64"
              "ELj128ELj128ELj128ELj{stages}ELj128ELj128ELj1ELb1ELj152ELj128ELb0ELb0ELNS_8GemmTypeE4ELb0EN7cutlass"
              "10bfloat16_tENS_8epilogue9transform16EpilogueIdentityELm100EEEvPijjjT23_14CUtensorMap_stSC_SC_")


def bf16_batched(n, k, rows, a, b, d, bn=256):
    """Interface: a, b, d, m. `a`, `b`, `d` are each operand's byte strides (row, head): A [b][h][k],
    B [h][d][k], D [b][h][d] as views of their buffers."""
    stages, smem = BF16_TILE[bn]
    maps = [desc(tm(0, 0, "bf16", [k, rows, 64], list(a), [64, 128, 1], 128)),
            desc(tm(0, 1, "bf16", [k, n, 64], list(b), [64, bn, 1], 128)),
            desc(tm(0, 2, "bf16", [n, rows, 64], list(d), [64, 128, 1], 128))]
    return {"params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch(
                "glm_deepgemm", BF16_ENTRY.format(bn=bn, stages=stages), [SMS, 1, 1], [256, 1, 1], smem,
                params=["i64", "i32", "i32", "i32", "bytes<24>", *["bytes<128>"] * 3],
                args=[{"i64": 0}, {"param": 3}, {"i32": n}, {"i32": k}, EPILOGUE, *maps], pdl=True)]}}


# ---- the lightning indexer over one sub-chunk of R rows: DeepGEMM's fp8 MQA logits (compressed: row r's
# logits at [r][0, ke - ks)) then vLLM's top-k of each row
MQA_ENTRY = ("_ZN9deep_gemm16sm100_mqa_logitsILj32ELj128ELb0ELb1ELb0ELb0ELj4ELj256ELj3ELj5ELj152ELj128ELj256E"
             "N7cutlass12float_e4m3_tEffLj2EEEvjjjPKjS4_S4_PT13_14CUtensorMap_stS7_S7_S7_S7_")
TOPK = 2048


def indexer(j, r, kv_cap):
    """Interface: iq [., 32, 128] u8, kg [kv_cap, 128] u8, ksf [kv_cap] f32, w [., 32] f32, ks, ke u32,
    zero, klen i32, logits f32, ind i32 [., 2048] out, then num_q, num_kv, stride. Every row pointer is
    the call's, offset to the sub-chunk's first row; the logits are the sub-chunk's own."""
    when = {"var": T, "min": j * r + 1}
    sf = tm(0, 2, "f32", [kv_cap, 1], [kv_cap * 4], [256, 1])
    maps = [desc(tm(0, 0, "u8", [128, r * 32], [128], [128, 128], 128)), desc(sf),
            desc(tm(0, 1, "u8", [128, kv_cap], [128], [128, 256], 128)), desc(sf),
            desc(tm(0, 3, "f32", [32, r], [128], [32, 4], 0))]
    mqa = launch("glm_deepgemm", MQA_ENTRY, [SMS, 1, 1], [384, 1, 1], 220160,
                 params=["i32", "i32", "i32", "in buffer<u32>", "in buffer<u32>", "i64", "out buffer<f32>",
                         *["bytes<128>"] * 5],
                 args=[{"param": 10}, {"param": 11}, {"param": 12}, {"param": 4}, {"param": 5}, {"i64": 0},
                       {"param": 8}, *maps], when=when, pdl=True)
    topk = launch("vllm_topk", "glm_topk_prefill", [{"clamp": [T, j * r, r]}, 1, 1], [512, 1, 1], TOPK * 4,
                  params=["in buffer<f32>", "in buffer<i32>", "in buffer<i32>", "out buffer<i32>", "i32", "i32",
                          "i32", "i32"],
                  args=[{"param": 8}, {"param": 6}, {"param": 7}, {"param": 9}, {"param": 12}, {"i32": 1},
                        {"i32": TOPK}, {"i32": 0}], when=when)
    return {"params": ["in buffer<u8>", "in buffer<u8>", "in buffer<f32>", "in buffer<f32>", "in buffer<u32>",
                       "in buffer<u32>", "in buffer<i32>", "in buffer<i32>", "out buffer<f32>", "out buffer<i32>",
                       "i32", "i32", "i32"],
            "impl": {"launches": [mqa, topk]}}


# ---- FlashMLA sm100 sparse prefill, 64 heads, d_qk 576 (512 latent | 64 rope), d_v 512
SCALE = 256 ** -0.5  # qk_nope 192 + rope 64


def flash_mla(rows, entry, kv):
    """`entry` is read off the cubin: its mangled TmaParams type runs to 1.5 KB. Interface: q [., 64, 576], q again
    (its rope view, offset 1024 bytes), kv [row][576] bf16 (`kv` its param type: the cache state, or the lifted
    stream), ind [., 2048] rows of kv (-1 masked), tlen [.], o [., 64, 512] out, max_logits, lse [., 64] f32 out."""
    i32 = lambda at, v: {"at": at, "i32": v}
    f32 = lambda at, v: {"at": at, "f32": v}
    ptr = lambda at, p: {"at": at, "param": p}
    s_q = lambda at: {"at": at, "var": T}
    params = {"pack": {"size": 144, "fields": [
        s_q(0), i32(4, 0x7FFFFFFF), i32(8, 64), i32(12, 1), i32(16, 576), i32(20, 512), i32(24, TOPK),
        f32(28, SCALE), f32(32, SCALE * math.log2(math.e)), ptr(40, 0), ptr(48, 2), ptr(56, 3), ptr(72, 4),
        i32(80, 64 * 576), i32(84, 576), i32(88, 576), i32(92, 576), i32(96, TOPK), i32(100, TOPK),
        ptr(104, 5), ptr(112, 6), ptr(120, 7), i32(128, SMS)]}}
    tma = {"pack": {"size": 1280, "fields": [
        i32(0x0, 64), i32(0x4, 512), s_q(0x8),
        tm(0x80, 0, "bf16", [512, 64, rows], [1152, 64 * 1152], [64, 64, 1], 128, 128),
        i32(0x180, 64), i32(0x184, 64), s_q(0x188),
        tm(0x200, 1, "bf16", [64, 64, rows], [1152, 64 * 1152], [32, 64, 1], 64, 128),
        i32(0x300, 64), i32(0x304, 512), s_q(0x308),
        tm(0x380, 5, "bf16", [512, 64, rows], [1024, 64 * 1024], [64, 64, 1], 128, 128),
        tm(0x480, 2, "bf16", [512, 0], [1152], [64, 1], 128, 256)]}}
    return {"params": ["in buffer<bf16>", "in buffer<bf16>", kv, "in buffer<i32>", "in buffer<i32>",
                       "out buffer<bf16>", "out buffer<f32>", "out buffer<f32>"],
            "impl": {"launches": [launch("glm_flashmla", entry, [T, 1, 1], [384, 1, 1], 230704,
                                         params=["bytes<144>", "bytes<1280>"], args=[params, tma])]}}


# ---- FlashInfer's routing over precomputed top-k (kern-k3-pp-prefill's prebuilt cubin):
# routingPrecomputed::KernelParams<float, 1024, 16>, runtime top-k 8
ROUTING_OFF = {
    "mIsPow2": 1, "mPtrExpertCounts": 8, "mPtrPermutedIdxSize": 16, "mPtrExpandedIdxToPermutedIdx": 24,
    "mPtrPermutedIdxToTokenIdx": 40, "mPtrCtaIdxXyToBatchIdx": 48, "mPtrCtaIdxXyToMnLimit": 56,
    "mPtrNumNonExitingCtas": 64, "mPtrTopKIds": 80, "mNumTokens": 96, "mNumExperts": 100, "mPaddingLog2": 104,
    "mTileTokensDim": 108, "mLocalExpertsStartIdx": 112, "mNumLocalExperts": 120, "mTopK": 176,
}
ROUTING_T = "_ZN3moe3dev7routing{}INS1_18routingPrecomputed12KernelParamsIfLi1024ELi16EEEEEvT_"
ROUTING_THREADS = 1024


def fast_div(d):
    """trtllm's IntFastDiv(d) as its four i32 fields (d, M, s, n_add_sign), Hacker's Delight 10-1."""
    two31 = 1 << 31
    ad = abs(d)
    t = two31 + ((d & 0xFFFFFFFF) >> 31)
    anc = t - 1 - t % ad
    p, q1, r1, q2, r2 = 31, two31 // anc, two31 - (two31 // anc) * anc, two31 // ad, two31 - (two31 // ad) * ad
    while True:
        p += 1
        q1, r1 = 2 * q1, 2 * r1
        if r1 >= anc:
            q1, r1 = q1 + 1, r1 - anc
        q2, r2 = 2 * q2, 2 * r2
        if r2 >= ad:
            q2, r2 = q2 + 1, r2 - ad
        delta = ad - r2
        if not (q1 < delta or (q1 == delta and r1 == 0)):
            break
    m = (q2 + 1) & 0xFFFFFFFF
    m = m - (1 << 32) if m >= two31 else m
    return [d, m, p - 32, 1 if d > 0 and m < 0 else (-1 if d < 0 and m > 0 else 0)]


assert fast_div(16) == [16, -2147483647, 3, 1]


def counts_len(experts):
    return max(512, 2 * experts)


def routing(experts, topk, tile):
    """Interface: ids i32 [tokens, topk], counts [counts_len], cta_batch, cta_limit, num_non_exiting,
    total_padded, route_map, exp2perm [tokens, topk]. After it, counts[e] is expert e's row count and
    expanded row (t, k) sits at exp2perm[t * topk + k]: expert e's rows from 128 * its CTA offset."""
    P = {n: i for i, n in enumerate(["ids", "counts", "cta_batch", "cta_limit", "num_non_exiting", "total_padded",
                                     "route_map", "exp2perm"])}
    ptr = lambda f, p: {"at": ROUTING_OFF[f], "param": P[p]}
    i32 = lambda f, v: {"at": ROUTING_OFF[f], "i32": v}
    fields = [
        {"at": ROUTING_OFF["mIsPow2"], "u8": 1},
        ptr("mPtrExpertCounts", "counts"), ptr("mPtrPermutedIdxSize", "total_padded"),
        ptr("mPtrExpandedIdxToPermutedIdx", "exp2perm"), ptr("mPtrPermutedIdxToTokenIdx", "route_map"),
        ptr("mPtrCtaIdxXyToBatchIdx", "cta_batch"), ptr("mPtrCtaIdxXyToMnLimit", "cta_limit"),
        ptr("mPtrNumNonExitingCtas", "num_non_exiting"), ptr("mPtrTopKIds", "ids"),
        {"at": ROUTING_OFF["mNumTokens"], "var": T}, i32("mNumExperts", experts),
        i32("mPaddingLog2", tile.bit_length() - 1), i32("mTileTokensDim", tile), i32("mLocalExpertsStartIdx", 0),
        i32("mNumLocalExperts", experts),
        *({"at": ROUTING_OFF["mTopK"] + 4 * j, "i32": w} for j, w in enumerate(fast_div(topk))),
    ]
    pack = {"pack": {"size": 192, "fields": sorted(fields, key=lambda f: f["at"])}}
    expanded = {"ceil_div": [{"mul": [T, topk]}, 8 * ROUTING_THREADS]}
    l = lambda kind, grid: launch("flashinfer_moe_routing", ROUTING_T.format(kind), [grid, 1, 1],
                                  [ROUTING_THREADS, 1, 1], params=["bytes<192>"], args=[pack])
    return {"params": ["in buffer<i32>", *["out buffer<i32>"] * 7],
            "impl": {"launches": [l("23routingInitExpertCounts", -(-counts_len(experts) // ROUTING_THREADS)),
                                  l("29routingIndicesHistogramKernel", expanded),
                                  l("27routingIndicesOffsetsKernel", expanded)]}}
