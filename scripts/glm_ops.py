"""The vendored kernels as manifest ops. Each launch's tensor maps and parameter blocks are laid out the
way the upstream host code builds them (DeepGEMM's `make_tma_*_desc`, FlashMLA's sparse prefill
`run()`, vLLM's `top_k_per_row_prefill`), so the manifest carries
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


def picked(mod, entry, params, flags, grid, block, rows):
    """An entry taking an optional flag buffer at `flags`: passed up to `rows` rows, nullptr past them."""
    null = [{"i64": 0} if i == flags else {"param": i} for i in range(len(params))]
    return {"params": params, "impl": {"launches": [
        launch(mod, entry, grid, block, when={"var": T, "max": rows}),
        launch(mod, entry, grid, block, params=[("i64" if i == flags else p) for i, p in enumerate(params)],
               args=null, when={"var": T, "min": rows + 1})]}}


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
FP8_TILE = {32: (10, 232448), 64: (7, 212992), 128: (5, 202752), 256: (3, 185344)}  # BN -> (stages, shared memory)
# BN -> (C/D swizzle, stages, shared memory) of the 2-CTA multicast tile (source/glm_fp8_tiles.cu)
FP8_MCB_TILE = {256: (128, 5, 204800), 224: (64, 6, 210944), 192: (128, 6, 215040), 176: (32, 7, 212992),
                160: (64, 7, 214016), 144: (32, 8, 226304), 96: (64, 9, 229376)}
# swap-AB, BN 128 over a 2-CTA cluster: BM -> (stages, shared memory) (source/glm_deepgemm.cu, glm_fp8_tiles.cu)
FP8_SWAP_TILE = {32: (11, 223232), 64: (10, 224256), 128: (8, 214016), 256: (6, 215040)}
FP8_ENTRY = ("_ZN9deep_gemm28sm100_fp8_fp4_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj128ELj128ELj128ELj0ELj0ELj0"
             "ELj{bm}ELj{bn}ELj128ELj{groups}ELj128ELj128ELj{swz}ELj{stages}ELj2ELj128ELj128ELj{mc}ELb{mca}ELj152ELb{swap}ELb0E"
             "LNS_8GemmTypeE{gemm}ELb{acc}EN7cutlass12float_e4m3_tES6_NS5_10bfloat16_tENS_8epilogue9transform"
             "16EpilogueIdentityEEEvPijjjT28_14CUtensorMap_stSD_SD_SD_SD_")


def fp8_gemm(n, k, rows, lda, ldc, sfa_rows, groups=1, bn=None, swap_bm=None, mcb=False, acc=False, when=None,
             tiled=False):
    """C[m, :n] (rows ldc apart) = A[m, :k] (lda apart) . W[n, k]^T with both operands' block scales.
    Interface: a, w, sfa, sfb, c, then m (dense) or the psum offsets of `groups` experts (grouped:
    expert g's rows end at psum[g] and start at the previous end rounded up to the block M, its weight
    rows at g * n, its scale words at g * k / 512). The tensor maps span `rows` rows of A and C.
    `swap_bm`: swap-AB with that block M (grouped: the psum alignment), BN 128, a 2-CTA cluster; `tiled` with it
    (grouped): W tile-major (glm_requant_tiled's 64-row x 128-byte tiles), source/dg/glm_moe_dn.cuh's copy.
    `mcb`: that BN on a 2-CTA cluster along M multicasting the weight tile (source/glm_fp8_tiles.cu); the
    pair needs an even count of 128-row blocks (DeepGEMM's multicast legality), so it runs m rounded up to
    256 rows, its extra rows inside the `rows` the maps span and read by no one.
    `acc` (swap_bm 64 or mcb): C += the product, the bf16 rounded product added by a TMA reduce-add."""
    grouped = groups > 1
    kw = k // 512
    if swap_bm:
        bm, bn, mc = swap_bm, 128, 2
        stages, smem = FP8_SWAP_TILE[bm]
        a_box, sfa_box, swz, cd_box = [128, bm // mc], [-(-bm // 128) * 128, 1], 128, [64, 16]
    elif mcb:
        bm, bn, mc = 128, mcb, 2
        swz, stages, smem = FP8_MCB_TILE[bn]
        a_box, sfa_box, cd_box = [128, 128], [128, 1], [swz // 2, 128]
    else:
        bm, mc = 128, 1
        bn = bn or (256 if n >= 4096 and n % 256 == 0 else 128)
        stages, smem = FP8_TILE[bn]
        swz = min(bn * 2, 128)
        a_box, sfa_box, cd_box = [128, 128], [128, 1], [swz // 2, 128]
    entry = FP8_ENTRY.format(bm=bm, bn=bn, groups=groups, stages=stages, mc=mc, mca=int(not mcb), swap=int(bool(swap_bm)), swz=swz, acc=int(acc),
                             gemm=5 if grouped else 0)
    assert not tiled or (swap_bm and grouped), "tile-major weights only on the grouped swap-AB copy"
    if tiled:
        entry = entry.replace("28sm100_fp8_fp4_gemm_1d1d_impl", "15glm_moe_dn_impl")
    params = ["in buffer<u8>", "in buffer<fp8e4m3>", "in buffer<u8>", "in buffer<u8>",
              "inout buffer<bf16>" if acc else "out buffer<bf16>",
              "in buffer<i32>" if grouped else "i32"]
    maps = [desc(tm(0, 0, "u8", [k, rows], [lda], a_box, 128)),
            desc(tm(0, 1, "u8", [128, 64, n * groups // 64 * (k // 128)], [128, 8192], [128, 64, 1], 128) if tiled else
                 tm(0, 1, "u8", [k, n * groups], [k], [128, bn // 2 if mcb else bn], 128)),
            desc(tm(0, 2, "i32", [sfa_rows, kw], [sfa_rows * 4], sfa_box)),
            desc(tm(0, 3, "i32", [n, kw * groups], [n * 4], [-(-bn // 128) * 128, 1])),
            desc(tm(0, 4, "bf16", [n, rows], [ldc * 2], cd_box, swz))]
    assert not mcb or rows % 256 == 0
    m = {"pack": {"size": 4, "fields": [{"at": 0, "expr": {"mul": [{"ceil_div": [T, 256]}, 256]}}]}} if mcb else \
        {"param": 5}
    head = [{"param": 5}, {"i32": rows}] if grouped else [{"i64": 0}, m]
    kparams = ["in buffer<i32>" if grouped else "i64", "bytes<4>" if mcb else "i32", "i32", "i32", "bytes<24>",
               *["bytes<128>"] * 5]
    extra = {"cluster": [mc, 1, 1]} if mc > 1 else {}
    if when:
        extra["when"] = when
    mod = "glm_fp8_tiles" if (swap_bm and not grouped) or mcb else "glm_deepgemm"
    return launch(mod, entry, [SMS, 1, 1], [256, 1, 1], smem, params=kparams,
                  args=[*head, {"i32": n}, {"i32": k}, EPILOGUE, *maps], pdl=True, **extra), params


GU_SILU_ENTRY = ("_ZN9deep_gemm20glm_moe_gu_silu_implILN4cute4UMMA5MajorE0ELS3_0ELj128ELj128ELj128ELj0ELj0ELj0ELj{bm}"
                 "ELj128ELj128ELj{groups}ELj128ELj128ELj128ELj{stages}ELj2ELj128ELj128ELj2ELb1ELj152ELb1ELb0E"
                 "LNS_8GemmTypeE5ELb0EN7cutlass12float_e4m3_tES6_NS5_10bfloat16_tENS_8epilogue9transform"
                 "16EpilogueIdentityEEEvPiPKiPKhPhSG_jjjT28_14CUtensorMap_stSI_SI_SI_SI_")
SILU_SMEM = 576  # sizeof(glm::SiluSmem)


def fp8_gu_silu(inter, k, x_rows, rows, groups, bm, when=None):
    """The routed gate-up GEMM (swap-AB at block M `bm`, BN 128, a 2-CTA cluster) over the tokens' rows, which a warp
    of its own gathers through the routing's row table, with act = bf16(silu(gate) * up) quantized per 128 columns in
    its epilogue (source/dg/glm_moe_gu_silu.cuh). Interface: x [x_rows, k] e4m3, w ([gate; up] of `inter` rows each, a group
    apart), sfa (the routed rows' scales), sfb, act [rows, inter] e4m3 out, its packed UE8M0 scales out, psum,
    the row table [rows]."""
    n, kw = 2 * inter, k // 512
    stages, smem = FP8_SWAP_TILE[bm]
    maps = [desc(tm(0, 0, "u8", [k, x_rows], [k], [128, 1], 128)),
            desc(tm(0, 1, "u8", [128, 64, n * groups // 64 * (k // 128)], [128, 8192], [128, 64, 1], 128)),
            desc(tm(0, 2, "i32", [rows, kw], [rows * 4], [-(-bm // 128) * 128, 1])),
            desc(tm(0, 3, "i32", [n, kw * groups], [n * 4], [64, 1])),
            desc(tm(0, 4, "u8", [inter, rows], [inter], [64, 16]))]
    l = launch("glm_deepgemm", GU_SILU_ENTRY.format(bm=bm, groups=groups, stages=stages), [SMS, 1, 1], [256, 1, 1],
               smem + SILU_SMEM,
               params=["in buffer<i32>", "in buffer<i32>", "in buffer<u8>", "out buffer<u8>", "out buffer<u8>", "i32", "i32",
                       "i32", "bytes<24>", *["bytes<128>"] * 5],
               args=[{"param": 6}, {"param": 7}, {"param": 0}, {"param": 4}, {"param": 5}, {"i32": rows}, {"i32": n},
                     {"i32": k}, EPILOGUE, *maps],
               pdl=True, cluster=[2, 1, 1], **({"when": when} if when else {}))
    params = ["in buffer<u8>", "in buffer<fp8e4m3>", "in buffer<u8>", "in buffer<u8>", "out buffer<u8>",
              "out buffer<u8>", "in buffer<i32>", "in buffer<i32>"]
    return l, params


def fp8_op(*a, **kw):
    l, params = fp8_gemm(*a, **kw)
    return {"params": params, "impl": {"launches": [l]}}


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


# Past MC_ROWS rows the v up-projection runs on a 2-CTA cluster along M that shares B (each CTA loads half of a
# head's 256 columns), over the rows rounded up to the cluster's 256: the pairs of M blocks must not straddle
# heads. The rows past the call's are computed into the buffers' spare rows and never read.
MC_ROWS, MC_ENTRY = 256, lambda e: e.replace("ELj4ELj128ELj128ELj1ELb1ELj152", "ELj6ELj128ELj128ELj2ELb0ELj152")


VUP_ENTRY = BF16_ENTRY.format(bn=256, stages=4).replace(
    "9transform16EpilogueIdentityELm100EEEvPijjjT23_14CUtensorMap_stSC_SC_", "9GlmQuantOELm100EEEvPijjjT23_14CUtensorMap_stSB_SB_")


def vup_quant(n, k, rows, a, b):
    """bf16_batched's v up-projection, D landing as e4m3 [b][h][n] with o_proj's packed UE8M0 scales of each
    128 columns ([h * n / 512][rows] words): a 1-CTA launch up to MC_ROWS rows, then the 2-CTA one.
    Interface: a, b, q, sf, m, m rounded up to 256."""
    maps = [[desc(tm(0, 0, "bf16", [k, rows, 64], list(a), [64, 128, 1], 128)),
             desc(tm(0, 1, "bf16", [k, n, 64], list(b), [64, box, 1], 128)),
             desc(tm(0, 2, "u8", [n, rows, 64], [64 * n, n], [128, 128, 1], 128))] for box in (256, 128)]
    epi = {"pack": {"size": 24, "fields": [{"at": 0, "param": 3}, {"at": 8, "i32": rows}, {"at": 12, "var": T},
                                           {"at": 16, "i32": n}, {"at": 20, "f32": 1.0}]}}
    params = ["i64", "i32", "i32", "i32", "bytes<24>", *["bytes<128>"] * 3]
    one = lambda entry, m, ms, **extra: launch("glm_vup", entry, [SMS, 1, 1], [256, 1, 1], 230400, params=params,
                                               args=[{"i64": 0}, {"param": m}, {"i32": n}, {"i32": k}, epi, *ms],
                                               pdl=True, **extra)
    return {"params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<u8>", "out buffer<u8>", "i32", "i32"],
            "impl": {"launches": [one(VUP_ENTRY, 4, maps[0], when={"var": T, "max": MC_ROWS}),
                                  one(MC_ENTRY(VUP_ENTRY), 5, maps[1], cluster=[2, 1, 1],
                                      when={"var": T, "min": MC_ROWS + 1})]}}


# ---- the lightning indexer over one sub-chunk of R rows: DeepGEMM's metadata kernel deals the (4-row q block, 384-key
# split) work evenly over the SMs, its fp8 MQA logits follow it on three math warpgroups, a split's keys in 128-key TMA
# tiles (bf16, compressed: row r's logits at [r][0, ke - ks)),
# then each row's top-k (glm_topk: vLLM's set, ascending, the same bytes every run; a row of at most 32k read once, a
# longer one's sample and then the row read once against a threshold)
MQA_ENTRY = ("_ZN9deep_gemm16sm100_mqa_logitsILj32ELj128ELb0ELb1ELb0ELb1ELj4ELj384ELj3ELj3ELj152ELj128ELj384E"
             "N7cutlass12float_e4m3_tE13__nv_bfloat16S3_Lj3EEEvjjjPKjS5_S5_PT13_14CUtensorMap_stS8_S8_S8_S8_")
META_ENTRY = "_ZN9deep_gemm5sched33sm100_mqa_logits_metadata_batchedILj4ELj384ELj152ELj1ELj1ELj15EEEvjjjPKjS3_Pjj"
TOPK = 2048
TOPK_SMEM = 93408  # sizeof(Smem) of source/k31/k31_topk.cu
BLOCK_Q = 4


def meta_words(r):
    """The schedule's i32 words: per-SM starts and counts, then a (base, splits) span per q block."""
    return (3 * SMS + 1) // 2 * 2 + 2 * -(-r // BLOCK_Q)


def indexer(j, r, kv_cap, pick_rows):
    """Interface: iq [., 32, 128] u8, kg [kv_cap, 128] u8, ksf [kv_cap] f32, w [., 32] bf16, ks, ke u32,
    zero, klen i32, logits bf16, ind i32 [., 2048] out, then num_q, num_kv, stride, the schedule
    [meta_words(r)] i32 (sub-chunk 0 writes the call's every sub-chunk's, one after another), then base u32 (the
    stream offset the picks take on), tlen i32 out, the pick flags u8 [ctx] (set by calls of at most pick_rows
    rows) and the call's rows. Every row pointer is the call's, offset to the sub-chunk's first row; the logits
    are the sub-chunk's own, the schedule its slice."""
    when = {"var": T, "min": j * r + 1}
    sf = tm(0, 2, "f32", [kv_cap, 1], [kv_cap * 4], [128, 1])
    maps = [desc(tm(0, 0, "u8", [128, r * 32], [128], [128, 128], 128)), desc(sf),
            desc(tm(0, 1, "u8", [128, kv_cap], [128], [128, 128], 128)), desc(sf),
            desc(tm(0, 3, "bf16", [32, r], [64], [32, 4], 0))]
    # sub-chunk 0 schedules the call's every sub-chunk, a block each, into its own slice of the schedule
    meta = [launch("glm_deepgemm", META_ENTRY, [{"ceil_div": [T, r]}, 1, 1], [256, 1, 1], 8 * -(-r // BLOCK_Q),
                   params=["i32", "i32", "i32", "in buffer<u32>", "in buffer<u32>", "out buffer<i32>", "i32"],
                   args=[{"param": 17}, {"i32": r}, {"param": 11}, {"param": 4}, {"param": 5}, {"param": 13},
                         {"i32": meta_words(r)}], when=when, pdl=True)] if j == 0 else []
    mqa = launch("glm_deepgemm", MQA_ENTRY, [SMS, 1, 1], [512, 1, 1], 202752,
                 params=["i32", "i32", "i32", "in buffer<u32>", "in buffer<u32>", "in buffer<i32>", "out buffer<bf16>",
                         *["bytes<128>"] * 5],
                 args=[{"param": 10}, {"param": 11}, {"param": 12}, {"param": 4}, {"param": 5}, {"param": 13},
                       {"param": 8}, *maps], when=when, pdl=True)
    topk_params = ["in buffer<bf16>", "i32", "in buffer<i32>", "out buffer<i32>", "in buffer<u32>", "out buffer<i32>",
                   "inout buffer<u8>"]
    topk_args = [{"param": 8}, {"param": 12}, {"param": 7}, {"param": 9}, {"param": 14}, {"param": 15}, {"param": 16}]
    topk = lambda lo, hi, picks: launch(
        "glm_topk", "glm_topk", [{"clamp": [T, j * r, r]}, 1, 1], [512, 1, 1], TOPK_SMEM,
        params=topk_params if picks else [*topk_params[:6], "i64"], args=topk_args if picks else [*topk_args[:6], {"i64": 0}],
        when={"var": T, "min": lo, **({"max": hi} if hi else {})}, pdl=True)
    topks = ([topk(j * r + 1, pick_rows, True)] if j * r < pick_rows else []) + [topk(max(j * r, pick_rows) + 1, 0, False)]
    return {"params": ["in buffer<u8>", "in buffer<u8>", "in buffer<f32>", "in buffer<bf16>", "in buffer<u32>",
                       "in buffer<u32>", "in buffer<i32>", "in buffer<i32>", "out buffer<bf16>", "out buffer<i32>",
                       "i32", "i32", "i32", "out buffer<i32>" if j == 0 else "in buffer<i32>", "in buffer<u32>",
                       "out buffer<i32>", "inout buffer<u8>", "i32"],
            "impl": {"launches": [*meta, mqa, *topks]}}


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
            "impl": {"launches": [launch("glm_flashmla2", entry, [T, 1, 1], [512, 1, 1], 231728,
                                         params=["bytes<144>", "bytes<1280>"], args=[params, tma])]}}


# ---- the routing tables (source/glm_moe.cu): a tally of each slice of ROUTE_SLICE tokens, then the experts'
# offsets and every routed row's place
ROUTE_SLICE = 256
SCATTER_ROWS = 4096  # up to it glm_moe_scatter writes the row table and scales; past it glm_moe_place does


def moe_route(experts, tiles, x_rows, a_rows, sf_words):
    """Interface: ids i32 [tokens, 8], hist i32 [slices, experts] (scratch), psum i32 [experts + 1] out, exp2perm
    i32 [tokens, 8] out, tokens, rows i32 [a_rows] out, xsf (the tokens' packed scales, `x_rows` a word row) in,
    asf (the grouped rows', `a_rows` a word row) out. Group 0, the shared expert, is rows [0, tokens) (psum[0]),
    routed expert e's rows start at its `tile`-aligned offset past them and end at psum[e + 1]; routed row (t, k)
    sits at p = exp2perm[t * 8 + k], an expert's rows in token order, rows[p] = t and its scale words at p.
    `tiles`: [(when, tile)]."""
    grid = [{"ceil_div": [T, ROUTE_SLICE]}, 1, 1]
    count = launch("glm_moe", "glm_moe_count", grid, [ROUTE_SLICE, 1, 1],
                   params=["in buffer<i32>", "i32", "out buffer<i32>"], args=[{"param": 0}, {"param": 4}, {"param": 1}])
    writes = [bool(w) and w.get("min", 1) > SCATTER_ROWS for w, _ in tiles]
    tables = lambda own: [{"param": 5}, {"param": 6}, {"i32": x_rows}, {"param": 7}, {"i32": a_rows}] if own else \
        [{"i64": 0}, {"i64": 0}, {"i32": x_rows}, {"i64": 0}, {"i32": a_rows}]
    place = [launch("glm_moe", "glm_moe_place", grid, [ROUTE_SLICE, 1, 1],
                    params=["in buffer<i32>", "i32", "in buffer<i32>", "i32", "out buffer<i32>", "out buffer<i32>",
                            *(["out buffer<i32>", "in buffer<u8>", "i32", "out buffer<u8>", "i32"] if own else
                              ["i64", "i64", "i32", "i64", "i32"])],
                    args=[{"param": 0}, {"param": 4}, {"param": 1}, {"i32": tile}, {"param": 2}, {"param": 3},
                          *tables(own)],
                    **({"when": w} if len(tiles) > 1 else {})) for (w, tile), own in zip(tiles, writes)]
    scatter = launch("glm_moe", "glm_moe_scatter", [{"ceil_div": [T, 256]}, sf_words + 1, 1], [256, 1, 1],
                     params=["in buffer<i32>", "i32", "out buffer<i32>", "in buffer<u8>", "i32", "out buffer<u8>", "i32"],
                     args=[{"param": 3}, {"param": 4}, {"param": 5}, {"param": 6}, {"i32": x_rows}, {"param": 7},
                           {"i32": a_rows}],
                     **({"when": {"var": T, "min": 1, "max": SCATTER_ROWS}} if any(writes) else {}))
    return {"params": ["in buffer<i32>", "out buffer<i32>", "out buffer<i32>", "out buffer<i32>", "i32",
                       "out buffer<i32>", "in buffer<u8>", "out buffer<u8>"],
            "impl": {"launches": [count, *place, scatter]}}
