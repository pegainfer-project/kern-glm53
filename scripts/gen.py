#!/usr/bin/env python3
"""GLM-5.3's packed prefill as one kern manifest, for `kern cut` into pipeline stages.

    KERN=<kern checkout> python3 scripts/gen.py > glm53.json

Program `prefill` runs up to --pack sequences' chunks back to back (`cu_seqlens`, at most --chunk
rows) through layers [0, --layers), then the head on each sequence's last row: `next_token` is one
token a sequence. Every layer keeps its own states (`kv.<l>`, and `idx.<l>` on an indexer layer),
and every stage starts with its own row plan (`L<l>.plan` at each --cuts layer), so a cut before
those labels moves only the residual stream (`h`, `res`).

A layer, as vLLM runs GLM-5 (DeepSeek V3.2's sparse attention) on Blackwell with DeepGEMM:
  - fused_add_rms_norm, the row quantized to e4m3 with UE8M0 scales per 128 columns;
  - [q_a | kv_a | k_pe | wk] in one fp8 GEMM; q_a / kv_a norms, k_pe rope into the latent cache
    (kv_norm | k_pe), the indexer key (LayerNorm, rope, e4m3) into its page-blocked cache;
  - [q_b | wq_b] in one fp8 GEMM, W_UK absorbed per head (bf16 batched GEMM), q_pe rope;
  - on an indexer layer (`full`), the indexer's fp8 MQA logits over every key of the row's
    sequence and the top 2048 of them; a `shared` layer reuses the last indexer layer's picks;
  - FlashMLA's sparse prefill over the picked latent rows, W_UV per head landing o_proj's e4m3 input, o_proj (fp8). With --kv fp8
    (the default) the latent cache is SGLang's fp8 DSA row (656 bytes: 512 e4m3 in tiles of 128 with
    f32 scales, 64 bf16 rope), lifted to bf16 sequence after sequence before the attention as SGLang
    does for a prefix (dequantize_k_cache_paged); --kv bf16 keeps [576] bf16 rows and attends in place;
  - the dense MLP (layers 0-2) or the MoE: sigmoid router with the score bias, top-8 of 256
    renormalized and scaled by 2.5, the routing tables (an expert's rows in token order), DeepGEMM's grouped fp8 GEMMs over
    each expert's rows only, the shared expert, one f32 sum.
Two branches run on lane 1 beside lane 0 (kern's program lanes): on a layer sharing the picks, the
latent cache's lift (kv_dequant) beside its query chain from q_b to q_post, joined by the attention; up
to 2048 rows, the shared expert beside the router and the routed experts, joined by the combine.
Weights are the checkpoint's e4m3 tensors re-quantized to UE8M0 block scales while they load (the
`derive` program, source/glm_prep.cu), the way vLLM prepares 128x128-block checkpoints for DeepGEMM
on Blackwell.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(1, str(pathlib.Path(os.environ["KERN"]) / "tools"))
import glm_ops as G  # noqa: E402
import kern_manifest  # noqa: E402
import once as once_mod  # noqa: E402
from once import Once, buf, seg  # noqa: E402
from pinned import module  # noqa: E402

# `once`'s modules resolve as the step's own do: this repository's pin first (pinned.py), then the index.
once_mod.variant = lambda family, **defines: types.SimpleNamespace(module=module(family, **defines))
once_mod.KERNELS["fill_bf16"] = ("glm_prep", "glm_fill_bf16", ["out buffer<bf16>", "i32", "f32"])

H, V = 6144, 154880
QL, KVL, ROPE, NOPE, VD, HEADS = 2048, 512, 64, 192, 256, 64
MLA = KVL + ROPE                       # 576, a latent cache row
KV_ROW = {"fp8": KVL + 16 + ROPE * 2, "bf16": MLA * 2}
IH, ID = 32, 128                       # indexer heads x dim
QB = HEADS * (NOPE + ROPE)             # 16384
IQB = IH * ID                          # 4096
EXPERTS, TOPK, INTER, DENSE_I = 256, 8, 2048, 12288
LAYERS, DENSE = 78, 3
FULL = {0, 1, 2} | set(range(6, LAYERS, 4))  # config.json indexer_types == "full"
PAGE = 64
PICK_ROWS = 2048                        # kv_dequant lifts only the picked rows of calls up to this many
IDX_ROW = ID + 4                       # e4m3 key then its f32 scale, page-blocked
NA = {True: QL + MLA + ID, False: QL + MLA}   # [q_a | kv_a | k_pe | wk]
NB = {True: QB + IQB, False: QB}
T, S, CTX = "tokens", "seqs", "ctx"
HF = "model."

i32 = lambda v: {"i32": v}
i64 = lambda v: {"i64": v}
var = lambda v: {"var": v}
expr = lambda e: {"expr": e}
st = lambda name, off=0: {"state": name, "offset": off} if off else {"state": name}
sf_bytes = lambda k, rows: k // 512 * rows * 4  # DeepGEMM's packed UE8M0 words, [k / 512][rows]


def mla_entry():
    cubin = pathlib.Path(__file__).resolve().parents[1] / "build" / "glm_flashmla2.cubin"
    out = subprocess.run(["cuobjdump", "-elf", str(cubin)], capture_output=True, text=True, check=True).stdout
    names = [l.split()[0][len(".text."):] for l in out.splitlines() if l.startswith(".text.")]
    (entry,) = [n for n in names if "sparse_attn_fwd_kernel" in n]
    return entry


# the shared expert's GEMMs by row count: BN 32 / 64 for few rows put 64-128 CTAs on the weight
SHARED_TILES = {"shared.gu": "256:n32,1024:n64,16384:n", "shared.dn": "256:n64,16384:n"}
LANE_ROWS = 2048  # the shared expert beside the routed path up to this many rows


def within(op, lo, hi):
    """`op` with its launches cut to row counts [lo, hi] (none outside it)."""
    ls = []
    for l in op["impl"]["launches"]:
        w = l.get("when", {})
        a, b = max(w.get("min", 1), lo), min(w.get("max", hi), hi)
        if a <= b:
            ls.append({**l, "when": {"var": T, "min": a, "max": b}})
    return {**op, "impl": {**op["impl"], "launches": ls}}

# the routed GEMMs' layout by row count (DeepGEMM's m-grouped choice: swap-AB, BN 128, a 2-CTA cluster)
MOE_TILES = "16384:s128"
# the dense GEMMs' layouts by row count, as moe_layouts reads them (s: swap-AB with that block M on a 2-CTA
# cluster, n: that BN): at few rows each puts more CTAs on its weight (fp8.o at 128 rows: 96 tiles of 128 weight
# rows x 64 tokens, not 24 of 128 x 256)
TILES = {**SHARED_TILES, "o": "512:s64,16384:n", "dense.dn": "512:s64,16384:n", "a.full": "256:n32,16384:n",
         "a.shared": "256:n32,16384:n", "b.shared": "256:n128,16384:n"}


def moe_layouts(spec, chunk):
    """`--moe-tiles` as [(when, block M or None)]: "512:s32,16384:n" runs swap-AB block M 32 up to 512 rows, then
    the plain BN-by-128 tile (block M 128) up to 16384."""
    out, lo = [], 1
    for part in spec.split(","):
        hi, cfg = part.split(":")
        hi = min(int(hi), chunk)
        out.append(({"var": T, "min": lo, "max": hi}, int(cfg[1:]) if cfg[1:] else None))
        lo = hi + 1
    assert lo > chunk, "the MoE layouts cover every row count"
    return out


def build(layers, chunk, pack, max_ctx, sub, cuts, kv_dtype="fp8", first=0, moe_tiles=MOE_TILES):
    assert chunk % sub == 0 and chunk % 128 == 0 and max_ctx % 256 == 0
    max_pages = max_ctx // PAGE
    layouts = moe_layouts(moe_tiles, chunk)
    tiles = [bm or 128 for _, bm in layouts]
    mcap = TOPK * chunk + EXPERTS * max(tiles)   # routed rows, each expert's padded to its block M
    kv0 = f"kv.{first}"
    head = first == 0 or layers == LAYERS
    fp8kv = kv_dtype == "fp8"

    ops = {
        "embed": G.plain("embedding", "kern_embedding_i64_bf16",
                         ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"], [T, 1, 1], [256, 1, 1]),
        "plan": G.plain("glm_glue", "glm_plan",
                        ["in buffer<i64>", "in buffer<i32>", "i32", "out buffer<i32>", "out buffer<u32>",
                         "out buffer<u32>", "out buffer<i32>", "out buffer<i32>", "out buffer<i32>", "i32"],
                        [{"ceil_div": [T, 256]}, 1, 1], [256, 1, 1]),
        "gemm_bf16": {"params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32", "i32"],
                      "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]}},
        "gemm_f32": {"params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32", "i32"],
                     "impl": {"launches": [{"entry": "extern:cublas_bf16_tn_f32"}]}},
        "silu_dense": G.nulled("glm_glue", "glm_silu_mul_quant",
                               ["in buffer<bf16>", "i32", "out buffer<u8>", "out buffer<u8>", "i32", "in buffer<i32>",
                                "i32"], {5}, [{"mul": [T, DENSE_I // 1024]}, 1, 1], [256, 1, 1]),
        "silu_shared": G.nulled("glm_glue", "glm_silu_rows", ["in buffer<bf16>", "in buffer<i32>", "out buffer<u8>",
                                                             "out buffer<u8>", "i32"], {1}, [T, 1, 1], [256, 1, 1]),
        "silu_moe": G.plain("glm_glue", "glm_silu_rows",
                            ["in buffer<bf16>", "in buffer<i32>", "out buffer<u8>", "out buffer<u8>", "i32"],
                            [{"mul": [T, TOPK]}, 1, 1], [256, 1, 1]),
        "q_post_full": G.plain("glm_glue", "glm_q_post",
                               ["in buffer<bf16>", "i32", "in buffer<bf16>", "in buffer<i32>", "inout buffer<bf16>",
                                "out buffer<u8>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"], [T, 1, 1], [256, 1, 1]),
        "q_post_shared": G.nulled("glm_glue", "glm_q_post",
                                  ["in buffer<bf16>", "i32", "in buffer<bf16>", "in buffer<i32>", "inout buffer<bf16>",
                                   "out buffer<u8>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"], {5, 6, 7},
                                  [T, 1, 1], [256, 1, 1]),
        "idx_gather": G.plain("glm_glue", "glm_idx_gather",
                              ["in state", "in buffer<i32>", "i32", "in buffer<i32>", "i32", "out buffer<u8>",
                               "out buffer<f32>", "out buffer<u8>"], [{"add": [{"ceil_div": [CTX, PAGE]}, S]}, 1, 1],
                              [256, 1, 1]),
        "topk_slots": G.plain("glm_glue", "glm_topk_slots",
                              ["inout buffer<i32>", "in buffer<i32>", "i32", "in buffer<i32>", "in buffer<i32>",
                               "out buffer<i32>", "i32"], [T, 1, 1], [256, 1, 1]),
        # the pick flags pay off while a chunk picks a small part of its context: up to PICK_ROWS rows
        "topk_stream": G.picked("glm_glue", "glm_topk_stream",
                                ["inout buffer<i32>", "in buffer<u32>", "in buffer<i32>", "out buffer<i32>",
                                 "inout buffer<u8>", "i32"], 4, [T, 1, 1], [256, 1, 1], PICK_ROWS),
        "kv_dequant": G.picked("glm_glue", "glm_kv_dequant",
                               ["in state", "in buffer<i32>", "i32", "in buffer<i32>", "i32", "in buffer<u8>",
                                "out buffer<bf16>"], 5, [{"add": [{"ceil_div": [CTX, PAGE]}, S]}, 1, 1], [256, 1, 1],
                               PICK_ROWS),
        "uk": G.bf16_batched(KVL, NOPE, chunk, (NB[True] * 2, (NOPE + ROPE) * 2), (NOPE * 2, KVL * NOPE * 2),
                             (HEADS * MLA * 2, MLA * 2)),
        "uv": G.vup_quant(VD, KVL, chunk, (HEADS * KVL * 2, KVL * 2), (KVL * 2, (NOPE + VD) * KVL * 2)),
        "flash_mla": G.flash_mla(chunk, mla_entry(), "in buffer<bf16>" if fp8kv else "in state"),
        "router_topk": {"params": ["in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "out buffer<i32>",
                                   "out buffer<f32>", "i32"],
                        "impl": {"launches": [{**module("k3_router_argmax", EXPERTS=EXPERTS, TOPK=TOPK),
                                               "entry": "kern_k3_router_topk", "grid": [T, 1, 1],
                                               "block": [256, 1, 1]}]}},
        "moe_route": G.moe_route(EXPERTS, [(w, bm or 128) for w, bm in layouts]),
        "moe_gather": G.plain("glm_glue", "glm_moe_gather",
                              ["in buffer<u8>", "in buffer<u8>", "i32", "in buffer<i32>", "out buffer<u8>",
                               "out buffer<u8>", "i32"], [T, 1, 1], [384, 1, 1]),
        "moe_combine": G.plain("glm_glue", "glm_moe_combine",
                               ["in buffer<bf16>", "in buffer<i32>", "in buffer<f32>", "in buffer<bf16>",
                                "out buffer<bf16>", "i32"], [T, 1, 1], [256, 1, 1]),
        "last_rows": {"params": ["out buffer<bf16>", "in buffer<bf16>", "in buffer<i64>", "i32", "i32"],
                      "impl": {"launches": [G.launch("copy_rows", "kern_last_rows_bf16", [S, 1, 1], [1024, 1, 1])]}},
        "head_norm": G.nulled("glm_glue", "glm_add_rms_quant",
                              ["in buffer<bf16>", "inout buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>",
                               "out buffer<u8>", "out buffer<u8>", "i32", "i32"], {4, 5}, [S, 1, 1], [256, 1, 1]),
        "argmax_partial": {"params": ["in buffer<f32>", "out buffer<f32>", "out buffer<i32>", "i32"],
                           "impl": {"launches": [{**module("k3_router_argmax", EXPERTS=EXPERTS, TOPK=TOPK),
                                                  "entry": "kern_k3_argmax_f32_partial", "grid": [S, 64, 1],
                                                  "block": [1024, 1, 1]}]}},
        "argmax_final": {"params": ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                         "impl": {"launches": [{**module("k3_router_argmax", EXPERTS=EXPERTS, TOPK=TOPK),
                                                "entry": "kern_k3_argmax_f32_final", "grid": [S, 1, 1],
                                                "block": [64, 1, 1]}]}},
    }
    norm_params = ["in buffer<bf16>", "inout buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "out buffer<u8>",
                   "out buffer<u8>", "i32", "i32"]
    ops["add_rms_quant"] = G.plain("glm_glue", "glm_add_rms_quant", norm_params, [T, 1, 1], [256, 1, 1])
    ops["add_quant"] = G.nulled("glm_glue", "glm_add_rms_quant", norm_params, {3}, [T, 1, 1], [256, 1, 1])
    ops["rms_quant"] = G.nulled("glm_glue", "glm_add_rms_quant", norm_params, {0}, [T, 1, 1], [256, 1, 1])
    prep_params = ["in buffer<bf16>", "i32", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<i32>",
                   "in buffer<i64>", "inout state", "inout state", "in buffer<bf16>", "in buffer<bf16>", "out buffer<u8>",
                   "out buffer<u8>", "i32", "i32"]
    prep = "glm_mla_prep_fp8" if fp8kv else "glm_mla_prep"
    ops["mla_prep_full"] = G.plain("glm_glue", prep, prep_params, [T, 1, 1], [256, 1, 1])
    ops["mla_prep_shared"] = G.nulled("glm_glue", prep, prep_params, {8, 9, 10}, [T, 1, 1], [256, 1, 1])
    for j in range(chunk // sub):
        ops[f"indexer.{j}"] = G.indexer(j, sub, max_ctx)

    gemms = {
        "a.full": (NA[True], H, chunk, H, NA[True], chunk, 1),
        "a.shared": (NA[False], H, chunk, H, NA[True], chunk, 1),
        "b.full": (NB[True], QL, chunk, QL, NB[True], chunk, 1),
        "b.shared": (NB[False], QL, chunk, QL, NB[True], chunk, 1),
        "o": (H, HEADS * VD, chunk, HEADS * VD, H, chunk, 1),
        "dense.gu": (2 * DENSE_I, H, chunk, H, 2 * DENSE_I, chunk, 1),
        "dense.dn": (H, DENSE_I, chunk, DENSE_I, H, chunk, 1),
        "shared.gu": (2 * INTER, H, chunk, H, 2 * INTER, chunk, 1),
        "shared.dn": (H, INTER, chunk, INTER, H, chunk, 1),
        "moe.gu": (2 * INTER, H, mcap, H, 2 * INTER, mcap, EXPERTS),
        "moe.dn": (H, INTER, mcap, INTER, H, mcap, EXPERTS),
    }
    for k, a in gemms.items():
        if k.startswith("moe."):
            ls = [G.fp8_gemm(*a, swap_bm=bm, when=w) for w, bm in layouts]
            ops["fp8." + k] = {"params": ls[0][1], "impl": {"launches": [l for l, _ in ls]}}
        elif k in TILES:
            kinds = [part.split(":")[1][:1] for part in TILES[k].split(",")]
            ls = [G.fp8_gemm(*a, **({"swap_bm": t} if kind == "s" else {"bn": t}), when=w)
                  for kind, (w, t) in zip(kinds, moe_layouts(TILES[k], chunk))]
            ops["fp8." + k] = {"params": ls[0][1], "impl": {"launches": [l for l, _ in ls]}}
        else:
            ops["fp8." + k] = G.fp8_op(*a)

    for k in ("fp8.shared.gu", "fp8.shared.dn", "silu_shared"):
        ops[k + ".lane"] = within(ops[k], 1, LANE_ROWS)
        ops[k] = within(ops[k], LANE_ROWS + 1, chunk)

    requant_params = ["in buffer<fp8e4m3>", "in buffer<f32>", "out buffer<fp8e4m3>", "out buffer<u8>", "i32", "i32",
                      "i32", "i32", "i32", "i32", "i64"]

    def requant_op(rows, k):
        name = f"requant.{rows}x{k}"
        ops.setdefault(name, G.plain("glm_prep", "glm_requant", requant_params, [-(-rows // 128), k // 128, 1],
                                     [256, 1, 1]))
        return name

    ops["kvb"] = G.plain("glm_prep", "glm_kvb", ["in buffer<fp8e4m3>", "in buffer<f32>", "out buffer<bf16>",
                                                  "out buffer<bf16>"], [HEADS * (NOPE + VD) * KVL // 256, 1, 1],
                         [256, 1, 1])

    # ---- buffers
    buffers = {
        "token_ids": {"dtype": "i64", "shape": [T], "kind": "input", "fill": "token", "domain": {"index_into": "embed"}},
        "slot_mapping": {"dtype": "i64", "shape": [T], "kind": "input", "fill": "slot", "domain": {"index_into": kv0}},
        "block_table": {"dtype": "i32", "shape": [S, max_pages], "kind": "input",
                        "domain": {"index_into": kv0, "stride": PAGE}},
        "seq_lens": {"dtype": "i32", "shape": [S], "kind": "input", "fill": "seq_len", "domain": {"min": 1}},
        "cu_seqlens": {"dtype": "i64", "shape": [pack + 1], "kind": "input", "fill": "cu_seqlens",
                       "domain": {"min": 0, "max": chunk, "monotone": True}},
    }
    if head:
        buffers["next_token"] = {"dtype": "i64", "shape": [S], "kind": "output", "fill": "tokens",
                                 "domain": {"index_into": "embed"}}
    states = {}
    programs = {}
    once = Once({"buffers": buffers, "ops": ops, "programs": programs})
    derive = Once({"buffers": buffers, "ops": ops, "programs": programs})

    def weight(name, shape, bind, dtype="bf16", kind="weight"):
        buffers[name] = {"dtype": dtype, "shape": list(shape), "kind": kind, "bind": bind}

    def carry(name, shape, dtype="bf16"):
        buffers[name] = {"dtype": dtype, "shape": list(shape), "kind": "carry"}

    def work(name, shape, dtype="bf16"):
        buffers[name] = {"dtype": dtype, "shape": list(shape), "kind": "workspace"}

    def fp8_weight(name, n, k):
        carry(name, [n, k], "fp8e4m3")
        carry(name + ".sf", [sf_bytes(k, n)], "u8")

    def requant(dst, n_total, tensor, rows, k, row0=0, group_rows=None, groups=1, dst_group_rows=0, sf_group=0,
                sf_ld=None):
        """Checkpoint `tensor` (or one per group) into rows [row0, row0 + rows) of fp8 weight `dst`."""
        src = f"{dst}.src.{len(derive.calls)}"
        names = [tensor] if isinstance(tensor, str) else tensor
        weight(src, [rows * groups, k], [seg(t + ".weight") for t in names], "fp8e4m3", "source")
        weight(src + ".s", [-(-rows // 128) * groups, k // 128], [seg(t + ".weight_scale_inv") for t in names], "f32",
               "source")
        derive.calls.append({"label": f"load.{src}", "op": requant_op(rows * groups, k),
                             "args": [buf(src), buf(src + ".s"), buf(dst), buf(dst + ".sf"), i32(rows * groups), i32(k),
                                      i32(group_rows or rows), i32(dst_group_rows), i32(row0), i32(sf_ld or n_total),
                                      i64(sf_group)]})

    weight("embed", [V, H], [seg(HF + "embed_tokens.weight")])
    if head:
        weight("gamma_final", [H], [seg(HF + "norm.weight")])
        weight("w_lm", [V, H], [seg("lm_head.weight")])
    carry("rope", [max_ctx, ROPE])
    once.rope(("rope", 0), ("rope", ROPE), max_ctx, ROPE // 2, ROPE, 8e6)
    carry("rs", [1])
    once.call("fill_bf16", [buf("rs"), i32(1), {"f32": 2.5}], 1)

    for name, w in (("res", H), ("h", H), ("xn", H), ("qa", NA[True]), ("qb", NB[True]), ("qmla", HEADS * MLA),
                    ("o", HEADS * KVL), ("sg", 2 * INTER), ("sh", H), ("dg", 2 * DENSE_I)):
        work(name, [T, w])
    for name, w in (("xq", H), ("qq", QL), ("iq", IQB), ("oq", HEADS * VD), ("sq", INTER), ("dq", DENSE_I)):
        work(name, [T, w], "u8")
    for name, k in (("xsf", H), ("qsf", QL), ("osf", HEADS * VD), ("ssf", INTER), ("dsf", DENSE_I)):
        work(name, [sf_bytes(k, chunk)], "u8")
    for name in ("pos", "klen", "row_seq", "zero", "tlen"):
        work(name, [T], "i32")
    for name in ("ks", "ke"):
        work(name, [T], "u32")
    work("iw", [T, IH])
    work("wout", [T, IH])
    work("kg", [max_ctx, ID], "u8")
    work("ksf", [max_ctx], "f32")
    work("pick", [max_ctx], "u8")
    work("logits", [sub * max_ctx])
    work("imeta", [G.meta_words(sub)], "i32")
    work("ind", [T, G.TOPK], "i32")
    if fp8kv:
        work("kvd", [max_ctx, MLA])
    work("ml", [T, HEADS], "f32")
    work("lse", [T, HEADS], "f32")
    work("rl", [T, EXPERTS], "f32")
    work("ids", [T, TOPK], "i32")
    work("wts", [T, TOPK], "f32")
    work("exp2perm", [T, TOPK], "i32")
    work("psum", [EXPERTS], "i32")
    work("hist", [-(-chunk // G.ROUTE_SLICE), EXPERTS], "i32")
    work("mq", [mcap, H], "u8")
    work("msf", [sf_bytes(H, mcap)], "u8")
    work("g13", [mcap, 2 * INTER])
    work("mq2", [mcap, INTER], "u8")
    work("msf2", [sf_bytes(INTER, mcap)], "u8")
    work("f2", [mcap, H])
    for name in ("hl", "rlast", "xf"):
        work(name, [S, H])
    work("lg", [S, V], "f32")
    work("pmax", [S, 64], "f32")
    work("pidx", [S, 64], "i32")

    # ---- program
    prog = []

    def step(label, op, *args, lane=0, after=()):
        prog.append({"label": label, "op": op, "args": list(args), **({"lane": lane} if lane else {}),
                     **({"after": list(after)} if after else {})})

    def gemm(label, key, a, w, c, m=var(T), a_sf="xsf", **lanes):
        grouped = key.startswith("moe.")
        step(label, "fp8." + key, buf(a), buf(w), buf(a_sf), buf(w + ".sf"), buf(c), buf("psum") if grouped else m,
             **lanes)

    def shared_expert(n, sfx, lane=0, after=None):
        first = {"lane": lane, "after": after} if lane else {"after": after}
        rest = {"lane": lane} if lane else {}
        gemm(n + "sgu" + sfx, "shared.gu" + sfx, "xq", n + "w_sgu", "sg", **first)
        step(n + "ssilu" + sfx, "silu_shared" + sfx, buf("sg"), buf("sq"), buf("ssf"), i32(chunk), **rest)
        gemm(n + "sdn" + sfx, "shared.dn" + sfx, "sq", n + "w_sdn", "sh", a_sf="ssf", **rest)

    def plan(l):
        step(f"L{l}.plan", "plan", buf("cu_seqlens"), buf("seq_lens"), var(S), buf("pos"), buf("ks"), buf("ke"),
             buf("klen"), buf("row_seq"), buf("zero"), var(T))

    plan(first)
    step("embed", "embed", buf("token_ids"), buf("embed"), buf("res"), var(T), i32(H))
    for l in range(first, layers):
        full = l in FULL
        n = f"L{l}."
        a = f"{HF}layers.{l}.self_attn."
        mlp = f"{HF}layers.{l}.mlp."
        if l in cuts and l != first:
            plan(l)
        states[f"kv.{l}"] = {"bytes_per_token": KV_ROW[kv_dtype]}
        if full:
            states[f"idx.{l}"] = {"bytes_per_token": IDX_ROW}
        weight(n + "gamma_in", [H], [seg(f"{HF}layers.{l}.input_layernorm.weight")])
        weight(n + "gamma_post", [H], [seg(f"{HF}layers.{l}.post_attention_layernorm.weight")])
        weight(n + "g_q", [QL], [seg(a + "q_a_layernorm.weight")])
        weight(n + "g_kv", [KVL], [seg(a + "kv_a_layernorm.weight")])
        fp8_weight(n + "w_a", NA[full], H)
        requant(n + "w_a", NA[full], a + "q_a_proj", QL, H)
        requant(n + "w_a", NA[full], a + "kv_a_proj_with_mqa", MLA, H, row0=QL)
        fp8_weight(n + "w_b", NB[full], QL)
        requant(n + "w_b", NB[full], a + "q_b_proj", QB, QL)
        fp8_weight(n + "w_o", H, HEADS * VD)
        requant(n + "w_o", H, a + "o_proj", H, HEADS * VD)
        weight(n + "kv_b.src", [HEADS * (NOPE + VD), KVL], [seg(a + "kv_b_proj.weight")], "fp8e4m3", "source")
        weight(n + "kv_b.src.s", [HEADS * (NOPE + VD) // 128, KVL // 128], [seg(a + "kv_b_proj.weight_scale_inv")],
               "f32", "source")
        carry(n + "kvb", [HEADS * (NOPE + VD), KVL])
        carry(n + "wuk", [HEADS, KVL, NOPE])
        derive.calls.append({"label": f"load.{n}kvb", "op": "kvb", "args": [buf(n + "kv_b.src"), buf(n + "kv_b.src.s"),
                                                                            buf(n + "kvb"), buf(n + "wuk")]})
        if full:
            requant(n + "w_a", NA[full], a + "indexer.wk", ID, H, row0=QL + MLA)
            requant(n + "w_b", NB[full], a + "indexer.wq_b", IQB, QL, row0=QB)
            weight(n + "ik_w", [ID], [seg(a + "indexer.k_norm.weight")])
            weight(n + "ik_b", [ID], [seg(a + "indexer.k_norm.bias")])
            weight(n + "w_p", [IH, H], [seg(a + "indexer.weights_proj.weight")])

        # attention
        if l == first:  # the bf16 row goes on only to the indexer's weights_proj and the router
            step(n + "norm_in", "rms_quant", buf("res"), buf(n + "gamma_in"), buf("xn"), buf("xq"), buf("xsf"),
                 i32(chunk), var(T))
        elif full:
            step(n + "norm_in", "add_rms_quant", buf("h"), buf("res"), buf(n + "gamma_in"), buf("xn"), buf("xq"),
                 buf("xsf"), i32(chunk), var(T))
        else:
            step(n + "norm_in", "add_quant", buf("h"), buf("res"), buf(n + "gamma_in"), buf("xq"), buf("xsf"),
                 i32(chunk), var(T))
        gemm(n + "a", "a.full" if full else "a.shared", "xq", n + "w_a", "qa")
        kv = st(f"kv.{l}")
        if full:
            step(n + "weights_proj", "gemm_bf16", buf("xn"), buf(n + "w_p"), buf("iw"), var(T), i32(IH), i32(H), i32(IH))
            step(n + "prep", "mla_prep_full", buf("qa"), i32(NA[True]), buf(n + "g_q"), buf(n + "g_kv"), buf("rope"),
                 buf("pos"), buf("slot_mapping"), kv, st(f"idx.{l}"), buf(n + "ik_w"), buf(n + "ik_b"), buf("qq"),
                 buf("qsf"), i32(chunk), var(T))
        else:
            step(n + "prep", "mla_prep_shared", buf("qa"), i32(NA[True]), buf(n + "g_q"), buf(n + "g_kv"), buf("rope"),
                 buf("pos"), buf("slot_mapping"), kv, buf("qq"), buf("qsf"), i32(chunk), var(T))
        if fp8kv and not full:  # the picks are the last indexer layer's: the lift runs beside the query chain
            step(n + "kv_dequant", "kv_dequant", kv, buf("block_table"), i32(max_pages), buf("seq_lens"), var(S),
                 buf("pick"), buf("kvd"), lane=1, after=[0])
        gemm(n + "b", "b.full" if full else "b.shared", "qq", n + "w_b", "qb", a_sf="qsf")
        step(n + "uk", "uk", buf("qb"), buf(n + "wuk"), buf("qmla"), var(T))
        if full:
            step(n + "q_post", "q_post_full", buf("qb"), i32(NB[True]), buf("rope"), buf("pos"), buf("qmla"), buf("iq"),
                 buf("iw"), buf("wout"), i32(1), var(T))
            step(n + "idx_gather", "idx_gather", st(f"idx.{l}"), buf("block_table"), i32(max_pages), buf("seq_lens"),
                 var(S), buf("kg"), buf("ksf"), buf("pick"))
            for j in range(chunk // sub):
                r0 = j * sub
                step(f"{n}indexer.{j}", f"indexer.{j}", buf("iq", r0 * IQB), buf("kg"), buf("ksf"),
                     buf("wout", r0 * IH * 2), buf("ks", r0 * 4), buf("ke", r0 * 4), buf("zero", r0 * 4),
                     buf("klen", r0 * 4), buf("logits"), buf("ind", r0 * G.TOPK * 4),
                     expr({"clamp": [T, r0, sub]}), var(CTX), expr({"mul": [{"ceil_div": [CTX, 256]}, 256]}),
                     buf("imeta"))
            if fp8kv:
                step(n + "topk_stream", "topk_stream", buf("ind"), buf("ks"), buf("klen"), buf("tlen"), buf("pick"),
                     var(T))
            else:
                step(n + "topk_slots", "topk_slots", buf("ind"), buf("block_table"), i32(max_pages), buf("row_seq"),
                     buf("klen"), buf("tlen"), var(T))
        else:
            step(n + "q_post", "q_post_shared", buf("qb"), i32(NB[True]), buf("rope"), buf("pos"), buf("qmla"),
                 i32(0), var(T))
        if fp8kv and full:
            step(n + "kv_dequant", "kv_dequant", kv, buf("block_table"), i32(max_pages), buf("seq_lens"), var(S),
                 buf("pick"), buf("kvd"))
        step(n + "attn", "flash_mla", buf("qmla"), buf("qmla", KVL * 2), buf("kvd") if fp8kv else kv, buf("ind"),
             buf("tlen"), buf("o"), buf("ml"), buf("lse"), after=[1] if fp8kv and not full else [])
        step(n + "uv", "uv", buf("o"), buf(n + "kvb", NOPE * KVL * 2), buf("oq"), buf("osf"), var(T))
        gemm(n + "o", "o", "oq", n + "w_o", "h", a_sf="osf")
        if l < DENSE:
            step(n + "norm_post", "add_quant", buf("h"), buf("res"), buf(n + "gamma_post"), buf("xq"), buf("xsf"),
                 i32(chunk), var(T))
        else:
            step(n + "norm_post", "add_rms_quant", buf("h"), buf("res"), buf(n + "gamma_post"), buf("xn"), buf("xq"),
                 buf("xsf"), i32(chunk), var(T))

        # MLP
        if l < DENSE:
            fp8_weight(n + "w_gu", 2 * DENSE_I, H)
            requant(n + "w_gu", 2 * DENSE_I, mlp + "gate_proj", DENSE_I, H)
            requant(n + "w_gu", 2 * DENSE_I, mlp + "up_proj", DENSE_I, H, row0=DENSE_I)
            fp8_weight(n + "w_dn", H, DENSE_I)
            requant(n + "w_dn", H, mlp + "down_proj", H, DENSE_I)
            gemm(n + "gu", "dense.gu", "xq", n + "w_gu", "dg")
            step(n + "silu", "silu_dense", buf("dg"), i32(DENSE_I), buf("dq"), buf("dsf"), i32(chunk), var(T))
            gemm(n + "dn", "dense.dn", "dq", n + "w_dn", "h", a_sf="dsf")
            continue
        weight(n + "w_r", [EXPERTS, H], [seg(mlp + "gate.weight")])
        weight(n + "bias", [EXPERTS], [seg(mlp + "gate.e_score_correction_bias")], "f32")
        fp8_weight(n + "w_13", EXPERTS * 2 * INTER, H)
        experts = lambda t: [f"{mlp}experts.{e}.{t}" for e in range(EXPERTS)]
        for row0, t in ((0, "gate_proj"), (INTER, "up_proj")):
            requant(n + "w_13", 2 * INTER, experts(t), INTER, H, row0=row0, groups=EXPERTS, dst_group_rows=2 * INTER,
                    sf_group=H // 512 * 2 * INTER)
        fp8_weight(n + "w_2", EXPERTS * H, INTER)
        requant(n + "w_2", H, experts("down_proj"), H, INTER, groups=EXPERTS, dst_group_rows=H,
                sf_group=INTER // 512 * H)
        fp8_weight(n + "w_sgu", 2 * INTER, H)
        requant(n + "w_sgu", 2 * INTER, mlp + "shared_experts.gate_proj", INTER, H)
        requant(n + "w_sgu", 2 * INTER, mlp + "shared_experts.up_proj", INTER, H, row0=INTER)
        fp8_weight(n + "w_sdn", H, INTER)
        requant(n + "w_sdn", H, mlp + "shared_experts.down_proj", H, INTER)

        # up to LANE_ROWS rows the shared expert runs on lane 1, beside the router, the routing and the routed GEMMs;
        # past it the routed path fills the GPU on its own and the shared expert follows it on lane 0
        shared_expert(n, ".lane", lane=1, after=[0])
        step(n + "router", "gemm_f32", buf("xn"), buf(n + "w_r"), buf("rl"), var(T), i32(EXPERTS), i32(H), i32(EXPERTS))
        step(n + "topk", "router_topk", buf("rl"), buf(n + "bias"), buf("rs"), buf("ids"), buf("wts"), var(T))
        step(n + "route", "moe_route", buf("ids"), buf("hist"), buf("psum"), buf("exp2perm"), var(T))
        step(n + "gather", "moe_gather", buf("xq"), buf("xsf"), i32(chunk), buf("exp2perm"), buf("mq"), buf("msf"),
             i32(mcap))
        gemm(n + "fc1", "moe.gu", "mq", n + "w_13", "g13", a_sf="msf")
        step(n + "silu", "silu_moe", buf("g13"), buf("exp2perm"), buf("mq2"), buf("msf2"), i32(mcap))
        gemm(n + "fc2", "moe.dn", "mq2", n + "w_2", "f2", a_sf="msf2")
        shared_expert(n, "", after=[1])
        step(n + "combine", "moe_combine", buf("f2"), buf("exp2perm"), buf("wts"), buf("sh"), buf("h"), var(T))

    if head:
        step("head.h", "last_rows", buf("hl"), buf("h"), buf("cu_seqlens"), i32(H), i32(H))
        step("head.res", "last_rows", buf("rlast"), buf("res"), buf("cu_seqlens"), i32(H), i32(H))
        step("head.norm", "head_norm", buf("hl"), buf("rlast"), buf("gamma_final"), buf("xf"), i32(0), var(S))
        step("head.lm", "gemm_f32", buf("xf"), buf("w_lm"), buf("lg"), var(S), i32(V), i32(H), i32(V))
        step("head.argmax", "argmax_partial", buf("lg"), buf("pmax"), buf("pidx"), i32(V))
        step("head.token", "argmax_final", buf("pmax"), buf("pidx"), buf("next_token"), i32(64))

    programs["prefill"] = kern_manifest.program(prog, groups=pack, rows=T, context=CTX)
    m = {
        "schema_version": kern_manifest.SCHEMA_VERSION,
        "model": f"glm-5.3/{layers}l-prefill",
        "vars": {T: {"max": chunk, "axis": "rows"}, S: {"max": pack, "axis": "groups"}, CTX: {"max": max_ctx}},
        "states": states,
        "buffers": buffers,
        "ops": ops,
        "programs": programs,
    }
    once.finish()
    derive.finish("derive", derive=True)
    called = {c["op"] for p in programs.values() for c in p["calls"]}
    m["ops"] = {k: v for k, v in ops.items() if k in called}
    named = {a["buf"] for p in programs.values() for c in p["calls"] for a in c["args"] if isinstance(a, dict) and "buf" in a}
    m["buffers"] = {k: v for k, v in m["buffers"].items() if v["kind"] != "workspace" or k in named}
    return kern_manifest.normalize(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=LAYERS, help="layers [0, N), with the embedding and the head")
    ap.add_argument("--chunk", type=int, default=16384, help="rows a call packs at most")
    ap.add_argument("--pack", type=int, default=16, help="sequences a call packs back to back")
    ap.add_argument("--max-ctx", type=int, default=1 << 20, help="the call's sequences' lengths summed, at most")
    ap.add_argument("--sub", type=int, default=1024, help="rows a call's indexer scores at once (logits sub x max-ctx bf16)")
    ap.add_argument("--cuts", default="18,38,58", help="layers a stage starts at, each an indexer layer")
    ap.add_argument("--kv", choices=("fp8", "bf16"), default="fp8", help="the latent cache's rows")
    ap.add_argument("--first", type=int, default=0,
                    help="first layer, fed the token embedding (a stage's stand-in for kern bench, which feeds a cut "
                         "stage's crossing h / res as it finds them)")
    ap.add_argument("--moe-tiles", default=MOE_TILES,
                    help="the routed GEMMs' layout by row count: MAX:s<block M> (swap-AB) or MAX:n (plain), comma-separated")
    a = ap.parse_args()
    cuts = {int(c) for c in a.cuts.split(",") if c}
    assert cuts <= FULL, "a stage starts at an indexer layer: the picks never cross a cut"
    json.dump(build(a.layers, a.chunk, a.pack, a.max_ctx, a.sub, cuts, a.kv, a.first, a.moe_tiles), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
