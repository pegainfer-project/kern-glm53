#!/usr/bin/env python3
"""Teacher-forced prompt logits of the PP4 stages, one stage a GPU in one process (kern-py).

    python3 scripts/kern_logits.py <stages dir> <kernels dir> <checkpoint> <prompts.json> <out.pt> [--chunks N]

prompts.json: {"name": [token ids]}. Each prompt runs through stage 0..3 (the residual stream copied from
GPU to GPU between them) in --chunks equal chunks; after the last stage every row's logits are the head's
math in torch (fused_add_rms_norm, then the f32 lm_head GEMM), and the last row's argmax is checked
against the manifest's own `next_token`. Saved per prompt: top-8 logprobs and ids of every position.
"""
import argparse
import json
import pathlib

import torch
from safetensors import safe_open

import kern

PAGE = 64
DT = {"bf16": torch.bfloat16, "u8": torch.uint8, "f32": torch.float32, "i32": torch.int32, "i64": torch.int64,
      "u32": torch.int32, "fp8e4m3": torch.uint8}


class Mem:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


class Stage:
    def __init__(self, path, kernels, ckpt, gpu):
        self.gpu = gpu
        torch.cuda.set_device(gpu)
        self.rt = kern.Runtime(str(path), str(kernels), [str(ckpt)], gpu)
        self.m = json.loads(self.rt.manifest())
        for p in sorted(n for n, p in self.m["programs"].items() if p.get("once")):
            self.rt.run(p, {})

    def view(self, name):
        ptr, nbytes = self.rt.region(name)
        return torch.as_tensor(Mem(ptr, nbytes), device=f"cuda:{self.gpu}").view(DT[self.m["buffers"][name]["dtype"]])

    def has(self, name):
        return name in self.m["buffers"]


def head_top(h, res, gamma, w32, k=8, block=2048):
    """Top-k logprobs and ids of every row, through the head's math: fused_add_rms_norm, f32 lm_head."""
    x = (h.float() + res.float()).bfloat16().float()
    x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16()
    x = (x.float() * gamma.float()).bfloat16().float()
    vals, ids, argmax = [], [], []
    for b in range(0, x.shape[0], block):
        lg = x[b : b + block] @ w32.T
        t = torch.log_softmax(lg, -1).topk(k, -1)
        vals.append(t.values.cpu())
        ids.append(t.indices.cpu())
    return torch.cat(vals), torch.cat(ids)


def call(stages, seqs):
    """One packed call through the four stages: `seqs` is [(ids, lo, hi, first page)], each sequence's rows
    lo..hi on top of its rows below lo already cached. Returns the last stage's (h, res) rows and next_token."""
    H = 6144
    n = sum(hi - lo for _, lo, hi, _ in seqs)
    ids = torch.tensor([t for i, lo, hi, _ in seqs for t in i[lo:hi]], dtype=torch.int64)
    pos = [torch.arange(lo, hi, dtype=torch.int64) for _, lo, hi, _ in seqs]
    slots = torch.cat([(p // PAGE + p0) * PAGE + p % PAGE for p, (_, _, _, p0) in zip(pos, seqs)])
    cu = torch.zeros(17, dtype=torch.int64)
    cu[1:] = torch.tensor([hi - lo for _, lo, hi, _ in seqs]).cumsum(0).tolist()[-1]
    cu[1 : len(seqs) + 1] = torch.tensor([hi - lo for _, lo, hi, _ in seqs]).cumsum(0)
    lens = torch.tensor([hi for _, _, hi, _ in seqs], dtype=torch.int32)
    for k, s in enumerate(stages):
        torch.cuda.set_device(s.gpu)
        if s.has("token_ids"):
            s.view("token_ids")[:n].copy_(ids)
        s.view("slot_mapping")[:n].copy_(slots)
        bt = s.view("block_table")
        bt = bt.view(16, bt.numel() // 16)
        for j, (_, _, hi, p0) in enumerate(seqs):
            np_ = -(-hi // PAGE)
            bt[j, :np_].copy_(torch.arange(p0, p0 + np_, dtype=torch.int32))
        s.view("seq_lens")[: len(seqs)].copy_(lens)
        s.view("cu_seqlens").copy_(cu)
        if k:
            for b in ("h", "res"):
                s.view(b)[: n * H].copy_(stages[k - 1].view(b)[: n * H].cpu())
        torch.cuda.synchronize(s.gpu)
        s.rt.run("prefill", {"tokens": n, "seqs": len(seqs), "ctx": int(lens.sum())})
    last = stages[-1]
    torch.cuda.synchronize(last.gpu)
    return (last.view("h")[: n * H].view(n, H).clone(), last.view("res")[: n * H].view(n, H).clone(),
            last.view("next_token")[: len(seqs)].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages")
    ap.add_argument("kernels")
    ap.add_argument("ckpt")
    ap.add_argument("prompts")
    ap.add_argument("out", help="out prefix: <out>-c<chunks>.pt per --chunks, <out>-packed.pt")
    ap.add_argument("--chunks", default="1", help="comma-separated chunk counts to run every prompt in")
    ap.add_argument("--pack", default="", help="comma-separated prompt names run packed into one call")
    a = ap.parse_args()
    # the runtime gives its pages all the memory free at load; what torch holds then stays torch's
    reserve = [torch.empty((24 if g == 3 else 2) << 30, dtype=torch.uint8, device=f"cuda:{g}") for g in range(4)]
    stages = [Stage(pathlib.Path(a.stages) / f"stage{s}.json", a.kernels, a.ckpt, s) for s in range(4)]
    del reserve
    last = stages[-1]
    idx = json.load(open(pathlib.Path(a.ckpt) / "model.safetensors.index.json"))["weight_map"]
    with safe_open(pathlib.Path(a.ckpt) / idx["lm_head.weight"], "pt", device=f"cuda:{last.gpu}") as f:
        w32 = f.get_tensor("lm_head.weight").float()
    with safe_open(pathlib.Path(a.ckpt) / idx["model.norm.weight"], "pt", device=f"cuda:{last.gpu}") as f:
        gamma = f.get_tensor("model.norm.weight")
    prompts = json.load(open(a.prompts))

    def record(h, r, toks, rows):
        torch.cuda.set_device(last.gpu)
        lp, tid = head_top(h, r, gamma, w32)
        return lp, tid, [int(tid[i, 0]) for i in rows], toks

    for c in map(int, a.chunks.split(",")):
        out = {}
        for name, ids in prompts.items():
            L = len(ids)
            bounds = [L * j // c for j in range(c + 1)]
            parts = [record(*call(stages, [(ids, lo, hi, 1)]), [hi - lo - 1]) for lo, hi in zip(bounds, bounds[1:])]
            out[name] = {"ids": ids, "top_lp": torch.cat([p[0] for p in parts]),
                         "top_id": torch.cat([p[1] for p in parts]),
                         "torch_head": [p[2][0] for p in parts], "next_token": [p[3][0] for p in parts]}
            print(f"c{c}", name, L, "next_token", out[name]["next_token"], "torch head", out[name]["torch_head"],
                  flush=True)
        torch.save(out, f"{a.out}-c{c}.pt")
    if a.pack:
        names = a.pack.split(",")
        seqs = [(prompts[k], 0, len(prompts[k]), 1 + 4096 * j) for j, k in enumerate(names)]
        ends = torch.tensor([len(prompts[k]) for k in names]).cumsum(0).tolist()
        lp, tid, heads, toks = record(*call(stages, seqs), [e - 1 for e in ends])
        out, lo = {}, 0
        for k, e, hd, tk in zip(names, ends, heads, toks):
            out[k] = {"ids": prompts[k], "top_lp": lp[lo:e], "top_id": tid[lo:e], "torch_head": [hd], "next_token": [tk]}
            lo = e
        print("packed", names, "next_token", toks, "torch head", heads, flush=True)
        torch.save(out, f"{a.out}-packed.pt")


if __name__ == "__main__":
    main()
