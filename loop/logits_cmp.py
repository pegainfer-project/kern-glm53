#!/usr/bin/env python3
"""loop/logits_cmp.py BASE.pt CAND.pt: two kern_logits recordings of the same prompts, position by position.

top-1 agreement; |dlp|, how far the candidate moved the logprob of the base's top-1 (mean and p99);
flips, positions whose top-1 changed, and of them the confident ones (base margin top-1 - top-2 > 1 nat),
which are bugs rather than near-ties; and whether every chunk's last-row next_token still matches."""
import sys

import torch

base, cand = torch.load(sys.argv[1]), torch.load(sys.argv[2])
tot = agree = conf = 0
dlp = []
heads = True
for name, b in base.items():
    c = cand[name]
    n = min(len(b["top_id"]), len(c["top_id"]))
    bt, ct = b["top_id"][:n, 0], c["top_id"][:n, 0]
    same = bt == ct
    tot += n
    agree += int(same.sum())
    margin = b["top_lp"][:n, 0] - b["top_lp"][:n, 1]
    conf += int(((~same) & (margin > 1.0)).sum())
    # the candidate's logprob of the base's top-1, where the candidate's top-8 holds it
    hit = c["top_id"][:n] == bt[:, None]
    held = hit.any(-1)
    cand_lp = torch.where(hit, c["top_lp"][:n], torch.zeros_like(c["top_lp"][:n])).sum(-1)
    dlp.append((cand_lp - b["top_lp"][:n, 0]).abs()[held])
    heads &= b.get("next_token", c.get("next_token")) == c.get("next_token")
d = torch.cat(dlp).float()
print(f"positions {tot} · top-1 {100 * agree / tot:.3f}% · flips {tot - agree} (confident {conf}) · "
      f"|dlp| mean {d.mean():.4f} p99 {d.quantile(0.99):.4f} · next_token {'same' if heads else 'DIFFERS'}")
