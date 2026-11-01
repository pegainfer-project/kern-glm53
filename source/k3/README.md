# From kern-k3-pp-prefill

Copied from `kern-k3-pp-prefill` at 08001b95 (kern's license, `LICENSE` of that repository).

- `copy_rows.cu`: unmodified. `kern_last_rows_bf16`, each sequence's last row of a packed call.
- `k3_router_argmax.cu`: the router top-k and the vocab argmax, built with `EXPERTS=256 TOPK=8`.
  Two changes: `TOPK` is guarded by `#ifndef` like `EXPERTS`, and the weight sum's shuffle mask is
  `(1u << TOPK) - 1u` instead of `0x0000ffffu` (a mask naming lanes that do not take part is
  undefined behaviour at eight). The math is GLM-5's as it is K3's: sigmoid scores, the score bias for
  the pick only, the picked scores renormalized and scaled by the routed scaling factor (2.5 here).
