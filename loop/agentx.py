#!/usr/bin/env python3
"""loop/agentx.py TRACES.jsonl > loop/weights.json: the AgentX corpus as weights on the bench grid.

Every request of every trace (subagent groups flattened, in time order) is served by an idle PP4 prefill
over an infinite prefix cache: its cached prefix is the leading 64-token blocks an earlier request of the
same trace already sent (the ids are chained, so a block id seen before is a prefix seen before), short
of its last block. The rest is cut into CHUNK-row chunks over a growing context. Four equal stages
pipeline the chunks, so the request's TTFT is (sum of its chunks' forward times + 3 x the last one's) / 4.
A chunk's time is read off the grid (ROWS x CTX, bilinear; fewer rows than the grid's first count as it),
so the corpus mean TTFT is a sum of grid times, each with the weight written here. A request of one chunk
is scenario `short` (a turn over its session's context), of several `long` (a fresh long prompt)."""
import bisect
import json
import sys

CHUNK, BLOCK = 16384, 64
ROWS = [128, 512, 2048, 8192, 16384]
CTX = [0, 65536, 131072, 196608, 262144]


def corners(grid, x):
    if x <= grid[0]:
        return [(0, 1.0)]
    if x >= grid[-1]:
        return [(len(grid) - 1, 1.0)]
    j = bisect.bisect_right(grid, x)
    f = (x - grid[j - 1]) / (grid[j] - grid[j - 1])
    return [(j - 1, 1 - f), (j, f)]


def requests(path):
    for line in open(path):
        t = json.loads(line)
        reqs = sorted((q for r in t["requests"] for q in (r["requests"] if r.get("type") == "subagent" else [r])),
                      key=lambda r: r["t"])
        seen = set()
        for r in reqs:
            h = r["hash_ids"]
            hit = next((i for i, b in enumerate(h) if b not in seen), len(h))
            seen.update(h)
            ctx = min(hit, max(0, len(h) - 1)) * BLOCK
            yield ctx, max(1, r["in"] - ctx)


def chunks(ctx, new):
    return [(min(CHUNK, new - o), ctx + o) for o in range(0, new, CHUNK)]


def main():
    w = {s: {} for s in ("short", "long")}
    n = {"short": 0, "long": 0}
    for ctx, new in requests(sys.argv[1]):
        cs = chunks(ctx, new)
        s = "short" if len(cs) == 1 else "long"
        n[s] += 1
        for i, (rows, c) in enumerate(cs):
            coef = 1.0 if len(cs) == 1 else (0.25 + 0.75 * (i == len(cs) - 1))
            for ri, rf in corners(ROWS, rows):
                for ci, cf in corners(CTX, c):
                    k = f"r{ROWS[ri]}-kv{CTX[ci]}"
                    w[s][k] = w[s].get(k, 0.0) + coef * rf * cf
    json.dump({"source": "semianalysis cc-traces-weka-062126-256k", "chunk": CHUNK, "rows": ROWS, "ctx": CTX,
               "requests": n, "weights": {s: dict(sorted(v.items())) for s, v in w.items()}}, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
