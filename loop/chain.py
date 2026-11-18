"""loop/chain.py AB_DIR LOOP_DIR PREV_ROW: a HISTORY.md row's scores, the row above's times the A/B's B / A medians.

Each A/B runs both sides on one GPU, so its ratio is the change; GPUs differ by a percent or more in absolute
terms, so the climb chains ratios from row 0 rather than quoting each A/B's absolute B side.
"""
import importlib.machinery, importlib.util, json, statistics, sys

out, loop, prev = sys.argv[1:4]
spec = importlib.util.spec_from_loader("score", importlib.machinery.SourceFileLoader("score", loop + "/score"))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
keys = ("short", "long", "all")
med = lambda side: {k: statistics.median(r[k] for r in side) for k in keys}
side = lambda s: med([sc.score(json.load(open(f"{out}/{s}-{i}.json")))[0] for i in (1, 2)])
a, b = side("a"), side("b")
cells = [c.strip() for c in prev.split("|")]
before = dict(zip(keys, map(float, cells[4:7])))
print(*(f"{before[k] * b[k] / a[k]:.2f}" for k in keys), f"{100 * (b['all'] / a['all'] - 1):+.2f}%")
