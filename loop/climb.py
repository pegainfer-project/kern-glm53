#!/usr/bin/env python3
"""loop/climb.py [HISTORY.md] [OUT.svg]: HISTORY.md's rows as the climb, one line a landed optimization: what it
did, its step of `all` (mean TTFT as % of row 0, a waterfall) coloured by the kernel family it touched, its Δ."""
import html
import pathlib
import re
import sys

here = pathlib.Path(__file__).resolve().parent
src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else here / "HISTORY.md"
out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else here / "climb.svg"

rows = [
    (int(c[0]), c[2], *map(float, c[3:6]))
    for c in ([x.strip() for x in line.split("|")[1:-1]] for line in src.read_text().splitlines()
              if re.match(r"^\| \d+ \|", line))
]
pct = lambda r, i: 100 * r[2 + i] / rows[0][2 + i]
families = [("MoE / MLP GEMMs and glue", "#2a78d6", "#3987e5"), ("attention (flash_mla, MLA)", "#eb6834", "#d95926"),
            ("indexer (logits, top-k)", "#1baf7a", "#199e70"), ("projections, norms, launches", "#eda100", "#c98500")]
rules = [(1, r"flash_mla|kv_dequant|v up-projection|q_pe|q_post|o_proj|quant_o"), (2, r"indexer|top-k|topk|logits"),
         (3, r"PDL|norms|projections"), (0, r"routed|moe|expert|silu|combine|gate-up|block M"), (3, r".")]
family = lambda what: next(k for k, p in rules if re.search(p, what, re.I))

W, RH, TOP, PL, PR = 1200, 17, 104, 700, 1080
n = len(rows) - 1
H = TOP + RH * n + 44
lo = min(pct(r, 2) for r in rows) - 1.5
x = lambda v: PL + (PR - PL) * (100 - v) / (100 - lo)
cut = lambda s, k=104: s if len(s) <= k else s[:k - 1].rstrip() + "…"

ticks = "".join(
    f'<line class="grid" x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{TOP - 6}" y2="{TOP + RH * n}"/>'
    f'<text class="tick" x="{x(v):.1f}" y="{TOP - 12}" text-anchor="middle">{v}%</text>'
    for v in range(5 * int(lo // 5 + 1), 101, 5))
lines = []
for j, (a, b) in enumerate(zip(rows, rows[1:])):
    yc = TOP + RH * j + RH / 2
    pa, pb, f = pct(a, 2), pct(b, 2), family(b[1])
    x0, x1 = sorted((x(pa), x(pb)))
    d = 100 * (b[4] / a[4] - 1)
    lines.append(
        f'<g><title>row {b[0]} ({b[1]}): all {a[4]:.2f} -> {b[4]:.2f} ms ({d:+.2f}%)</title>'
        f'<text class="num" x="{40}" y="{yc + 4:.1f}" text-anchor="end">{b[0]}</text>'
        f'<text class="lab" x="48" y="{yc + 4:.1f}">{html.escape(cut(b[1]))}</text>'
        f'<line class="link" x1="{x(pa):.1f}" x2="{x(pa):.1f}" y1="{yc - RH:.1f}" y2="{yc + 5:.1f}"/>'
        f'<rect class="f{f}" x="{x0:.1f}" y="{yc - 5:.1f}" width="{max(x1 - x0, 2):.1f}" height="10" rx="2"/>'
        f'<text class="{"num" if d < 0 else "up"}" x="{PR + 12}" y="{yc + 4:.1f}">{d:+.2f}%</text></g>')
legend = "".join(
    f'<rect class="f{k}" x="{48 + 210 * k}" y="{TOP - 50}" width="10" height="10" rx="2"/>'
    f'<text class="tick" x="{62 + 210 * k}" y="{TOP - 41}">{name}</text>' for k, (name, *_) in enumerate(families))
last = rows[-1]
summary = (f"row {n}: short {pct(last, 0) - 100:+.1f}% ({last[2]:.0f} ms), long {pct(last, 1) - 100:+.1f}% "
           f"({last[3]:.0f} ms), all {pct(last, 2) - 100:+.1f}% ({last[4]:.0f} ms); each landing through the numerics gate")
style = "".join(f".f{k}{{fill:{c}}}" for k, (_, c, _) in enumerate(families))
dark = "".join(f".f{k}{{fill:{c}}}" for k, (_, _, c) in enumerate(families))
title = "GLM-5.3 PP4 prefill: mean TTFT over the AgentX corpus, every landed optimization"
out.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{title}">
<title>{title}</title>
<style>
svg{{font:12px system-ui,-apple-system,"Segoe UI",sans-serif}}
.bg{{fill:#fcfcfb}}.grid{{stroke:#e4e3df}}.link{{stroke:#b9b8b2;stroke-width:1}}
.tick,.num,.sub{{fill:#52514e}}.lab,.ttl{{fill:#0b0b0b}}.up{{fill:#0b0b0b;font-weight:600}}
.ttl{{font-size:15px;font-weight:600}}.num{{font-variant-numeric:tabular-nums}}{style}
@media (prefers-color-scheme:dark){{.bg{{fill:#1a1a19}}.grid{{stroke:#3a3936}}.link{{stroke:#6b6a64}}
.tick,.num,.sub{{fill:#c3c2b7}}.lab,.ttl,.up{{fill:#f0efec}}{dark}}}
</style>
<rect class="bg" width="{W}" height="{H}" rx="8"/>
<text class="ttl" x="16" y="26">{title}</text>
<text class="sub" x="16" y="45">{summary}</text>
{legend}{ticks}{"".join(lines)}
<text class="tick" x="{(PL + PR) / 2:.0f}" y="{H - 14}" text-anchor="middle">all: mean TTFT as % of row 0 (lower to the right)</text>
<text class="tick" x="{PR + 12}" y="{TOP - 12}">Δ all</text>
</svg>
''')
print(f"{out}: {n} landings, all {pct(last, 2) - 100:+.2f}%")
