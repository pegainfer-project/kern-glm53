#!/usr/bin/env python3
"""loop/climb.py [HISTORY.md] [OUT.svg]: HISTORY.md's rows as the climb, short / long / all as % of row 0."""
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
base = rows[0][2:]
series = [("short", 0, "#2a78d6", "#3987e5"), ("long", 1, "#eb6834", "#d95926"), ("all", 2, "#1baf7a", "#199e70")]
pct = lambda r, i: 100 * r[2 + i] / base[i]

W, H, L, R, T, B = 960, 440, 64, 196, 56, 48
lo = min(pct(r, i) for r in rows for _, i, *_ in series)
y0, y1 = 5 * int((lo - 1) // 5), 100
n = rows[-1][0]
x = lambda k: L + (W - L - R) * k / n
y = lambda v: T + (H - T - B) * (y1 - v) / (y1 - y0)


def step(i):
    pts = [(x(rows[0][0]), y(pct(rows[0], i)))]
    for a, b in zip(rows, rows[1:]):
        pts += [(x(b[0]), y(pct(a, i))), (x(b[0]), y(pct(b, i)))]
    pts.append((x(n) + 6, pts[-1][1]))
    return " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)


grid = "".join(
    f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
    f'<text class="tick" x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v}%</text>'
    for v in range(y0, y1 + 1, 5))
xt = "".join(f'<text class="tick" x="{x(k):.1f}" y="{H - B + 20}" text-anchor="middle">{k}</text>'
             for k in range(0, n + 1, 5))
lines = "".join(f'<polyline class="s-{name}" points="{step(i)}"/>' for name, i, *_ in series)
ends = sorted(((y(pct(rows[-1], i)), name, pct(rows[-1], i), rows[-1][2 + i]) for name, i, *_ in series))
for j in range(1, len(ends)):
    if ends[j][0] - ends[j - 1][0] < 16:
        ends[j] = (ends[j - 1][0] + 16, *ends[j][1:])
labels = "".join(
    f'<text class="lab" x="{x(n) + 12:.1f}" y="{ly + 4:.1f}"><tspan class="k-{name}">■</tspan> {name} '
    f'{v - 100:+.1f}%<tspan class="sub"> {ms:.0f} ms</tspan></text>'
    for ly, name, v, ms in ends)
pp = 42  # from this row the ratio is the real PP4 stage 0's (HISTORY.md's header)
notes = (f'<line class="grid" x1="{x(pp):.1f}" x2="{x(pp):.1f}" y1="{T + 4}" y2="{H - B}" stroke-dasharray="3 3"/>'
         f'<text class="note" x="{x(pp) - 6:.1f}" y="{T + 16}" text-anchor="end">measured on PP4 stage 0 from row {pp}</text>'
         if n >= pp else "")
style = "".join(f".s-{s}{{stroke:{lc}}}.k-{s}{{fill:{lc}}}" for s, _, lc, _ in series)
dark = "".join(f".s-{s}{{stroke:{dc}}}.k-{s}{{fill:{dc}}}" for s, _, _, dc in series)
title = f"GLM-5.3 PP4 prefill: mean TTFT over the AgentX corpus, rows 0-{n}, % of row 0"
out.write_text(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{title}">
<title>{title}</title>
<style>
svg{{font:12px system-ui,-apple-system,"Segoe UI",sans-serif}}
.bg{{fill:#fcfcfb}}.grid{{stroke:#e4e3df;stroke-width:1}}.tick,.sub,.note{{fill:#52514e}}.lab,.ttl{{fill:#0b0b0b}}
.ttl{{font-size:14px;font-weight:600}}.lab{{font-size:13px}}
polyline{{fill:none;stroke-width:2;stroke-linejoin:round}}{style}
@media (prefers-color-scheme:dark){{.bg{{fill:#1a1a19}}.grid{{stroke:#3a3936}}.tick,.sub,.note{{fill:#c3c2b7}}.lab,.ttl{{fill:#f0efec}}{dark}}}
</style>
<rect class="bg" width="{W}" height="{H}" rx="8"/>
<text class="ttl" x="{L}" y="24">{title}</text>
<text class="sub" x="{L}" y="42">one step per landed optimization (loop/HISTORY.md), each through the numerics gate</text>
{grid}{xt}{notes}{lines}{labels}
<text class="tick" x="{(L + W - R) / 2:.0f}" y="{H - 10}" text-anchor="middle">landed optimization (HISTORY.md row)</text>
</svg>
''')
print(f"{out}: {len(rows)} rows, all {pct(rows[-1], 2) - 100:+.2f}%")
