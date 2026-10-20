"""Block-level diagram of the prefix mask on ONE real intrusion head, running example, fixed inputs.

Shared token strip in the middle. Above: the head at END reads every source (arrow width = its real attention
weight). Below: the same head with END->prefix masked: prefix arrows cut, every surviving arrow scaled by
1/(1-a).
usage: python 88_fig_mask_compact.py <model> <layer> <head>
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from xp_common import XP, ROOT

MODEL, LAY, HEAD = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
xp = XP(MODEL, "cuda:0"); m = xp.m
CTX, QUERY = "Computers are useful.", "The capital of Germany is"
PROMPT = f"{CTX} {QUERY}"
toks = m.to_tokens(PROMPT); T = toks.shape[1]
npre = len(m.tokenizer.encode(CTX, add_special_tokens=False))
with torch.no_grad():
    _, c = m.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_pattern") or n.endswith("hook_v")
                            or n == "ln_final.hook_scale")
A = c["pattern", LAY][0, HEAD, -1].float()
w = xp.values(c, LAY)[:, HEAD, :] @ xp.WO[LAY][HEAD]
d = xp.WUf[:, xp.fid("Berlin")] - xp.WUf[:, sorted(xp.forms("Computers"))].mean(-1)
contrib = ((A[:, None] * w) @ d / c["ln_final.hook_scale"][0, -1, 0]).detach().cpu().numpy()
A = A.detach().cpu().numpy()
a = float(A[1:npre + 1].sum())
P, U = float(contrib[1:npre + 1].sum()), float(contrib[0] + contrib[npre + 1:].sum())
UM = U / (1 - a)
print(f"{MODEL} L{LAY}H{HEAD}: a={a:.3f}  P={P:+.3f}  U={U:+.3f}  P+U={P+U:+.3f}  U/(1-a)={UM:+.3f}")

# word-level chips (sum attention over sub-word pieces); group: bos / prefix / query
tok = m.tokenizer(PROMPT, return_offsets_mapping=True, add_special_tokens=False)
offs = tok["offset_mapping"]; shift = T - len(offs)
words, spans = [], []
for wd in [("BOS", None)] + [(x, None) for x in ["Computers", "are", "useful", ".", "The", "capital", "of", "Germany", "is"]]:
    words.append(wd[0])
cursor, pos_of = 0, []
for wd in words[1:]:
    s0 = PROMPT.index(wd, cursor); e0 = s0 + len(wd); cursor = e0
    pos_of.append([i + shift for i, (x0, x1) in enumerate(offs) if x0 < e0 and x1 > s0])
pos_of = [[0]] + pos_of
att = [float(A[p].sum()) for p in pos_of]
grp = ["bos"] + ["pre"] * 4 + ["qry"] * 5
assert abs(sum(att) - 1) < 1e-3, sum(att)

# ------------------------------------------------------------------ drawing
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#ffffff"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a8a69e"
TINT = {"bos": "#ecebe7", "pre": "#fde3d5", "qry": "#dcebfb"}
EDGE = {"bos": GRAY, "pre": ORANGE, "qry": BLUE}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK, "pdf.fonttype": 42})
FW, FH = 7.0, 4.0
fig = plt.figure(figsize=(FW, FH)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW); ax.set_ylim(-FH / 2, FH / 2); ax.set_aspect("equal"); ax.axis("off")
fig.canvas.draw(); R = fig.canvas.get_renderer()
def textw(s, size, weight="normal"):
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight); bb = t.get_window_extent(R); t.remove()
    inv = ax.transData.inverted(); return inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]

# token strip
CH, PADX, GAPX, X0 = 0.26, 0.055, 0.05, 0.35
xs, ws, x = [], [], X0
for wd in words:
    wdt = textw(wd, 8, "semibold" if wd == "is" else "normal") + 2 * PADX
    xs.append(x + wdt / 2); ws.append(wdt); x += wdt + GAPX
XEND = xs[-1]
for i, wd in enumerate(words):
    ax.add_patch(FancyBboxPatch((xs[i] - ws[i] / 2, -CH / 2), ws[i], CH, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=TINT[grp[i]], ec=INK if wd == "is" else "none", lw=1.1, zorder=3))
    ax.text(xs[i], 0, wd, ha="center", va="center", fontsize=8, zorder=4,
            fontweight="semibold" if wd == "is" else "normal", color=MUTED if wd == "BOS" else INK)

# the head as a composition bar: orange share a = prefix part P, blue share 1-a = the rest U
HY, BW, BH = 1.30, 1.05, 0.30
BL = XEND - BW / 2
def bar(y, split):
    outline = FancyBboxPatch((BL, y - BH / 2), BW, BH, boxstyle="round,pad=0,rounding_size=0.06",
                             fc="none", ec=INK, lw=1.0, zorder=7)
    ax.add_patch(outline)
    segs = [(BL, split * BW, ORANGE), (BL + split * BW, (1 - split) * BW, BLUE)]
    for x0, wdt, col in segs:
        if wdt <= 0: continue
        r = Rectangle((x0, y - BH / 2), wdt, BH, fc=col, ec="none", zorder=6)
        ax.add_patch(r); r.set_clip_path(outline)
    return segs
from matplotlib.patches import Rectangle
top = bar(HY, a); bot = bar(-HY, 0.0)
lab = dict(ha="center", va="center", color=SURF, fontweight="bold", zorder=8)
def span(x0, x1, y, txt):
    ax.plot([x0 + 0.015, x1 - 0.015], [y, y], color=INK2, lw=0.7, zorder=5)
    for xx in (x0 + 0.015, x1 - 0.015): ax.plot([xx, xx], [y - 0.03, y + 0.03], color=INK2, lw=0.7, zorder=5)
    ax.text((x0 + x1) / 2, y + 0.045, txt, ha="center", va="bottom", fontsize=7.5, color=INK2, zorder=5)
xs_split = BL + a * BW
yPU = HY + BH / 2 + 0.12
ax.text((BL + xs_split) / 2, yPU, "P", ha="center", va="center", fontsize=9.5, color=ORANGE, fontweight="bold")
ax.text((xs_split + BL + BW) / 2, yPU, "U", ha="center", va="center", fontsize=9.5, color=BLUE, fontweight="bold")
ySP = yPU + 0.15
span(BL, xs_split, ySP, f"a = {a:.2f}"); span(xs_split, BL + BW, ySP, "1 − a")
ax.text(BL + BW / 2, -HY - BH / 2 - 0.13, "U / (1 − a)", ha="center", va="center", fontsize=9.5,
        color=BLUE, fontweight="bold")
for y in (HY, -HY):
    ax.text(BL + BW / 2, y, f"Head L{LAY}.H{HEAD}", fontsize=8.5, **lab)
for y in (HY, -HY):
    ax.plot([XEND, XEND], [np.sign(y) * CH / 2, y - np.sign(y) * BH / 2], color="#c3c2b7", lw=0.8,
            ls=(0, (2, 2)), zorder=1)

# attention arcs: prefix arcs feed the orange part, all others the blue part
LWMAX = 7.5
def curve(x0, y0, x1, y1, bend):
    return MPath([(x0, y0), ((x0 + x1) / 2, bend), (x1, y1)], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
def fan(sign, weights, masked):
    src = list(range(len(words) - 1))
    pre = [i for i in src if grp[i] == "pre"]; rest = [i for i in src if grp[i] != "pre"]
    if masked:
        slots = {i: BL + 0.06 + (a * BW - 0.12) * r / max(len(pre) - 1, 1) for r, i in enumerate(pre)}
        slots.update({i: BL + 0.08 + (BW - 0.30) * r / max(len(rest) - 1, 1) for r, i in enumerate(rest)})
    else:
        slots = {i: BL + 0.05 + (a * BW - 0.10) * r / max(len(pre) - 1, 1) for r, i in enumerate(pre)}
        slots.update({i: BL + a * BW + 0.06 + ((1 - a) * BW - 0.30) * r / max(len(rest) - 1, 1)
                      for r, i in enumerate(rest)})
    for i in src:
        y0, y1 = sign * CH / 2, sign * (HY - BH / 2)
        bend = sign * (0.55 + (HY - BH / 2 - 0.45) * (0.45 + 0.75 * (XEND - xs[i]) / (XEND - xs[0])))
        path = curve(xs[i], y0, slots[i], y1, bend)
        if masked and grp[i] == "pre":
            ax.add_patch(PathPatch(path, fc="none", ec=ORANGE, lw=0.8, ls=(0, (1.5, 2)), alpha=0.45, zorder=2))
            mid = path.interpolated(40).vertices[20]
            ax.text(mid[0], mid[1], "×", ha="center", va="center", fontsize=10, color=ORANGE, fontweight="bold", zorder=4)
        elif weights[i] > 0.004:
            ax.add_patch(PathPatch(path, fc="none", ec=EDGE[grp[i]], lw=max(0.6, LWMAX * weights[i] / max_w),
                                   alpha=0.9, capstyle="round", zorder=2))
    if weights[-1] > 0.004:                                   # END reading itself
        lx = xs[-1] + ws[-1] / 2 - 0.04
        loop = MPath([(lx, sign * CH / 2), (BL + BW + 0.12, sign * 0.62), (BL + BW - 0.10, sign * (HY - BH / 2))],
                     [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3])
        ax.add_patch(PathPatch(loop, fc="none", ec=BLUE, lw=max(0.6, LWMAX * weights[-1] / max_w), alpha=0.9, zorder=2))
keep = [0.0 if grp[i] == "pre" else att[i] / (1 - a) for i in range(len(words))]
max_w = max(max(att), max(keep))
fan(+1, att, masked=False)
fan(-1, keep, masked=True)

# what the head's write favours (net direct push on u_Berlin - u_Computers)
TAG_R = []
def tag(y, favours_berlin):
    word, col = ("Berlin", BLUE) if favours_berlin else ("Computers", ORANGE)
    x0 = BL + BW + 0.40
    ax.annotate("", xy=(x0 - 0.05, y), xytext=(BL + BW + 0.07, y),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, mutation_scale=8), zorder=3)
    wdt = textw(word, 8.5) + 0.14
    TAG_R.append(x0 + wdt)
    ax.add_patch(FancyBboxPatch((x0, y - CH / 2), wdt, CH, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=SURF, ec=col, lw=1.2, zorder=3))
    ax.text(x0 + wdt / 2, y, word, ha="center", va="center", fontsize=8.5, color=INK, style="italic", zorder=4)


ax.text(xs[7], -0.80, f"× {1/(1-a):.2f}", ha="center", va="center", fontsize=8.5, color=BLUE, zorder=6,
        bbox=dict(boxstyle="round,pad=0.12", fc=SURF, ec="none"))
ax.text(0.08, HY, "a)", fontsize=11, fontweight="semibold", va="center")
ax.text(0.08, -HY, "b)", fontsize=11, fontweight="semibold", va="center")
from matplotlib.transforms import Bbox
x_right = BL + BW + 0.12                      # head box + END self-loop
y_top, y_bot = ySP + 0.24, -HY - BH / 2 - 0.26
crop = Bbox([[0.0, y_bot + FH / 2], [x_right, y_top + FH / 2]])      # data units are inches here
out = ROOT / "figs" / f"mask_compact_{MODEL}_L{LAY}H{HEAD}"
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches=crop)
print("saved", f"{out}.pdf")
