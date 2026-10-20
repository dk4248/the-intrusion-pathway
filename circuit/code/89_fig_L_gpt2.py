"""L-shape, GPT-2 XL only: every head's retrieval score (x) vs intrusion score (y, paper Eq. 3).
Top-20 of each score coloured; dot area = the head's larger score as a share of that score's maximum.
"""
import numpy as np, torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
SURF, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e9e8e3", "#c3c2b7"
BLUE, ORANGE, OTHER = "#2a78d6", "#eb6834", "#c9c7bf"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "pdf.fonttype": 42, "figure.facecolor": SURF, "axes.facecolor": SURF})
K = 20
S = torch.load(ROOT / "results" / "SEP_scores_gpt2-xl.pt")
si = S["int|raw|prefix"].flatten().numpy(); sr = S["ret|raw"].flatten().numpy()
N = len(si)
I = set(np.argsort(-si)[:K].tolist()); R = set(np.argsort(-sr)[:K].tolist())
v = np.maximum(np.clip(sr, 0, None) / sr.max(), np.clip(si, 0, None) / si.max())
size = 4 + 150 * v
cat = np.array(["o"] * N, dtype=object)
for i in I - R: cat[i] = "I"
for i in R - I: cat[i] = "R"
for i in I & R: cat[i] = "B"

fig, ax = plt.subplots(figsize=(3.3, 3.0))
ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.axhline(0, color=AXIS, lw=0.8, zorder=1); ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
XL = -0.30                                     # heads with lower retrieval scores are pinned at the edge
o = (cat == "o") & (sr >= XL)
ax.scatter(sr[o], si[o], s=size[o], c=OTHER, edgecolors="none", alpha=0.7, zorder=2)
off = (cat == "o") & (sr < XL)
ax.scatter(np.full(off.sum(), XL), si[off], s=14, marker="<", c=OTHER, edgecolors="none", alpha=0.9,
           zorder=2, clip_on=False)
for c, col in (("R", BLUE), ("I", ORANGE)):
    mm = cat == c
    ax.scatter(sr[mm], si[mm], s=size[mm], c=col, edgecolors=SURF, linewidths=0.7, zorder=3)
mm = cat == "B"
ax.scatter(sr[mm], si[mm], s=size[mm], c=ORANGE, edgecolors=BLUE, linewidths=1.8, zorder=4)
ax.set_xlim(XL, sr.max() * 1.06); ax.set_ylim(-0.015, si.max() * 1.08)
ax.set_xlabel("Retrieval score")
ax.set_ylabel("Intrusion score")
ax.tick_params(length=2, labelsize=7.5)
nb = len(I & R)
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=6.5, mfc=ORANGE, mec=SURF, label="Intrusion"),
                   Line2D([], [], marker="o", ls="", ms=6.5, mfc=BLUE, mec=SURF, label="Retrieval"),
                   Line2D([], [], marker="o", ls="", ms=6.5, mfc=ORANGE, mec=BLUE, mew=1.8, label=f"Both ({nb})")],
          loc="upper right", frameon=False, fontsize=8, handletextpad=0.3, borderaxespad=0.2)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(ROOT / "figs" / f"L_shape_gpt2-xl.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("saved figs/L_shape_gpt2-xl.pdf  | pinned:", int(off.sum()), " both:", nb, " max s_int", si.max(), " max s_ret", sr.max())
