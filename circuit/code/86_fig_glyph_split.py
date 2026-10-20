"""Running example, frozen intrusion heads, as two separate PDFs per model:
  figs/heat/<model>/nctx.pdf        a)  query alone
  figs/heat/<model>/short_irr.pdf   b)  "Computers are useful." + query   (carries the colour bar)
Same cell size, colour scale, square-size scale and fixed margins in both files (no tight cropping), so the
two can be placed at equal scale and bottom-aligned with the query rows lining up.
Cell (row = destination token, column = source token), BOS omitted:
  square area   mean attention of the Istar heads
  square colour sum over Istar heads of A_h[q,k] <v_k W_O^h, u_Berlin - u_Computers> / s_q
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, FuncNorm
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from xp_common import XP, ROOT

MODEL = sys.argv[1] if len(sys.argv) > 1 else "pythia-6.9b"
xp = XP(MODEL, "cuda:0"); m = xp.m
QUERY, CTX, GOLD, INTR = "The capital of Germany is", "Computers are useful.", "Berlin", "Computers"
gid, iid, heads = xp.fid(GOLD), sorted(xp.forms(INTR)), xp.Istar

@torch.no_grad()
def maps(prompt):
    toks = m.to_tokens(prompt)
    _, c = m.run_with_cache(toks, names_filter=lambda n: n.endswith("hook_pattern") or n.endswith("hook_v")
                            or n == "ln_final.hook_scale")
    T = toks.shape[1]; sc = c["ln_final.hook_scale"][0, :, 0].float()
    att = torch.zeros(T, T, device=xp.dev); mar = torch.zeros_like(att)
    udir = xp.WUf[:, gid] - xp.WUf[:, iid].mean(-1)
    for L, h in heads:
        A = c["pattern", L][0, h].float()
        w = xp.values(c, L)[:, h, :] @ xp.WO[L][h]
        att += A / len(heads)
        mar += A * (w @ udir)[None, :] / sc[:, None]
    lab = [m.tokenizer.decode([int(t)]).strip() for t in toks[0]]
    return lab[1:], att[1:, 1:].cpu().numpy(), mar[1:, 1:].cpu().numpy()

L0, A0, M0 = maps(QUERY)
L1, A1, M1 = maps(f"{CTX} {QUERY}")

INK, INK2, GRIDC, SURF = "#0b0b0b", "#52514e", "#e9e8e3", "#ffffff"
BLUE, ORANGE, NEUT = "#2a78d6", "#eb6834", "#d9d8d2"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "pdf.fonttype": 42, "figure.facecolor": SURF})
cmap = LinearSegmentedColormap.from_list("m", [ORANGE, "#f3a37f", NEUT, "#7fb0ea", BLUE])
AMAX = max(A0.max(), A1.max())
V = max(np.abs(M0[-1]).max(), np.abs(M1[-1]).max())
norm = FuncNorm((lambda x: np.sign(x) * np.sqrt(np.abs(x)), lambda y: np.sign(y) * y ** 2), vmin=-V, vmax=V)

CELL, BM, TM = 0.46, 1.05, 0.50          # inches; identical in both files
OUT = ROOT / "figs" / "heat" / MODEL; OUT.mkdir(parents=True, exist_ok=True)

def render(lab, A, M, title, fname, left, right, ylabel, colorbar):
    n = len(lab)
    W, H = left + n * CELL + right, BM + n * CELL + TM
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, BM / H, n * CELL / W, n * CELL / H])
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5); ax.set_aspect("equal")
    for q in range(n):
        for k in range(q + 1):
            ax.add_patch(Rectangle((k - 0.5, q - 0.5), 1, 1, fc="#f7f6f3", ec=GRIDC, lw=0.6, zorder=1))
            s = 0.94 * np.sqrt(max(A[q, k], 0.0) / AMAX)
            if s > 0.02:
                ax.add_patch(Rectangle((k - s / 2, q - s / 2), s, s, fc=cmap(norm(M[q, k])), ec="none", zorder=2))
    ax.add_patch(Rectangle((-0.5, n - 1.5), n, 1, fill=False, ec=INK, lw=1.4, zorder=3))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(lab, rotation=55, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(lab)
    ax.get_yticklabels()[-1].set_fontweight("bold"); ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", pad=8)
    ax.set_xlabel("Source token", fontsize=9.5, color=INK2, labelpad=4)
    if ylabel:
        ax.set_ylabel("Destination token", fontsize=9.5, color=INK2, labelpad=6)
    if colorbar:
        cx = left + n * CELL + 0.35
        cax = fig.add_axes([cx / W, (BM + 0.12 * n * CELL) / H, 0.13 / W, 0.76 * n * CELL / H])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, extend="both")
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.5, length=2.5, width=0.6, color=INK2)
        step = [x for x in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0) if V / x <= 4.5][0]
        cand = sorted({round(v, 6) for v in (step / 2, step, 2 * step, 4 * step) if v <= V * 1.0001})
        tk = [-t for t in reversed(cand)] + [0.0] + cand
        cb.set_ticks(tk); cb.set_ticklabels([f"{t:g}" for t in tk])
        cb.set_label("Net logit contribution  (Berlin − Computers)", fontsize=9, color=INK2, labelpad=8)
        cax.text(0.5, 1.06, "Berlin", transform=cax.transAxes, ha="center", va="bottom", fontsize=8.5, color=INK2)
        cax.text(0.5, -0.06, "Computers", transform=cax.transAxes, ha="center", va="top", fontsize=8.5, color=INK2)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=300)
    plt.close(fig)

render(L0, A0, M0, "a)", "nctx", left=1.05, right=0.15, ylabel=True, colorbar=False)
render(L1, A1, M1, "b)", "short_irr", left=0.85, right=1.35, ylabel=False, colorbar=True)
print("saved", OUT / "nctx.pdf", "and", OUT / "short_irr.pdf")
