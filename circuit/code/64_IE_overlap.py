"""Intrusion (I) vs entrainment (E): every available E set, compared against s_int.

E sources, per model, whichever exist:
  published  Niu et al.'s own saved mask (Llama-3.1-8B / country_capital_city / epoch 406)
  padbug     63_niu_entrainment.py with their search() position bug replicated
  correct    63_niu_entrainment.py reading the true last position
I is the manuscript s_int (primary variant of 61_sep_scores.py). For each (I_k, E) pair:
overlap, chance k|E|/N, hypergeometric P(overlap >= observed), and where E's heads sit on s_int.
The padbug-vs-published comparison is the validation of the port.
"""
import json, math, sys
from pathlib import Path
from collections import Counter
import torch
ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = {"Llama-3.1-8B": [(0,4),(0,8),(0,24),(0,27),(0,28),(1,2),(1,25),(2,10),(4,9),(7,7),(7,21),
    (7,26),(8,12),(8,29),(9,8),(9,10),(9,21),(11,2),(11,9),(11,20),(12,12),(13,0),(13,19),(13,28),
    (14,17),(14,19),(14,20),(14,29),(15,10),(15,28),(16,19),(16,20),(16,30),(21,8),(26,5),(26,7)]}
NL = {"Llama-3.1-8B": (32, 32), "pythia-6.9b": (32, 32), "gpt2-xl": (48, 25)}

def hyp(obs, K, n, N):
    if obs <= 0: return 1.0
    return sum(math.comb(K, x) * math.comb(N - K, n - x)
               for x in range(obs, min(K, n) + 1)) / math.comb(N, n)

for m, (nL, nH) in NL.items():
    N = nL * nH
    sp = ROOT / "results" / f"SEP_scores_{m}.pt"
    if not sp.exists(): continue
    si = torch.load(sp)["int|raw|prefix"].flatten()
    order = torch.argsort(si, descending=True)
    Ik = lambda k: {(int(i) // nH, int(i) % nH) for i in order[:k]}
    pct = lambda L, h: 100 * float((si < si[L * nH + h]).float().mean())
    Es = {}
    if m in PUBLISHED: Es["published"] = set(PUBLISHED[m])
    for tag, suf in (("padbug", "_padbug"), ("correct", "")):
        f = ROOT / "results" / f"NIU_E_{m}_country_capital_city_s0{suf}.json"
        if f.exists():
            d = json.load(f.open()); Es[tag] = {tuple(x) for x in d["E"]}
    print(f"\n=== {m} (N={N} heads) ===")
    band = lambda L: "early" if L < nL // 3 else ("mid" if L < 2 * (nL // 3) else "late")
    for tag, E in Es.items():
        bd = dict(Counter(band(L) for L, _ in E))
        med = sorted(pct(L, h) for L, h in E)
        print(f"  E_{tag:9s} |E|={len(E):3d}  bands {bd}  "
              f"s_int percentile of E heads: median {med[len(med)//2] if med else float('nan'):.1f}")
    if "published" in Es and "padbug" in Es:
        a_, b_ = Es["published"], Es["padbug"]
        ov = len(a_ & b_)
        print(f"  VALIDATION padbug vs published: overlap {ov} of |pub|={len(a_)} |padbug|={len(b_)}, "
              f"chance {len(a_)*len(b_)/N:.2f}, P={hyp(ov, len(a_), len(b_), N):.1e}, "
              f"Jaccard {ov/max(len(a_|b_),1):.3f}")
    for tag, E in Es.items():
        if not E:
            print(f"  I vs E_{tag}: E is EMPTY -- no heads switched off, nothing to overlap"); continue
        print(f"  I vs E_{tag}:")
        print(f"    {'k':>4s} {'overlap':>8s} {'chance':>7s} {'P(>=obs)':>9s}")
        for k in (5, 10, 20, 40, 60, 100, 150):
            I = Ik(k); ov = len(I & E)
            print(f"    {k:4d} {ov:8d} {k*len(E)/N:7.2f} {hyp(ov, len(E), k, N):9.1e}")
