#!/usr/bin/env python
"""
Alpha-recovery injection — introduce a retrieval-penalty alpha into the intrusion selection and
inject the re-ranked I*_alpha sets into HEAD_SETS.json so the knockout can measure recovery/cost at
each alpha. alpha=0 is the paper's I* (top-K s_int excluding top-2K s_ret). alpha>0 additionally
penalizes retrieval-ish heads: score = z(s_int) - alpha*z(s_ret).
"""
import json, sys, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # repository root

RES = REPO / "intrusion_heads/results"
CONDS = ["nctx", "golden", "related", "counterfactual", "short_irr", "long_irr"]
SIDX = {s: i for i, s in enumerate(
    ["att_to_ctx_excl", "att_to_gold", "att_to_dstr", "att_max", "att_entropy",
     "dla_gold", "dla_dstr", "z_norm"])}
ALPHAS = [0.5, 1.0, 2.0]
model = sys.argv[1]

arr = np.load(RES / f"{model}__head_scores.npz")["scores"]
N, C, L, H, S = arr.shape
ci = {c: i for i, c in enumerate(CONDS)}
gi, iis, iil, ni = ci["golden"], ci["short_irr"], ci["long_irr"], ci["nctx"]
s_ret = (arr[:, gi, :, :, SIDX["att_to_gold"]] * arr[:, gi, :, :, SIDX["dla_gold"]]).mean(0)
ds = np.clip(arr[:, ni, :, :, SIDX["dla_gold"]] - arr[:, iis, :, :, SIDX["dla_gold"]], 0, None)
dl = np.clip(arr[:, ni, :, :, SIDX["dla_gold"]] - arr[:, iil, :, :, SIDX["dla_gold"]], 0, None)
s_int = 0.5 * ((arr[:, iis, :, :, SIDX["att_to_ctx_excl"]] * ds).mean(0)
               + (arr[:, iil, :, :, SIDX["att_to_ctx_excl"]] * dl).mean(0))
fr, fi = s_ret.flatten(), s_int.flatten()
zr = (fr - fr.mean()) / (fr.std() + 1e-9)
zi = (fi - fi.mean()) / (fi.std() + 1e-9)
excl = {(int(i // H), int(i % H)) for i in fr.argsort()[-40:][::-1]}   # top-2K retrieval, always excluded

hs = json.load(open(RES / "HEAD_SETS.json"))
for a in ALPHAS:
    score = zi - a * zr
    out, seen = [], set()
    for i in score.argsort()[::-1]:
        lh = (int(i // H), int(i % H))
        if lh in excl or lh in seen:
            continue
        out.append([lh[0], lh[1], float(fi[i])]); seen.add(lh)
        if len(out) >= 20:
            break
    hs[model][f"Istar_alpha{a}"] = out
    print(f"{model} alpha={a}: I*_alpha top-10 = {[[l,h] for l,h,_ in out[:10]]}")
json.dump(hs, open(RES / "HEAD_SETS.json", "w"), indent=2)
