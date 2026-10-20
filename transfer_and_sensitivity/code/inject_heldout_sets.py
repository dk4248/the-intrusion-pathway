#!/usr/bin/env python
"""
Held-out knockout, GPU prep — compute fold-A-discovered I* (relation split AND sample split) for a model,
inject into HEAD_SETS.json as temp keys, and save the held-out (fold-B) sample_id lists so the
knockout recovery can be measured on the disjoint fold. Run knockout with --set-name = the temp key.
"""
import json, sys, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # repository root

RES = REPO / "intrusion_heads/results"
OUT = REPO / "transfer_and_sensitivity/results"
DATA = REPO / "lre_dataset/data/samples_6x250.json"
CONDS = ["nctx", "golden", "related", "counterfactual", "short_irr", "long_irr"]
SIDX = {s: i for i, s in enumerate(
    ["att_to_ctx_excl", "att_to_gold", "att_to_dstr", "att_max", "att_entropy",
     "dla_gold", "dla_dstr", "z_norm"])}
FOLD_A_REL = {"landmark_in_country", "city_in_country", "country_language"}
model = sys.argv[1] if len(sys.argv) > 1 else "Llama-3.1-8B"

samples = json.load(open(DATA))["samples"]
sids = [s["sample_id"] for s in samples]
relations = np.array([s["relation"] for s in samples])
arr = np.load(RES / f"{model}__head_scores.npz")["scores"]
H = arr.shape[3]


def raw_scores(mask):
    a = arr[mask]; ci = {c: i for i, c in enumerate(CONDS)}
    gi, iis, iil, ni = ci["golden"], ci["short_irr"], ci["long_irr"], ci["nctx"]
    ret = (a[:, gi, :, :, SIDX["att_to_gold"]] * a[:, gi, :, :, SIDX["dla_gold"]]).mean(0)
    ds = np.clip(a[:, ni, :, :, SIDX["dla_gold"]] - a[:, iis, :, :, SIDX["dla_gold"]], 0, None)
    dl = np.clip(a[:, ni, :, :, SIDX["dla_gold"]] - a[:, iil, :, :, SIDX["dla_gold"]], 0, None)
    intr = 0.5 * ((a[:, iis, :, :, SIDX["att_to_ctx_excl"]] * ds).mean(0)
                  + (a[:, iil, :, :, SIDX["att_to_ctx_excl"]] * dl).mean(0))
    return ret, intr


def clean_set(ret, intr, K=10, excl_mult=2):
    fr, fi = ret.flatten(), intr.flatten()
    excl = {(int(i // H), int(i % H)) for i in fr.argsort()[-excl_mult * 20:][::-1]}
    out, seen = [], set()
    for i in fi.argsort()[::-1]:
        l, h = int(i // H), int(i % H)
        if (l, h) in excl or (l, h) in seen:
            continue
        out.append([l, h, float(fi[i])]); seen.add((l, h))
        if len(out) >= 20:  # store 20, knockout slices top-k
            break
    return out


mA_rel = np.isin(relations, list(FOLD_A_REL))
rng = np.random.default_rng(0)
perm = rng.permutation(len(arr)); mA_samp = np.zeros(len(arr), bool); mA_samp[perm[:len(arr) // 2]] = True

sets = {
    "Istar_relfoldA": clean_set(*raw_scores(mA_rel)),      # discovered on fold-A relations
    "Istar_sampfoldA": clean_set(*raw_scores(mA_samp)),    # discovered on fold-A samples (seed0)
}
heldout_ids = {
    "Istar_relfoldA": [sids[i] for i in range(len(sids)) if not mA_rel[i]],   # fold-B relations
    "Istar_sampfoldA": [sids[i] for i in range(len(sids)) if not mA_samp[i]], # fold-B samples
}
insample_ids = {
    "Istar_relfoldA": [sids[i] for i in range(len(sids)) if mA_rel[i]],
    "Istar_sampfoldA": [sids[i] for i in range(len(sids)) if mA_samp[i]],
}

hs = json.load(open(RES / "HEAD_SETS.json"))
for k, v in sets.items():
    hs[model][k] = v
json.dump(hs, open(RES / "HEAD_SETS.json", "w"), indent=2)
json.dump({"heldout_ids": heldout_ids, "insample_ids": insample_ids},
          open(OUT / f"e1_folds_{model}.json", "w"), indent=2)
print(f"injected {list(sets.keys())} into HEAD_SETS[{model}]")
for k, v in sets.items():
    print(f"  {k} top-10: {[[l,h] for l,h,_ in v[:10]]}  | held-out n={len(heldout_ids[k])}")
