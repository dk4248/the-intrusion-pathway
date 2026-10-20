#!/usr/bin/env python
"""
Instruction-tuned transfer, part 1 (Table E.1) — compute instruct head-sets IDENTICALLY to
aggregate_head_scores.py, append instruct to HEAD_SETS.json (base keys preserved), and compute the
base<->instruct comparison: does RLHF preserve the intrusion-head population?

Metrics:
  - retrieval_set / intrusion_set / clean_intrusion_set (I*) for instruct
  - |R n I| for instruct (disjointness holds under RLHF?)
  - Jaccard(base I*, instruct I*) at K=10,20  (population preserved?)
  - Spearman(s_int base, s_int instruct) over late-block heads (ranking preserved?)
  - Spearman(s_ret base, s_ret instruct)
Because the instruct checkpoint shares the base architecture, (L,H) indices align 1:1.
"""
import json, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # repository root

RES = REPO / "intrusion_heads/results"
OUT = REPO / "transfer_and_sensitivity/results"
CONDS = ["nctx", "golden", "related", "counterfactual", "short_irr", "long_irr"]
SIDX = {s: i for i, s in enumerate(
    ["att_to_ctx_excl", "att_to_gold", "att_to_dstr", "att_max", "att_entropy",
     "dla_gold", "dla_dstr", "z_norm"])}
TOP_K = 20
BASE = "Llama-3.1-8B"
INSTRUCT = "Llama-3.1-8B-Instruct"


def scores_and_sets(mname):
    """Replicates aggregate_head_scores.py exactly."""
    arr = np.load(RES / f"{mname}__head_scores.npz")["scores"]  # [N,C,L,H,S]
    N, C, L, H, S = arr.shape
    ci = {c: i for i, c in enumerate(CONDS)}
    gi, ii_s, ii_l, ni = ci["golden"], ci["short_irr"], ci["long_irr"], ci["nctx"]
    ret = (arr[:, gi, :, :, SIDX["att_to_gold"]] * arr[:, gi, :, :, SIDX["dla_gold"]]).mean(0)
    drop_s = np.clip(arr[:, ni, :, :, SIDX["dla_gold"]] - arr[:, ii_s, :, :, SIDX["dla_gold"]], 0, None)
    drop_l = np.clip(arr[:, ni, :, :, SIDX["dla_gold"]] - arr[:, ii_l, :, :, SIDX["dla_gold"]], 0, None)
    intr = 0.5 * ((arr[:, ii_s, :, :, SIDX["att_to_ctx_excl"]] * drop_s).mean(0)
                  + (arr[:, ii_l, :, :, SIDX["att_to_ctx_excl"]] * drop_l).mean(0))
    fr, fi = ret.flatten(), intr.flatten()
    top_ret = fr.argsort()[-TOP_K:][::-1]
    top_intr = fi.argsort()[-TOP_K:][::-1]
    retrieval_set = [[int(i // H), int(i % H), float(fr[i])] for i in top_ret]
    intrusion_set = [[int(i // H), int(i % H), float(fi[i])] for i in top_intr]
    ret_lh = {(l, h) for l, h, _ in retrieval_set}
    intr_lh = {(l, h) for l, h, _ in intrusion_set}
    top_ret_wide = {(int(i // H), int(i % H)) for i in fr.argsort()[-2 * TOP_K:][::-1]}
    clean, seen = [], set()
    for i in top_intr.tolist() + fi.argsort()[::-1].tolist():
        l, h = int(i // H), int(i % H)
        if (l, h) in top_ret_wide or (l, h) in seen:
            continue
        clean.append([l, h, float(fi[i])]); seen.add((l, h))
        if len(clean) >= TOP_K:
            break
    return dict(L=L, H=H, ret=ret, intr=intr,
                retrieval_set=retrieval_set, intrusion_set=intrusion_set,
                clean_intrusion_set=clean, overlap=sorted(ret_lh & intr_lh),
                overlap_n=len(ret_lh & intr_lh))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def jacc(A, B, K):
    a = {(l, h) for l, h, _ in A[:K]}; b = {(l, h) for l, h, _ in B[:K]}
    return len(a & b) / len(a | b) if (a | b) else 0.0


base = scores_and_sets(BASE)
inst = scores_and_sets(INSTRUCT)
L, H = inst["L"], inst["H"]
late = np.zeros((L, H), bool); late[L // 2:] = True
lf = late.flatten()

cmp = dict(
    instruct=dict(retrieval_set=inst["retrieval_set"], intrusion_set=inst["intrusion_set"],
                  clean_intrusion_set=inst["clean_intrusion_set"],
                  overlap=inst["overlap"], overlap_n=inst["overlap_n"]),
    base_overlap_n=base["overlap_n"],
    jaccard_Istar_base_instruct={K: round(jacc(base["clean_intrusion_set"], inst["clean_intrusion_set"], K), 3)
                                 for K in (10, 20)},
    jaccard_intrusion_set=round(jacc(base["intrusion_set"], inst["intrusion_set"], 20), 3),
    jaccard_retrieval_set=round(jacc(base["retrieval_set"], inst["retrieval_set"], 20), 3),
    spearman_s_int_all=round(spearman(base["intr"].flatten(), inst["intr"].flatten()), 3),
    spearman_s_int_late=round(spearman(base["intr"].flatten()[lf], inst["intr"].flatten()[lf]), 3),
    spearman_s_ret_all=round(spearman(base["ret"].flatten(), inst["ret"].flatten()), 3),
    spearman_s_ret_late=round(spearman(base["ret"].flatten()[lf], inst["ret"].flatten()[lf]), 3),
)
json.dump(cmp, open(OUT / "e5_head_sets_comparison.json", "w"), indent=2)

# append instruct to HEAD_SETS.json (preserve base keys)
hs_path = RES / "HEAD_SETS.json"
hs = json.load(open(hs_path))
hs[INSTRUCT] = dict(retrieval_set=inst["retrieval_set"], intrusion_set=inst["intrusion_set"],
                    clean_intrusion_set=inst["clean_intrusion_set"],
                    overlap=[[l, h] for l, h in inst["overlap"]], overlap_n=inst["overlap_n"])
json.dump(hs, open(hs_path, "w"), indent=2)

md = [f"# Base↔instruct head-set comparison, Table E.1 ({BASE} vs {INSTRUCT})\n",
      "Identical scoring to aggregate_head_scores.py. (L,H) indices align 1:1 (shared arch).\n",
      f"- **Disjointness holds under RLHF?** |R∩I| = base **{base['overlap_n']}/20** vs instruct **{inst['overlap_n']}/20**",
      f"- **Intrusion population preserved?** Jaccard(I* base, I* instruct) = "
      f"K10 **{cmp['jaccard_Istar_base_instruct'][10]}**, K20 **{cmp['jaccard_Istar_base_instruct'][20]}**; "
      f"intrusion_set(20) Jaccard **{cmp['jaccard_intrusion_set']}**; retrieval_set(20) Jaccard **{cmp['jaccard_retrieval_set']}**",
      f"- **Ranking preserved?** Spearman s_int (late-block) **{cmp['spearman_s_int_late']}** (all heads {cmp['spearman_s_int_all']}); "
      f"Spearman s_ret (late-block) **{cmp['spearman_s_ret_late']}** (all {cmp['spearman_s_ret_all']})",
      f"\nInstruct I* (clean_intrusion_set, top-10): {[[l,h] for l,h,_ in inst['clean_intrusion_set'][:10]]}",
      f"Instruct intrusion_set (top-10): {[[l,h] for l,h,_ in inst['intrusion_set'][:10]]}",
      f"Base I* (top-10): {[[l,h] for l,h,_ in base['clean_intrusion_set'][:10]]}"]
open(OUT / "e5_head_sets_comparison.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("\nAppended", INSTRUCT, "to HEAD_SETS.json")
