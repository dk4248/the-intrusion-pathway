#!/usr/bin/env python
"""
Held-out validation of the intrusion set + K/alpha/exclusion sensitivity (offline part).

Shows I* is NOT an in-sample artifact of the s_int-based selection: the head population discovered on
one split matches the population discovered on a disjoint split (relations and samples), and is
stable to K, the retrieval-exclusion radius, and an introduced retrieval-penalty alpha.

The GPU held-out KNOCKOUT recovery (discover on A, knock out measured on B) is a separate step;
here we establish set/score stability, which is the load-bearing anti-circularity evidence.
Offline numpy over head_scores.npz.
"""
import json, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # repository root

RES = REPO / "intrusion_heads/results"
OUT = REPO / "transfer_and_sensitivity/results"
DATA = REPO / "lre_dataset/data/samples_6x250.json"
MODELS = ["Llama-3.1-8B", "Mistral-7B-v0.1", "Qwen2.5-7B", "pythia-6.9b"]
CONDS = ["nctx", "golden", "related", "counterfactual", "short_irr", "long_irr"]
SIDX = {s: i for i, s in enumerate(
    ["att_to_ctx_excl", "att_to_gold", "att_to_dstr", "att_max", "att_entropy",
     "dla_gold", "dla_dstr", "z_norm"])}
relations = [s["relation"] for s in json.load(open(DATA))["samples"]]
relations = np.array(relations)
# relation-balanced 2-fold (by sample count): ~125 each, disjoint relations
FOLD_A_REL = {"landmark_in_country", "city_in_country", "country_language"}
FOLD_B_REL = {"person_plays_instrument", "object_superclass", "country_capital_city"}


def raw_scores(arr, mask):
    """s_ret, s_int per (L,H) over the masked samples — exactly aggregate_head_scores.py."""
    a = arr[mask]
    ci = {c: i for i, c in enumerate(CONDS)}
    gi, ii_s, ii_l, ni = ci["golden"], ci["short_irr"], ci["long_irr"], ci["nctx"]
    ret = (a[:, gi, :, :, SIDX["att_to_gold"]] * a[:, gi, :, :, SIDX["dla_gold"]]).mean(0)
    ds = np.clip(a[:, ni, :, :, SIDX["dla_gold"]] - a[:, ii_s, :, :, SIDX["dla_gold"]], 0, None)
    dl = np.clip(a[:, ni, :, :, SIDX["dla_gold"]] - a[:, ii_l, :, :, SIDX["dla_gold"]], 0, None)
    intr = 0.5 * ((a[:, ii_s, :, :, SIDX["att_to_ctx_excl"]] * ds).mean(0)
                  + (a[:, ii_l, :, :, SIDX["att_to_ctx_excl"]] * dl).mean(0))
    return ret, intr


def clean_set(ret, intr, K=20, excl_mult=2, alpha=0.0):
    """I*: top-K by (intr - alpha*z(ret)) excluding top-(excl_mult*K) by ret."""
    H = ret.shape[1]
    fr, fi = ret.flatten(), intr.flatten()
    score = fi.copy()
    if alpha:
        z = (fr - fr.mean()) / (fr.std() + 1e-9)
        score = (fi - fi.mean()) / (fi.std() + 1e-9) - alpha * z
    excl = {(int(i // H), int(i % H)) for i in fr.argsort()[-excl_mult * K:][::-1]}
    out, seen = [], set()
    for i in score.argsort()[::-1]:
        l, h = int(i // H), int(i % H)
        if (l, h) in excl or (l, h) in seen:
            continue
        out.append((l, h)); seen.add((l, h))
        if len(out) >= K:
            break
    return out


def jacc(A, B):
    a, b = set(A), set(B)
    return round(len(a & b) / len(a | b), 3) if (a | b) else 0.0


def spearman(a, b):
    ra = a.argsort().argsort().astype(float); rb = b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return round(float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12)), 3)


report = {}
for m in MODELS:
    arr = np.load(RES / f"{m}__head_scores.npz")["scores"]
    L, H = arr.shape[2], arr.shape[3]
    late = np.zeros(L * H, bool); late[(L // 2) * H:] = True
    full_ret, full_intr = raw_scores(arr, np.ones(len(arr), bool))
    I_full = clean_set(full_ret, full_intr, K=20)

    # 1) relation-held-out
    mA = np.isin(relations, list(FOLD_A_REL)); mB = np.isin(relations, list(FOLD_B_REL))
    rA, iA = raw_scores(arr, mA); rB, iB = raw_scores(arr, mB)
    IA, IB = clean_set(rA, iA, 20), clean_set(rB, iB, 20)
    rel_holdout = dict(nA=int(mA.sum()), nB=int(mB.sum()),
                       jaccard_Istar_A_vs_B=jacc(IA, IB),
                       spearman_s_int_late=spearman(iA.flatten()[late], iB.flatten()[late]),
                       spearman_s_ret_late=spearman(rA.flatten()[late], rB.flatten()[late]))

    # 2) sample-held-out (10 seeds, 50/50)
    js, sp = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(arr)); half = len(arr) // 2
        m1 = np.zeros(len(arr), bool); m1[perm[:half]] = True
        r1, i1 = raw_scores(arr, m1); r2, i2 = raw_scores(arr, ~m1)
        js.append(len(set(clean_set(r1, i1, 20)) & set(clean_set(r2, i2, 20))) / 20.0)
        sp.append(spearman(i1.flatten()[late], i2.flatten()[late]))
    samp_holdout = dict(mean_overlap_frac=round(float(np.mean(js)), 3),
                        mean_spearman_s_int_late=round(float(np.mean(sp)), 3))

    # 3) K sensitivity (Jaccard of top-K I* vs top-20 restricted to K)
    Ksens = {K: jacc(clean_set(full_ret, full_intr, K), I_full[:K]) for K in (5, 10, 15, 30)}
    # 4) exclusion-radius sensitivity (Jaccard vs 2K baseline)
    excl_sens = {f"{e}K": jacc(clean_set(full_ret, full_intr, 20, excl_mult=e), I_full) for e in (1, 2, 3)}
    # 5) alpha retrieval-penalty sensitivity (Jaccard vs alpha=0)
    alpha_sens = {a: jacc(clean_set(full_ret, full_intr, 20, alpha=a), I_full) for a in (0.5, 1.0, 2.0)}

    report[m] = dict(relation_held_out=rel_holdout, sample_held_out=samp_holdout,
                     K_sensitivity=Ksens, exclusion_radius_sensitivity=excl_sens,
                     alpha_penalty_sensitivity=alpha_sens)

json.dump(report, open(OUT / "e1_held_out.json", "w"), indent=2)

lines = ["# Held-out validation + K/alpha/exclusion sensitivity (offline set-stability)\n",
         "Anti-circularity: I* discovered on a disjoint split matches I* on the other split.\n",
         "## 1. Relation-held-out (discover on fold A relations, test on disjoint fold B relations)",
         "| Model | nA | nB | Jaccard(I*_A,I*_B) | Spearman s_int (late) | Spearman s_ret (late) |",
         "|---|---|---|---|---|---|"]
for m in MODELS:
    r = report[m]["relation_held_out"]
    lines.append(f"| {m} | {r['nA']} | {r['nB']} | {r['jaccard_Istar_A_vs_B']} | "
                 f"{r['spearman_s_int_late']} | {r['spearman_s_ret_late']} |")
lines += ["\n## 2. Sample-held-out (10 seeds, 50/50 split)",
          "| Model | mean overlap frac | mean Spearman s_int (late) |", "|---|---|---|"]
for m in MODELS:
    s = report[m]["sample_held_out"]
    lines.append(f"| {m} | {s['mean_overlap_frac']} | {s['mean_spearman_s_int_late']} |")
lines += ["\n## 3-5. K / exclusion-radius / alpha-penalty sensitivity (Jaccard vs baseline I*)",
          "| Model | K5 | K10 | K15 | K30 | excl 1K | excl 3K | a=0.5 | a=1 | a=2 |", "|---|---|---|---|---|---|---|---|---|---|"]
for m in MODELS:
    K = report[m]["K_sensitivity"]; e = report[m]["exclusion_radius_sensitivity"]; a = report[m]["alpha_penalty_sensitivity"]
    lines.append(f"| {m} | {K[5]} | {K[10]} | {K[15]} | {K[30]} | {e['1K']} | {e['3K']} | {a[0.5]} | {a[1.0]} | {a[2.0]} |")
lines.append("\n**Note:** only 6 relations exist (3 per fold), heavily imbalanced — relation-held-out is a "
             "coarse test; sample-held-out (10 seeds) is the higher-power one. There is NO alpha in the intrusion-head "
             "s_int code: its only weighting is the 0.5 short/long mix. alpha here is an "
             "*introduced* retrieval-penalty robustness knob; low sensitivity = I* is not alpha-defined. "
             "GPU held-out knockout recovery (discover-on-A, measure-on-B) is the confirming step, run separately.")
open(OUT / "e1_held_out.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
