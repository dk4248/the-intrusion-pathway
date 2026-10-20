#!/usr/bin/env python
"""Assemble the complete K/alpha sweep tables (Tables F.1-F.2).
Table F.1: K sweep on knockout RECOVERY (K_sweep_pareto.json).
Set stability: K/alpha/exclusion sweep on I* set-stability JACCARD (e1_held_out.json).
Table F.2: alpha sweep on knockout RECOVERY (e1_alpha_*.json, K=10) + alpha=0 baseline (K_sweep clean-10).
"""
import json, glob, os, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # repository root

RES = REPO / "intrusion_heads/results"
RB = REPO / "transfer_and_sensitivity/results"
MODELS = ["Llama-3.1-8B", "Mistral-7B-v0.1", "Qwen2.5-7B", "pythia-6.9b"]
CONDS = ["nctx", "golden", "related", "counterfactual", "short_irr", "long_irr"]


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def ko_recovery(path):
    d = json.load(open(path))
    by = {c: [] for c in CONDS}
    for r in d["records"]:
        by[r["condition"]].append(r["delta_lg"])
    m = {c: nanmean(v) for c, v in by.items()}
    return m["short_irr"], m["long_irr"], -m["golden"], m["nctx"]


L = []
# ---- Table F.1: K sweep on recovery ----
pareto = json.load(open(RES / "K_sweep_pareto.json"))["models"]
L += ["# Complete K / alpha sweep (Tables F.1–F.2)\n",
      "## Table F.1 — K sweep on knockout RECOVERY (deployable clean set + full set)",
      "rec = mean Δℓ_g under irrelevant (recovery); cost = golden cost. From K_sweep_pareto.json.\n",
      "| Model | set | " + " | ".join(f"K={k}" for k in [5, 8, 10, 12, 15, 20]) + " |",
      "|" + "---|" * 8]
def cell_A(m, setn, k):
    d = pareto[m][setn].get(str(k))
    if d:
        return d["rec_short"], d["cost_gold"]
    # fill the two cells that used a different K grid: clean-K20, full-K10
    p = None
    if setn == "clean" and k == 20:
        p = RB / f"e1_fill_{m}_clean20.json"
    elif setn == "full" and k == 10:
        p = RB / f"e1_fill_{m}_full10.json"
    if p and p.exists() and os.path.getsize(p) > 1000:
        rs, rl, cost, nctx = ko_recovery(p)
        return rs, cost
    return None


for m in MODELS:
    for setn in ["clean", "full"]:
        cells = []
        for k in [5, 8, 10, 12, 15, 20]:
            c = cell_A(m, setn, k)
            cells.append(f"{c[0]:+.3f}/{c[1]:+.3f}" if c else "—")
        L.append(f"| {m} | {setn} (rec_short/cost) | " + " | ".join(cells) + " |")

# ---- Set stability: K/alpha/exclusion Jaccard ----
ho = json.load(open(RB / "e1_held_out.json"))
L += ["\n## Set stability — K / alpha / exclusion sweep on I* (Jaccard vs baseline I*)",
      "| Model | K5 | K10 | K15 | K30 | excl 1K | excl 3K | α=0.5 | α=1 | α=2 |",
      "|" + "---|" * 10]
for m in MODELS:
    K = ho[m]["K_sensitivity"]; e = ho[m]["exclusion_radius_sensitivity"]; a = ho[m]["alpha_penalty_sensitivity"]
    L.append(f"| {m} | {K['5']} | {K['10']} | {K['15']} | {K['30']} | {e['1K']} | {e['3K']} | "
             f"{a['0.5']} | {a['1.0']} | {a['2.0']} |")

# ---- Table F.2: alpha sweep on recovery ----
L += ["\n## Table F.2 — alpha (retrieval-penalty) sweep on knockout RECOVERY, K=10",
      "α=0 is the paper's I* (clean-K10 from K_sweep). α>0: score = z(s_int) − α·z(s_ret).\n",
      "| Model | α | rec_short | rec_long | golden cost | nctx |",
      "|---|---|---|---|---|---|"]
missing = []
for m in MODELS:
    # alpha=0 baseline from K_sweep clean-10
    d0 = pareto[m]["clean"]["10"]
    L.append(f"| {m} | 0 (paper I*) | {d0['rec_short']:+.3f} | {d0['rec_long']:+.3f} | {d0['cost_gold']:+.3f} | +0.000 |")
    for a in ["0.5", "1.0", "2.0"]:
        p = RB / f"e1_alpha_{m}_a{a}.json"
        if p.exists() and os.path.getsize(p) > 1000:
            rs, rl, cost, nctx = ko_recovery(p)
            L.append(f"| {m} | {a} | {rs:+.3f} | {rl:+.3f} | {cost:+.3f} | {nctx:+.3f} |")
        else:
            L.append(f"| {m} | {a} | (pending) | | | |"); missing.append(f"{m}_a{a}")

if missing:
    L.append(f"\n**Pending alpha runs:** {missing}")
report = "\n".join(L)
open(RB / "e1_ksweep_full_table.md", "w").write(report + "\n")
print(report)
