#!/usr/bin/env python
"""Assemble the LRE 24-cell K x alpha cross grid per model (Tables F.3-F.6).
Rows = alpha in {0,0.5,1,2}; cols = K in {5,8,10,12,15,20}. Cell = rec_short / golden_cost.
Sources: alpha=0 -> K_sweep_pareto(clean) [K<=15] + e1_fill(clean20) [K=20];
         alpha>0,K=10 -> e1_alpha_*; alpha>0,K!=10 -> e1_cross_*.
"""
import json, os, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # repository root

RES = REPO / "intrusion_heads/results"
RB = REPO / "transfer_and_sensitivity/results"
MODELS = ["Llama-3.1-8B", "Mistral-7B-v0.1", "Qwen2.5-7B", "pythia-6.9b"]
CONDS = ["nctx", "golden", "related", "counterfactual", "short_irr", "long_irr"]
KS = [5, 8, 10, 12, 15, 20]
ALPHAS = [0.0, 0.5, 1.0, 2.0]
pareto = json.load(open(RES / "K_sweep_pareto.json"))["models"]


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def ko(path):
    if not (os.path.exists(path) and os.path.getsize(path) > 1000):
        return None
    d = json.load(open(path)); by = {c: [] for c in CONDS}
    for r in d["records"]:
        by[r["condition"]].append(r["delta_lg"])
    m = {c: nanmean(v) for c, v in by.items()}
    return m["short_irr"], -m["golden"]   # rec_short, golden_cost


def cell(m, a, K):
    if a == 0.0:
        if K <= 15:
            d = pareto[m]["clean"].get(str(K))
            return (d["rec_short"], d["cost_gold"]) if d else None
        return ko(RB / f"e1_fill_{m}_clean20.json")   # K=20
    astr = f"{a}"
    if K == 10:
        return ko(RB / f"e1_alpha_{m}_a{astr}.json")
    return ko(RB / f"e1_cross_{m}_a{astr}_K{K}.json")


L = ["# LRE 24-cell K x alpha cross grid per model (Tables F.3–F.6)",
     "Cell = rec_short (short-irrelevant Δℓ_g recovery) / golden_cost, logits. K=10,α=0 = the paper's I*.\n"]
grid = {}
n_missing = 0
for m in MODELS:
    L.append(f"## {m}")
    L.append("| α \\ K | " + " | ".join(f"K={k}" for k in KS) + " |")
    L.append("|" + "---|" * (len(KS) + 1))
    grid[m] = {}
    for a in ALPHAS:
        cells = []
        for K in KS:
            c = cell(m, a, K)
            grid[m][f"a{a}_K{K}"] = c
            if c is None:
                n_missing += 1; cells.append("(pending)")
            else:
                cells.append(f"{c[0]:+.3f}/{c[1]:+.3f}")
        L.append(f"| α={a} | " + " | ".join(cells) + " |")
    L.append("")

json.dump(grid, open(RB / "e1_cross_grid.json", "w"), indent=2)
if n_missing:
    L.append(f"\n**{n_missing} cells pending** (cross sweep still running).")
report = "\n".join(L)
open(RB / "e1_cross_grid_table.md", "w").write(report + "\n")
print(report)
