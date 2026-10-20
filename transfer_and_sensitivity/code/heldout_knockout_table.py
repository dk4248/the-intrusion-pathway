#!/usr/bin/env python
"""
Held-out knockout, GPU aggregation — measure held-out knockout recovery. For each fold, the I* was discovered on
fold-A; here we measure recovery on the DISJOINT fold-B (held-out) vs fold-A (in-sample).
If held-out recovery ~ in-sample recovery, the intrusion set is not an in-sample fit.
"""
import json, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # repository root

OUT = REPO / "transfer_and_sensitivity/results"
CONDS = ["nctx", "golden", "related", "counterfactual", "short_irr", "long_irr"]
model = "Llama-3.1-8B"
folds = json.load(open(OUT / f"e1_folds_{model}.json"))


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def cond_means(records, id_set):
    by = {c: [] for c in CONDS}
    for r in records:
        if r["sample_id"] in id_set:
            by[r["condition"]].append(r["delta_lg"])
    return {c: nanmean(v) for c, v in by.items()}


rows = []
for tag, kof in [("relation-held-out", "e1_ko_llama_relfoldA.json"),
                 ("sample-held-out", "e1_ko_llama_sampfoldA.json")]:
    key = "Istar_relfoldA" if "rel" in tag else "Istar_sampfoldA"
    ko = json.load(open(OUT / kof))
    ho = set(folds["heldout_ids"][key]); ins = set(folds["insample_ids"][key])
    m_ho = cond_means(ko["records"], ho)
    m_in = cond_means(ko["records"], ins)
    rows.append((tag, m_in, m_ho))

lines = [f"# Held-out knockout recovery ({model}, I* discovered on fold-A, K10)\n",
         "Recovery = mean Δℓ_g under irrelevant; cost = −mean Δℓ_g under golden. In-sample = fold-A "
         "(where I* was discovered); held-out = disjoint fold-B.\n",
         "| Split | where | short_irr rec | long_irr rec | golden cost | nctx |",
         "|---|---|---|---|---|---|"]
for tag, m_in, m_ho in rows:
    lines.append(f"| {tag} | in-sample (A) | {m_in['short_irr']:+.3f} | {m_in['long_irr']:+.3f} | "
                 f"{-m_in['golden']:+.3f} | {m_in['nctx']:+.4f} |")
    lines.append(f"| {tag} | **held-out (B)** | **{m_ho['short_irr']:+.3f}** | **{m_ho['long_irr']:+.3f}** | "
                 f"{-m_ho['golden']:+.3f} | {m_ho['nctx']:+.4f} |")
lines.append("\n**Reading:** if held-out (B) recovery ≈ in-sample (A) recovery, the intrusion set discovered "
             "on one split transfers to the disjoint split — recovery is not an in-sample fit to the "
             "selection criterion. Sample-held-out is the high-power test (I* overlap 0.85); relation-held-out "
             "is coarse (3 imbalanced relations/fold) and expected to be noisier — functional redundancy: the "
             "invariant is the recovery, not exact set identity.")
json.dump({tag: dict(in_sample=m_in, held_out=m_ho) for tag, m_in, m_ho in rows},
          open(OUT / "e1_heldout_ko.json", "w"), indent=2)
open(OUT / "e1_heldout_ko.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
