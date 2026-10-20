"""Table E.4: class-conditional attention knockout at the LRE level, base vs instruct (Llama-3.1-8B).

Cells are mean Δℓ_g (knockout − baseline, in logits) per condition: recovery under short / long
irrelevant context, golden cost (= −Δℓ_g under golden), and nctx (0 by construction).

Inputs
  base     intrusion_heads/results/K_sweep_pareto.json            (aggregate_K_sweep.py)
  instruct transfer_and_sensitivity/results/e5_knockout_instruct_{K10_clean,K20_full}.json, from
           intrusion_heads/code/run_attention_knockout.py --model meta-llama/Llama-3.1-8B-Instruct
             --top-k 10 --set-name clean_intrusion_set --output-path <...>_K10_clean.json
             --top-k 20 --set-name intrusion_set       --output-path <...>_K20_full.json
           (HEAD_SETS.json must first contain the instruct sets: see instruct_head_sets.py)
"""
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # repository root
PARETO = REPO / "intrusion_heads/results/K_sweep_pareto.json"
RES = REPO / "transfer_and_sensitivity/results"
BASE = "Llama-3.1-8B"


def nanmean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(v) / len(v) if v else float("nan")


def instruct_row(path):
    by = {}
    for r in json.load(open(path))["records"]:
        by.setdefault(r["condition"], []).append(r["delta_lg"])
    m = {c: nanmean(v) for c, v in by.items()}
    return m["short_irr"], m["long_irr"], -m["golden"], m["nctx"]


def base_row(pareto, label, K):
    c = pareto["models"][BASE][label][str(K)]
    return c["rec_short"], c["rec_long"], c["cost_gold"], 0.0   # nctx: no context edges to mask


pareto = json.load(open(PARETO))
rows = [("I* clean (K10)", "Base", base_row(pareto, "clean", 10)),
        ("I* clean (K10)", "Instruct", instruct_row(RES / "e5_knockout_instruct_K10_clean.json")),
        ("Full (K20)", "Base", base_row(pareto, "full", 20)),
        ("Full (K20)", "Instruct", instruct_row(RES / "e5_knockout_instruct_K20_full.json"))]

lines = ["| Set | Checkpoint | short_irr rec. | long_irr rec. | golden cost | nctx |",
         "|---|---|---|---|---|---|"]
for s, ck, (a, b, c, d) in rows:
    lines.append(f"| {s} | {ck} | {a:+.3f} | {b:+.3f} | {c:+.3f} | {d:+.3f} |")
report = "\n".join(lines)
print(report)
RES.mkdir(parents=True, exist_ok=True)
(RES / "instruct_knockout_table.md").write_text(report + "\n")
