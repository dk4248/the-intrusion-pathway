"""Matched base-vs-instruct downstream table (Table E.5).

Computes mean recovery on the SAME task points for Llama-3.1-8B (base) and
Llama-3.1-8B-Instruct, so the two columns are apples-to-apples. Only task points
with a real distractor drop (clean > distract) enter the mean; n is reported.

Tiers:
  oracle  = run_downstream.py per-step resid_pre patch @ L30   (fields clean_acc/w_acc/patched_acc)
  K10     = run_downstream_head_ko.py, top-10 clean_intrusion_set (clean_acc/distract_acc/hko_acc)
GSM-IC (single-token, gsm_ic/) is reported separately: it has one distractor,
no k axis, and its recovery is only defined where clean > distract.
"""
import json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # repository root

DS = REPO / "downstream/results"
IC = REPO / "gsm_ic/results"
OUT = REPO / "transfer_and_sensitivity/results"
BASE, INST = "Llama-3.1-8B", "Llama-3.1-8B-Instruct"

FAMILIES = {
    "GSM-DC":   ["gsmdc_irr4", "gsmdc_irr7"],
    "HotpotQA": ["hotpotqa_k4", "hotpotqa_k8"],
    "PubMedQA": ["pubmedqa_k2", "pubmedqa_k4"],
}


def rec(clean, distract, interv):
    if None in (clean, distract, interv):
        return None
    d = clean - distract
    return (interv - distract) / d if d > 1e-9 else None   # undefined if no real drop


def oracle(model, task):
    p = DS / f"{model}__{task}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return rec(d.get("clean_acc"), d.get("w_acc"), d.get("patched_acc"))


def hko(model, task, sfx="K10_clean"):
    p = DS / f"{model}__{task}__hko_{sfx}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return rec(d.get("clean_acc"), d.get("distract_acc"), d.get("hko_acc"))


def gsmic(model):
    p = IC / f"{model}.json"
    if not p.exists():
        return None, None, None
    d = json.loads(p.read_text())
    recs = d["records"]
    n = len(recs)
    clean = sum(r["n_correct"] for r in recs) / n
    distract = sum(r["w_correct"] for r in recs) / n
    best = None
    for L in [str(x) for x in d["layers_swept"]]:
        pa = sum(r["per_layer"][L]["correct"] for r in recs) / n
        rr = rec(clean, distract, pa)
        if rr is not None and (best is None or rr > best[1]):
            best = (L, rr)
    return clean, distract, best


def agg(fn, model, tasks):
    vals = [v for v in (fn(model, t) for t in tasks) if v is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def cell(model, tasks, fn):
    m, n = agg(fn, model, tasks)
    return "n/a" if m is None else f"{m*100:.1f}% (n={n})"


rows = []
for fam, tasks in FAMILIES.items():
    rows.append(("Oracle patch @L30", fam,
                 cell(BASE, tasks, oracle), cell(INST, tasks, oracle)))
for fam, tasks in FAMILIES.items():
    rows.append(("Deployable knockout (K=10)", fam,
                 cell(BASE, tasks, hko), cell(INST, tasks, hko)))

# GSM-IC
bc, bd, bb = gsmic(BASE)
ic, idd, ib = gsmic(INST)


def ic_cell(c, dd, b):
    if c is None:
        return "n/a"
    if b is None:
        return f"undefined (clean {c:.3f} ≤ distract {dd:.3f})"
    return f"{b[1]*100:.1f}% (L{b[0]})"


md = ["# Matched base-vs-instruct downstream recovery (Table E.5)", "",
      "Same task points for both checkpoints. recovery = (intervened − distract)/(clean − distract); "
      "only points with a real distractor drop (clean > distract) enter the mean.", "",
      "| Intervention (tier) | Task | Llama-3.1-8B (base) | Llama-3.1-8B-Instruct |",
      "|---|---|---|---|"]
md.append(f"| Oracle patch @L30 | GSM-IC (single-token) | {ic_cell(bc,bd,bb)} | {ic_cell(ic,idd,ib)} |")
for tier, fam, b, i in rows:
    md.append(f"| {tier} | {fam} | {b} | {i} |")
md += ["", f"GSM-IC base: clean={bc:.3f} distract={bd:.3f}; instruct: clean={ic:.3f} distract={idd:.3f}.",
       "", "Task points: " + "; ".join(f"{f} = {', '.join(t)}" for f, t in FAMILIES.items())]

p = OUT / "e5_base_vs_instruct_matched.md"
p.write_text("\n".join(md))
print("\n".join(md))
print(f"\nWROTE {p}")
