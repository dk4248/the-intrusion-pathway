"""Tables D.2 and D.3: GSM-IC accuracy under residual patching.

Table D.2 (single-token argmax, run_gsm_ic_defense.py): per model, clean / +irrelevant accuracy and the
patched accuracy at the best layer of the sweep; Δ = clean − irrelevant; recovery =
(patched − irrelevant) / (clean − irrelevant). Models whose clean accuracy is at or below
--min-clean are listed as omitted (the paper omits them: near-floor clean accuracy).

Table D.3 (per-step chain of thought, run_gsm_ic_cot_perstep.py): clean / +irrelevant / +patch
accuracy and recovery.

Usage:
    python table_gsm_ic.py [--results ../results] [--results-cot ../results_cot_perstep]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(ROOT / "results"))
ap.add_argument("--results-cot", default=str(ROOT / "results_cot_perstep"))
ap.add_argument("--min-clean", type=float, default=0.16)
args = ap.parse_args()


def recovery(clean, dis, pat):
    return None if clean <= dis else 100.0 * (pat - dis) / (clean - dis)


def fmt_rec(r):
    return "undefined" if r is None else f"{r:.1f}%"


print("Table D.2 — GSM-IC single-token argmax (patched = best layer of the sweep)")
print(f"{'model':<22}{'clean':>7}{'+irr':>7}{'+patch':>8}{'best L':>8}{'Δ':>8}{'rec.':>10}")
for p in sorted(Path(args.results).glob("*.json")):
    d = json.load(open(p))
    recs = d["records"]
    n = len(recs)
    if n == 0:
        continue
    clean = sum(r.get("n_correct", 0) for r in recs) / n
    dis = sum(r.get("w_correct", 0) for r in recs) / n
    layers = sorted({L for r in recs for L in (r.get("per_layer") or {})}, key=int)
    acc = {L: sum(((r.get("per_layer") or {}).get(L) or {}).get("correct", 0) for r in recs) / n
           for L in layers}
    best_L = max(acc, key=acc.get)
    pat = acc[best_L]
    tag = "" if clean > args.min_clean else "   (omitted: clean accuracy at floor)"
    print(f"{d['model_short']:<22}{clean:7.3f}{dis:7.3f}{pat:8.3f}{best_L:>8}{clean - dis:+8.3f}"
          f"{fmt_rec(recovery(clean, dis, pat)):>10}{tag}")

print("\nTable D.3 — GSM-IC per-step chain of thought")
print(f"{'model':<22}{'n':>5}{'clean':>7}{'+irr':>7}{'+patch':>8}{'rec.':>10}")
for p in sorted(Path(args.results_cot).glob("*.json")):
    d = json.load(open(p))
    c, w, pt = d["clean_acc"], d["distract_acc"], d["patched_acc"]
    print(f"{d['model_short']:<22}{d['n_problems']:>5}{c:7.2f}{w:7.2f}{pt:8.2f}{fmt_rec(recovery(c, w, pt)):>10}")
