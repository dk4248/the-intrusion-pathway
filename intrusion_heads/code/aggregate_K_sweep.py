"""Aggregate the LRE K-sweep results into a Pareto curve.

For each (model, K, set):
  irrelevant_recovery = mean(delta_lg under short_irr) - mean(delta_lg under nctx)
                      = mean(delta_lg) for short_irr (since nctx delta is 0 by design)
                      (same for long_irr)
  golden_cost = -mean(delta_lg under golden)   (positive = hurts golden)

Save: results/K_sweep_pareto.json
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
REPO = Path(__file__).resolve().parents[2]  # repository root

RES = REPO / "intrusion_heads/results/knockout_K_sweep"
OUT = RES.parent

MODELS = ["Llama-3.1-8B", "Mistral-7B-v0.1", "Qwen2.5-7B", "pythia-6.9b"]
K_VALUES = [5, 8, 10, 12, 15, 20]
SETS = ["clean", "full"]


def main():
    out = {"models": {}}
    for m in MODELS:
        out["models"][m] = {}
        print(f"\n=== {m} ===")
        print(f"  {'set':<6} {'K':>3} | "
              f"{'Δ_sh_irr':>10} {'Δ_lg_irr':>10} {'Δ_gold':>10} "
              f"{'rec_short':>10} {'cost_gold':>10} {'net':>10}")
        for set_name in SETS:
            for K in K_VALUES:
                p = RES / f"{m}__K{K}_{set_name}.json"
                if not p.exists():
                    continue
                d = json.load(p.open())
                by_cond = defaultdict(list)
                for r in d["records"]:
                    by_cond[r["condition"]].append(r["delta_lg"])
                rec_short = float(np.mean(by_cond["short_irr"]))
                rec_long = float(np.mean(by_cond["long_irr"]))
                gold = float(np.mean(by_cond["golden"]))
                cost_gold = -gold   # positive = hurts
                net = (rec_short + rec_long) / 2 + gold   # net signed effect averaging irr - cost
                out["models"][m].setdefault(set_name, {})[str(K)] = {
                    "rec_short": rec_short, "rec_long": rec_long,
                    "delta_gold": gold, "cost_gold": cost_gold,
                    "net": net,
                    "n_top_intrusion": d.get("n_top_intrusion", K),
                }
                print(f"  {set_name:<6} {K:>3} | "
                      f"{rec_short:>+10.3f} {rec_long:>+10.3f} {gold:>+10.3f} "
                      f"{rec_short:>+10.3f} {cost_gold:>+10.3f} {net:>+10.3f}")
    (OUT / "K_sweep_pareto.json").write_text(json.dumps(out, indent=2))
    print(f"\nWROTE {OUT}/K_sweep_pareto.json")


if __name__ == "__main__":
    main()
