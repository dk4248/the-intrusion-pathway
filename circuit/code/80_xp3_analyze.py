"""Experiment 3 analysis -> results/XP3_summary_{model}.json and a printed report.

For each intervention on the receivers (END-mediated: q_cop / q_attn / q_mlp / q_full, bare and filler
donors separately; direct access: acc_mask / acc_kvfill), the chain
    query change (cosine, relative norm) -> prefix attention change -> distractor-token attention change
    -> receiver write change (gold, D) -> final change (gold logit, log-prob, margin, P(D))
at set level for Istar / Ronly / CTRL receivers, and per Istar receiver (split by whether any copier
sits below it). Self-patch controls must be exactly 0.
"""
import json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
MODELS = sys.argv[1:] or ["Llama-3.1-8B", "pythia-6.9b", "gpt2-xl"]
rng = np.random.default_rng(0)

def boot(x, B=2000):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) == 0: return (float("nan"),) * 3
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def f3(t, p=3): return f"{t[0]:+.{p}f} [{t[1]:+.{p}f},{t[2]:+.{p}f}]"

for mk in MODELS:
    fp = ROOT / "results" / f"XP3_{mk}.pt"
    if not fp.exists(): continue
    D = torch.load(fp, weights_only=False); rows = D["rows"]; pops = D["pops"]; IV = D["IV"]; COP = D["copier"]
    S = {"model": mk, "n": len(rows), "set": {}, "per_receiver": {}}
    print(f"\n{'#'*96}\n# {mk}: Experiment 3 -- copier route vs direct access (held-out n={len(rows)})\n{'#'*96}")
    sp = [abs(r["res"][c][(k, "Istar", "set")]["gold"]) for r in rows for c in ("far", "long")
          for k in ("self_cop", "self_full")]
    print(f"  self-patch max |d gold| = {max(sp):.1e} (must be ~0)")
    istar = pops["Istar"]
    has_cop = {h: any(c[0] < h[0] for c in COP) for h in istar}
    print(f"  Istar receivers with >=1 copier below them: {sum(has_cop.values())}/{len(istar)}  "
          f"(copier layers {sorted({c[0] for c in COP})})")
    for c, dk in (("far", ["ans", "lit"]), ("long", ["lit_long"])):
        print(f"\n-- {c}: set level")
        print(f"  {'recv':6s} {'intervention':12s} {'cos(q)':>7s} {'|dq|/|q|':>8s} {'d a_pfx':>8s} {'d a_D':>7s} "
              f"{'dw gold':>8s} " + " ".join(f"{'dw '+d:>9s}" for d in dk) + f" {'d logp [95% CI]':>26s} {'d gold':>7s} "
              f"{'d margin':>8s} " + " ".join(f"{'dlogit_'+d:>12s}" for d in dk))
        for pn in pops:
            for iv in IV:
                key = (iv, pn, "set")
                if key not in rows[0]["res"][c]: continue
                R = [r["res"][c][key] for r in rows if key in r["res"][c]]
                ent = {"cos": boot([x["chain"]["dq_cos"] for x in R]), "rel": boot([x["chain"]["dq_rel"] for x in R]),
                       "da": boot([x["chain"]["a1"] - x["chain"]["a0"] for x in R]),
                       "dD": boot([x["chain"]["d1"] - x["chain"]["d0"] for x in R]),
                       "dw": {t: boot([x["dw"].get(t, np.nan) for x in R]) for t in ["gold"] + dk}}
                for k in ["gold", "logp", "margin", "acc"] + [f"P_{d}" for d in dk]:
                    ent[k] = boot([x.get(k, np.nan) for x in R])
                for d in dk:
                    ent[f"dlogit_{d}"] = boot([x["gold"] - x.get(f"m_vs_{d}", np.nan) for x in R])
                S["set"][f"{c}|{pn}|{iv}"] = ent
                print(f"  {pn:6s} {iv:12s} {ent['cos'][0]:7.3f} {ent['rel'][0]:8.3f} {ent['da'][0]:+8.3f} {ent['dD'][0]:+7.3f} "
                      f"{ent['dw']['gold'][0]:+8.3f} " + " ".join(f"{ent['dw'][d][0]:+9.3f}" for d in dk) +
                      f" {f3(ent['logp']):>26s} {ent['gold'][0]:+7.3f} {ent['margin'][0]:+8.3f} " +
                      " ".join(f"{ent[f'dlogit_{d}'][0]:+12.3f}" for d in dk))
        print(f"  per Istar receiver (mean d logp; receivers with / without a copier below):")
        for iv in IV:
            with_c, without = [], []
            for h in istar:
                key = (iv, "Istar", h)
                v = [r["res"][c][key]["logp"] for r in rows if key in r["res"][c]]
                if not v: continue
                (with_c if has_cop[h] else without).append(np.mean(v))
                S["per_receiver"][f"{c}|{iv}|{h[0]}.{h[1]}"] = {"logp": float(np.mean(v)), "has_copier": has_cop[h],
                    "da": float(np.mean([r["res"][c][key]["chain"]["a1"] - r["res"][c][key]["chain"]["a0"] for r in rows if key in r["res"][c]]))}
            print(f"    {iv:12s} with copier: n={len(with_c):2d} mean {np.mean(with_c) if with_c else np.nan:+.4f} "
                  f"max {max(with_c) if with_c else np.nan:+.4f} | without: n={len(without):2d} "
                  f"mean {np.mean(without) if without else np.nan:+.4f}")
    json.dump(S, (ROOT / "results" / f"XP3_summary_{mk}.json").open("w"))
    print(f"\n  -> results/XP3_summary_{mk}.json")
