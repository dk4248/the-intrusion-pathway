"""Experiment 2 analysis -> results/XP2_summary_{model}.json and a printed report.

Per population and condition, the four inserted writes (rm = prefix write removed, amp = non-prefix
amplified with prefix kept, both = local hard mask, gain = same scalar gain on the full write) and the
natural mask, at set level (all heads at once) and per head. Readouts are changes vs the unedited run
of the SAME batch: gold logit, gold log-prob, margin, accuracy, P(D_*), and the immediate change of
the edited write on gold and each D. Interactions: set-level effect vs the sum of single-head effects.
Denominators for recovery shares come from Experiment 1 (same facts, same fp32 model).
Checks printed: orig (every head patched with its own write) must be 0; single-head both == mask.
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
MODELS = sys.argv[1:] or ["Llama-3.1-8B", "pythia-6.9b", "gpt2-xl"]
rng = np.random.default_rng(0)
WR = ["rm", "amp", "both", "gain", "mask"]

def boot(x, B=2000):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) == 0: return (float("nan"),) * 3
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def f3(t, p=3): return f"{t[0]:+.{p}f} [{t[1]:+.{p}f},{t[2]:+.{p}f}]"

for mk in MODELS:
    fp = ROOT / "results" / f"XP2_{mk}.pt"
    if not fp.exists(): continue
    D = torch.load(fp, weights_only=False); rows = D["rows"]; pops = D["pops"]
    X1 = {r["sid"]: r for r in torch.load(ROOT / "results" / f"XP1_{mk}.pt", weights_only=False)["rows"]}
    S = {"model": mk, "n": len(rows), "set": {}, "single": {}, "interaction": {}, "checks": {}}
    print(f"\n{'#'*96}\n# {mk}: Experiment 2 -- four writes (held-out n={len(rows)})\n{'#'*96}")
    # checks
    orig = [abs(r["res"][c][("orig|ctx", pn, "set")]["gold"]) for r in rows for c in ("far", "long", "golden")
            for pn in pops]
    bm = [abs(r["res"]["far"][("both|ctx", "Istar", h)]["gold"] - r["res"]["far"][("mask|ctx", "Istar", h)]["gold"])
          for r in rows for h in pops["Istar"] if ("both|ctx", "Istar", h) in r["res"]["far"]]
    S["checks"] = {"orig_max_abs_dgold": float(max(orig)), "single_both_vs_mask_max_abs": float(max(bm)) if bm else None}
    print(f"  checks: orig max|dgold| {max(orig):.1e} (must be ~0);  single-head |both - mask| max "
          + (f"{max(bm):.1e}" if bm else "n/a (set-only run)"))

    def loss(c):          # context effect on log-prob, from Experiment 1 (irr - bare)
        return np.array([X1[r["sid"]]["runs"][c]["readout"]["logp"] - X1[r["sid"]]["runs"]["bare"]["readout"]["logp"]
                         for r in rows])
    for c, dk in (("far", ["ans", "lit", "assoc"]), ("long", ["lit_long"]), ("golden", ["ans", "lit"]),
                  ("ev_irr", ["ans", "lit", "assoc"])):
        spans = ["ctx", "oracle"] if c == "ev_irr" else ["ctx"]
        ref = loss(c) if c in ("far", "long") else None
        if c == "ev_irr":
            # harm of the irrelevant sentence when evidence is present: ev_irr - golden (both unedited)
            ev = np.array([r["res"]["ev_irr"]["base"]["logp"] for r in rows])
            gd = np.array([X1[r["sid"]]["runs"]["golden"]["readout"]["logp"] for r in rows])
            bb = np.array([X1[r["sid"]]["runs"]["bare"]["readout"]["logp"] for r in rows])
            ref = ev - gd
            S["ev_irr_levels"] = {"bare": boot(bb), "golden": boot(gd), "ev_irr": boot(ev), "harm": boot(ref)}
            print(f"\n  gold log-prob levels: bare {bb.mean():+.3f} | golden {gd.mean():+.3f} | golden+irrelevant "
                  f"{ev.mean():+.3f}  -> irrelevant harm with evidence present {ref.mean():+.3f}")
        print(f"\n-- {c}" + (f"   (context effect on gold log-prob: {ref.mean():+.3f})" if ref is not None else ""))
        print(f"  {'pop':6s} {'write':12s} {'d gold':>8s} {'d logp [95% CI]':>26s} {'d margin':>9s} {'d acc':>7s} "
              + " ".join(f"{'dlogit_'+d:>13s}" for d in dk) + "  | immediate write: gold  " + "  ".join(dk))
        for pn in pops:
            for sp in spans:
                for w in WR:
                    key = (f"{w}|{sp}", pn, "set")
                    if key not in rows[0]["res"][c]: continue
                    ent = {k: boot([r["res"][c][key].get(k, np.nan) for r in rows])
                           for k in ["gold", "logp", "margin", "acc"] + [f"P_{d}" for d in dk]}
                    for d in dk:     # strongest-distractor logit change = d gold - d (gold - max_D)
                        ent[f"dlogit_{d}"] = boot([r["res"][c][key]["gold"] - r["res"][c][key].get(f"m_vs_{d}", np.nan)
                                                   for r in rows])
                    if w != "mask":
                        ent["imm"] = {t: boot([r["res"][c][key]["imm"].get(t, np.nan) for r in rows])
                                      for t in ["gold"] + dk}
                    if ref is not None: ent["logp_share"] = float(ent["logp"][0] / -ref.mean())
                    S["set"][f"{c}|{pn}|{w}|{sp}"] = ent
                    imm = ("  " + f"{ent['imm']['gold'][0]:+7.3f} " + " ".join(f"{ent['imm'][d][0]:+7.3f}" for d in dk)) if "imm" in ent else ""
                    print(f"  {pn:6s} {w+'|'+sp:12s} {ent['gold'][0]:+8.3f} {f3(ent['logp']):>26s} {ent['margin'][0]:+9.3f} "
                          f"{ent['acc'][0]:+7.3f} " + " ".join(f"{ent[f'dlogit_{d}'][0]:+13.3f}" for d in dk) + imm)
        # single heads and interactions (not for ev_irr, which is set-level only)
        if c == "ev_irr": continue
        if not any((f"rm|ctx", pn, h) in rows[0]["res"][c] for pn, hs in pops.items() for h in hs): continue   # set-only run
        for pn, heads in pops.items():
            for w in ["rm", "amp", "both", "gain"]:
                per_head = []
                for h in heads:
                    key = (f"{w}|ctx", pn, h)
                    v = [r["res"][c][key]["logp"] for r in rows if key in r["res"][c]]
                    per_head.append(np.mean(v) if v else np.nan)
                per_head = np.array(per_head)
                S["single"][f"{c}|{pn}|{w}"] = {"logp_per_head": per_head.tolist(), "heads": [list(h) for h in heads]}
                # interaction: set effect vs sum of singles, per fact
                sums = [sum(r["res"][c][(f"{w}|ctx", pn, h)]["logp"] for h in heads) for r in rows]
                sets = [r["res"][c][(f"{w}|ctx", pn, "set")]["logp"] for r in rows]
                S["interaction"][f"{c}|{pn}|{w}"] = {"sum_single": boot(sums), "set": boot(sets),
                                                     "set_minus_sum": boot(np.array(sets) - np.array(sums))}
        print(f"  single heads (mean d logp per head; set effect vs sum of single-head effects):")
        for pn in pops:
            line = []
            for w in ["rm", "amp", "both", "gain"]:
                ph = np.array(S["single"][f"{c}|{pn}|{w}"]["logp_per_head"]); it = S["interaction"][f"{c}|{pn}|{w}"]
                line.append(f"{w}: heads>0 {int((ph > 0).sum())}/{len(ph)} max {np.nanmax(ph):+.3f} | "
                            f"set {it['set'][0]:+.3f} vs sum {it['sum_single'][0]:+.3f}")
            print(f"    {pn:6s} " + "  ||  ".join(line))
    json.dump(S, (ROOT / "results" / f"XP2_summary_{mk}.json").open("w"))
    print(f"\n  -> results/XP2_summary_{mk}.json")
