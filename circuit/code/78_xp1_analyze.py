"""Experiment 1 analysis -> results/XP1_summary_{model}.json and a printed report.

A  head level: gold-support decrease (-delta_g) vs distractor-support increase (delta_D), per head,
   bare and matched-filler references separately; quadrant counts per population (sign of the mean,
   and 95%-CI classification); per-fact co-occurrence for the frozen set.
B  source-resolved change of each population's summed write, per target:
       vs bare:  P^irr + (U_r^irr - U_r^bare) for r in bos/subj/rel/end
       vs fill:  (P^irr - P^fill) + (U_r^irr - U_r^fill)
C  interventions: joint changes of gold logit / log-prob / P(D) / margins / accuracy, same forward
   pass; share of facts where the SAME edit raises gold log-prob AND lowers P(D).
"""
import json, sys, math, random
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
MODELS = sys.argv[1:] or ["Llama-3.1-8B", "pythia-6.9b", "gpt2-xl"]
REG = ["bos", "prefix", "subj", "rel", "end"]
TI = {"gold": 0, "ans": 1, "lit": 2, "assoc": 3, "lit_long": 4}
rng = np.random.default_rng(0)

def boot(x, B=2000):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) == 0: return (float("nan"),) * 3
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    if len(a) < 5: return float("nan")
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])

def fmt(t, p=3): return f"{t[0]:+.{p}f} [{t[1]:+.{p}f},{t[2]:+.{p}f}]"

for mk in MODELS:
    fp = ROOT / "results" / f"XP1_{mk}.pt"
    if not fp.exists(): continue
    D = torch.load(fp, weights_only=False); rows = D["rows"]; pops = D["pops"]
    nL, nH = rows[0]["runs"]["bare"]["dla"].shape[1:]
    S = {"model": mk, "n": len(rows), "pops": {k: [list(x) for x in v] for k, v in pops.items()}}
    print(f"\n{'#'*96}\n# {mk}: Experiment 1 (held-out n={len(rows)})\n{'#'*96}")

    # ------------------------------------------------ context effect at the output (what must be reversed)
    print("\n-- output-level context effect (irr - bare), mean [95% CI] over facts")
    S["context_effect"] = {}
    for c, dk in (("far", ["ans", "lit", "assoc"]), ("long", ["lit_long"]), ("golden", ["ans", "lit"])):
        ce = {}
        for k in ["gold", "logp", "margin", "acc"] + [f"P_{d}" for d in dk]:
            ce[k] = boot([r["runs"][c]["readout"][k] - r["runs"]["bare"]["readout"][k] for r in rows
                          if k in r["runs"][c]["readout"]])
        base_P = {f"P_{d}": boot([r["runs"][c]["readout"].get(f"P_{d}", float("nan")) for r in rows]) for d in dk}
        S["context_effect"][c] = {"delta": ce, "level": base_P}
        print(f"  {c:7s} " + "  ".join(f"{k} {v[0]:+.3f}" for k, v in ce.items()) +
              "   | level " + "  ".join(f"{k}={v[0]:.4f}" for k, v in base_P.items()))

    # ------------------------------------------------ A. head level
    def deltas(c, ref, t):
        """[facts, nL, nH] delta of DLA to target t: cond c minus reference."""
        i = TI[t]
        return np.stack([(r["runs"][c]["dla"][i].float() - r["runs"][ref]["dla"][i].float()).numpy() for r in rows])
    S["head_level"] = {}
    print("\n-- A. head level: x = gold-support DECREASE, y = distractor-support INCREASE (raw DLA)")
    for c, ref, dts in (("far", "bare", ["ans", "lit", "assoc"]), ("far", "fill_far", ["ans", "lit", "assoc"]),
                        ("long", "bare", ["lit_long"]), ("long", "fill_long", ["lit_long"])):
        G_all = -deltas(c, ref, "gold")
        for t in dts:
            Dd = deltas(c, ref, t)
            okf = ~np.isnan(Dd).reshape(len(rows), -1).any(1)      # facts where this vocabulary exists
            G, Dd = G_all[okf], Dd[okf]
            key = f"{c}|{ref}|{t}"
            gm, dm = G.mean(0), Dd.mean(0)                         # [nL, nH]
            ent = {"gdec_mean": gm.tolist(), "dinc_mean": dm.tolist(), "n_facts": int(okf.sum()), "pops": {}}
            print(f"\n  [{key}] (n={int(okf.sum())})  all-head Spearman(gold decrease, distractor increase) = "
                  f"{spearman(gm.ravel(), dm.ravel()):+.3f}")
            for pn, heads in pops.items():
                g = np.stack([G[:, L, h] for L, h in heads], 1)    # [facts, heads]
                d = np.stack([Dd[:, L, h] for L, h in heads], 1)
                mg, md = g.mean(0), d.mean(0)
                q = dict(both=int(((mg > 0) & (md > 0)).sum()), gold_only=int(((mg > 0) & (md <= 0)).sum()),
                         distr_only=int(((mg <= 0) & (md > 0)).sum()), neither=int(((mg <= 0) & (md <= 0)).sum()))
                # CI-based classification per head
                sig = {"both": 0, "gold_only": 0, "distr_only": 0, "neither": 0}
                for j in range(len(heads)):
                    gl = boot(g[:, j], 1000); dl = boot(d[:, j], 1000)
                    gs, ds = gl[1] > 0, dl[1] > 0
                    sig["both" if gs and ds else "gold_only" if gs else "distr_only" if ds else "neither"] += 1
                # per-fact co-occurrence
                rho_h = [spearman(g[:, j], d[:, j]) for j in range(len(heads))]
                both_frac = float(((g > 0) & (d > 0)).mean()); indep = float((g > 0).mean() * (d > 0).mean())
                gs_, ds_ = g.sum(1), d.sum(1)
                ent["pops"][pn] = dict(quadrant_mean=q, quadrant_ci=sig, rho_heads_median=float(np.nanmedian(rho_h)),
                                       both_frac=both_frac, both_indep=indep,
                                       set_gdec=boot(gs_), set_dinc=boot(ds_), set_rho=spearman(gs_, ds_),
                                       set_both_frac=float(((gs_ > 0) & (ds_ > 0)).mean()),
                                       across_heads_rho=spearman(mg, md))
                e = ent["pops"][pn]
                print(f"    {pn:6s} head means: both {q['both']:2d} gold-only {q['gold_only']:2d} distr-only "
                      f"{q['distr_only']:2d} neither {q['neither']:2d} | CI: both {sig['both']:2d} gold {sig['gold_only']:2d} "
                      f"distr {sig['distr_only']:2d} none {sig['neither']:2d}")
                print(f"           set per fact: gold dec {fmt(e['set_gdec'])}  distr inc {fmt(e['set_dinc'])}  "
                      f"rho(fact) {e['set_rho']:+.2f}  both>0 on {100*e['set_both_frac']:.0f}% of facts | "
                      f"per head rho median {e['rho_heads_median']:+.2f}; head-fact cells both>0 "
                      f"{100*both_frac:.0f}% (indep {100*indep:.0f}%) | across-heads rho {e['across_heads_rho']:+.2f}")
            S["head_level"][key] = ent

    # ------------------------------------------------ B. source-resolved change of the populations' writes
    print("\n-- B. source-resolved change of each population's summed END write (raw DLA; mean over facts)")
    S["sources"] = {}
    for c, ref, dts in (("far", "bare", ["gold", "ans", "lit", "assoc"]), ("far", "fill_far", ["gold", "ans", "lit", "assoc"]),
                        ("long", "bare", ["gold", "lit_long"]), ("long", "fill_long", ["gold", "lit_long"]),
                        ("golden", "bare", ["gold"])):
        for pn, heads in pops.items():
            for t in dts:
                i = TI[t]; comp = {}
                for ri, rn in enumerate(REG):
                    v = [sum(float(r["runs"][c]["dla_reg"][i, L, h, ri]) - float(r["runs"][ref]["dla_reg"][i, L, h, ri])
                             for L, h in heads) for r in rows]
                    comp[rn] = boot(v)
                tot = boot([sum(float(r["runs"][c]["dla"][i, L, h]) - float(r["runs"][ref]["dla"][i, L, h])
                                for L, h in heads) for r in rows])
                S["sources"][f"{c}|{ref}|{pn}|{t}"] = {"total": tot, **comp}
                if pn == "Istar" or t == "gold":
                    lab = "P" if ref == "bare" else "dP"
                    print(f"  {c:6s} vs {ref:9s} {pn:6s} {t:8s} total {tot[0]:+7.3f} = {lab} {comp['prefix'][0]:+7.3f}"
                          f" + dU[subj {comp['subj'][0]:+6.3f} rel {comp['rel'][0]:+6.3f} end {comp['end'][0]:+6.3f}"
                          f" bos {comp['bos'][0]:+6.3f}]")
    # attention mass moved into the prefix
    S["amass"] = {}
    for c in ("far", "long", "golden"):
        for pn, heads in pops.items():
            mv = {rn: boot([np.mean([float(r["runs"][c]["amass"][L, h, ri]) - float(r["runs"]["bare"]["amass"][L, h, ri])
                                     for L, h in heads]) for r in rows]) for ri, rn in enumerate(REG)}
            S["amass"][f"{c}|{pn}"] = mv
            if pn == "Istar":
                print(f"  attention mass change {c:6s} {pn}: " + "  ".join(f"{k} {v[0]:+.3f}" for k, v in mv.items()))

    # ------------------------------------------------ C. interventions, joint readouts
    print("\n-- C. interventions: change vs unedited run, same forward pass (mean [95% CI])")
    S["interventions"] = {}
    for c, dk in (("far", ["ans", "lit", "assoc"]), ("long", ["lit_long"]), ("golden", ["ans", "lit"])):
        ce_logp = np.array([r["runs"][c]["readout"]["logp"] - r["runs"]["bare"]["readout"]["logp"] for r in rows])
        for nm in rows[0]["iv"][c]:
            if nm == "base": continue
            ent = {}
            for k in ["gold", "logp", "margin", "acc"] + [f"P_{d}" for d in dk] + [f"m_vs_{d}" for d in dk]:
                v = [r["iv"][c][nm][k] - r["iv"][c]["base"][k] for r in rows if k in r["iv"][c]["base"]]
                ent[k] = boot(v)
            # same edit, same fact: gold log-prob up AND distractor probability down
            for d in dk:
                a_ = np.array([r["iv"][c][nm]["logp"] - r["iv"][c]["base"]["logp"] for r in rows])
                b_ = np.array([r["iv"][c][nm].get(f"P_{d}", np.nan) - r["iv"][c]["base"].get(f"P_{d}", np.nan) for r in rows])
                ent[f"joint_{d}"] = dict(frac_logp_up_P_down=float(((a_ > 0) & (b_ < 0)).mean()),
                                         frac_logp_up=float((a_ > 0).mean()), frac_P_down=float((b_ < 0).mean()),
                                         rho=spearman(a_, -b_))
            ent["logp_recovery_ratio"] = float(ent["logp"][0] / -ce_logp.mean()) if c != "golden" and ce_logp.mean() < 0 else float("nan")
            S["interventions"][f"{c}|{nm}"] = ent
            # distractor LOGIT change (P(D) is ~1e-3 at the output): d max_D logit = d gold - d (gold - max_D)
            for d in dk:
                ent[f"dlogit_{d}"] = boot([(r["iv"][c][nm]["gold"] - r["iv"][c]["base"]["gold"]) -
                                           (r["iv"][c][nm].get(f"m_vs_{d}", np.nan) - r["iv"][c]["base"].get(f"m_vs_{d}", np.nan))
                                           for r in rows])
                a_ = np.array([r["iv"][c][nm]["gold"] - r["iv"][c]["base"]["gold"] for r in rows])
                b_ = np.array([(r["iv"][c][nm]["gold"] - r["iv"][c]["base"]["gold"]) -
                               (r["iv"][c][nm].get(f"m_vs_{d}", np.nan) - r["iv"][c]["base"].get(f"m_vs_{d}", np.nan)) for r in rows])
                ent[f"joint_logit_{d}"] = float(((a_ > 0) & (b_ < 0)).mean())
            dd = dk[0] if c != "far" else "lit"
            j = ent[f"joint_{dd}"]
            dl = ent[f"dlogit_{dd}"]
            print(f"  {c:6s} {nm:20s} gold {ent['gold'][0]:+.3f}  logp {fmt(ent['logp'])}  margin {ent['margin'][0]:+.3f}  "
                  f"acc {ent['acc'][0]:+.3f} | P_{dd} {ent[f'P_{dd}'][0]:+.4f}  max-{dd} logit {fmt(dl)} | same fact: "
                  f"logp up & P down {100*j['frac_logp_up_P_down']:.0f}%, gold up & {dd}-logit down "
                  f"{100*ent[f'joint_logit_{dd}']:.0f}%")
    json.dump(S, (ROOT / "results" / f"XP1_summary_{mk}.json").open("w"))
    print(f"\n  -> results/XP1_summary_{mk}.json")
