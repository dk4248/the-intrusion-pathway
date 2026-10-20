"""C1 analysis -> summary tables for the MLP-enrichment experiments.

E1.1  subject enrichment: MLP write onto A_s above a norm-matched control, by layer.
E1.3  distractor enrichment: same at the distractor position, against A_d and against the
      identity direction (the distractor token itself).
E1.2  causal: does cancelling those MLP writes change the answer, and is add-back exact?
E1.4  preservation: is subject enrichment the same with and without an irrelevant prefix?
E1.5  origin: does the MLP's association content survive killing earlier attention into
      that position?
"""
import json, sys, random, argparse
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent
rng = random.Random(0)

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="Llama-3.1-8B,pythia-6.9b,gpt2-xl")
ap.add_argument("--band", type=float, default=0.05, help="equivalence half-width, E1.4")
a = ap.parse_args()

def boot(per_sample, n=2000):
    ks = list(per_sample)
    if not ks:
        return (float("nan"),) * 3
    def mean(sel):
        v = [x for k in sel for x in per_sample[k]]
        return sum(v) / len(v) if v else float("nan")
    obs = mean(ks)
    d = sorted(mean([rng.choice(ks) for _ in ks]) for _ in range(n))
    return obs, d[int(0.025 * n)], d[int(0.975 * n)]

def group(rows, keyf, valf):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = valf(r)
        if v is None:
            continue
        g[keyf(r)][r["sid"]].append(v)
    return g

for mk in a.models.split(","):
    ep = ROOT / "results" / f"E1_enrich_{mk}.json"
    cp = ROOT / "results" / f"E1_causal_{mk}.json"
    if not ep.exists():
        continue
    E = json.load(ep.open())["rows"]
    nL = max(r["layer"] for r in E) + 1
    print(f"\n{'#'*100}\n# {mk}   ({nL} layers)\n{'#'*100}")

    # ---------- E1.1 / E1.3 : MLP write above a norm-matched control ----------
    def excess(cond, position, bank, ctrl="ctrl", key="dmlp"):
        return group([r for r in E if r["cond"] == cond and r["position"] == position],
                     lambda r: r["layer"],
                     lambda r: (r[f"{key}_{bank}"] - r.get(f"{key}_{ctrl}", 0.0))
                     if f"{key}_{bank}" in r else None)

    def line(tag, g):
        """Report the peak over layers BEFORE the last one separately from the final layer.
        The final layer's MLP at a position directly shapes that position's own next-token
        prediction, which at the subject position is entity-related by definition; that is a
        different phenomenon from mid-layer enrichment available for later extraction."""
        if not g:
            print(f"  {tag:46s} (no data)"); return
        st = {L: boot(v) for L, v in g.items()}
        pre = {L: v for L, v in st.items() if L < nL - 1}
        pk = max(pre, key=lambda L: pre[L][0]) if pre else max(st, key=lambda L: st[L][0])
        o, lo, hi = st[pk]
        fin = st.get(nL - 1, (float("nan"),) * 3)
        marks = " ".join(f"L{L}:{st[L][0]:+.2f}" for L in range(0, nL, max(1, nL // 6)))
        print(f"  {tag:44s} peak(<last) L{pk:<3d} {o:+.3f} [{lo:+.3f},{hi:+.3f}] "
              f"{'CI>0' if lo > 0 else 'n.s.'} | final L{nL-1}: {fin[0]:+.3f}")
        print(f"     {marks}")

    print("\n-- E1.1  SUBJECT position: MLP write onto subject attributes (minus control)")
    for cond in ["nctx", "far_irr", "long_irr"]:
        line(f"A_s @subj [{cond}] vs norm-ctrl", excess(cond, "subj", "A_s"))
    line("A_s @subj [nctx]    vs OTHER subject's attrs", excess("nctx", "subj", "A_s", "A_sX"))
    line("A_s @subj [far_irr] vs OTHER subject's attrs", excess("far_irr", "subj", "A_s", "A_sX"))
    # vs norm-matched control, AND vs another entity's wiki cloud (controls for the
    # encyclopedic boilerplate that every Wikipedia summary shares)
    line("A_sW (wiki) @subj [nctx] vs norm-ctrl", excess("nctx", "subj", "A_sW", "ctrlW"))
    line("A_sW (wiki) @subj [nctx] vs OTHER entity", excess("nctx", "subj", "A_sW", "A_sO"))
    line("A_sW (wiki) @subj [far_irr] vs OTHER entity", excess("far_irr", "subj", "A_sW", "A_sO"))

    print("\n-- E1.1b Geva attributes-rate: fraction of the MLP's top-20 in the attribute set")
    for cond in ["nctx", "far_irr"]:
        g = group([r for r in E if r["cond"] == cond and r["position"] == "subj"],
                  lambda r: r["layer"], lambda r: r.get("rate_A_sW"))
        gc = group([r for r in E if r["cond"] == cond and r["position"] == "subj"],
                   lambda r: r["layer"], lambda r: r.get("rate_ctrlW"))
        if g:
            st = {L: boot(v)[0] for L, v in g.items()}
            sc = {L: boot(v)[0] for L, v in gc.items()} if gc else {}
            pk = max(st, key=lambda L: st[L])
            print(f"  [{cond:8s}] peak rate L{pk}: {st[pk]:.3f}  "
                  f"(control {sc.get(pk, float('nan')):.3f})")

    print("\n-- E1.3  DISTRACTOR position: does the MLP enrich an IRRELEVANT token?")
    line("A_d @dstr vs norm-ctrl", excess("far_irr", "dstr", "A_d"))
    line("A_d @dstr vs OTHER distractor's category", excess("far_irr", "dstr", "A_d", "A_dX"))
    line("A_dW (wiki) @dstr vs norm-ctrl", excess("far_irr", "dstr", "A_dW", "ctrlW"))
    line("A_dW (wiki) @dstr vs OTHER distractor", excess("far_irr", "dstr", "A_dW", "A_dO"))
    line("ident (the token itself)   @dstr", excess("far_irr", "dstr", "ident"))
    line("A_s  (query subj attrs)    @dstr  <- expect ~0", excess("far_irr", "dstr", "A_s"))

    # ---------- E1.4 : preservation ----------
    print("\n-- E1.4  is subject enrichment PRESERVED under an irrelevant prefix?")
    base = defaultdict(dict)
    for r in E:
        if r["position"] == "subj" and "dmlp_A_s" in r:
            base[(r["sid"], r["layer"])][r["cond"]] = r["dmlp_A_s"] - r.get("dmlp_ctrl", 0)
    for cond in ["far_irr", "long_irr"]:
        diff = defaultdict(list)
        for (sid, L), v in base.items():
            if "nctx" in v and cond in v:
                diff[sid].append(v[cond] - v["nctx"])
        o, lo, hi = boot(diff)
        verdict = "EQUIVALENT" if (lo > -a.band and hi < a.band) else "NOT equivalent"
        print(f"   {cond:9s} - nctx: {o:+.4f} [{lo:+.4f},{hi:+.4f}]  band +/-{a.band} -> {verdict}")

    # ---------- E1.2 / E1.5 : causal ----------
    if not cp.exists():
        print("\n  (causal results not present yet)")
        continue
    C = json.load(cp.open())["rows"]
    gate = [r for r in C if r["mode"] == "addback"]
    if gate:
        print(f"\n-- GATE  add-back reproduces the clean run: "
              f"max|d gold logit| = {max(abs(r['d_gold_logit']) for r in gate):.2e} "
              f"over n={len(gate)}")
    print("\n-- E1.2/E1.3c  effect of CANCELLING the MLP write (d gold logit at END)")
    print(f"   {'position':10s} {'band':9s} {'zero [95% CI]':>26s} {'resample':>10s} "
          f"{'d local assoc':>14s}")
    for cond in ["nctx", "far_irr"]:
        print(f"   --- {cond} ---")
        for position in ["subj", "dstr", "qry_next", "ctrl_pfx", "ctrl_null"]:
            for band in ["early", "mid", "late_mid", "late"]:
                z = group([r for r in C if r["cond"] == cond and r["position"] == position
                           and r["band"] == band and r["mode"] == "zero"],
                          lambda r: 0, lambda r: r.get("d_gold_logit"))
                rs = group([r for r in C if r["cond"] == cond and r["position"] == position
                            and r["band"] == band and r["mode"] == "resample"],
                           lambda r: 0, lambda r: r.get("d_gold_logit"))
                lo_ = group([r for r in C if r["cond"] == cond and r["position"] == position
                             and r["band"] == band and r["mode"] == "zero"],
                            lambda r: 0, lambda r: r.get("d_local"))
                if not z:
                    continue
                o, l, h = boot(z[0]); ro = boot(rs[0])[0] if rs else float("nan")
                lc = boot(lo_[0])[0] if lo_ else float("nan")
                print(f"   {position:10s} {band:9s} {o:+9.3f} [{l:+7.3f},{h:+7.3f}] "
                      f"{ro:+10.3f} {lc:+14.3f}")

    print("\n-- E1.5  MLP content after killing ALL attention into that position beforehand")
    for cond in ["nctx", "far_irr"]:
        for position in ["subj", "dstr"]:
            g = group([r for r in C if r["cond"] == cond and r["position"] == position
                       and r["mode"] == "attnoff_before"],
                      lambda r: r["band"], lambda r: r.get("d_local"))
            if not g:
                continue
            parts = " ".join(f"{b}:{boot(v)[0]:+.3f}" for b, v in sorted(g.items()))
            print(f"   [{cond:8s}] {position:6s} d local assoc -> {parts}")
