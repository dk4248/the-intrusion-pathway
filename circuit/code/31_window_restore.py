"""Is the middle block load-bearing, or is its divergence injected and then cancelled?

Measuring WHERE the END-position divergence appears cannot separate
"this layer caused the damage" from "this layer wrote a difference that later cancels out".
This is the causal version.

For a window of layers W, run the prefixed prompt but overwrite attn_out and/or mlp_out at the
END position with the values those sublayers produced in the NO-CONTEXT run. Sublayers in W then
inject exactly what they would have injected with no context; everything outside W runs normally
on the modified stream. Measure the gold logit and the scale-free margin at END.

    recovery(W) = (metric_patched - metric_ctx) / (metric_nctx - metric_ctx)

recovery ~ 1 for a window means the damage was injected there. recovery ~ 0 means the window's
divergence does not survive to the output -- that window IS effectively a passthrough for damage.
Patching W = all layers must give recovery ~ 1; it is the correctness check on the whole setup.
"""
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=40)
ap.add_argument("--conds", default="golden,far_irr,long_irr")
a = ap.parse_args()
m = load(a.model, a.device); nL = m.cfg.n_layers
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"][: a.limit]
CONDS = a.conds.split(",")

T3 = nL // 3
WINDOWS = {"early": range(0, T3), "mid": range(T3, 2 * T3), "late": range(2 * T3, nL),
           "all": range(0, nL)}

def fid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False); return i[0] if i else None

def readout(logits, gid):
    lg = logits[0, -1].float()
    gold = float(lg[gid])
    other = lg.clone(); other[gid] = -1e9
    return gold, gold - float(other.max())          # raw gold logit, scale-free margin

res = defaultdict(lambda: defaultdict(list))
with torch.no_grad():
    for s in S:
        gid = fid(s["gold"])
        if gid is None: continue
        tn = m.to_tokens(s["query"])
        ln, cn = m.run_with_cache(tn)
        a_n = {L: cn["attn_out", L][0, -1].clone() for L in range(nL)}
        m_n = {L: cn["mlp_out", L][0, -1].clone() for L in range(nL)}
        del cn
        g_n, mg_n = readout(ln, gid)

        for cond in CONDS:
            v = s["variants"].get(cond, [])
            if not v: continue
            tc = m.to_tokens(f"{v[0]['context']} {s['query']}")
            g_c, mg_c = readout(m(tc), gid)
            res[cond]["_nctx"].append((g_n, mg_n)); res[cond]["_ctx"].append((g_c, mg_c))

            for wname, W in WINDOWS.items():
                for part in ("both", "attn", "mlp"):
                    if wname == "all" and part != "both": continue
                    def mk(src):
                        def hk(x, hook):
                            x[:, -1, :] = src
                            return x
                        return hk
                    hooks = []
                    for L in W:
                        if part in ("both", "attn"):
                            hooks.append((f"blocks.{L}.hook_attn_out", mk(a_n[L])))
                        if part in ("both", "mlp"):
                            hooks.append((f"blocks.{L}.hook_mlp_out", mk(m_n[L])))
                    lp = m.run_with_hooks(tc, fwd_hooks=hooks)
                    res[cond][f"{wname}:{part}"].append(readout(lp, gid))

def mean(xs, i): return sum(x[i] for x in xs) / len(xs)

import random as _r
def boot_recov(ctx, nctx, pat, i, n=2000):
    """Paired bootstrap over query facts of (mean_pat - mean_ctx) / (mean_nctx - mean_ctx)."""
    N = len(ctx); out = []
    rg = _r.Random(0)
    for _ in range(n):
        idx = [rg.randrange(N) for _ in range(N)]
        c = sum(ctx[j][i] for j in idx) / N
        g = sum(nctx[j][i] for j in idx) / N
        q = sum(pat[j][i] for j in idx) / N
        if abs(g - c) < 1e-9: continue
        out.append((q - c) / (g - c))
    out.sort()
    if not out: return float("nan"), float("nan")
    return out[int(.025 * len(out))], out[int(.975 * len(out))]

print(f"\n=== {a.model}: restoring no-context sublayer outputs at END over a layer window ===")
print(f"  layers 0-{T3-1} early | {T3}-{2*T3-1} mid | {2*T3}-{nL-1} late\n")
out = {}
for cond in CONDS:
    if not res[cond]["_ctx"]: continue
    gc, mc = mean(res[cond]["_ctx"], 0), mean(res[cond]["_ctx"], 1)
    gn, mn = mean(res[cond]["_nctx"], 0), mean(res[cond]["_nctx"], 1)
    print(f"  --- {cond} ---   n={len(res[cond]['_ctx'])}")
    print(f"  no context : gold {gn:7.3f}   margin {mn:7.3f}")
    print(f"  prefixed   : gold {gc:7.3f}   margin {mc:7.3f}   "
          f"(damage: gold {gc-gn:+.3f}, margin {mc-mn:+.3f})")
    print(f"  {'restored window':>18s} {'gold':>8s} {'recov':>7s} {'95% CI':>16s} {'margin':>8s} {'recov':>7s} {'95% CI':>16s}")
    for wname in WINDOWS:
        for part in ("both", "attn", "mlp"):
            k = f"{wname}:{part}"
            if k not in res[cond]: continue
            g, mg = mean(res[cond][k], 0), mean(res[cond][k], 1)
            rg = (g - gc) / (gn - gc) if abs(gn - gc) > 1e-6 else float("nan")
            rm = (mg - mc) / (mn - mc) if abs(mn - mc) > 1e-6 else float("nan")
            lo_g, hi_g = boot_recov(res[cond]["_ctx"], res[cond]["_nctx"], res[cond][k], 0)
            lo_m, hi_m = boot_recov(res[cond]["_ctx"], res[cond]["_nctx"], res[cond][k], 1)
            print(f"  {k:>18s} {g:8.3f} {100*rg:6.1f}% [{100*lo_g:6.1f},{100*hi_g:6.1f}] "
                  f"{mg:8.3f} {100*rm:6.1f}% [{100*lo_m:6.1f},{100*hi_m:6.1f}]")
            out[f"{cond}|{k}"] = dict(gold=g, margin=mg, recov_gold=rg, recov_margin=rm,
                                      ci_gold=[lo_g, hi_g], ci_margin=[lo_m, hi_m],
                                      raw=res[cond][k])
    out[f"{cond}|_ctx"] = dict(gold=gc, margin=mc, raw=res[cond]["_ctx"])
    out[f"{cond}|_nctx"] = dict(gold=gn, margin=mn, raw=res[cond]["_nctx"])
    print()
json.dump(out, (ROOT / "results" / f"window_restore_{a.model}.json").open("w"), indent=1)
