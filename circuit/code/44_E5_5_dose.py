"""E5.5b — graded dose, not only on/off.

Every causal number so far is a binary edit: a head's END write is either its context value or its
bare value. That cannot distinguish "this set carries a proportional share of the damage" from
"this set trips a threshold". Dose does:

    z[END,h] <- z_ctx[END,h] + alpha * (z_nctx[END,h] - z_ctx[END,h])

alpha=0 is a no-op (recovery must be exactly 0 -- the correctness check) and alpha=1 is the binary
edit from E5.1. A proportional mechanism gives recovery roughly linear in alpha; a threshold
mechanism gives a knee; a saturating curve means most of the effect is achieved by partial
attenuation and the set is not acting as an all-or-nothing switch.

The activity-matched control gets the same dose curve, so "frozen beats control" can be checked at
every dose rather than only at full strength.
"""
import argparse, json, sys, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--freeze_on", default="combined")
ap.add_argument("--conds", default="far_irr,long_irr")
ap.add_argument("--topk", type=int, default=20)
ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.25")
a = ap.parse_args()
m = load(a.model, a.device); nL, nH = m.cfg.n_layers, m.cfg.n_heads
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"]
HELD = set(json.load((ROOT / "data" / "c5_split.json").open())["heldout"])
man = json.load((ROOT / "results" / f"E5_1_manifest_{a.model}.json").open())
rows = man["per_condition"][a.freeze_on]["intrusion_top"][: a.topk]
FROZEN = [(int(r[0]), int(r[1])) for r in rows]
ALPHAS = [float(x) for x in a.alphas.split(",")]

# activity-matched control, same construction as 38
SP = torch.load(ROOT / "results" / f"E5_1_spaired_{a.model}.pt")[a.freeze_on]
qs = torch.quantile(SP.abs().flatten(), torch.linspace(0, 1, 11))
def decile(L, h): return int((qs < float(SP[L, h].abs())).sum().clamp(max=10)) - 1
pool = defaultdict(list)
for L in range(nL):
    for h in range(nH): pool[decile(L, h)].append((L, h))
rg = random.Random(909)
CTRL = [rg.choice(pool[decile(L, h)]) for L, h in FROZEN]

def fid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False); return i[0] if i else None
def readout(lg, gid):
    x = lg[0, -1].float(); g = float(x[gid]); o = x.clone(); o[gid] = -1e9
    return g, g - float(o.max())

def hooks(heads, z_n, alpha):
    byL = defaultdict(list)
    for L, h in heads: byL[L].append(h)
    hk = []
    for L, hs in byL.items():
        idx = torch.tensor(sorted(set(hs)), device=z_n[L].device)
        def mk(L=L, idx=idx):
            def f(x, hook):
                x[:, -1, idx, :] = x[:, -1, idx, :] + alpha * (z_n[L][idx] - x[:, -1, idx, :])
                return x
            return f
        hk.append((f"blocks.{L}.attn.hook_z", mk()))
    return hk

res = defaultdict(lambda: defaultdict(list))
with torch.no_grad():
    for s in S:
        if s["sample_id"] not in HELD: continue
        gid = fid(s["gold"])
        if gid is None: continue
        ln, cn = m.run_with_cache(m.to_tokens(s["query"]),
                                  names_filter=lambda n: n.endswith("hook_z"))
        z_n = {L: cn["z", L][0, -1].clone() for L in range(nL)}; del cn
        gn = readout(ln, gid)
        for cond in a.conds.split(","):
            v = s["variants"].get(cond) or []
            if not v: continue
            tc = m.to_tokens(f"{v[0]['context']} {s['query']}")
            res[cond]["_nctx"].append(gn); res[cond]["_ctx"].append(readout(m(tc), gid))
            for al in ALPHAS:
                for tag, hs in (("frozen", FROZEN), ("ctrl", CTRL)):
                    res[cond][f"{tag}@{al}"].append(
                        readout(m.run_with_hooks(tc, fwd_hooks=hooks(hs, z_n, al)), gid))

def mn(xs, i): return sum(x[i] for x in xs) / len(xs)
print(f"\n=== {a.model}: E5.5 dose-response (held-out n={len(res[a.conds.split(',')[0]]['_ctx'])}) ===")
out = {}
for cond in a.conds.split(","):
    if not res[cond]["_ctx"]: continue
    gc, mc = mn(res[cond]["_ctx"], 0), mn(res[cond]["_ctx"], 1)
    gN, mN = mn(res[cond]["_nctx"], 0), mn(res[cond]["_nctx"], 1)
    print(f"\n  --- {cond} ---  damage gold {gc-gN:+.3f}")
    print(f"  {'alpha':>6s} {'frozen gold':>12s} {'ctrl gold':>10s} {'frozen margin':>14s} "
          f"{'ratio f/c':>10s}")
    for al in ALPHAS:
        f = mn(res[cond][f"frozen@{al}"], 0); c = mn(res[cond][f"ctrl@{al}"], 0)
        fm = mn(res[cond][f"frozen@{al}"], 1)
        rf = (f - gc) / (gN - gc) if abs(gN - gc) > 1e-6 else float("nan")
        rc = (c - gc) / (gN - gc) if abs(gN - gc) > 1e-6 else float("nan")
        rfm = (fm - mc) / (mN - mc) if abs(mN - mc) > 1e-6 else float("nan")
        ratio = f"{rf/rc:.1f}x" if rc > 0.03 else "n/a"
        print(f"  {al:6.2f} {100*rf:11.1f}% {100*rc:9.1f}% {100*rfm:13.1f}% {ratio:>10s}")
        out[f"{cond}|{al}"] = dict(frozen=rf, ctrl=rc, frozen_margin=rfm)
    r1 = out[f"{cond}|1.0"]["frozen"]; rh = out[f"{cond}|0.5"]["frozen"]
    print(f"  half dose delivers {100*rh/r1:.0f}% of the full-dose effect "
          f"({'saturating' if rh/r1 > 0.6 else 'roughly proportional' if rh/r1 > 0.4 else 'threshold-like'})")
json.dump(out, (ROOT / "results" / f"E5_5_dose_{a.model}.json").open("w"), indent=1)
