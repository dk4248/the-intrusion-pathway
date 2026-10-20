"""E5.1 validation + E5.2 per-head causal contribution.

The E5.1 screen is correlational (a logit-lens read of each head's own write) and was computed on
the DISCOVERY half only. This validates the frozen set by intervention on the HELD-OUT half, and
reports per-head causal effects in every condition (E5.2).

Intervention: restore a head's no-context write at END, z[END,h] <- z_nctx[END,h], per query head
(GQA-safe). Recovery is the usual (patched - ctx) / (nctx - ctx) on the gold logit and on the
scale-free margin.

Controls -- the screen ranks heads, so SOME set will look good; these say how good is good:
  rand        uniformly random heads, size-matched
  rand_layer  random heads drawn from the SAME layers as the frozen set
  rand_act    random heads matched on activity (mean |dla| decile under ctx)
The frozen set has to beat all three, not just `rand`.

Besides recovery on the gold logit and margin, the script reports how often the edit changes the
ARGMAX (final prediction), not just the logit.
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
ap.add_argument("--manifest_tag", default="")
ap.add_argument("--freeze_on", default="combined", help="condition whose frozen set is tested")
ap.add_argument("--conds", default="golden,far_irr,long_irr")
ap.add_argument("--topk", type=int, default=20)
ap.add_argument("--n_single", type=int, default=10)
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
m = load(a.model, a.device); nL, nH = m.cfg.n_layers, m.cfg.n_heads
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"]
split = json.load((ROOT / "data" / "c5_split.json").open())
HELD = set(split["heldout"])
man = json.load((ROOT / "results" / f"E5_1_manifest_{a.model}{a.manifest_tag}.json").open())
rows = man["per_condition"][a.freeze_on]["intrusion_top"][: a.topk]
FROZEN = [(int(r[0]), int(r[1])) for r in rows]
ACT = {(int(r[0]), int(r[1])): r[3] for r in rows}
print(f"  frozen on '{a.freeze_on}' (discovery), tested on {len(HELD)} held-out facts")
print(f"  frozen set: {FROZEN}")

# activity table over all heads, for activity-matched controls
allact = {}
for L in range(nL):
    for h in range(nH): allact[(L, h)] = 0.0
for r in man["per_condition"][a.freeze_on]["intrusion_top"]: allact[(int(r[0]), int(r[1]))] = r[3]
SP = torch.load(ROOT / "results" / f"E5_1_spaired_{a.model}{a.manifest_tag}.pt")[a.freeze_on]

def fid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False); return i[0] if i else None

def readout(lg, gid):
    x = lg[0, -1].float(); g = float(x[gid]); o = x.clone(); o[gid] = -1e9
    return g, g - float(o.max()), int(x.argmax())

allheads = [(L, h) for L in range(nL) for h in range(nH)]
layers_of = [L for L, _ in FROZEN]
# activity deciles from |s_paired| over all heads
absflat = SP.abs().flatten()
qs = torch.quantile(absflat, torch.linspace(0, 1, 11))
def decile(L, h):
    v = float(SP[L, h].abs()); return int((qs < v).sum().clamp(max=10)) - 1
FROZ_DEC = [decile(L, h) for L, h in FROZEN]

ABAR = torch.zeros(nL, nH)
for L, h, v in man.get("a_bar_all", []): ABAR[int(L), int(h)] = v
qa = torch.quantile(ABAR.flatten(), torch.linspace(0, 1, 11))
def adec(L, h): return int((qa < float(ABAR[L, h])).sum().clamp(max=10)) - 1
FROZ_ADEC = [adec(L, h) for L, h in FROZEN]
apool = defaultdict(list)
for L, h in allheads: apool[adec(L, h)].append((L, h))

SETS = {"frozen": FROZEN}
for s in range(a.seeds):
    rg = random.Random(7000 + s)
    SETS[f"rand{s}"] = rg.sample(allheads, len(FROZEN))
    SETS[f"randlayer{s}"] = [(L, rg.randrange(nH)) for L in layers_of]
    pool = defaultdict(list)
    for L, h in allheads: pool[decile(L, h)].append((L, h))
    SETS[f"randact{s}"] = [rg.choice(pool[d]) for d in FROZ_DEC]
    SETS[f"randattn{s}"] = [rg.choice(apool[d]) for d in FROZ_ADEC]
SINGLES = {f"h{L}.{h}": [(L, h)] for L, h in FROZEN[: a.n_single]}

def hooks_for(heads, z_n):
    byL = defaultdict(list)
    for L, h in heads: byL[L].append(h)
    hk = []
    for L, hs in byL.items():
        idx = torch.tensor(sorted(set(hs)), device=z_n[L].device)
        def mk(L=L, idx=idx):
            def f(x, hook):
                x[:, -1, idx, :] = z_n[L][idx]; return x
            return f
        hk.append((f"blocks.{L}.attn.hook_z", mk()))
    return hk

res = defaultdict(lambda: defaultdict(list))
write = defaultdict(lambda: defaultdict(list))
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
            v = s["variants"].get(cond, [])
            if not v: continue
            tc = m.to_tokens(f"{v[0]['context']} {s['query']}")
            res[cond]["_nctx"].append(gn); res[cond]["_ctx"].append(readout(m(tc), gid))
            for name, heads in {**SETS, **SINGLES}.items():
                res[cond][name].append(readout(m.run_with_hooks(tc, fwd_hooks=hooks_for(heads, z_n)), gid))

def mn(xs, i): return sum(x[i] for x in xs) / len(xs)
print(f"\n=== {a.model}: E5.1 held-out validation / E5.2 per-head causal  n={len(res[a.freeze_on]['_ctx'])} ===")
out = {"frozen_set": FROZEN, "freeze_on": a.freeze_on}
for cond in a.conds.split(","):
    if not res[cond]["_ctx"]: continue
    gc, mc = mn(res[cond]["_ctx"], 0), mn(res[cond]["_ctx"], 1)
    gN, mN = mn(res[cond]["_nctx"], 0), mn(res[cond]["_nctx"], 1)
    print(f"\n  --- {cond} ---  damage gold {gc-gN:+.3f}  margin {mc-mN:+.3f}")
    print(f"  {'set':>14s} {'gold recov':>11s} {'margin recov':>13s}")
    def rec(name):
        r = res[cond][name]
        g, mg = mn(r, 0), mn(r, 1)
        return ((g - gc) / (gN - gc) if abs(gN - gc) > 1e-6 else float("nan"),
                (mg - mc) / (mN - mc) if abs(mN - mc) > 1e-6 else float("nan"))
    fg, fm = rec("frozen")
    print(f"  {'frozen':>14s} {100*fg:10.1f}% {100*fm:12.1f}%   ({len(FROZEN)} heads)")
    out[f"{cond}|frozen"] = dict(gold=fg, margin=fm)
    for tag in ("rand", "randlayer", "randact", "randattn"):
        gs = [rec(f"{tag}{s}")[0] for s in range(a.seeds)]
        ms = [rec(f"{tag}{s}")[1] for s in range(a.seeds)]
        print(f"  {tag:>14s} {100*sum(gs)/len(gs):10.1f}% {100*sum(ms)/len(ms):12.1f}%   "
              f"(range {100*min(gs):.0f}–{100*max(gs):.0f}%)")
        out[f"{cond}|{tag}"] = dict(gold=sum(gs)/len(gs), margin=sum(ms)/len(ms),
                                    lo=min(gs), hi=max(gs))
    print(f"  {'single heads':>14s}")
    for name in SINGLES:
        g, mg = rec(name)
        flips = sum(1 for x, y in zip(res[cond][name], res[cond]["_ctx"]) if x[2] != y[2])
        print(f"  {name:>14s} {100*g:10.1f}% {100*mg:12.1f}%   argmax changed {flips}/{len(res[cond][name])}")
        out[f"{cond}|{name}"] = dict(gold=g, margin=mg, argmax_flips=flips)
json.dump(out, (ROOT / "results" / f"E5_1_validate_{a.model}{a.manifest_tag}.json").open("w"), indent=1)
