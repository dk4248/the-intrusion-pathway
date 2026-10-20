"""Causal separability of intrusion and retrieval heads, swept over k.

A double dissociation is the test, because the two populations are defined by different
conditions: retrieval heads do their job under GOLDEN context, intrusion heads (paper Eq. 3)
are defined by what they do under IRRELEVANT context.

    separable  <=>  intervening on I-only heads helps under irrelevant context and leaves golden
                    alone, while intervening on R-only heads hurts golden and leaves irrelevant alone

Interventions:
  knockout  the paper's defence: pre-softmax, attn_scores[h, END, ctx] = -1e9 for
            every head in the set, so softmax redistributes to non-context positions. It is a
            no-op on a bare query by construction (no context positions), so it is run on
            golden / far_irr / long_irr.
  zero      the head's END output set to 0 -- a cruder ablation, reported for comparison only.

Sets per k, from the PRIMARY score variant of 61_sep_scores.py (raw DLA, prefix mass):
  I_only = I_k \\ R_k,  R_only = R_k \\ I_k,  both = I_k n R_k,  I*_k (paper's clean set)
  rand   = layer-matched to I_only, 3 seeds
Held-out facts only.
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
ap.add_argument("--ks", default="5,10,20,40")
ap.add_argument("--limit", type=int, default=60)
ap.add_argument("--variant", default="raw|prefix")
a = ap.parse_args()
m = load(a.model, a.device); nL, nH = m.cfg.n_layers, m.cfg.n_heads
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"]
HELD = set(json.load((ROOT / "data" / "c5_split.json").open())["heldout"])
SEP = json.load((ROOT / "results" / f"SEP_scores_{a.model}.json").open())
sweep = {r["k"]: r for r in SEP["variants"][a.variant]["sweep"]}
KS = [int(x) for x in a.ks.split(",")]
assert all(k in sweep for k in KS), f"k values must be in the 61 sweep: {sorted(sweep)}"
assert "blocks.0.attn.hook_attn_scores" in m.hook_dict, "hook_attn_scores missing"

rg = random.Random(1234)
SETS = {}
for k in KS:
    I = {tuple(x) for x in sweep[k]["I"]}; R = {tuple(x) for x in sweep[k]["R"]}
    SETS[f"I_only@{k}"] = sorted(I - R)
    SETS[f"R_only@{k}"] = sorted(R - I)
    SETS[f"both@{k}"] = sorted(I & R)
    SETS[f"Istar@{k}"] = sorted(tuple(x) for x in sweep[k]["Istar"])
    for sd in range(3):
        SETS[f"rand{sd}@{k}"] = [(L, rg.choice([h for h in range(nH) if (L, h) not in I | R]))
                                 for L, _ in SETS[f"I_only@{k}"]]
SETS = {n: v for n, v in SETS.items() if v}

def fid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False); return i[0] if i else None
def readout(lg, gid):
    x = lg[0, -1].float(); g = float(x[gid]); o = x.clone(); o[gid] = -1e9
    return g, g - float(o.max())

def knock_hooks(heads, P):
    byL = defaultdict(list)
    for L, h in heads: byL[L].append(h)
    idx = torch.tensor(P, device=a.device); hk = []
    for L, hs in byL.items():
        hh = torch.tensor(sorted(set(hs)), device=a.device)
        def mk(hh=hh):
            def f(x, hook):
                sub = x[:, hh, -1, :]
                sub[..., idx] = -1e9
                x[:, hh, -1, :] = sub
                return x
            return f
        hk.append((f"blocks.{L}.attn.hook_attn_scores", mk()))
    return hk
def zero_hooks(heads):
    byL = defaultdict(list)
    for L, h in heads: byL[L].append(h)
    hk = []
    for L, hs in byL.items():
        hh = torch.tensor(sorted(set(hs)), device=a.device)
        def mk(hh=hh):
            def f(x, hook):
                x[:, -1, hh, :] = 0.0; return x
            return f
        hk.append((f"blocks.{L}.attn.hook_z", mk()))
    return hk

CONDS = ["bare", "golden", "far_irr", "long_irr"]
res = defaultdict(lambda: defaultdict(list)); n = 0
with torch.no_grad():
    for s in S:
        if s["sample_id"] not in HELD or n >= a.limit: continue
        gid = fid(s["gold"])
        if gid is None: continue
        n += 1
        for cond in CONDS:
            if cond == "bare":
                txt, P = s["query"], []
            else:
                v = s["variants"].get(cond) or []
                if not v: continue
                txt = f"{v[0]['context']} {s['query']}"
            t = m.to_tokens(txt)
            if cond != "bare":
                p_last = min(m.to_tokens(v[0]["context"]).shape[1] - 1, t.shape[1] - 2)
                P = list(range(1, p_last + 1))
            res[cond]["_base"].append(readout(m(t), gid))
            for name, heads in SETS.items():
                if P:
                    res[cond][f"KO|{name}"].append(
                        readout(m.run_with_hooks(t, fwd_hooks=knock_hooks(heads, P)), gid))
                res[cond][f"ZERO|{name}"].append(
                    readout(m.run_with_hooks(t, fwd_hooks=zero_hooks(heads)), gid))

def d(cond, key, i=0):
    r = res[cond][key]
    if not r: return float("nan")
    b = res[cond]["_base"]
    return sum(x[i] - y[i] for x, y in zip(r, b)) / len(r)

out = {"n": n, "sets": {k: [list(x) for x in v] for k, v in SETS.items()}}
print(f"\n=== {a.model}: causal separability, k-sweep (held-out n={n}, variant {a.variant}) ===")
print(f"  Δ gold logit vs the un-intervened run. SEPARABLE predicts:")
print(f"    KO I_only : Δ(far_irr) > 0  and  Δ(golden) ≈ 0")
print(f"    KO R_only : Δ(golden) < 0   and  Δ(far_irr) ≈ 0")
for iv in ("KO", "ZERO"):
    print(f"\n  ===== {iv} {'(paper defence: END->ctx pre-softmax knockout)' if iv=='KO' else '(head END output zeroed)'} =====")
    cols = ["golden", "far_irr", "long_irr"] + (["bare"] if iv == "ZERO" else [])
    print(f"  {'set':>11s} {'|set|':>5s} " + " ".join(f"{c:>9s}" for c in cols))
    for k in KS:
        for base in ("I_only", "R_only", "both", "Istar"):
            name = f"{base}@{k}"
            if name not in SETS: continue
            row = [d(c, f"{iv}|{name}") for c in cols]
            print(f"  {name:>11s} {len(SETS[name]):5d} " + " ".join(f"{x:+9.3f}" for x in row))
            out[f"{iv}|{name}"] = dict(zip(cols, row))
        rr = [[d(c, f"{iv}|rand{sd}@{k}") for c in cols] for sd in range(3) if f"rand{sd}@{k}" in SETS]
        if rr:
            mean = [sum(x[j] for x in rr) / len(rr) for j in range(len(cols))]
            print(f"  {'rand@'+str(k):>11s} {len(SETS.get(f'rand0@{k}',[])):5d} "
                  + " ".join(f"{x:+9.3f}" for x in mean) + "   (layer-matched to I_only, 3 seeds)")
            out[f"{iv}|rand@{k}"] = dict(zip(cols, mean))
        print()
json.dump(out, (ROOT / "results" / f"SEP_causal_{a.model}.json").open("w"), indent=1)
