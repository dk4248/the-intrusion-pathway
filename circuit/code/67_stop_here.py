"""Where along the pathway can intrusion be stopped, and at what cost to useful evidence?

Every intervention is applied to the same held-out facts and scored with the same two numbers:

    recovery  = (gold_edit_irr - gold_irr) / (gold_nctx - gold_irr)             share of the irrelevant-
                                                                                context loss undone
    cost      = (gold_golden - gold_edit_golden) / (gold_golden - gold_nctx)   share of the benefit of
                                                                                evidential context lost

A selective defence has high recovery and ~zero cost. Stages, in pathway order:

  src_mlp      zero the MLP writes at every PREFIX position in the upper-half layers, where C1
               measured enrichment -- stop the associated information forming at the source
  copier_ko    pre-softmax knockout of END -> prefix attention for the 20 copier heads (C2, E2.3)
  mid_ko       the same knockout for EVERY middle-block attention head
  mid_restore  every middle-block head's END write replaced by its context-free value (the
               localisation reported in the manuscript)
  intr_ko      the knockout for the I-only intrusion heads (top-20 s_int minus top-20 s_ret)
  intr_restore the intrusion heads' END write replaced by its context-free value

mid_ko and mid_restore act on the SAME heads, so they are the like-for-like comparison of
"block its access" versus "replace its write". Bare queries are unaffected by every edit here by
construction (no prefix positions; restoring context-free values into a context-free run), so the
two conditions that carry information are the irrelevant ones and golden.
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
ap.add_argument("--limit", type=int, default=90)
ap.add_argument("--split", default="heldout", choices=["heldout", "all"],
                help="heldout for anything that uses a selected head set; 'all' is valid only for the "
                     "unselected middle-block and MLP edits")
ap.add_argument("--stages", default="src_mlp,copier_ko,mid_ko,mid_restore,intr_ko,intr_restore")
ap.add_argument("--tag", default="")
a = ap.parse_args()
m = load(a.model, a.device); nL, nH = m.cfg.n_layers, m.cfg.n_heads
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"]
_held = set(json.load((ROOT / "data" / "c5_split.json").open())["heldout"])
HELD = ([s for s in S if s["sample_id"] in _held] if a.split == "heldout" else S)[: a.limit]
STAGES = a.stages.split(",")
assert a.split == "heldout" or not ({"copier_ko", "intr_ko", "intr_restore"} & set(STAGES)), \
    "selected head sets must be evaluated on held-out facts only"
T3 = nL // 3
MID = list(range(T3, 2 * T3))
UPPER = list(range(nL // 2, nL - 1))
COPIER = [tuple(x) for x in json.load((ROOT / "results" / f"E2_3_{a.model}.json").open())["top_heads"]]
sw = {r["k"]: r for r in json.load((ROOT / "results" / f"SEP_scores_{a.model}.json").open())
      ["variants"]["raw|prefix"]["sweep"]}
I20 = {tuple(x) for x in sw[20]["I"]}; R20 = {tuple(x) for x in sw[20]["R"]}
I_ONLY = sorted(I20 - R20)
assert "blocks.0.attn.hook_attn_scores" in m.hook_dict

def fid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False); return i[0] if i else None
def readout(lg, gid):
    x = lg[0, -1].float(); g = float(x[gid]); o = x.clone(); o[gid] = -1e9
    return g, g - float(o.max())

def ko(heads, P):
    byL = defaultdict(list)
    for L, h in heads: byL[L].append(h)
    idx = torch.tensor(P, device=a.device); hk = []
    for L, hs in byL.items():
        hh = torch.tensor(sorted(set(hs)), device=a.device)
        def mk(hh=hh):
            def f(x, hook):
                sub = x[:, hh, -1, :]; sub[..., idx] = -1e9; x[:, hh, -1, :] = sub; return x
            return f
        hk.append((f"blocks.{L}.attn.hook_attn_scores", mk()))
    return hk
def restore_layers(Ls, An):
    def mk(L):
        def f(x, hook):
            x[:, -1, :] = An[L].to(x.dtype); return x
        return f
    return [(f"blocks.{L}.hook_attn_out", mk(L)) for L in Ls]
def restore_heads(heads, Zn):
    byL = defaultdict(list)
    for L, h in heads: byL[L].append(h)
    hk = []
    for L, hs in byL.items():
        hh = torch.tensor(sorted(set(hs)), device=a.device)
        def mk(L=L, hh=hh):
            def f(x, hook):
                x[:, -1, hh, :] = Zn[L][hh].to(x.dtype); return x
            return f
        hk.append((f"blocks.{L}.attn.hook_z", mk()))
    return hk
def zero_mlp(Ls, P):
    idx = torch.tensor(P, device=a.device)
    def f(x, hook):
        x[:, idx, :] = 0.0; return x
    return [(f"blocks.{L}.hook_mlp_out", f) for L in Ls]

ALL_MID = [(L, h) for L in MID for h in range(nH)]
CONDS = ["golden", "far_irr", "long_irr"]
res = defaultdict(lambda: defaultdict(list))
with torch.no_grad():
    for s in HELD:
        gid = fid(s["gold"])
        if gid is None: continue
        ln, cn = m.run_with_cache(m.to_tokens(s["query"]),
                                  names_filter=lambda n: n.endswith("hook_z") or n.endswith("hook_attn_out"))
        Zn = {L: cn["z", L][0, -1].clone() for L in range(nL)}
        An = {L: cn["attn_out", L][0, -1].clone() for L in range(nL)}
        del cn
        res["nctx"]["_base"].append(readout(ln, gid))
        for cond in CONDS:
            v = s["variants"].get(cond) or []
            if not v: continue
            t = m.to_tokens(f"{v[0]['context']} {s['query']}")
            P = list(range(1, min(m.to_tokens(v[0]["context"]).shape[1] - 1, t.shape[1] - 2) + 1))
            res[cond]["_base"].append(readout(m(t), gid))
            edits = {
                "src_mlp": zero_mlp(UPPER, P),
                "copier_ko": ko(COPIER, P),
                "mid_ko": ko(ALL_MID, P),
                "mid_restore": restore_layers(MID, An),
                "intr_ko": ko(I_ONLY, P),
                "intr_restore": restore_heads(sorted(I20), Zn),
            }
            for name, hk in edits.items():
                if name not in STAGES: continue
                res[cond][name].append(readout(m.run_with_hooks(t, fwd_hooks=hk), gid))

def boot(fn, n, B=1000):
    rg = random.Random(0); vals = []
    for _ in range(B):
        idx = [rg.randrange(n) for _ in range(n)]
        v = fn(idx)
        if v == v: vals.append(v)
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]
def mean(xs, idx, i=0): return sum(xs[j][i] for j in idx) / len(idx)

N = len(res["nctx"]["_base"]); ALL = list(range(N))
bn = res["nctx"]["_base"]; bg = res["golden"]["_base"]
out = {"model": a.model, "n": N, "sets": {"copier": COPIER, "I_only": I_ONLY, "I20": sorted(I20),
       "mid_layers": MID, "upper_mlp_layers": UPPER}}
print(f"\n=== {a.model}: stopping intrusion along the pathway (held-out n={N}) ===")
print(f"  golden benefit {mean(bg,ALL)-mean(bn,ALL):+.3f} | far_irr loss "
      f"{mean(res['far_irr']['_base'],ALL)-mean(bn,ALL):+.3f} | long_irr loss "
      f"{mean(res['long_irr']['_base'],ALL)-mean(bn,ALL):+.3f}\n")
print(f"  {'stage':>13s} {'far recovery':>20s} {'long recovery':>20s} {'golden cost':>20s}")
for name in [x for x in ("src_mlp", "copier_ko", "mid_ko", "mid_restore", "intr_ko", "intr_restore")
             if x in STAGES]:
    row = {}
    for cond in ("far_irr", "long_irr"):
        bi, ed = res[cond]["_base"], res[cond][name]
        f = lambda idx: (mean(ed, idx) - mean(bi, idx)) / (mean(bn, idx) - mean(bi, idx)) \
            if abs(mean(bn, idx) - mean(bi, idx)) > 1e-6 else float("nan")
        row[cond] = (f(ALL), *boot(f, N))
    eg = res["golden"][name]
    fc = lambda idx: (mean(bg, idx) - mean(eg, idx)) / (mean(bg, idx) - mean(bn, idx)) \
        if abs(mean(bg, idx) - mean(bn, idx)) > 1e-6 else float("nan")
    row["golden_cost"] = (fc(ALL), *boot(fc, N))
    row["abs"] = {c: mean(res[c][name], ALL) - mean(res[c]["_base"], ALL) for c in CONDS}
    out[name] = row
    fmt = lambda t: f"{100*t[0]:6.1f}% [{100*t[1]:5.0f},{100*t[2]:5.0f}]"
    print(f"  {name:>13s} {fmt(row['far_irr']):>20s} {fmt(row['long_irr']):>20s} {fmt(row['golden_cost']):>20s}")
print(f"\n  absolute Δ gold (logits):")
for name in [x for x in ("mid_ko", "mid_restore", "intr_ko") if x in STAGES]:
    print(f"    {name:>12s} " + "  ".join(f"{c} {out[name]['abs'][c]:+.3f}" for c in CONDS))
out["per_sample"] = {c: {k: v for k, v in res[c].items()} for c in res}   # for CIs downstream
out["split"] = a.split
json.dump(out, (ROOT / "results" / f"STOP_{a.model}{a.tag}.json").open("w"), indent=1)
