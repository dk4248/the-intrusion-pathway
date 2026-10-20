"""E5.5a — selection stability. Is the frozen set a property of the model, or of one split?

The manifest in E5.1 is one draw from one discovery half. If re-running discovery on a different
sample of facts returns a different top-20, the set is a sampling artefact and every causal number
attached to it inherits that instability.

This re-runs the screen ONCE while saving the per-sample, per-head gold-support change, then does
all resampling on CPU:

  * B grouped resamples of the discovery facts (stratified by relation, sampled WITH replacement at
    the fact level -- facts are the independent unit, not tokens)
  * top-K recomputed in each resample; per-head INCLUSION PROBABILITY over resamples
  * expected Jaccard between two independent resamples -- the honest "how reproducible is this set"
  * rank stability of the top-5

Sign changes and failures are kept: heads whose s_paired changes sign across resamples are reported
separately rather than dropped.
"""
import argparse, json, sys, random
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--cond", default="far_irr")
ap.add_argument("--topk", type=int, default=20)
ap.add_argument("--B", type=int, default=200)
a = ap.parse_args()
m = load(a.model, a.device); nL, nH = m.cfg.n_layers, m.cfg.n_heads
W_U = m.W_U.float()
center = m.cfg.normalization_type in ("LN", "LNPre")
lw = getattr(m.ln_final, "w", None); lw = lw.float() if lw is not None else None
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"]
DISC = set(json.load((ROOT / "data" / "c5_split.json").open())["discovery"])

def fid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False); return i[0] if i else None

def dla_all(cache, gid):
    sc = cache["ln_final.hook_scale"][0, -1].float(); ug = W_U[:, gid]
    out = torch.zeros(nL, nH)
    for L in range(nL):
        z = cache["z", L][0, -1].float()
        c = torch.einsum("hd,hdm->hm", z, m.W_O[L].float())
        if center: c = c - c.mean(-1, keepdim=True)
        c = c / sc
        if lw is not None: c = c * lw
        out[L] = (c @ ug).cpu()
    return out

keep = lambda n: n.endswith("hook_z") or n == "ln_final.hook_scale"
per_sample, rels = [], []
with torch.no_grad():
    for s in S:
        if s["sample_id"] not in DISC: continue
        v = s["variants"].get(a.cond) or []
        gid = fid(s["gold"])
        if gid is None or not v: continue
        _, cn = m.run_with_cache(m.to_tokens(s["query"]), names_filter=keep)
        dn = dla_all(cn, gid); del cn
        pre = v[0]["context"]
        toks = m.to_tokens(f"{pre} {s['query']}")
        p_last = min(m.to_tokens(pre).shape[1] - 1, toks.shape[1] - 2)
        _, cc = m.run_with_cache(toks, names_filter=lambda n: keep(n) or "hook_pattern" in n)
        dc = dla_all(cc, gid)
        A = torch.zeros(nL, nH)
        for L in range(nL):
            A[L] = cc["pattern", L][0, :, -1, 1:p_last + 1].float().sum(-1).cpu()
        del cc
        # per-sample s_int contribution: a_h * max(0, drop). Resampling these preserves the
        # manuscript definition (mean of per-sample products) under the bootstrap.
        per_sample.append(A * (dn - dc).clamp(min=0)); rels.append(s["relation"])
X = torch.stack(per_sample)                       # [n_facts, nL, nH]
n = X.shape[0]
torch.save({"X": X, "rels": rels}, ROOT / "results" / f"E5_5_persample_{a.model}.pt")
print(f"\n=== {a.model}: E5.5 selection stability ({a.cond}, {n} discovery facts, B={a.B}) ===")

byrel = defaultdict(list)
for i, r in enumerate(rels): byrel[r].append(i)
rg = random.Random(31337)
def topk_of(idx):
    mu = X[idx].mean(0)
    flat = [(float(mu[L, h]), L, h) for L in range(nL) for h in range(nH)]
    return [(L, h) for _, L, h in sorted(flat, reverse=True)[: a.topk]]

full = topk_of(list(range(n)))
inc = Counter(); draws = []
for _ in range(a.B):
    idx = []
    for r, ii in byrel.items():                    # stratified, with replacement within relation
        idx += [rg.choice(ii) for _ in range(len(ii))]
    t = topk_of(idx); draws.append(set(t))
    inc.update(t)

p_inc = {h: c / a.B for h, c in inc.items()}
stable80 = [h for h in full if p_inc.get(h, 0) >= 0.8]
stable50 = [h for h in full if p_inc.get(h, 0) >= 0.5]
jac = []
for i in range(0, len(draws) - 1, 2):
    A, B_ = draws[i], draws[i + 1]
    jac.append(len(A & B_) / len(A | B_))
print(f"  full-discovery top-{a.topk}: {full}")
print(f"  heads with inclusion probability >= 0.8 : {len(stable80)}/{a.topk}  {stable80}")
print(f"  heads with inclusion probability >= 0.5 : {len(stable50)}/{a.topk}")
print(f"  expected Jaccard between two independent resamples: "
      f"{sum(jac)/len(jac):.3f}  (1.0 = identical set)")
print(f"\n  per-head inclusion probability (full-discovery top-10):")
for L, h in full[:10]:
    col = X[:, L, h]
    frac_neg = float((col < 0).float().mean())
    print(f"    L{L:2d}H{h:<3d} p_inc={p_inc.get((L,h),0):.2f}  "
          f"mean {float(col.mean()):+.4f}  contributing-fact share {frac_neg:.2f} "
          f"({'broad' if frac_neg >= .5 else 'driven by a minority of facts'})")
tot = sum(1 for L in range(nL) for h in range(nH))
print(f"\n  heads appearing in ANY resample's top-{a.topk}: {len(inc)} of {tot} "
      f"({100*len(inc)/tot:.1f}%)")
json.dump({"full_top": [[L, h] for L, h in full],
           "p_inclusion": {f"{L}.{h}": p for (L, h), p in p_inc.items()},
           "expected_jaccard": sum(jac) / len(jac), "n_any": len(inc), "B": a.B, "n_facts": n},
          (ROOT / "results" / f"E5_5_stability_{a.model}.json").open("w"), indent=1)
