"""E7.6 — per-example specificity and breadth of the frozen heads' write.

Measuring breadth with a category/cluster RATIO is fragile because its denominator is unstable, and
averaging over examples whose gold answers differ manufactures breadth, because the union of many
examples' correct answers looks broad even if each example's write is narrow.

This avoids both. Each frozen head's END write, and separately its IRRELEVANT-MINUS-MATCHED-CLEAN
difference, is decoded PER EXAMPLE against independently fixed token sets, and only then averaged:

  gold        the example's own gold token
  other_ans   other valid answers of the query relation, gold removed
  cat_query   category words of the QUERY's answer type
  cat_dstr    category words of the DISTRACTOR's type
  literal     the literal token ids present in the prefix
  ctrl        a frequency- and norm-matched random control set of the same size

Differences, not ratios. The matched-clean reference is the length-matched filler prefix, following
E5.3 -- against a bare query every number inherits the prefix-presence cost.
"""
import argparse, json, sys, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load
from lre_pool import load_all_relations
from assoc_vocab import category_cloud, other_valid_answers

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--topk", type=int, default=20)
ap.add_argument("--freeze_on", default="combined")
ap.add_argument("--limit", type=int, default=90)
a = ap.parse_args()
m = load(a.model, a.device); nL, nH = m.cfg.n_layers, m.cfg.n_heads
W_U = m.W_U.float()
center = m.cfg.normalization_type in ("LN", "LNPre")
lw = getattr(m.ln_final, "w", None); lw = lw.float() if lw is not None else None
REL = load_all_relations()
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"]
HELD = set(json.load((ROOT / "data" / "c5_split.json").open())["heldout"])
FILLER = json.load((ROOT / "data" / "e53_factorial.json").open())["samples"][0]["filler_pool"]
man = json.load((ROOT / "results" / f"E5_1_manifest_{a.model}.json").open())
FROZEN = [(int(r[0]), int(r[1])) for r in man["per_condition"][a.freeze_on]["intrusion_top"][: a.topk]]
FROZ_L = sorted({L for L, _ in FROZEN})
rg = random.Random(24601)

def fid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False); return i[0] if i else None
def ntok(t): return len(m.tokenizer.encode(t, add_special_tokens=False))
def ids_of(words, drop=()):
    out = []
    for w in words:
        i = fid(w)
        if i is not None and i not in drop: out.append(i)
    return sorted(set(out))

# frequency proxy for the control set: unembedding norm rank
unorm = W_U.norm(dim=0)
order = torch.argsort(unorm)
rank_of = torch.empty_like(order); rank_of[order] = torch.arange(len(order), device=order.device)

def matched_ctrl(ids, n):
    """random ids matched to `ids` on unembedding-norm rank decile."""
    out = []
    for i in ids[:n]:
        r = int(rank_of[i]); lo, hi = max(0, r - 2000), min(len(order) - 1, r + 2000)
        out.append(int(order[rg.randrange(lo, hi + 1)]))
    return out

def write_of(cache, sc, heads):
    """summed END write of the given heads, in logit space (pre-dot)."""
    tot = None
    for L, h in heads:
        z = cache["z", L][0, -1, h].float()
        o = z @ m.W_O[L, h].float()
        if center: o = o - o.mean()
        o = o / sc
        if lw is not None: o = o * lw
        tot = o if tot is None else tot + o
    return tot

SETS = ["gold", "other_ans", "cat_query", "cat_dstr", "literal", "ctrl"]
agg = defaultdict(lambda: defaultdict(list))
keep = lambda n: n.endswith("hook_z") or n == "ln_final.hook_scale"
n_used = 0
with torch.no_grad():
    for s in S:
        if s["sample_id"] not in HELD: continue
        gid = fid(s["gold"])
        if gid is None: continue
        v = s["variants"]["far_irr"][0]
        irr = v["context"]; d_ent = v.get("context_subject"); d_rel = REL.get(v.get("context_relation", ""))
        q_rel = REL.get(s["relation"])
        if q_rel is None or d_rel is None: continue
        want = ntok(irr)
        cand = [f for f in FILLER if abs(ntok(f) - want) <= 1] or FILLER
        pad = cand[hash(s["sample_id"]) % len(cand)]
        idsets = {
            "gold": [gid],
            "other_ans": ids_of(other_valid_answers(s["relation"], s["gold"]) or [], drop={gid})[:40],
            "cat_query": ids_of(category_cloud(s["subject"], q_rel["range"], exclude=[s["gold"]]), drop={gid})[:40],
            "cat_dstr": ids_of(category_cloud(d_ent, d_rel["range"], exclude=[s["gold"]]), drop={gid})[:40],
            "literal": sorted(set(m.tokenizer.encode(irr, add_special_tokens=False)) - {gid})[:40],
        }
        idsets["ctrl"] = matched_ctrl(idsets["cat_query"] or [gid], len(idsets["cat_query"]) or 1)
        if not idsets["cat_query"]: continue
        outs = {}
        for tag, pre in (("irr", irr), ("pad", pad)):
            _, c = m.run_with_cache(m.to_tokens(f"{pre} {s['query']}"), names_filter=keep)
            sc = c["ln_final.hook_scale"][0, -1].float()
            outs[tag] = write_of(c, sc, FROZEN); del c
        n_used += 1
        for name in SETS:
            ii = idsets[name]
            if not ii: continue
            t = torch.tensor(ii, device=W_U.device)
            for tag in ("irr", "pad"):
                agg[tag][name].append(float((outs[tag] @ W_U[:, t]).mean()))
            agg["diff"][name].append(
                float(((outs["irr"] - outs["pad"]) @ W_U[:, t]).mean()))

print(f"\n=== {a.model}: E7.6 per-example decoding of the frozen heads' write (n={n_used}) ===")
print(f"  mean logit-lens value of the {len(FROZEN)}-head END write, decoded per example then averaged")
print(f"  reference for `diff` is the LENGTH-MATCHED FILLER prefix, not the bare query\n")
print(f"  {'token set':>11s} {'irr':>9s} {'filler':>9s} {'irr - filler':>13s}")
out = {}
for name in SETS:
    if not agg["diff"][name]: continue
    i_, p_, d_ = (sum(agg[t][name]) / len(agg[t][name]) for t in ("irr", "pad", "diff"))
    print(f"  {name:>11s} {i_:9.4f} {p_:9.4f} {d_:+13.4f}")
    out[name] = dict(irr=i_, pad=p_, diff=d_, n=len(agg["diff"][name]))
g = out.get("gold", {}).get("diff", float("nan"))
print(f"\n  breadth check -- is the difference aimed at the distractor's category, or at nothing?")
for name in ("cat_dstr", "cat_query", "literal", "other_ans"):
    if name in out:
        print(f"    {name:>10s} - ctrl = {out[name]['diff'] - out['ctrl']['diff']:+.4f}")
print(f"    {'gold':>10s} - ctrl = {g - out['ctrl']['diff']:+.4f}")
json.dump(out, (ROOT / "results" / f"E7_6_specificity_{a.model}.json").open("w"), indent=1)
