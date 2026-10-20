"""E5.1 — intrusion-head screen. Definition of the intrusion score (Eq. 3).

    s_int(h) = 1/2 * sum_{c in COND}  E_i [ a_h^(i,c) * d_h^(i,c) ]

    a_h^(i,c) = sum_{p in prefix} A_h^(i,c)[END, p]               attention mass END -> prefix
    d_h^(i,c) = max(0, DLA_h^(i,nctx)(g_i) - DLA_h^(i,c)(g_i))    clamped DROP in gold support
    DLA_h(g)  = < o_h , u_g >,  o_h = the head's own END write, final-LN scale frozen at END

DEFINITIONS, fixed here and used by every downstream script:
  * prefix positions are 1..p_last inclusive; position 0 is BOS and is NOT part of the prefix.
  * the mean is of PER-SAMPLE PRODUCTS, never a product of separately averaged quantities.
  * the clamp means samples where the head's gold support IMPROVES contribute 0, not a negative.
  * COND = the two irrelevant conditions, weighted 1/2 each (here far_irr and long_irr, mirroring
    the manuscript's short_irr/long_irr).
  * a head qualifying on s_int typically still writes POSITIVELY toward gold; it qualifies because
    that support decreased. This is by design, not a defect.

Also recorded, for downstream controls and for comparison:
  s_ret(h)   = E_i DLA_h^(i,nctx)(g_i)       retrieval score, gold support written with no context
  a_bar(h)   = E_i a_h^(i,c)                 mean prefix attention mass -- the activity variable the
                                             score weights by, so controls can be matched on it
  ablations  no_attn / no_clamp / signed     kept as diagnostics ONLY

Discovery facts only (frozen split `data/c5_split.json`); every causal number downstream is held-out.
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
ap.add_argument("--conds", default="far_irr,long_irr")
ap.add_argument("--topk", type=int, default=20)
ap.add_argument("--split_seed", type=int, default=20260922)
a = ap.parse_args()
m = load(a.model, a.device); nL, nH = m.cfg.n_layers, m.cfg.n_heads
W_U = m.W_U.float()
center = m.cfg.normalization_type in ("LN", "LNPre")
lw = getattr(m.ln_final, "w", None); lw = lw.float() if lw is not None else None
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"]
CONDS = a.conds.split(",")

split_path = ROOT / "data" / "c5_split.json"
if split_path.exists():
    split = json.load(split_path.open()); print(f"  reusing frozen split {split_path.name}")
else:
    byrel = defaultdict(list)
    for s in S: byrel[s["relation"]].append(s["sample_id"])
    rg = random.Random(a.split_seed); disc, held = [], []
    for r, ids in sorted(byrel.items()):
        ids = sorted(ids); rg.shuffle(ids); k = len(ids) // 2
        disc += ids[:k]; held += ids[k:]
    split = {"seed": a.split_seed, "discovery": sorted(disc), "heldout": sorted(held)}
    json.dump(split, split_path.open("w"), indent=1)
DISC = set(split["discovery"])
print(f"  discovery {len(split['discovery'])} | held-out {len(split['heldout'])}")

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

keepz = lambda n: n.endswith("hook_z") or n == "ln_final.hook_scale"
keepc = lambda n: keepz(n) or "hook_pattern" in n
ACC = {k: {c: torch.zeros(nL, nH) for c in CONDS}
       for k in ("s_int", "no_attn", "no_clamp", "signed", "a_bar")}
CNT = defaultdict(int); ret = torch.zeros(nL, nH); n_ret = 0
with torch.no_grad():
    for s in S:
        if s["sample_id"] not in DISC: continue
        gid = fid(s["gold"])
        if gid is None: continue
        _, cn = m.run_with_cache(m.to_tokens(s["query"]), names_filter=keepz)
        d_n = dla_all(cn, gid); del cn
        ret += d_n; n_ret += 1
        for cond in CONDS:
            v = s["variants"].get(cond) or []
            if not v: continue
            pre = v[0]["context"]
            toks = m.to_tokens(f"{pre} {s['query']}")
            p_last = min(m.to_tokens(pre).shape[1] - 1, toks.shape[1] - 2)
            _, cc = m.run_with_cache(toks, names_filter=keepc)
            d_c = dla_all(cc, gid)
            A = torch.zeros(nL, nH)
            for L in range(nL):
                A[L] = cc["pattern", L][0, :, -1, 1:p_last + 1].float().sum(-1).cpu()
            del cc
            drop = d_n - d_c
            cl = drop.clamp(min=0)
            ACC["s_int"][cond] += A * cl
            ACC["no_attn"][cond] += cl
            ACC["no_clamp"][cond] += A * drop
            ACC["signed"][cond] += drop
            ACC["a_bar"][cond] += A
            CNT[cond] += 1
ret /= max(n_ret, 1)

FIN = {k: {c: ACC[k][c] / max(CNT[c], 1) for c in CONDS} for k in ACC}
COMB = {k: sum(FIN[k][c] for c in CONDS) / len(CONDS) for k in ACC}

def top(t, k):
    fl = [(float(t[L, h]), L, h) for L in range(nL) for h in range(nH)]
    return sorted(fl, reverse=True)[:k]

T3 = nL // 3
band = lambda L: "early" if L < T3 else ("mid" if L < 2 * T3 else "late")
man = {"model": a.model, "n_layers": nL, "n_heads": nH, "split_seed": a.split_seed,
       "n_discovery_used": n_ret, "conds": CONDS,
       "screen": "Eq. 3: s_int =mean_i [ a_h * max(0, DLA_nctx - DLA_ctx) ]",
       "per_condition": {}}
print(f"\n=== {a.model}: E5.1 s_int screen (Eq. 3), discovery n={n_ret} ===")
for cond in CONDS + ["combined"]:
    t = COMB["s_int"] if cond == "combined" else FIN["s_int"][cond]
    ab = COMB["a_bar"] if cond == "combined" else FIN["a_bar"][cond]
    rows = top(t, a.topk)
    dist = defaultdict(int)
    for _, L, _ in rows: dist[band(L)] += 1
    print(f"\n  --- {cond} --- top-{a.topk}  (early/mid/late = "
          f"{dist['early']}/{dist['mid']}/{dist['late']})")
    print(f"  {'L':>3s} {'H':>3s} {'s_int':>9s} {'a_bar':>8s} {'s_ret':>9s}")
    for v, L, h in rows[:8]:
        print(f"  {L:3d} {h:3d} {v:9.4f} {float(ab[L,h]):8.4f} {float(ret[L,h]):9.4f}")
    man["per_condition"][cond] = {
        "intrusion_top": [[L, h, v, float(ab[L, h]), float(ret[L, h])] for v, L, h in rows],
        "retrieval_top": [[L, h, v] for v, L, h in top(ret, a.topk)],
        "band_counts": dict(dist), "n": CNT.get(cond, n_ret)}
man["per_condition"]["golden"] = man["per_condition"]["combined"]   # downstream compat
man["retrieval_score_nctx"] = [[L, h, float(ret[L, h])] for L in range(nL) for h in range(nH)]
man["a_bar_all"] = [[L, h, float(COMB["a_bar"][L, h])] for L in range(nL) for h in range(nH)]
json.dump(man, (ROOT / "results" / f"E5_1_manifest_{a.model}.json").open("w"), indent=1)
torch.save({c: FIN["s_int"][c] for c in CONDS} | {"combined": COMB["s_int"]},
           ROOT / "results" / f"E5_1_spaired_{a.model}.pt")
torch.save({k: COMB[k] for k in COMB}, ROOT / "results" / f"E5_1_allscores_{a.model}.pt")
print(f"\n  manifest -> results/E5_1_manifest_{a.model}.json  (s_int; 'combined' is the 1/2-average)")
