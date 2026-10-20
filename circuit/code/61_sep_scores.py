"""Are intrusion and retrieval heads separable? Score-level, with a k-sweep.

Definitions taken from the paper's own code (intrusion_heads/code/aggregate_head_scores.py
and run_stage4_head_scores.py):

  s_ret(h) = mean_i  A_h^(golden)[END, p_g] * DLA_h^(golden)(gold)          retrieval score
             p_g = first occurrence of the gold token id among non-END positions of the golden prompt
  s_int(h) = 1/2 sum_c mean_i  a_h^(c) * max(0, DLA_h^(nctx) - DLA_h^(c))   paper Eq. 3

Three knobs where versions of this analysis could differ, all computed so the conclusion can be
checked for dependence on them:
  DLA  'raw'    (z W_O) . (w * W_U[:,g])     paper code (fold_ln folds w into W_U; no scale division)
       'scaled' raw / final-LN scale at END   the frozen-scale convention used elsewhere in circuit/
  a_h  'prefix' attention END -> positions 1..p_last          paper TEXT
       'nonBE'  attention END -> all non-BOS, non-END positions paper CODE (includes query tokens)
Primary = raw DLA + prefix mass (text definition, code's DLA). Others are robustness variants.

Outputs per model: every head's s_int and s_ret under each variant, and a k-sweep of
  |I_k n R_k|, the chance expectation k^2/N, hypergeometric P(overlap >= observed), enrichment,
  Jaccard, and the clean set I*_k (top-k by s_int excluding top-2k by s_ret, as the paper defines).
Discovery facts only (frozen split).
"""
import argparse, json, sys, math
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--conds", default="far_irr,long_irr")
a = ap.parse_args()
m = load(a.model, a.device); nL, nH = m.cfg.n_layers, m.cfg.n_heads; N = nL * nH
W_U = m.W_U.float()
lw = getattr(m.ln_final, "w", None); lw = lw.float() if lw is not None else None
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"]
DISC = set(json.load((ROOT / "data" / "c5_split.json").open())["discovery"])
CONDS = a.conds.split(",")

def fid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False); return i[0] if i else None

def dla_both(cache, gid):
    """[nL,nH] DLA to gold, raw (paper code) and scaled (frozen final-LN scale)."""
    sc = cache["ln_final.hook_scale"][0, -1].float()
    ug = W_U[:, gid] * (lw if lw is not None else 1.0)     # fold_ln equivalent
    raw = torch.zeros(nL, nH)
    for L in range(nL):
        z = cache["z", L][0, -1].float()
        o = torch.einsum("hd,hdm->hm", z, m.W_O[L].float())
        raw[L] = (o @ ug).cpu()
    return raw, raw / float(sc)

keep = lambda n: n.endswith("hook_z") or n == "ln_final.hook_scale" or "hook_pattern" in n
acc = defaultdict(lambda: torch.zeros(nL, nH)); cnt = defaultdict(int); n_pg_miss = 0
with torch.no_grad():
    for s in S:
        if s["sample_id"] not in DISC: continue
        gid = fid(s["gold"])
        if gid is None: continue
        _, cn = m.run_with_cache(m.to_tokens(s["query"]), names_filter=keep)
        dn_raw, dn_sc = dla_both(cn, gid); del cn
        # --- golden: retrieval score ---
        g = s["variants"].get("golden") or []
        if g:
            toks = m.to_tokens(f"{g[0]['context']} {s['query']}")
            hits = (toks[0, :-1] == gid).nonzero().flatten()
            if len(hits):
                pg = int(hits[0])
                _, cg = m.run_with_cache(toks, names_filter=keep)
                d_raw, d_sc = dla_both(cg, gid)
                A_pg = torch.stack([cg["pattern", L][0, :, -1, pg].float().cpu() for L in range(nL)])
                acc["ret|raw"] += A_pg * d_raw; acc["ret|scaled"] += A_pg * d_sc
                cnt["ret"] += 1; del cg
            else:
                n_pg_miss += 1
        # --- irrelevant conditions: intrusion score ---
        for cond in CONDS:
            v = s["variants"].get(cond) or []
            if not v: continue
            pre = v[0]["context"]
            toks = m.to_tokens(f"{pre} {s['query']}"); T = toks.shape[1]
            p_last = min(m.to_tokens(pre).shape[1] - 1, T - 2)
            _, cc = m.run_with_cache(toks, names_filter=keep)
            dc_raw, dc_sc = dla_both(cc, gid)
            Apre = torch.stack([cc["pattern", L][0, :, -1, 1:p_last + 1].float().sum(-1).cpu()
                                for L in range(nL)])
            Anbe = torch.stack([cc["pattern", L][0, :, -1, 1:T - 1].float().sum(-1).cpu()
                                for L in range(nL)])
            del cc
            for dl, dnv, dcv in (("raw", dn_raw, dc_raw), ("scaled", dn_sc, dc_sc)):
                drop = (dnv - dcv).clamp(min=0)
                acc[f"int|{dl}|prefix|{cond}"] += Apre * drop
                acc[f"int|{dl}|nonBE|{cond}"] += Anbe * drop
            cnt[cond] += 1

SC = {}
for dl in ("raw", "scaled"):
    SC[f"ret|{dl}"] = acc[f"ret|{dl}"] / max(cnt["ret"], 1)
    for am in ("prefix", "nonBE"):
        SC[f"int|{dl}|{am}"] = sum(acc[f"int|{dl}|{am}|{c}"] / max(cnt[c], 1) for c in CONDS) / len(CONDS)
torch.save(SC, ROOT / "results" / f"SEP_scores_{a.model}.pt")

def ranks(t):
    f = t.flatten(); o = torch.argsort(f); r = torch.empty_like(o, dtype=torch.float64)
    r[o] = torch.arange(len(f), dtype=torch.float64); return r
def spearman(x, y):
    rx, ry = ranks(x), ranks(y); rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (rx.norm() * ry.norm()))
def topk(t, k):
    f = t.flatten(); idx = torch.argsort(f, descending=True)[:k]
    return {(int(i) // nH, int(i) % nH) for i in idx}
def hyper_sf(obs, N_, K, n):
    """P(X >= obs), X ~ Hypergeometric(N_, K, n)."""
    tot = math.comb(N_, n)
    return sum(math.comb(K, x) * math.comb(N_ - K, n - x) for x in range(obs, min(K, n) + 1)) / tot

print(f"\n=== {a.model}: intrusion vs retrieval, SCORE level (discovery n={cnt[CONDS[0]]}; "
      f"golden samples with gold in prefix: {cnt['ret']}, missing: {n_pg_miss}) ===")
KS = [5, 10, 15, 20, 30, 40, 60, 80, 100, 150]
out = {"model": a.model, "N_heads": N, "variants": {}}
for dl in ("raw", "scaled"):
    for am in ("prefix", "nonBE"):
        si, sr = SC[f"int|{dl}|{am}"], SC[f"ret|{dl}"]
        tag = f"DLA={dl}, a_h={am}" + ("   <-- PRIMARY" if (dl, am) == ("raw", "prefix") else "")
        rho = spearman(si, sr)
        print(f"\n  [{tag}]  Spearman(s_int, s_ret) over all {N} heads = {rho:+.3f}")
        print(f"  {'k':>4s} {'|I∩R|':>6s} {'chance':>7s} {'enrich':>7s} {'P(≥obs)':>9s} {'Jaccard':>8s} "
              f"{'|I*_k|':>7s}")
        rows = []
        for k in KS:
            I, R = topk(si, k), topk(sr, k)
            ov = len(I & R); ch = k * k / N
            p = hyper_sf(ov, N, k, k) if ov > 0 else 1.0
            R2 = topk(sr, 2 * k)
            f = si.flatten(); order = torch.argsort(f, descending=True)
            Istar = []
            for i in order.tolist():
                hl = (i // nH, i % nH)
                if hl not in R2: Istar.append(hl)
                if len(Istar) == k: break
            print(f"  {k:4d} {ov:6d} {ch:7.2f} {ov/ch if ch else float('nan'):6.1f}x {p:9.1e} "
                  f"{ov/len(I|R):8.3f} {len(Istar):7d}")
            rows.append(dict(k=k, overlap=ov, chance=ch, enrich=ov / ch, p=p,
                             jaccard=ov / len(I | R),
                             I=sorted(I), R=sorted(R), Istar=sorted(Istar)))
        out["variants"][f"{dl}|{am}"] = dict(spearman=rho, sweep=rows)
json.dump(out, (ROOT / "results" / f"SEP_scores_{a.model}.json").open("w"), indent=1)
