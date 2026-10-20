"""E3.3 — the STRONG hypothesis: is the cloud passed through unchanged up to a scalar gain?

Only meaningful inside the measured cloud subspace B (rank 12, identity direction projected
out). A head's OV is W_V W_O, of rank <= d_head, so it cannot be an identity map on an
arbitrary higher-dimensional residual; the question is whether B is approximately an invariant
subspace with ONE gain.

    a = B x_ln      input cloud coordinates at the source
    b = B w_p       output cloud coordinates of the head's write for that source
    c fitted on the discovery half only, then frozen; error/cosine on the held-out half.

No free linear decoder is fitted -- that would be able to rotate a transformed output back onto
the input and call the transformation "unchanged".
Also reported: how much of the input actually lies inside B.
"""
import argparse, json, sys, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load
from lre_pool import load_relations
from assoc_vocab import category_cloud, wiki_attr_cloud

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=60); ap.add_argument("--heads", type=int, default=8)
a = ap.parse_args()
m = load(a.model, a.device); nL, nH = m.cfg.n_layers, m.cfg.n_heads
W_U = m.W_U.float(); R = load_relations()
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"][: a.limit]
HEADS = [tuple(x) for x in json.load((ROOT / "results" / f"E2_3_{a.model}.json").open())["top_heads"][: a.heads]]
rng = random.Random(0)

def fid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False); return i[0] if i else None

per = defaultdict(lambda: {"a": [], "b": [], "frac_in": []})
with torch.no_grad():
    for s in S:
        v = s["variants"].get("far_irr", [])
        if not v: continue
        d = v[0]["distractor"]; crel = R.get(v[0]["context_relation"])
        if not crel: continue
        prompt = f"{v[0]['context']} {s['query']}"
        toks = m.to_tokens(prompt); dp = v[0]["context"].rfind(d)
        if dp < 0: continue
        p = min(m.to_tokens(prompt[:dp + len(d)]).shape[1] - 1, toks.shape[1] - 1)
        tid = int(toks[0, p]); uT = W_U[:, tid].float(); uT = uT / uT.norm()
        ids = sorted({j for j in (fid(w) for w in
                      category_cloud(d, crel["range"], exclude=[s["gold"]])
                      + wiki_attr_cloud(d, exclude=[s["gold"]])[:40]) if j is not None})[:40]
        if len(ids) < 6: continue
        A = W_U[:, torch.tensor(ids, device=W_U.device)].T.float()
        A = A / A.norm(dim=1, keepdim=True)
        M = A - (A @ uT).unsqueeze(1) * uT.unsqueeze(0)
        Q, _ = torch.linalg.qr(M.T.double()); B = Q.T.float()[:12]
        _, cache = m.run_with_cache(toks, names_filter=lambda n: ("hook_v" in n or
                                     "ln1.hook_normalized" in n))
        for (L, H) in HEADS:
            x_ln = cache[f"blocks.{L}.ln1.hook_normalized"][0, p].float()
            if x_ln.ndim > 1: x_ln = x_ln[0]
            vv = cache["v", L][0].float(); n_kv = vv.shape[1]
            if n_kv != nH: vv = vv.repeat_interleave(nH // n_kv, dim=1)
            w = vv[p, H] @ m.blocks[L].attn.W_O[H].float()
            per[(L, H)]["a"].append(B @ x_ln)
            per[(L, H)]["b"].append(B @ w)
            per[(L, H)]["frac_in"].append(float((B @ x_ln).norm() / x_ln.norm()))
        del cache

print(f"\n=== {a.model}: E3.3 is the cloud passed through unchanged (one gain, held out)? ===")
print(f"  {'head':9s} {'input var in B':>14s} {'fitted gain':>12s} {'held-out rel err':>17s} "
      f"{'cosine':>8s}")
rows = []
for (L, H), dd in per.items():
    n = len(dd["a"]); h = n // 2
    if h < 3: continue
    Aa = torch.stack(dd["a"]); Bb = torch.stack(dd["b"])
    c = float((Aa[:h] * Bb[:h]).sum() / (Aa[:h] * Aa[:h]).sum().clamp_min(1e-9))
    resid = Bb[h:] - c * Aa[h:]
    err = float((resid.norm(dim=1) / Bb[h:].norm(dim=1).clamp_min(1e-9)).mean())
    cos = float(torch.nn.functional.cosine_similarity(Bb[h:], Aa[h:], dim=1).mean())
    fi = sum(dd["frac_in"]) / len(dd["frac_in"])
    rows.append((err, cos))
    print(f"  L{L:2d}H{H:<4d} {fi:14.3f} {c:12.4f} {err:17.3f} {cos:8.3f}")
if rows:
    print(f"\n  median held-out relative error {sorted(e for e, _ in rows)[len(rows)//2]:.3f} · "
          f"median cosine {sorted(c for _, c in rows)[len(rows)//2]:+.3f}")
    print("  (error ~1 and cosine ~0 mean the cloud coordinates are NOT passed through; "
          "'unchanged up to a gain' would need error << 1 and cosine near 1)")
json.dump([dict(err=e, cos=c) for e, c in rows],
          (ROOT / "results" / f"E3_3_{a.model}.json").open("w"), indent=1)
