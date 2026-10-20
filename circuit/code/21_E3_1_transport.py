"""E3.1 — does a copier head transport ASSOCIATION information, beyond token-identity geometry?

The trap: related-token decoding caused solely by non-orthogonal unembedding vectors does not
establish an extra transported association signal. A head that writes NOTHING
but the pure identity direction u_t already yields a positive projection on t's associates,
because <u_t, u_a> != 0. So the null is not zero — it is a GAIN-MATCHED IDENTITY-ONLY vector.

Per attended source p with token t:
    w_p       = v_p W_O                      the head's unweighted OV write for that source
    c         = <w_p, u_hat_t>               its identity coefficient
    assoc     = mean_{a in A_d(t)} <w_p, u_hat_a>          (t itself excluded from A_d)
    null      = mean_{a in A_d(t)} <c u_hat_t, u_hat_a>    what pure identity alone would give
    EXCESS    = assoc - null                 <- the only quantity that can support transport

Controls: the same excess computed against ANOTHER token's association list (matched in kind),
and against norm-matched random directions. Everything is per-source, never prompt-averaged.
"""
import argparse, json, sys, time, random
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
ap.add_argument("--limit", type=int, default=70); ap.add_argument("--heads", type=int, default=10)
a = ap.parse_args()

m = load(a.model, a.device)
nL, nH = m.cfg.n_layers, m.cfg.n_heads
W_U = m.W_U.float()
uni = W_U.norm(dim=0).float()
R = load_relations()
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"][: a.limit]
E23 = json.load((ROOT / "results" / f"E2_3_{a.model}.json").open())
HEADS = [tuple(x) for x in E23["top_heads"][: a.heads]]
gen = torch.Generator().manual_seed(0)
CTRL_H = [(int(torch.randint(nL, (1,), generator=gen)), int(torch.randint(nH, (1,), generator=gen)))
          for _ in range(a.heads)]
rng = random.Random(0)

alpha_ids = [i for i in range(0, m.cfg.d_vocab, 11)
             if m.tokenizer.decode([i]).strip().isalpha()
             and len(m.tokenizer.decode([i]).strip()) > 3][:3000]

def first_id(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False)
    return i[0] if i else None

def dirs(words, cap=40):
    ids = sorted({j for j in (first_id(w) for w in words) if j is not None})[:cap]
    if not ids:
        return None
    U = W_U[:, torch.tensor(ids, device=W_U.device)].T.float()
    return U / U.norm(dim=1, keepdim=True)

acc = defaultdict(lambda: defaultdict(list))
t0 = time.time()
with torch.no_grad():
    for s in S:
        v = s["variants"].get("far_irr", [])
        if not v:
            continue
        d = v[0]["distractor"]; crel = R.get(v[0]["context_relation"])
        if not crel:
            continue
        prompt = f"{v[0]['context']} {s['query']}"
        toks = m.to_tokens(prompt)
        dp = v[0]["context"].rfind(d)
        if dp < 0:
            continue
        p = min(m.to_tokens(prompt[:dp + len(d)]).shape[1] - 1, toks.shape[1] - 1)
        tid = int(toks[0, p])
        uT = W_U[:, tid].float(); uT = uT / uT.norm()

        A = dirs(category_cloud(d, crel["range"], exclude=[s["gold"]])
                 + wiki_attr_cloud(d, exclude=[s["gold"]])[:40])
        # control association list: another distractor's associations, same construction
        od = None
        for s2 in S:
            for v2 in s2["variants"].get("far_irr", [])[:1]:
                if v2.get("distractor") and v2["distractor"] != d and R.get(v2["context_relation"]):
                    od, ocrel = v2["distractor"], R[v2["context_relation"]]
                    break
            if od:
                break
        A_other = dirs(category_cloud(od, ocrel["range"], exclude=[s["gold"]])
                       + wiki_attr_cloud(od, exclude=[s["gold"]])[:40]) if od else None
        rnd_ids = [alpha_ids[rng.randrange(len(alpha_ids))] for _ in range(40)]
        A_rand = W_U[:, torch.tensor(rnd_ids, device=W_U.device)].T.float()
        A_rand = A_rand / A_rand.norm(dim=1, keepdim=True)
        if A is None:
            continue

        _, cache = m.run_with_cache(toks, names_filter=lambda n: "hook_v" in n)
        for tag, hs in (("copier", HEADS), ("control", CTRL_H)):
            for (L, H) in hs:
                vv = cache["v", L][0].float()
                n_kv = vv.shape[1]
                if n_kv != nH:
                    vv = vv.repeat_interleave(nH // n_kv, dim=1)
                w = vv[p, H] @ m.blocks[L].attn.W_O[H].float()      # unweighted OV write
                if w.norm() < 1e-8:
                    continue
                c = float(w @ uT)
                for nm, bank in (("assoc", A), ("other", A_other), ("rand", A_rand)):
                    if bank is None:
                        continue
                    real = float((bank @ w).mean())
                    null = float(c * (bank @ uT).mean())
                    acc[tag][f"{nm}_excess"].append(real - null)
                    if nm == "assoc":
                        _own = real - null
                    if nm == "other":
                        # paired, per observation: how much of the excess is specific to
                        # THIS token's associates rather than association-like content in general
                        acc[tag]["own_minus_other"].append(_own - (real - null))
                    if nm == "assoc":
                        acc[tag]["identity_c"].append(c)
                        acc[tag]["assoc_raw"].append(real)
                        acc[tag]["assoc_null"].append(null)
        del cache

def mci(xs, n=2000):
    if not xs:
        return (float("nan"),) * 3
    mu = sum(xs) / len(xs)
    d = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return mu, d[int(.025 * n)], d[int(.975 * n)]

print(f"\n=== {a.model}: E3.1 transported association beyond identity geometry "
      f"({len(acc['copier']['assoc_excess'])} head x source observations, {time.time()-t0:.0f}s) ===")
print(f"  {'population':10s} {'identity c':>11s} {'assoc raw':>10s} {'identity-only null':>19s} "
      f"{'EXCESS [95% CI]':>28s}")
for tag in ("copier", "control"):
    c = mci(acc[tag]["identity_c"]); r = mci(acc[tag]["assoc_raw"])
    nu = mci(acc[tag]["assoc_null"]); e = mci(acc[tag]["assoc_excess"])
    print(f"  {tag:10s} {c[0]:11.4f} {r[0]:10.4f} {nu[0]:19.4f} "
          f"{e[0]:+12.4f} [{e[1]:+.4f},{e[2]:+.4f}]")
print(f"\n  control association lists (should be ~0 if the excess is specific to the token's own associates):")
for tag in ("copier", "control"):
    o = mci(acc[tag]["other_excess"]); rr = mci(acc[tag]["rand_excess"])
    print(f"    {tag:8s} other-token associates {o[0]:+.4f} [{o[1]:+.4f},{o[2]:+.4f}] | "
          f"norm-matched random {rr[0]:+.4f} [{rr[1]:+.4f},{rr[2]:+.4f}]")
print(f"\n  PAIRED token-specificity (own associates minus other token's, same observation):")
for tag in ("copier", "control"):
    q = mci(acc[tag]["own_minus_other"])
    print(f"    {tag:8s} {q[0]:+.4f} [{q[1]:+.4f},{q[2]:+.4f}]  "
          f"{'CI>0' if q[1] > 0 else 'n.s.'}  (n={len(acc[tag]['own_minus_other'])})")
json.dump({k: {kk: mci(vv)[0] for kk, vv in d.items()} for k, d in acc.items()},
          (ROOT / "results" / f"E3_1_{a.model}.json").open("w"), indent=1)
