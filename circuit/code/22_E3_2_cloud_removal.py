"""E3.2 — remove a source's cloud without deleting its token or changing attention.

The cloud subspace for a source token t is built from its association directions and
then has the MEASURED IDENTITY DIRECTION REMOVED and is re-orthonormalised, so the edit cannot
work merely by deleting the token itself:

    B = orthonormal basis of { u_a : a in A_d(t) }  with u_hat_t projected out

The edit touches ONLY head h's value contribution at that position -- Q, K and the whole
attention pattern are untouched, which is what "without changing attention" requires:

    dv_h = (Pi_B x_ln) W_V^(h)          x_ln = ln1-normalised residual at p
    z_h[END] <- z_h[END] - alpha_h[END,p] * dv_h

It is applied at hook_z, NOT at hook_v. Under GQA a value slot is shared by several query heads
(Llama-3.1-8B: 8 kv slots for 32 heads), so editing hook_v would silently change 4 heads at
once; z is per query head, so this isolates exactly one head while reusing its original
attention weight. The head's modified write is computed analytically as w' = w - dv_h W_O^(h).

Conditions: full · cloud-removed · identity-removed (the contrast) · random subspace matched in
rank and removed-norm · exact add-back (GATE).

Readouts: the association excess in the head's own write at that source (the E3.1 statistic),
whether the identity projection survived, and the downstream gold logit / margin at END.
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
ap.add_argument("--limit", type=int, default=60); ap.add_argument("--heads", type=int, default=8)
a = ap.parse_args()

m = load(a.model, a.device)
nL, nH = m.cfg.n_layers, m.cfg.n_heads
W_U = m.W_U.float(); R = load_relations()
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"][: a.limit]
HEADS = [tuple(x) for x in json.load((ROOT / "results" / f"E2_3_{a.model}.json").open())["top_heads"][: a.heads]]
BY_L = defaultdict(list)
for L, H in HEADS:
    BY_L[L].append(H)
rng = random.Random(0)
torch.manual_seed(0)

def fid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False)
    return i[0] if i else None

def bank_dirs(words, cap=40):
    ids = sorted({j for j in (fid(w) for w in words) if j is not None})[:cap]
    if not ids:
        return None
    U = W_U[:, torch.tensor(ids, device=W_U.device)].T.float()
    return U / U.norm(dim=1, keepdim=True)

def excess(w, A, uT):
    """the E3.1 statistic: association projection minus what pure identity would give"""
    c = float(w @ uT)
    return float((A @ w).mean() - c * (A @ uT).mean()), c

res = defaultdict(lambda: defaultdict(list))
t0 = time.time()
with torch.no_grad():
    for s in S:
        v = s["variants"].get("far_irr", [])
        if not v:
            continue
        d = v[0]["distractor"]; crel = R.get(v[0]["context_relation"])
        gid = fid(s["gold"])
        if not crel or gid is None:
            continue
        prompt = f"{v[0]['context']} {s['query']}"
        toks = m.to_tokens(prompt)
        dp = v[0]["context"].rfind(d)
        if dp < 0:
            continue
        p = min(m.to_tokens(prompt[:dp + len(d)]).shape[1] - 1, toks.shape[1] - 1)
        END = toks.shape[1] - 1
        tid = int(toks[0, p])
        uT = W_U[:, tid].float(); uT = uT / uT.norm()
        A = bank_dirs(category_cloud(d, crel["range"], exclude=[s["gold"]])
                      + wiki_attr_cloud(d, exclude=[s["gold"]])[:40])
        if A is None:
            continue

        base_logits, cache = m.run_with_cache(toks)
        # cloud basis with the identity direction projected out, then re-orthonormalised
        M = A - (A @ uT).unsqueeze(1) * uT.unsqueeze(0)
        Q, _ = torch.linalg.qr(M.T.double())
        B = Q.T.float()[:12]                                   # rank-12 cloud subspace
        rank = B.shape[0]
        Rr = torch.randn(rank, B.shape[1], device=B.device)
        Qr, _ = torch.linalg.qr(Rr.T.double())
        Brand = Qr.T.float()[:rank]

        def comp_for(kind, x_ln):
            if kind == "cloud":
                return (B @ x_ln) @ B
            if kind == "identity":
                return (x_ln @ uT) * uT
            if kind == "random":
                return (Brand @ x_ln) @ Brand
            return torch.zeros_like(x_ln)

        outs = {}
        for kind in ("full", "cloud", "identity", "random", "addback"):
            hooks, ex_list, id_list = [], [], []
            for L, Hs in BY_L.items():
                x_ln = cache[f"blocks.{L}.ln1.hook_normalized"][0, p].float()
                if x_ln.ndim > 1:
                    x_ln = x_ln[0]
                comp = comp_for("cloud" if kind == "addback" else kind, x_ln)
                W_V = m.blocks[L].attn.W_V.float(); W_O = m.blocks[L].attn.W_O.float()
                vv = cache["v", L][0].float()
                n_kv = vv.shape[1]
                if n_kv != nH:
                    vv = vv.repeat_interleave(nH // n_kv, dim=1)
                dz = {}
                for H in Hs:
                    dv = comp @ W_V[H]                                  # [d_head]
                    alpha = float(cache["pattern", L][0, H, END, p])
                    dz[H] = alpha * dv
                    w = vv[p, H] @ W_O[H] - (dv @ W_O[H] if kind not in ("full", "addback") else 0)
                    e, c = excess(w, A, uT)
                    ex_list.append(e); id_list.append(c)
                def mk(dd=dz, kind=kind):
                    def hk(x, hook):
                        if kind == "full":
                            return x
                        for H, d_ in dd.items():
                            if kind == "addback":
                                x[:, END, H, :] = x[:, END, H, :] - d_.to(x.dtype) + d_.to(x.dtype)
                            else:
                                x[:, END, H, :] = x[:, END, H, :] - d_.to(x.dtype)
                        return x
                    return hk
                hooks.append((f"blocks.{L}.attn.hook_z", mk()))
            with m.hooks(fwd_hooks=hooks):
                lg = m(toks, return_type="logits")
            row = lg[0, END].float(); lse = torch.logsumexp(row, -1)
            mk_ = row.clone(); mk_[gid] = float("-inf")
            outs[kind] = dict(gold=float(row[gid]), margin=float(row[gid] - mk_.max()),
                              logprob=float(row[gid] - lse),
                              cloud=sum(ex_list) / len(ex_list),
                              ident=sum(id_list) / len(id_list))
        for k, o in outs.items():
            for f, val in o.items():
                res[k][f].append(val)
        del cache

def mci(xs, n=1500):
    mu = sum(xs) / len(xs)
    d = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return mu, d[int(.025 * n)], d[int(.975 * n)]

print(f"\n=== {a.model}: E3.2 cloud removal at the source's value input "
      f"(n={len(res['full']['gold'])}, {len(HEADS)} heads, {time.time()-t0:.0f}s) ===")
g = mci(res["addback"]["gold"])[0] - mci(res["full"]["gold"])[0]
print(f"  GATE add-back vs full: d gold logit = {g:+.2e}")
print(f"\n  {'condition':12s} {'cloud in write':>15s} {'identity kept':>14s} "
      f"{'gold logit':>11s} {'gold margin':>12s}")
for k in ("full", "cloud", "identity", "random"):
    c = mci(res[k]["cloud"]); i = mci(res[k]["ident"])
    gl = mci(res[k]["gold"]); mg = mci(res[k]["margin"])
    print(f"  {k:12s} {c[0]:15.4f} {i[0]:14.4f} {gl[0]:11.3f} {mg[0]:12.3f}")
base_c = mci(res["full"]["cloud"])[0]
for k in ("cloud", "identity", "random"):
    dd = [x - y for x, y in zip(res[k]["cloud"], res["full"]["cloud"])]
    q = mci(dd)
    print(f"  d cloud-in-write, {k:9s} = {q[0]:+.4f} [{q[1]:+.4f},{q[2]:+.4f}]")
json.dump({k: {f: mci(v)[0] for f, v in d.items()} for k, d in res.items()},
          (ROOT / "results" / f"E3_2_{a.model}.json").open("w"), indent=1)
