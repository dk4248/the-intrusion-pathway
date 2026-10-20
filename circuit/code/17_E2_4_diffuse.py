"""E2.4 — distributed copying with unequal attention, measured by held-out RECONSTRUCTION.

G0.2 shows the attention ratio cannot support this claim: rho_attn equals ||u*||^2/<o,u*>, i.e.
1/gain, and is blind to anything the head writes orthogonal to u*. The version used here asks how
much of the head's prefix-derived output is actually reconstructed by the attention-weighted
sum of source-token directions.

    b_h    = sum_{p in content} alpha_p u_hat_{t_p}        (attention-weighted identity target)
    o_h    = sum_{p in content} alpha_p (v_p W_O)          (actual prefix-derived output)
    c_h    = <o_h, b_h> / <b_h, b_h>                       (ONE scalar gain, fit on discovery)
    error  = ||o_h - c_h b_h|| / ||o_h||                   (evaluated on HELD-OUT samples)

References, all with the observed o_h held fixed:
    best-single   b from the single highest-attention source only
    uniform       b with alpha replaced by 1/k
    shuf-attn     b with alpha permuted across content positions
    shuf-token    b with the token identities permuted across content positions
Also reported: cosine(o_h, b_h) and the effective source count 1 / sum(alpha_tilde^2).
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=80)
ap.add_argument("--heads", type=int, default=10)
a = ap.parse_args()

m = load(a.model, a.device)
nL, nH = m.cfg.n_layers, m.cfg.n_heads
W_U = m.W_U.float()
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"][: a.limit]
E23 = json.load((ROOT / "results" / f"E2_3_{a.model}.json").open())
HEADS = [tuple(x) for x in E23["top_heads"][: a.heads]]
gen = torch.Generator().manual_seed(0)
rand_heads = [(int(torch.randint(nL, (1,), generator=gen)), int(torch.randint(nH, (1,), generator=gen)))
              for _ in range(a.heads)]
STOP = set(m.tokenizer.encode(" the of a an and is in to for on at .", add_special_tokens=False))

per = {h: {"o": [], "b": [], "b1": [], "bu": [], "bsa": [], "bst": [], "eff": []}
       for h in HEADS + rand_heads}

with torch.no_grad():
    for s in S:
        v = s["variants"].get("far_irr", [])
        if not v:
            continue
        prompt = f"{v[0]['context']} {s['query']}"
        toks = m.to_tokens(prompt)
        n_ctx = m.to_tokens(v[0]["context"]).shape[1] - 1
        keep = [p for p in range(1, min(n_ctx + 1, toks.shape[1]))
                if int(toks[0, p]) not in STOP
                and m.tokenizer.decode([int(toks[0, p])]).strip()[:1].isalnum()]
        if len(keep) < 3:
            continue
        END = toks.shape[1] - 1
        _, c = m.run_with_cache(toks, names_filter=lambda n: ("hook_pattern" in n or "hook_v" in n))
        ids = torch.tensor([int(toks[0, p]) for p in keep], device=W_U.device)
        U = W_U[:, ids].T.float(); U = U / U.norm(dim=1, keepdim=True)        # [k, d]
        kpos = torch.tensor(keep, device=W_U.device)
        for (L, H) in per:
            pat = c["pattern", L][0, H, END, kpos].float()                    # [k]
            vv = c["v", L][0].float()
            n_kv = vv.shape[1]
            if n_kv != nH:
                vv = vv.repeat_interleave(nH // n_kv, dim=1)
            W_O = m.blocks[L].attn.W_O[H].float()                             # [d_head, d]
            w = vv[kpos, H] @ W_O                                             # [k, d]
            o = (pat.unsqueeze(1) * w).sum(0)
            if o.norm() < 1e-6:
                continue
            at = pat / pat.sum().clamp_min(1e-9)
            b = (pat.unsqueeze(1) * U).sum(0)
            top = int(torch.argmax(pat))
            b1 = pat[top] * U[top]
            bu = (pat.sum() / len(keep)) * U.sum(0)
            perm = torch.randperm(len(keep), generator=gen).to(U.device)
            bsa = (pat[perm].unsqueeze(1) * U).sum(0)
            bst = (pat.unsqueeze(1) * U[perm]).sum(0)
            d = per[(L, H)]
            d["o"].append(o); d["b"].append(b); d["b1"].append(b1)
            d["bu"].append(bu); d["bsa"].append(bsa); d["bst"].append(bst)
            d["eff"].append(float(1.0 / (at ** 2).sum()))
        del c

def fit_eval(o, b, per_sample=False):
    """Gain fitted on the first half, error evaluated on the second half.

    NOTE on interpretation: b is a single direction, o lives in d_model. With an optimal gain
    the error is exactly sqrt(1 - cos^2), so the ABSOLUTE error is near 1 for any rank-1 target
    in high dimensions and carries no information on its own. The criterion is therefore
    COMPARATIVE -- does this target beat best-single-source, uniform, and the two shuffles --
    and the unexplained remainder is reported, not treated as a refutation.
    """
    n = len(o); h = n // 2
    if h < 2:
        return (float("nan"), float("nan"), []) if per_sample else (float("nan"), float("nan"))
    O = torch.stack(o); B = torch.stack(b)
    cg = float((O[:h] * B[:h]).sum() / (B[:h] * B[:h]).sum().clamp_min(1e-9))
    R = O[h:] - cg * B[h:]
    ps = (R.norm(dim=1) / O[h:].norm(dim=1)).tolist()
    err = sum(ps) / len(ps)
    cos = float(torch.nn.functional.cosine_similarity(O[h:], B[h:], dim=1).mean())
    return (err, cos, ps) if per_sample else (err, cos)

print(f"\n=== {a.model}: E2.4 held-out reconstruction of the prefix-derived write ===")
print(f"{'head':9s} {'eff k':>6s} {'gain-fit err':>12s} {'cos':>7s} | {'single':>8s} "
      f"{'uniform':>8s} {'shufattn':>9s} {'shuftok':>8s} | paired shuf-real [95% CI]")
import random as _rnd
def _paired_ci(x, y, n=2000):
    """bootstrap CI of mean(y - x): positive = the attention-weighted target fits BETTER"""
    r = _rnd.Random(0); d = [b - aa for aa, b in zip(x, y)]
    mu = sum(d) / len(d)
    draws = sorted(sum(r.choice(d) for _ in d) / len(d) for _ in range(n))
    return mu, draws[int(.025 * n)], draws[int(.975 * n)]

def block(name, hs):
    print(f"  -- {name}")
    wins = 0; tot = 0
    for h in hs:
        d = per[h]
        if len(d["o"]) < 6:
            continue
        e, cs, pe = fit_eval(d["o"], d["b"], True)
        e1, _, p1 = fit_eval(d["o"], d["b1"], True); eu, _, pu = fit_eval(d["o"], d["bu"], True)
        ea, _, pa = fit_eval(d["o"], d["bsa"], True); et, _, pt = fit_eval(d["o"], d["bst"], True)
        eff = sum(d["eff"]) / len(d["eff"])
        mu, lo, hi = _paired_ci(pe, pa)       # shuffled-attention minus real
        tot += 1; wins += int(lo > 0)
        print(f"  L{h[0]:2d}H{h[1]:<4d} {eff:6.1f} {e:12.3f} {cs:7.3f} | {e1:8.3f} "
              f"{eu:8.3f} {ea:9.3f} {et:8.3f} | shuf-real {mu:+.4f} [{lo:+.4f},{hi:+.4f}]"
              f"{'  *' if lo > 0 else ''}")
    print(f"     -> attention-weighted beats shuffled-attention with CI>0 on {wins}/{tot} heads")
block("E2.3 copier heads", HEADS)
block("random control heads", rand_heads)
print("\n  error < 1.0 means the target explains some of the write; 1.0 = no better than predicting 0.")
json.dump({f"L{L}H{H}": dict(n=len(per[(L,H)]["o"])) for (L, H) in per},
          (ROOT / "results" / f"E2_4_{a.model}.json").open("w"), indent=1)
