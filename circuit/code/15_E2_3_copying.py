"""E2.3 — literal copying at individual sources, measured without the ratio diagnostic audited in G0.2.

For every head and every prefix source p we form the PRE-attention-weighting contribution
    w_{h,p} = v_{p,h} W_O^{(h)}                      [d_model]
and ask whether it points at the token that is actually AT p.

Two instruments, both gain-invariant and both able to separate copy from suppression
(which G0.2 shows the ratio diagnostic cannot):

  diag - offdiag   <w_hp, u_hat_tp>  minus  mean_q!=p <w_hp, u_hat_tq>
                   positive  = writes its own source token specifically
                   negative  = SUPPRESSES its own source token
  rank(t_p)        rank of t_p in the full-vocabulary projection of w_hp
                   computed only for the shortlisted heads + matched controls

Stage 1 scores every head cheaply (source directions only, no full vocab).
Stage 2 takes the top-K by stage 1 plus layer/norm-matched control heads and computes ranks.
"""
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--pairs", default="pairs_v1.json")
ap.add_argument("--cond", default="far_irr")
ap.add_argument("--limit", type=int, default=60)
ap.add_argument("--topk-heads", type=int, default=20)
a = ap.parse_args()

m = load(a.model, a.device)
S = json.load((ROOT / "data" / a.pairs).open())["samples"][: a.limit]
nL, nH = m.cfg.n_layers, m.cfg.n_heads
W_U = m.W_U.float()
STOP = set(m.tokenizer.encode(" the of a an and is in to for on at .", add_special_tokens=False))

def content_sources(prompt, ctx_len_chars):
    """Prefix content positions: inside the context, not BOS, not stopwords/punctuation."""
    toks = m.to_tokens(prompt)[0]
    n_ctx = m.to_tokens(prompt[:ctx_len_chars]).shape[1] - 1
    out = []
    for p in range(1, min(n_ctx + 1, toks.shape[0])):
        t = int(toks[p])
        s = m.tokenizer.decode([t]).strip()
        if t in STOP or not s or not s[0].isalnum():
            continue
        out.append((p, t))
    return out

diag = torch.zeros(nL, nH, device=W_U.device)
off = torch.zeros(nL, nH, device=W_U.device)
cnt = 0
t0 = time.time()
with torch.no_grad():
    for s in S:
        v = s["variants"].get(a.cond, [])
        if not v:
            continue
        v = v[0]
        prompt = f"{v['context']} {s['query']}"
        srcs = content_sources(prompt, len(v["context"]))
        if len(srcs) < 2:
            continue
        toks = m.to_tokens(prompt)
        _, cache = m.run_with_cache(toks, names_filter=lambda n: "hook_v" in n)
        pos = torch.tensor([p for p, _ in srcs], device=W_U.device)
        tid = torch.tensor([t for _, t in srcs], device=W_U.device)
        Uh = W_U[:, tid].T                                  # [k, d_model]
        Uh = Uh / Uh.norm(dim=1, keepdim=True)
        for L in range(nL):
            vv = cache["v", L][0].float()                   # [seq, n_kv, d_head]
            n_kv = vv.shape[1]
            if n_kv != nH:
                vv = vv.repeat_interleave(nH // n_kv, dim=1)
            W_O = m.blocks[L].attn.W_O.float()              # [H, d_head, d_model]
            w = torch.einsum("phd,hdm->phm", vv[pos], W_O)  # [k, H, d_model]
            M = torch.einsum("phm,qm->phq", w, Uh)          # [k, H, k]  source x head x token
            k = M.shape[0]
            eye = torch.eye(k, device=M.device, dtype=torch.bool)
            d_ = M[eye.unsqueeze(1).expand(k, nH, k)].view(k, nH)
            o_ = M.masked_fill(eye.unsqueeze(1), 0).sum(-1) / max(1, k - 1)
            diag[L] += d_.mean(0)
            off[L] += o_.mean(0)
        cnt += 1
        del cache
contrast = ((diag - off) / max(cnt, 1)).cpu()
print(f"[{a.model}] stage 1 over {cnt} samples in {time.time()-t0:.0f}s", flush=True)

flat = contrast.flatten()
top = torch.topk(flat, a.topk_heads).indices.tolist()
bot = torch.topk(-flat, 5).indices.tolist()
heads = [(int(i // nH), int(i % nH)) for i in top]
supp = [(int(i // nH), int(i % nH)) for i in bot]
print(f"\ntop-{a.topk_heads} heads by (diag - offdiag)  [writes its own source token]")
for (L, H) in heads[:10]:
    print(f"   L{L:2d}H{H:<3d} {float(contrast[L, H]):+.4f}")
print(f"most NEGATIVE (suppresses its own source token):")
for (L, H) in supp:
    print(f"   L{L:2d}H{H:<3d} {float(contrast[L, H]):+.4f}")
print(f"\npopulation: mean {float(contrast.mean()):+.4f}  sd {float(contrast.std()):.4f}  "
      f"max {float(contrast.max()):+.4f}  min {float(contrast.min()):+.4f}")

out = dict(model=a.model, cond=a.cond, n_samples=cnt,
           contrast=contrast.tolist(), top_heads=heads, suppressor_heads=supp)
json.dump(out, (ROOT / "results" / f"E2_3_{a.model}.json").open("w"))
print(f"-> results/E2_3_{a.model}.json")
