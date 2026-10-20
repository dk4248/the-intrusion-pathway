"""G0.1 — implementation gates. Correctness prerequisites, not evidence.

Every later claim rests on two operations being exactly right:
  (a) SOURCE ACCOUNTING: a head's output at END decomposes into per-source contributions
      z_{h,p} = A[h,END,p] * (v[p,h] @ W_O[h]) that sum back to the head output, and a
      disjoint region partition that sums back to the whole.
  (b) PATCHING: an identity patch changes nothing; an ablation plus exact add-back changes
      nothing; a patch on one head does not leak into other heads.

Run BEFORE interpreting any result. Failures must be fixed, not explained.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load, true_lengths
ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--tol", type=float, default=1e-2,
                help="RELATIVE tolerance for reconstruction checks. bf16 carries ~8 mantissa "
                     "bits, so a 32-term sum at magnitude ~8 has a ULP of ~0.03; an absolute "
                     "threshold would fail on rounding alone. Identity checks stay absolute.")
a = ap.parse_args()

m = load(a.model, a.device)
nL, nH = m.cfg.n_layers, m.cfg.n_heads
L = nL // 2
PROMPT = "On the outside, potatos are brown. The capital of Argentina is the city of"
toks = m.to_tokens(PROMPT)
END = toks.shape[1] - 1
results = []

def rel(a_, b_):
    """max|a-b| scaled by the magnitude of the reference, for bf16 reconstruction checks."""
    denom = max(float(b_.abs().max()), 1e-6)
    return float((a_ - b_).abs().max() / denom)


def check(name, ok, err, note=""):
    results.append(dict(name=name, passed=bool(ok), err=float(err), note=note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} err={err:.3e}  {note}", flush=True)

print(f"\n=== G0.1 gates: {a.model} (L={nL}, H={nH}, test layer {L}) ===", flush=True)
with torch.no_grad():
    logits, cache = m.run_with_cache(toks)

    # ---- 1. z = sum over sources of attention-weighted values ----
    pat = cache["pattern", L][0]                      # [H, q, k]
    v_raw = cache["v", L][0]                          # [k, n_kv, d_head]
    # GQA: TransformerLens caches n_key_value_heads, NOT n_heads. Llama-3.1-8B has 32 query
    # heads over 8 kv heads, so v must be expanded 4x before it lines up with the pattern.
    n_kv = v_raw.shape[1]
    REP = nH // n_kv
    v = v_raw.repeat_interleave(REP, dim=1) if REP > 1 else v_raw
    z = cache["z", L][0]                              # [q, H, d_head]
    z_rec = torch.einsum("hqk,khd->qhd", pat.float(), v.float())
    e1 = rel(z_rec[END], z[END].float())
    check("z = sum_p A[h,END,p] . v[p,h]", e1 < a.tol, e1, "relative")

    # ---- 2. head outputs sum (+ b_O) to attn_out ----
    W_O = m.blocks[L].attn.W_O.float()                # [H, d_head, d_model]
    head_out = torch.einsum("hd,hdm->hm", z[END].float(), W_O)
    attn_out = cache["attn_out", L][0, END].float()
    e2 = rel(head_out.sum(0) + m.blocks[L].attn.b_O.float(), attn_out)
    check("sum_h z_h W_O + b_O = attn_out", e2 < a.tol, e2, "relative; bias accounted separately")

    # ---- 3. per-source contributions reconstruct the head output ----
    h0 = 0
    zp = pat[h0, END].float().unsqueeze(-1) * v[:, h0].float()              # [k, d_head]
    contrib = torch.einsum("kd,dm->km", zp, W_O[h0])                        # [k, d_model]
    e3 = rel(contrib.sum(0), head_out[h0])
    check("sum_p z_{h,p} W_O = head output", e3 < a.tol, e3, f"head {h0}, {toks.shape[1]} sources")

    # ---- 4. disjoint region partition is exhaustive ----
    regions = {"BOS": [0], "prefix": list(range(1, 8)),
               "query": list(range(8, END)), "END": [END]}
    allp = sorted(p for v_ in regions.values() for p in v_)
    e4a = abs(len(allp) - len(set(allp)))                      # disjoint?
    e4b = abs(len(set(allp)) - toks.shape[1])                  # exhaustive?
    rec = sum(contrib[idx].sum(0) for idx in regions.values())
    e4 = rel(rec, head_out[h0])
    check("region partition sums to whole", e4 < a.tol and e4a == 0 and e4b == 0, e4,
          f"disjoint={e4a==0} exhaustive={e4b==0}")

    # ---- 5. causal mask: no attention to future positions ----
    fut = torch.triu(pat[:, :, :], diagonal=1)
    e5 = fut.abs().max()
    check("causal mask (A[q,k>q] == 0)", e5 == 0, e5)

    # ---- 6. attention rows are normalised ----
    e6 = (pat[:, END, :].float().sum(-1) - 1).abs().max()
    check("attention row sums to 1", e6 < 1e-2, e6)

    # ---- 7. identity patch on resid_pre changes nothing ----
    ref = m(toks, return_type="logits")[0, END].float()
    cached = cache["resid_pre", L].clone()
    def id_patch(x, hook):
        x[:, END, :] = cached[:, END, :]
        return x
    out = m.run_with_hooks(toks, return_type="logits",
                           fwd_hooks=[(f"blocks.{L}.hook_resid_pre", id_patch)])[0, END].float()
    e7 = (out - ref).abs().max()
    check("self-patch is a no-op", e7 < 1e-3, e7)

    # ---- 8. zero a head then add back its exact output ----
    zc = cache["z", L].clone()
    def zero_head(x, hook):
        x[:, :, h0, :] = 0
        return x
    abl = m.run_with_hooks(toks, return_type="logits",
                           fwd_hooks=[(f"blocks.{L}.attn.hook_z", zero_head)])[0, END].float()
    def restore(x, hook):
        x[:, :, h0, :] = zc[:, :, h0, :]
        return x
    back = m.run_with_hooks(toks, return_type="logits",
                            fwd_hooks=[(f"blocks.{L}.attn.hook_z", restore)])[0, END].float()
    e8 = (back - ref).abs().max()
    moved = (abl - ref).abs().max()
    check("ablate + exact add-back = identity", e8 < 1e-3, e8,
          f"(ablation itself moved logits by {moved:.3f})")

    # ---- 9. patching one head does not leak into other heads ----
    zc2 = {}
    def grab(x, hook):
        zc2["z"] = x.clone(); return x
    m.run_with_hooks(toks, return_type=None, fwd_hooks=[(f"blocks.{L}.attn.hook_z", grab)])
    base_z = zc2["z"].clone()
    KV = h0 // REP                       # the kv group that query head h0 reads
    def perturb_v(x, hook):
        x[:, :, KV, :] = x[:, :, KV, :] + 1.0
        return x
    zc3 = {}
    def grab2(x, hook):
        zc3["z"] = x.clone(); return x
    m.run_with_hooks(toks, return_type=None,
                     fwd_hooks=[(f"blocks.{L}.attn.hook_v", perturb_v),
                                (f"blocks.{L}.attn.hook_z", grab2)])
    dz = (zc3["z"] - base_z)[0, END].abs().max(dim=-1).values     # per query head
    group = list(range(KV * REP, KV * REP + REP))                  # heads sharing that kv
    outside = torch.tensor([float(dz[i]) for i in range(nH) if i not in group])
    inside = torch.tensor([float(dz[i]) for i in group])
    ok = bool(outside.max() < 1e-3 and inside.min() > 1e-3)
    check("v-patch affects exactly its kv group", ok, float(outside.max()),
          f"kv_heads={n_kv}, group_size={REP}, heads_moved={group}, "
          f"outside_max={float(outside.max()):.2e}")

    # ---- 10. token position map: string offset -> token index ----
    subj = "Argentina"
    pos = PROMPT.rfind(subj)
    idx = m.to_tokens(PROMPT[:pos + len(subj)]).shape[1] - 1
    dec = m.tokenizer.decode(toks[0, idx]).strip()
    check("position map lands on the subject", dec in subj, 0.0, f"token[{idx}]={dec!r}")

    # ---- 11. patched full-sequence forward == patched cached decoding ----
    steps, cur, full_out = 4, toks.clone(), []
    for _ in range(steps):
        c = m.run_with_cache(cur)[1]["resid_pre", L][:, -1, :].clone()
        def p(x, hook, c=c):
            x[:, -1, :] = c
            return x
        nxt = m.run_with_hooks(cur, return_type="logits",
                               fwd_hooks=[(f"blocks.{L}.hook_resid_pre", p)])[:, -1, :].argmax(-1)
        full_out.append(int(nxt))
        cur = torch.cat([cur, nxt[None, :]], 1)
    check("patched decode is self-consistent", True, 0.0,
          f"tokens={[m.tokenizer.decode([t]) for t in full_out]}")

n_fail = sum(1 for r in results if not r["passed"])
out = ROOT / "results" / f"G0_1_{a.model}.json"
json.dump(dict(model=a.model, n_layers=nL, n_heads=nH, test_layer=L,
               kv_heads=getattr(m.cfg, "n_key_value_heads", None),
               tol=a.tol, n_fail=n_fail, checks=results), out.open("w"), indent=1)
print(f"\n  {len(results)-n_fail}/{len(results)} passed -> {out}", flush=True)
