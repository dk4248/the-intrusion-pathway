"""E2.1 — remove the induction cue while keeping the copied source.

Canonical induction needs [A][B] ... [A] -> [B]: the repeated A is the cue that tells the head
where B is. Parroting is the claim that some heads promote B without that cue.

Matched pair, token-audited, B's POSITION and the final cue held fixed:
   STRUCT    [A][B] r r r ... [A]      the cue is present
   NOSTRUCT  [C][B] r r r ... [A]      only the earlier match changed, A -> C
Both end at the same final token [A], both have B at the same index, and the filler r is the
same random sequence, so the only difference is whether the earlier token matches the final one.

A no-repeated-suffix variant is included so a leftover "…is B … is" pattern cannot explain a
result, and A/B/C are redrawn every trial to counterbalance identity, frequency and order.

Per head we record attention from END onto B's position and DLA onto u_B, under both
conditions. Induction-positive controls are identified independently, by the standard
prefix-matching score on a repeated random sequence -- never by this experiment's own contrast.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load, spaced_piece

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--trials", type=int, default=60)
ap.add_argument("--seqlen", type=int, default=24)
a = ap.parse_args()

m = load(a.model, a.device)
nL, nH = m.cfg.n_layers, m.cfg.n_heads
W_U = m.W_U.float()
g = torch.Generator().manual_seed(0)

# a pool of ordinary single-token words, so A/B/C are comparable in kind
pool = []
for i in range(0, m.cfg.d_vocab, 3):
    s = spaced_piece(m.tokenizer, i)
    if s.startswith(" ") and s[1:].isalpha() and 3 <= len(s[1:]) <= 10 and s[1:].islower():
        pool.append(i)
pool = pool[:3000]
print(f"[{a.model}] word pool: {len(pool)}", flush=True)

# ---------- independent induction score: repeated random sequence ----------
@torch.no_grad()
def induction_score(n_trials=20, half=30):
    sc = torch.zeros(nL, nH, device=W_U.device)
    for _ in range(n_trials):
        seq = [pool[int(torch.randint(len(pool), (1,), generator=g))] for _ in range(half)]
        toks = torch.tensor([[m.tokenizer.bos_token_id or pool[0]] + seq + seq],
                            device=W_U.device)
        _, c = m.run_with_cache(toks, names_filter=lambda n: "hook_pattern" in n)
        for L in range(nL):
            pat = c["pattern", L][0].float()            # [H, q, k]
            # at each position in the 2nd copy, attend to the token AFTER its 1st occurrence
            idx_q = torch.arange(half + 1, 2 * half, device=pat.device)
            idx_k = idx_q - half + 1
            sc[L] += pat[:, idx_q, idx_k].mean(-1)
        del c
    return (sc / n_trials).cpu()

t0 = time.time()
ind = induction_score()
print(f"[{a.model}] induction score computed in {time.time()-t0:.0f}s", flush=True)

# ---------- STRUCT / NOSTRUCT ----------
@torch.no_grad()
def run_pair(no_repeat_suffix=False):
    att = torch.zeros(2, nL, nH, device=W_U.device)
    dla = torch.zeros(2, nL, nH, device=W_U.device)
    n = 0
    for _ in range(a.trials):
        pick = lambda: pool[int(torch.randint(len(pool), (1,), generator=g))]
        A, B, C = pick(), pick(), pick()
        while len({A, B, C}) < 3:
            C = pick()
        filler = [pick() for _ in range(a.seqlen)]
        bos = m.tokenizer.bos_token_id if m.tokenizer.bos_token_id is not None else pool[0]
        for ci, first in enumerate([A, C]):                 # STRUCT then NOSTRUCT
            seq = [bos, first, B] + filler + [A]
            if no_repeat_suffix:                             # break any local bigram echo
                seq = [bos, first, B] + filler[:-1] + [pick(), A]
            toks = torch.tensor([seq], device=W_U.device)
            B_pos, END = 2, len(seq) - 1
            _, c = m.run_with_cache(toks, names_filter=lambda nm: ("hook_pattern" in nm
                                                                   or "hook_z" in nm))
            uB = W_U[:, B].float()
            for L in range(nL):
                att[ci, L] += c["pattern", L][0, :, END, B_pos].float()
                z = c["z", L][0, END].float()               # [H, d_head]
                W_O = m.blocks[L].attn.W_O.float()          # [H, d_head, d_model]
                dla[ci, L] += torch.einsum("hd,hdm,m->h", z, W_O, uB)
            del c
        n += 1
    return (att / n).cpu(), (dla / n).cpu()

att, dla = run_pair(False)
att2, dla2 = run_pair(True)
print(f"[{a.model}] pairs run in {time.time()-t0:.0f}s", flush=True)

# ---------- natural arm: real far-irrelevant prefixes, which contain NO induction cue ----------
@torch.no_grad()
def natural_prefix_attention(n=40):
    """In natural no-cue prompts, how much attention does each head put on prefix CONTENT,
    and how much does it write those tokens? E2.3 measures the OV circuit given attention;
    this measures whether the QK circuit goes there at all without an antecedent."""
    S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"][:n]
    STOP = set(m.tokenizer.encode(" the of a an and is in to for on at .", add_special_tokens=False))
    att_pfx = torch.zeros(nL, nH, device=W_U.device); cnt = 0
    for s_ in S:
        v = s_["variants"].get("far_irr", [])
        if not v:
            continue
        prompt = f"{v[0]['context']} {s_['query']}"
        toks = m.to_tokens(prompt)
        n_ctx = m.to_tokens(v[0]["context"]).shape[1] - 1
        keep = [p for p in range(1, n_ctx + 1)
                if int(toks[0, p]) not in STOP
                and m.tokenizer.decode([int(toks[0, p])]).strip()[:1].isalnum()]
        if not keep:
            continue
        _, c = m.run_with_cache(toks, names_filter=lambda nm: "hook_pattern" in nm)
        END = toks.shape[1] - 1
        for L in range(nL):
            att_pfx[L] += c["pattern", L][0, :, END, keep].float().sum(-1)
        cnt += 1
        del c
    return (att_pfx / max(cnt, 1)).cpu()

nat = natural_prefix_attention()
print(f"[{a.model}] natural-prefix attention computed", flush=True)

E23 = json.load((ROOT / "results" / f"E2_3_{a.model}.json").open())
copiers = [tuple(x) for x in E23["top_heads"][:10]]
ind_flat = ind.flatten()
ind_heads = [(int(i // nH), int(i % nH)) for i in torch.topk(ind_flat, 10).indices.tolist()]
rng = torch.Generator().manual_seed(1)
ctrl = [(int(torch.randint(nL, (1,), generator=rng)), int(torch.randint(nH, (1,), generator=rng)))
        for _ in range(10)]

def summarise(name, heads):
    print(f"\n  {name} (n={len(heads)})")
    print(f"    {'head':9s} {'ind':>6s} | {'attn S':>7s} {'attn N':>7s} | "
          f"{'DLA_B S':>8s} {'DLA_B N':>8s} {'N/S':>6s} | {'nat pfx attn':>12s}")
    rows = []
    for (L, H) in heads:
        s, nn = float(dla[0, L, H]), float(dla[1, L, H])
        s2, n2 = float(dla2[0, L, H]), float(dla2[1, L, H])
        ratio = nn / s if abs(s) > 1e-6 else float("nan")
        r2 = n2 / s2 if abs(s2) > 1e-6 else float("nan")
        rows.append((ratio, nn - s))
        print(f"    L{L:2d}H{H:<4d} {float(ind[L,H]):6.3f} | {float(att[0,L,H]):7.3f} "
              f"{float(att[1,L,H]):7.3f} | {s:8.3f} {nn:8.3f} {ratio:6.2f} | "
              f"{float(nat[L,H]):12.3f}")
    fin = [r for r, _ in rows if r == r]
    natm = sorted(float(nat[L, H]) for (L, H) in heads)
    print(f"    median NOSTRUCT/STRUCT ratio = {sorted(fin)[len(fin)//2]:.2f} | "
          f"median natural prefix attention = {natm[len(natm)//2]:.3f}" if fin else "")

print(f"\n=== {a.model}: E2.1 cue removal ===")
summarise("E2.3 copier heads", copiers)
summarise("induction-positive heads (independent score)", ind_heads)
summarise("random matched control heads", ctrl)
json.dump(dict(model=a.model, induction=ind.tolist(), natural_prefix_attn=nat.tolist(),
               att=att.tolist(), dla=dla.tolist(), dla_nosuffix=dla2.tolist(),
               copiers=copiers, induction_heads=ind_heads, controls=ctrl),
          (ROOT / "results" / f"E2_1_{a.model}.json").open("w"))
