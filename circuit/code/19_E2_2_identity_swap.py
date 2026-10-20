"""E2.2 — change the source identity and follow the output.

On cue-broken prompts (far_irr has no [A]…[A] antecedent by construction) we replace one prefix
source token B by a matched token D and ask whether what the head writes FOLLOWS the identity.

Two levels:
  NATURAL    rewrite the prompt text B -> D and rerun everything (attention free to change)
  FIXED-ATTN keep the B-prompt's attention pattern, patch ONLY the value at that position with
             the D-prompt's value, so routing is held constant and only the source content moves

Readouts per head, at the swapped source position p:
    follow = (<w_D, u_D> - <w_D, u_B>) - (<w_B, u_D> - <w_B, u_B>)
             positive = the write tracks whichever token is actually there
and at END: the head's own output projected on u_B and u_D, plus the model's final logits.

Controls: matched donor tokens (same pool, length-matched), an edit at an UNRELATED position,
and layer/norm-matched control heads. Token-audited: a swap is kept only if it leaves the token
count of the prompt unchanged.
"""
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load, spaced_piece

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=80); ap.add_argument("--heads", type=int, default=10)
a = ap.parse_args()

m = load(a.model, a.device)
nL, nH = m.cfg.n_layers, m.cfg.n_heads
W_U = m.W_U.float()
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"][: a.limit]
HEADS = [tuple(x) for x in json.load((ROOT / "results" / f"E2_3_{a.model}.json").open())["top_heads"][: a.heads]]
gen = torch.Generator().manual_seed(0)
CTRL = [(int(torch.randint(nL, (1,), generator=gen)), int(torch.randint(nH, (1,), generator=gen)))
        for _ in range(a.heads)]
STOP = set(m.tokenizer.encode(" the of a an and is in to for on at .", add_special_tokens=False))

pool = []
for i in range(0, m.cfg.d_vocab, 3):
    s = spaced_piece(m.tokenizer, i)
    if s.startswith(" ") and s[1:].isalpha() and 3 <= len(s[1:]) <= 10 and s[1:].islower():
        pool.append(s[1:])
pool = pool[:2000]

def toklen(t):
    return m.to_tokens(t).shape[1]

@torch.no_grad()
def per_source(cache, L, H, p):
    vv = cache["v", L][0].float()
    n_kv = vv.shape[1]
    if n_kv != nH:
        vv = vv.repeat_interleave(nH // n_kv, dim=1)
    return vv[p, H] @ m.blocks[L].attn.W_O[H].float()          # [d_model]

acc = defaultdict(lambda: defaultdict(list))
n_ok = 0
t0 = time.time()
with torch.no_grad():
    for s in S:
        v = s["variants"].get("far_irr", [])
        if not v:
            continue
        ctx = v[0]["context"]
        words = [w for w in ctx.replace(".", "").split() if w.isalpha() and w.lower() not in
                 {"the", "of", "a", "an", "and", "is", "in", "to", "on", "at", "are"}]
        if not words:
            continue
        B_word = words[len(words) // 2]
        D_word = None
        for cand in [pool[int(torch.randint(len(pool), (1,), generator=gen))] for _ in range(25)]:
            if cand.lower() == B_word.lower():
                continue
            t2 = ctx.replace(B_word, cand, 1)
            if toklen(f"{t2} {s['query']}") == toklen(f"{ctx} {s['query']}"):
                D_word = cand; ctx_D = t2; break
        if D_word is None:
            continue
        pB = f"{ctx} {s['query']}"; pD = f"{ctx_D} {s['query']}"
        tB, tD = m.to_tokens(pB), m.to_tokens(pD)
        diff = (tB[0] != tD[0]).nonzero().flatten()
        if len(diff) != 1:                       # token-audited: exactly one token changed
            continue
        p = int(diff[0]); END = tB.shape[1] - 1
        idB, idD = int(tB[0, p]), int(tD[0, p])
        uB = W_U[:, idB].float(); uB = uB / uB.norm()
        uD = W_U[:, idD].float(); uD = uD / uD.norm()

        lB, cB = m.run_with_cache(tB)
        lD, cD = m.run_with_cache(tD)
        for tag, heads in (("copier", HEADS), ("control", CTRL)):
            for (L, H) in heads:
                wB = per_source(cB, L, H, p)
                wD = per_source(cD, L, H, p)
                follow = ((wD @ uD - wD @ uB) - (wB @ uD - wB @ uB)).item()
                acc[tag]["follow"].append(follow)
                # fixed-attention local effect: B-prompt routing, D-prompt value at p only
                aB = cB["pattern", L][0, H, END, p].float()
                d_o = aB * (wD - wB)                      # change in the head's END output
                acc[tag]["fixed_dB"].append(float(d_o @ uB))
                acc[tag]["fixed_dD"].append(float(d_o @ uD))
        # natural rerun, final distribution
        rB, rD = lB[0, END].float(), lD[0, END].float()
        acc["model"]["nat_dB"].append(float(rD[idB] - rB[idB]))
        acc["model"]["nat_dD"].append(float(rD[idD] - rB[idD]))
        n_ok += 1
        del cB, cD

def mci(xs, n=2000):
    import random
    r = random.Random(0)
    mu = sum(xs) / len(xs)
    d = sorted(sum(r.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return mu, d[int(.025 * n)], d[int(.975 * n)]

print(f"\n=== {a.model}: E2.2 source-identity swap ({n_ok} audited swaps, "
      f"{time.time()-t0:.0f}s) ===")
print("  follow = write moves toward whichever token is at that position (higher = identity-tracking)")
for tag in ("copier", "control"):
    f = mci(acc[tag]["follow"]); b = mci(acc[tag]["fixed_dB"]); d = mci(acc[tag]["fixed_dD"])
    print(f"  {tag:8s} follow {f[0]:+8.3f} [{f[1]:+.3f},{f[2]:+.3f}] | fixed-attn END write: "
          f"u_B {b[0]:+.4f}  u_D {d[0]:+.4f}  (D-B {d[0]-b[0]:+.4f})")
nb, nd = mci(acc["model"]["nat_dB"]), mci(acc["model"]["nat_dD"])
print(f"  natural rerun, final logits: u_B {nb[0]:+.3f} [{nb[1]:+.3f},{nb[2]:+.3f}]  "
      f"u_D {nd[0]:+.3f} [{nd[1]:+.3f},{nd[2]:+.3f}]")
json.dump({k: {kk: vv for kk, vv in d.items()} for k, d in acc.items()},
          (ROOT / "results" / f"E2_2_{a.model}.json").open("w"))
