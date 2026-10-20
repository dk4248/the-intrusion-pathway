"""Experiment 1 -- do the same heads and the same edits act on BOTH sides of the prediction?

Frozen populations (xp_common): Istar (paper's intrusion set), Ronly, CTRL. Held-out facts only.

A. Head-level changes, every head, every fact, gold and each distractor vocabulary separately:
       delta_g = DLA^irr(g) - DLA^ref(g)        delta_D = mean_{t in D} [DLA^irr(t) - DLA^ref(t)]
   with ref = bare query, and separately ref = token-exact matched filler (never pooled).
   D is fixed from the input: D_ans (the far context's answer), D_lit (literal prefix content words),
   D_assoc (association words, no literal tokens). long_irr has D_lit only.
B. Source decomposition of each head's END write, bare query included:
       o_h = sum_p A[END,p] v_p W_O  split into  bos | prefix | subj | rel | end
       delta o_h (irr - bare) = P_h^irr + (U_h^irr - U_h^bare)          (P_h^bare = 0)
       delta o_h (irr - fill) = (P_h^irr - P_h^fill) + (U_h^irr - U_h^fill)
   Each region decoded onto gold and every D. Exactness is checked per fact (sum of regions = z).
C. Interventions on each population, same facts, joint readouts from the SAME forward pass:
       restore_bare  population's END z <- bare-query value            (the existing clean-write edit)
       restore_fill  population's END z <- matched-filler value
       mask          pre-softmax END -> prefix knockout (the existing prefix-attention mask)
   readouts: gold logit, gold log-prob, P(D_ans), P(D_lit), P(D_assoc), margin (vs strongest non-gold
   token and vs strongest literal token), accuracy. Conditions far / long / golden.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from xp_common import XP, ROOT, run_variants, lt, byL

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=90)
a = ap.parse_args()
xp = XP(a.model, a.device); m = xp.m; nL, nH = xp.nL, xp.nH; dev = a.device
REG = ["bos", "prefix", "subj", "rel", "end"]
TGT = ["gold", "ans", "lit", "assoc", "lit_long"]
CONDS = ["bare", "far", "long", "golden", "fill_far", "fill_long"]
keep = lambda n: n.endswith("hook_z") or n.endswith("hook_v") or n.endswith("hook_pattern")

@torch.no_grad()
def analyse_run(ex, cond):
    """Final readout, per-head DLA per target, region decomposition, region attention mass."""
    toks = ex["toks"][cond]; T = toks.shape[1]
    lg, cache = m.run_with_cache(toks, names_filter=keep)
    Rm = torch.stack([ex["regions"][cond][r] for r in REG]).float().to(dev)      # [5, T]
    ids = {"gold": [ex["gid"]], "ans": ex["D"]["ans"], "lit": ex["D"]["lit"],
           "assoc": ex["D"]["assoc"], "lit_long": ex["D"]["lit_long"]}
    dla_tot = torch.full((len(TGT), nL, nH), float("nan"))
    dla_reg = torch.full((len(TGT), nL, nH, len(REG)), float("nan"))
    amass = torch.zeros(nL, nH, len(REG)); zerr = 0.0; zend = torch.zeros(nL, nH, xp.WO[0].shape[1])
    for L in range(nL):
        A = cache["pattern", L][0, :, -1, :].float()                 # [H, T]
        v = xp.values(cache, L)                                      # [T, H, dh]
        zr = torch.einsum("ht,rt,thd->hrd", A, Rm, v)                # [H, 5, dh]
        z = cache["z", L][0, -1].float()                             # [H, dh]
        zerr = max(zerr, float((zr.sum(1) - z).abs().max() / (z.abs().max() + 1e-6)))
        zend[L] = z.cpu()
        o_r = torch.einsum("hrd,hdm->hrm", zr, xp.WO[L])             # [H, 5, dm]
        o = torch.einsum("hd,hdm->hm", z, xp.WO[L])
        for ti, t in enumerate(TGT):
            if not ids[t]: continue
            U = xp.WUf[:, ids[t]]
            dla_reg[ti, L] = (o_r @ U).mean(-1).cpu()
            dla_tot[ti, L] = (o @ U).mean(-1).cpu()
        amass[L] = (A @ Rm.T).cpu()
    del cache
    rd = xp.readout(lg[0, -1], ex)
    return dict(readout=rd, dla=dla_tot.half(), dla_reg=dla_reg.half(), amass=amass.half(),
                zerr=zerr), zend

def pop_variants(pops, z_donor, cond, ex):
    """restore (from donor z) and mask variants for each population."""
    V, names = [{}], ["base"]
    P = list(range(1, ex["p_last"][cond] + 1))
    for pn, heads in pops.items():
        for tag, zd in z_donor.items():
            zz = {L: (lt(hs, dev), zd[L][sorted(hs)].to(dev)) for L, hs in byL(heads).items()}
            V.append({"z": zz}); names.append(f"restore_{tag}|{pn}")
        if P:
            V.append({"mask": {L: (lt(hs, dev), lt(P, dev)) for L, hs in byL(heads).items()}})
            names.append(f"mask|{pn}")
    return V, names

rows = []; t0 = time.time(); worst = 0.0
for s in xp.HELD[: a.limit]:
    ex = xp.example(s)
    if ex is None: continue
    rec = {"sid": ex["sid"], "relation": ex["relation"], "nD": {k: len(v) for k, v in ex["D"].items()},
           "runs": {}, "iv": {}}
    Z = {}
    for c in CONDS:
        r, Z[c] = analyse_run(ex, c)
        rec["runs"][c] = r; worst = max(worst, r["zerr"])
    grab = {L: lt(hs, dev) for L, hs in byL(sorted(set(xp.Istar) | set(xp.Ronly) | set(xp.CTRL))).items()}
    for c, donors in (("far", {"bare": Z["bare"], "fill": Z["fill_far"]}),
                      ("long", {"bare": Z["bare"], "fill": Z["fill_long"]}),
                      ("golden", {"bare": Z["bare"]})):
        V, names = pop_variants(xp.POPS, donors, c, ex)
        lg, G = run_variants(xp, ex["toks"][c], V, grab=grab)
        out = {}
        for b, nm in enumerate(names):
            rd = xp.readout(lg[b], ex)
            # the populations' own summed write after the edit, per target
            for pn, heads in xp.POPS.items():
                o = sum(torch.einsum("d,dm->m", G[L][0][b, sorted(grab[L].tolist()).index(h)],
                                     xp.WO[L][h]) for L, h in heads)
                for t, ids in (("gold", [ex["gid"]]), ("ans", ex["D"]["ans"]), ("lit", ex["D"]["lit"]),
                               ("assoc", ex["D"]["assoc"])):
                    if ids: rd[f"w_{pn}_{t}"] = float((o @ xp.WUf[:, ids]).mean())
            out[nm] = rd
        rec["iv"][c] = out
    rows.append(rec)
    if len(rows) % 10 == 0:
        print(f"  [{a.model}] {len(rows)} facts  {time.time()-t0:.0f}s  worst region-sum error {worst:.2e}",
              flush=True)

torch.save(dict(model=a.model, pops=xp.POPS, copier=xp.COPIER, REG=REG, TGT=TGT, rows=rows),
           ROOT / "results" / f"XP1_{a.model}.pt")
print(f"[{a.model}] saved {len(rows)} facts; worst relative region-sum error {worst:.2e}")
