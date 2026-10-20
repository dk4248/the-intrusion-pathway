"""Experiment 3 -- is earlier copying upstream of the intrusion heads, or do they read the prefix directly?

Receivers: the frozen intrusion heads (Istar), each one separately and all at once; Ronly and CTRL as
receiver sets for comparison. Senders: the E2.3 copier heads strictly below the receiver's layer.
Recipient: an irrelevant-context run (far, long). Donors, reported separately: the bare query (END is
alignable; no prefix position is invented) and a token-exact matched filler.

END-mediated input (query side only -- prefix keys and values untouched):
    q_cop    receiver query input at END  =  resid_pre_L[END] - sum_senders o_c^recip + sum o_c^donor
    q_attn   same with every earlier attention layer's END write as the sender   (broader)
    q_mlp    same with every earlier MLP's END write                             (broader)
    q_full   the whole END query input replaced by the donor's resid_pre_L[END]  (broadest)
    q = ln1(input) W_Q + b_Q, recomputed for that head only and patched before rotary.
Direct contextual access (END information untouched):
    acc_mask    the receiver's END -> prefix attention knocked out (renormalised)
    acc_kvfill  the receiver reads the FILLER run's keys and values at prefix positions (same length),
                recomputed for that head only, so GQA partner heads are unaffected

The chain is recorded for every intervention: query change (cosine, relative norm) -> attention change
(prefix mass, mass on the distractor-answer tokens) -> receiver write change (raw DLA on gold and each D)
-> final prediction (gold logit, log-prob, P(D_*), margins, accuracy) from a forward pass in which only
the receiver's computation is edited. Self-patches (donor = recipient) must reproduce the base.
"""
import argparse, sys, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from xp_common import XP, ROOT, run_variants, lt, byL

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=90)
ap.add_argument("--set-only", action="store_true",
                help="skip the per-receiver variants (report only, not used by any Appendix table)")
a = ap.parse_args()
xp = XP(a.model, a.device); m = xp.m; dev = a.device; nL, nH = xp.nL, xp.nH
ROT = "blocks.0.attn.hook_rot_q" in m.hook_dict
QN, KN = ("hook_rot_q", "hook_rot_k") if ROT else ("hook_q", "hook_k")
RECV_POPS = {"Istar": xp.Istar, "Ronly": xp.Ronly, "CTRL": xp.CTRL}
ALLR = sorted(set(xp.Istar) | set(xp.Ronly) | set(xp.CTRL))
RL = sorted({L for L, _ in ALLR})
COP = xp.COPIER
IV = ["q_cop_bare", "q_cop_fill", "q_attn_bare", "q_attn_fill", "q_mlp_bare", "q_mlp_fill",
      "q_full_bare", "q_full_fill", "acc_mask", "acc_kvfill"]

def names(n):
    return (n.endswith("hook_attn_out") or n.endswith("hook_mlp_out") or n.endswith("hook_resid_pre")
            or n.endswith("attn.hook_z") or n.endswith("attn.hook_v") or n.endswith(f"attn.{KN}")
            or n.endswith(f"attn.{QN}") or n.endswith("attn.hook_q") or n.endswith("attn.hook_pattern"))

@torch.no_grad()
def grab(toks):
    _, c = m.run_with_cache(toks, names_filter=names)
    out = {"resid": {L: c["resid_pre", L][0, -1].float() for L in RL},
           "attn": torch.stack([c["attn_out", L][0, -1].float() for L in range(nL)]),
           "mlp": torch.stack([c["mlp_out", L][0, -1].float() for L in range(nL)]),
           "zc": {(L, h): c["z", L][0, -1, h].float() for L, h in COP},
           "q": {L: c["q", L][0, -1].float() for L in RL},
           "qr": {L: c[f"blocks.{L}.attn.{QN}"][0, -1].float() for L in RL},
           "k": {L: c[f"blocks.{L}.attn.{KN}"][0].float() for L in RL},
           "v": {L: c["v", L][0].float() for L in RL},
           "pat": {L: c["pattern", L][0, :, -1, :].float() for L in RL},
           "z": {L: c["z", L][0, -1].float() for L in RL}}
    del c
    return out

def q_of(L, h, x):
    """pre-rotary query of head h at layer L from a residual input vector x [d_model]."""
    y = m.blocks[L].ln1(x[None, None, :].to(torch.float32))[0, 0].float()
    return y @ m.blocks[L].attn.W_Q[h].float() + m.blocks[L].attn.b_Q[h].float()

def cop_write(G, senders):
    return sum((G["zc"][c] @ xp.WO[c[0]][c[1]] for c in senders), torch.zeros(m.cfg.d_model, device=dev))

def kv_swap_z(R, F, L, h, prefix):
    """receiver z at END with filler keys/values at prefix positions (head h only)."""
    kvh = h // xp.REP
    k = R["k"][L][:, kvh, :].clone(); v = R["v"][L][:, kvh, :].clone()
    idx = torch.tensor(prefix, device=dev, dtype=torch.long)
    k[idx] = F["k"][L][idx, kvh, :]; v[idx] = F["v"][L][idx, kvh, :]
    sc = (R["qr"][L][h] @ k.T) / m.blocks[L].attn.attn_scale
    A = torch.softmax(sc, -1)
    return A @ v, A

rows = []; t0 = time.time(); worst_q = worst_k = 0.0
for s in xp.HELD[: a.limit]:
    ex = xp.example(s)
    if ex is None: continue
    rec = {"sid": ex["sid"], "relation": ex["relation"], "res": {}}
    Gb = grab(ex["toks"]["bare"])
    for cond, fillc in (("far", "fill_far"), ("long", "fill_long")):
        R = grab(ex["toks"][cond]); F = grab(ex["toks"][fillc])
        prefix = list(range(1, ex["p_last"][cond] + 1)); idxP = lt(prefix, dev)
        # distractor-answer token positions (far only)
        dpos = []
        if cond == "far" and ex["D"]["ans"]:
            toks = ex["toks"][cond][0].tolist()
            dpos = [p for p in prefix if toks[p] in set(ex["D"]["ans"]) | set(ex["D"]["lit"])]
        # self-checks: recomputed query and recomputed z (with the recipient's own K/V) match the cache
        for L, h in ALLR:
            worst_q = max(worst_q, float((q_of(L, h, R["resid"][L]) - R["q"][L][h]).abs().max()
                                         / (R["q"][L][h].abs().max() + 1e-9)))
            zc, _ = kv_swap_z(R, R, L, h, prefix)
            worst_k = max(worst_k, float((zc - R["z"][L][h]).abs().max() / (R["z"][L][h].abs().max() + 1e-9)))
        donors = {"bare": Gb, "fill": F}

        def q_new(kind, dn, L, h):
            D = donors[dn]; x = R["resid"][L]
            if kind == "cop":
                snd = [c for c in COP if c[0] < L]
                if not snd: return None
                x = x - cop_write(R, snd) + cop_write(D, snd)
            elif kind == "attn":
                x = x - R["attn"][:L].sum(0) + D["attn"][:L].sum(0)
            elif kind == "mlp":
                x = x - R["mlp"][:L].sum(0) + D["mlp"][:L].sum(0)
            elif kind == "full":
                x = D["resid"][L]
            return q_of(L, h, x)

        def variant(iv, heads):
            v = {}
            if iv.startswith("q_"):
                _, kind, dn = iv.split("_")
                for L, hs in byL(heads).items():
                    qs = [(h, q_new(kind, dn, L, h)) for h in sorted(hs)]
                    qs = [(h, q) for h, q in qs if q is not None]
                    if qs: v.setdefault("q", {})[L] = (lt([h for h, _ in qs], dev), torch.stack([q for _, q in qs]))
            elif iv == "acc_mask":
                v["mask"] = {L: (lt(hs, dev), idxP) for L, hs in byL(heads).items()}
            elif iv == "acc_kvfill":
                v["z"] = {L: (lt(hs, dev), torch.stack([kv_swap_z(R, F, L, h, prefix)[0] for h in sorted(hs)]))
                          for L, hs in byL(heads).items()}
            return v

        V, meta = [{}], [("base", None, None)]
        for L, h in ([] if a.set_only else xp.Istar):
            for iv in IV:
                vv = variant(iv, [(L, h)])
                if vv: V.append(vv); meta.append((iv, "Istar", (L, h)))
        for pn, heads in RECV_POPS.items():
            for iv in IV:
                vv = variant(iv, heads)
                if vv: V.append(vv); meta.append((iv, pn, "set"))
        # self-patch controls: donor = recipient must reproduce the base exactly
        donors["self"] = R
        for kind in ("cop", "full"):
            vv = {}
            for L, hs in byL(xp.Istar).items():
                qs = [(h, q_new(kind, "self", L, h)) for h in sorted(hs)]
                qs = [(h, q) for h, q in qs if q is not None]
                if qs: vv.setdefault("q", {})[L] = (lt([h for h, _ in qs], dev), torch.stack([q for _, q in qs]))
            V.append(vv); meta.append((f"self_{kind}", "Istar", "set"))
        gr = {L: lt([h for (LL, h) in ALLR if LL == L], dev) for L in RL}
        lg, G = run_variants(xp, ex["toks"][cond], V, grab=gr)
        base = xp.readout(lg[0], ex)
        out = {"base": base, "dpos": dpos}
        tg = {"gold": [ex["gid"]], "ans": ex["D"]["ans"], "lit": ex["D"]["lit"], "assoc": ex["D"]["assoc"],
              "lit_long": ex["D"]["lit_long"]}
        for b, (iv, pn, who) in enumerate(meta):
            if iv == "base": continue
            rd = xp.readout(lg[b], ex)
            d = {k: rd[k] - base[k] for k in rd if k in base}
            heads = [who] if who != "set" else RECV_POPS[pn]
            chain = {"dq_cos": [], "dq_rel": [], "a0": [], "a1": [], "d0": [], "d1": []}
            dw = None
            for L, h in heads:
                j = gr[L].tolist().index(h)
                z1, p1 = G[L][0][b, j], G[L][1][b, j]
                z0, p0 = G[L][0][0, j], G[L][1][0, j]
                w = (z1 - z0) @ xp.WO[L][h]; dw = w if dw is None else dw + w
                chain["a0"].append(float(p0[idxP].sum())); chain["a1"].append(float(p1[idxP].sum()))
                if dpos:
                    ip = lt(dpos, dev)
                    chain["d0"].append(float(p0[ip].sum())); chain["d1"].append(float(p1[ip].sum()))
                if iv.startswith("q_"):
                    _, kind, dn = iv.split("_")
                    qn = q_new(kind, dn, L, h)
                    if qn is not None:
                        q0 = R["q"][L][h]
                        chain["dq_cos"].append(float(torch.nn.functional.cosine_similarity(qn, q0, dim=0)))
                        chain["dq_rel"].append(float((qn - q0).norm() / q0.norm()))
            d["chain"] = {k: (sum(v) / len(v) if v else float("nan")) for k, v in chain.items()}
            d["dw"] = {t: float((dw @ xp.WUf[:, ids]).mean()) for t, ids in tg.items() if ids}
            if who != "set" and iv.startswith("q_cop"):
                d["n_senders"] = len([c for c in COP if c[0] < who[0]])
            out[(iv, pn, who)] = d
        rec["res"][cond] = out
    rows.append(rec)
    if len(rows) % 5 == 0:
        print(f"  [{a.model}] {len(rows)} facts {time.time()-t0:.0f}s  q-recompute err {worst_q:.1e}  "
              f"z-recompute err {worst_k:.1e}", flush=True)

torch.save(dict(model=a.model, pops=RECV_POPS, copier=COP, IV=IV, rows=rows),
           ROOT / "results" / f"XP3_{a.model}.pt")
print(f"[{a.model}] saved {len(rows)} facts; q-recompute err {worst_q:.1e}; z-recompute err {worst_k:.1e}")
