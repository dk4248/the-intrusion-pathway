"""Experiment 2 -- which part of the prefix mask produces its effect?

For a head at FIXED recipient inputs its END write splits exactly into a prefix part and the rest,
    z = z_P + z_U,   z_P = sum_{p in prefix} A[END,p] v_p,   a = sum_{p in prefix} A[END,p]
and a hard prefix-attention mask gives z_U / (1 - a). Inserted writes (hook_z at END):
    orig   z_P + z_U                 baseline (must equal the unpatched run)
    rm     z_U                       prefix information removed, its attention NOT given to others
    amp    z_P + z_U / (1 - a)       non-prefix contributions amplified, prefix write kept
    both   z_U / (1 - a)             the local result of the hard mask
    gain   (z_P + z_U) / (1 - a)     activation-gain control: the same scalar gain, nothing removed
plus the natural mask (pre-softmax knockout) -- identical to `both` for a single head, and at set
level the version where downstream set members see upstream edits.

Granularity: every head of Istar, Ronly and CTRL individually, then each complete population at
once (interactions measured, not summed). Conditions: far, long, golden (the cost), and
golden+irrelevant (ev_irr: irrelevant sentence, golden sentence, query) where the frozen mask is
applied to the whole context (deployable) and, as an oracle reference, to the irrelevant span only.
Readouts per variant, same forward pass: gold logit, gold log-prob, P(D_ans), P(D_lit), P(D_assoc),
margins, accuracy; and the immediate change of the edited write projected on gold and each D.
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from xp_common import XP, ROOT, run_variants, lt, byL

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=90)
ap.add_argument("--set-only", action="store_true",
                help="skip the per-head variants (report-only single-head/interaction lines, not used by any "
                     "Appendix table); set-level variants are unchanged (fp32 batch noise < 1e-4)")
a = ap.parse_args()
xp = XP(a.model, a.device); m = xp.m; dev = a.device
ALL = sorted(set(xp.Istar) | set(xp.Ronly) | set(xp.CTRL))
LAYERS = sorted({L for L, _ in ALL})
WRITES = ["rm", "amp", "both", "gain"]
keep = lambda n: any(n == f"blocks.{L}.attn.{k}" for L in LAYERS for k in ("hook_pattern", "hook_v", "hook_z"))

@torch.no_grad()
def parts(ex, cond, span):
    """z, z_P, a for every head in ALL at END of `cond`, prefix = positions `span`."""
    _, c = m.run_with_cache(ex["toks"][cond], names_filter=keep)
    out, err = {}, 0.0
    idx = torch.tensor(span, device=dev, dtype=torch.long)
    for L in LAYERS:
        A = c["pattern", L][0, :, -1, :].float(); v = xp.values(c, L); z = c["z", L][0, -1].float()
        for h in [h for (LL, h) in ALL if LL == L]:
            zf = A[h] @ v[:, h, :]
            err = max(err, float((zf - z[h]).abs().max() / (z[h].abs().max() + 1e-9)))
            zP = A[h, idx] @ v[idx, h, :] if len(span) else torch.zeros_like(zf)
            out[(L, h)] = (z[h], zP, float(A[h, idx].sum()) if len(span) else 0.0)
    del c
    return out, err

def write(kind, z, zP, a_):
    zU = z - zP; g = 1.0 / max(1.0 - a_, 1e-6)
    return {"orig": z, "rm": zU, "amp": zP + zU * g, "both": zU * g, "gain": z * g}[kind]

TG = lambda ex: {"gold": [ex["gid"]], "ans": ex["D"]["ans"], "lit": ex["D"]["lit"],
                 "assoc": ex["D"]["assoc"], "lit_long": ex["D"]["lit_long"]}

def imm(ex, edits):
    """immediate change of the summed write of edited heads, per target (raw DLA)."""
    d = None
    for (L, h), (z_new, z_old) in edits.items():
        w = (z_new - z_old) @ xp.WO[L][h]
        d = w if d is None else d + w
    return {t: float((d @ xp.WUf[:, ids]).mean()) for t, ids in TG(ex).items() if ids}

def build(ex, cond, P, heads_mode, span_tag="ctx"):
    V, meta = [{}], [("base", None, None)]
    span = list(range(1, ex["p_last"][cond] + 1)) if span_tag == "ctx" else \
        list(range(1, ex["irr_span_evirr"] + 1))
    idxP = lt(span, dev)
    if heads_mode == "single":
        for pn, heads in xp.POPS.items():
            for (L, h) in heads:
                z, zP, a_ = P[(L, h)]
                for k in WRITES:
                    V.append({"z": {L: (lt([h], dev), write(k, z, zP, a_)[None])}})
                    meta.append((f"{k}|{span_tag}", pn, (L, h)))
                if pn == "Istar":
                    V.append({"mask": {L: (lt([h], dev), idxP)}}); meta.append((f"mask|{span_tag}", pn, (L, h)))
    for pn, heads in xp.POPS.items():
        for k in ["orig"] + WRITES:
            zz = {}
            for L, hs in byL(heads).items():
                zz[L] = (lt(hs, dev), torch.stack([write(k, *P[(L, h)]) for h in sorted(hs)]))
            V.append({"z": zz}); meta.append((f"{k}|{span_tag}", pn, "set"))
        V.append({"mask": {L: (lt(hs, dev), idxP) for L, hs in byL(heads).items()}})
        meta.append((f"mask|{span_tag}", pn, "set"))
    return V, meta

rows = []; t0 = time.time(); worst = 0.0
for s in xp.HELD[: a.limit]:
    ex = xp.example(s)
    if ex is None: continue
    rec = {"sid": ex["sid"], "relation": ex["relation"], "res": {}}
    for cond, mode, spans in (("far", "single", ["ctx"]), ("long", "single", ["ctx"]),
                              ("golden", "single", ["ctx"]), ("ev_irr", "set", ["ctx", "oracle"])):
        if a.set_only: mode = "set"
        out = {}
        for sp in spans:
            span = list(range(1, ex["p_last"][cond] + 1)) if sp == "ctx" else list(range(1, ex["irr_span_evirr"] + 1))
            P, err = parts(ex, cond, span); worst = max(worst, err)
            V, meta = build(ex, cond, P, mode, sp)
            lg = run_variants(xp, ex["toks"][cond], V)
            base = xp.readout(lg[0], ex)
            out.setdefault("base", base)
            for b, (kind, pn, who) in enumerate(meta):
                if kind == "base": continue
                rd = xp.readout(lg[b], ex)
                d = {k: rd[k] - base[k] for k in rd if k in base}
                if who == "set":
                    heads = xp.POPS[pn]; wk = kind.split("|")[0]
                    if wk != "mask":
                        d["imm"] = imm(ex, {(L, h): (write(wk, *P[(L, h)]), P[(L, h)][0]) for (L, h) in heads})
                    d["a_mean"] = sum(P[(L, h)][2] for (L, h) in heads) / len(heads)
                elif kind.split("|")[0] != "mask":
                    wk = kind.split("|")[0]; L, h = who
                    d["imm"] = imm(ex, {(L, h): (write(wk, *P[(L, h)]), P[(L, h)][0])})
                    d["a"] = P[(L, h)][2]
                out[(kind, pn, who)] = d
        rec["res"][cond] = out
    rows.append(rec)
    if len(rows) % 5 == 0:
        print(f"  [{a.model}] {len(rows)} facts  {time.time()-t0:.0f}s  worst z-reconstruction {worst:.1e}", flush=True)

torch.save(dict(model=a.model, pops=xp.POPS, rows=rows), ROOT / "results" / f"XP2_{a.model}.pt")
print(f"[{a.model}] saved {len(rows)} facts; worst z-reconstruction {worst:.1e}")
