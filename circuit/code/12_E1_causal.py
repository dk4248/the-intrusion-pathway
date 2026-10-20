"""E1.2 / E1.3-causal / E1.5 — do the MLP writes MATTER, and where does their content come from?

Interventions, all applied to `mlp_out` at ONE position and a band of layers:
  zero      cancel the MLP write            (Geva et al. 2023 sec.4 sublayer knockout)
  resample  replace with a matched DONOR's write at the same role position
            (a non-zero baseline: cancelling to 0 is off-distribution, resampling is not)
  addback   restore the exact cached write  -> must reproduce the clean run (a GATE)
  attnoff   zero attn_out at the same position for all layers BEFORE the band, then read the
            MLP write inside the band  -> E1.5: is the MLP's association content created
            locally, or inherited through earlier attention?

Positions: subj      the query's subject token
           dstr      the distractor token inside the prefix
           qry_next  the token right after the subject -- relation-bearing, reported as a
                     comparison, NOT as a null control
           ctrl_pfx  an early prefix function word   } genuine null controls
           ctrl_null a query function word           }
Readouts:  FINAL - gold logit / log-prob / gold-vs-best-competitor margin / argmax at END.
           LOCAL - projection at the edited position onto the association bank minus a
                   norm-matched control bank, i.e. did the edit remove the association content.
"""
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load
from lre_pool import load_relations
from assoc_vocab import (subject_attr_cloud, category_cloud, wiki_attr_cloud,
                         query_category_words, other_valid_answers)

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--pairs", default="pairs_v1.json")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--gate-n", type=int, default=5, help="samples on which to run the addback gate")
ap.add_argument("--bands", default="early,mid,late_mid,late", help="bands for zero/resample (Table B.2 uses early, mid)")
ap.add_argument("--skip-e15", action="store_true", help="skip the E1.5 attention-kill runs (not in the Appendix)")
ap.add_argument("--shard", default="", help="k/n: only samples with index %% n == k; donors keep global indices")
a = ap.parse_args()

m = load(a.model, a.device)
R = load_relations()
S = json.load((ROOT / "data" / a.pairs).open())["samples"]
if a.limit:
    S = S[: a.limit]
nL = m.cfg.n_layers
W_U = m.W_U
uni_norm = W_U.norm(dim=0).float()
BANDS = {"early": range(0, nL // 4), "mid": range(nL // 4, nL // 2),
         "late_mid": range(nL // 2, 3 * nL // 4), "late": range(3 * nL // 4, nL)}

def first_id(w):
    ids = m.tokenizer.encode(" " + str(w), add_special_tokens=False)
    return ids[0] if ids else None

def ids_of(ws, cap=64):
    out = [first_id(w) for w in ws]
    return sorted({i for i in out if i is not None})[:cap]

alpha_ids = [i for i in range(0, m.cfg.d_vocab, 7)
             if m.tokenizer.decode([i]).strip().isalpha()
             and len(m.tokenizer.decode([i]).strip()) > 3][:4000]
alpha_norm = uni_norm[alpha_ids]

def ctrl_ids(target, k=40):
    if not target:
        return []
    tgt = uni_norm[torch.tensor(target, device=uni_norm.device)].mean()
    order = torch.argsort((alpha_norm - tgt).abs())
    ban = set(target)
    out = []
    for j in order.tolist():
        if alpha_ids[j] not in ban:
            out.append(alpha_ids[j])
        if len(out) >= k:
            break
    return out

def char_to_tok(prompt, ce):
    return m.to_tokens(prompt[:ce]).shape[1] - 1

@torch.no_grad()
def final_metrics(logits, gid):
    row = logits[0, -1].float()
    lse = torch.logsumexp(row, -1)
    masked = row.clone(); masked[gid] = float("-inf")
    return dict(gold_logit=float(row[gid]), gold_logprob=float(row[gid] - lse),
                margin=float(row[gid] - masked.max()),
                hit=int(int(row.argmax()) == gid))

@torch.no_grad()
def local_assoc(cache, pos, bank, ctrl):
    """Association content at the edited position, read from the last layer's residual."""
    x = cache["resid_post", nL - 1][0, pos].float()
    if m.cfg.normalization_type in ("LN", "LNPre"):
        x = x - x.mean(-1, keepdim=True)
    x = x / x.norm() * (x.shape[-1] ** 0.5)
    p = x @ W_U.float()
    b = float(p[torch.tensor(bank, device=p.device)].mean()) if bank else float("nan")
    c = float(p[torch.tensor(ctrl, device=p.device)].mean()) if ctrl else float("nan")
    return b - c

rows = []
t0 = time.time()
SK, SN = map(int, a.shard.split("/")) if a.shard else (0, 1)
RUN_BANDS = {b: BANDS[b] for b in a.bands.split(",")}
for si, s in enumerate(S):
    if si % SN != SK:
        continue
    gid = first_id(s["gold"])
    if gid is None:
        continue
    A_s = ids_of(subject_attr_cloud(s["subject"], exclude=[s["gold"]])
                 + wiki_attr_cloud(s["subject"], exclude=[s["gold"]])[:80])
    donor = S[(si + 7) % len(S)]                      # matched donor, fixed offset
    for cond in ["nctx", "far_irr"]:
        vs = [None] if cond == "nctx" else s["variants"].get("far_irr", [])[:1]
        for v in vs:
            prompt = s["query"] if v is None else f"{v['context']} {s['query']}"
            toks = m.to_tokens(prompt)
            END = toks.shape[1] - 1
            pos, banks = {}, {}
            cs = prompt.rfind(s["subject"])
            if cs >= 0:
                pos["subj"] = min(char_to_tok(prompt, cs + len(s["subject"])), END)
                banks["subj"] = A_s
            if v is not None and v.get("distractor"):
                d = v["distractor"]
                dp = v["context"].rfind(d)
                crel = R.get(v["context_relation"])
                if dp >= 0:
                    pos["dstr"] = min(char_to_tok(prompt, dp + len(d)), END)
                    banks["dstr"] = ids_of(
                        category_cloud(d, crel["range"], exclude=[s["gold"]])
                        + wiki_attr_cloud(d, exclude=[s["gold"]])[:80]) if crel else []
                pos["ctrl_pfx"] = max(1, min(2, END))       # an early prefix token
                banks["ctrl_pfx"] = banks.get("dstr", A_s)
            if "subj" in pos and pos["subj"] + 1 < END:
                pos["qry_next"] = pos["subj"] + 1      # relation-bearing, NOT a null control
                banks["qry_next"] = A_s
            qstart = char_to_tok(prompt, prompt.rfind(s["query"])) if v is not None else 1
            nullp = max(1, min(qstart + 1, END - 1))
            if nullp not in pos.values():
                pos["ctrl_null"] = nullp               # a function word in the query
                banks["ctrl_null"] = A_s

            base_logits, base_cache = m.run_with_cache(toks)
            base = final_metrics(base_logits, gid)
            cached_mlp = {L: base_cache["mlp_out", L].clone() for L in range(nL)}
            donor_mlp = None

            for pname, p in pos.items():
                bank = banks.get(pname) or []
                ctrl = ctrl_ids(bank)
                base_loc = local_assoc(base_cache, p, bank, ctrl)
                rows.append(dict(sid=s["sample_id"], cond=cond, position=pname, band="none",
                                 mode="baseline", layer_n=0, **base, local=base_loc))
                for bname, band in RUN_BANDS.items():
                    for mode in ["zero", "resample"] + (["addback"] if si < a.gate_n else []):
                        if mode == "resample" and donor_mlp is None:
                            dtoks = m.to_tokens(donor["query"])
                            _, dc = m.run_with_cache(dtoks)
                            dend = dtoks.shape[1] - 1
                            donor_mlp = {L: dc["mlp_out", L][0, dend].clone() for L in range(nL)}
                            del dc
                        hooks = []
                        for L in band:
                            def mk(L=L, p=p, mode=mode):
                                def hk(x, hook):
                                    if mode == "zero":
                                        x[:, p, :] = 0
                                    elif mode == "resample":
                                        x[:, p, :] = donor_mlp[L].to(x.dtype)
                                    else:
                                        x[:, p, :] = cached_mlp[L][:, p, :].to(x.dtype)
                                    return x
                                return hk
                            hooks.append((f"blocks.{L}.hook_mlp_out", mk()))
                        with m.hooks(fwd_hooks=hooks):      # run_with_cache takes no fwd_hooks
                            lg, ca = m.run_with_cache(toks)
                        met = final_metrics(lg, gid)
                        loc = local_assoc(ca, p, bank, ctrl)
                        rows.append(dict(sid=s["sample_id"], cond=cond, position=pname,
                                         band=bname, mode=mode, layer_n=len(list(band)),
                                         **met, local=loc,
                                         d_gold_logit=met["gold_logit"] - base["gold_logit"],
                                         d_margin=met["margin"] - base["margin"],
                                         d_local=loc - base_loc))
                        del ca
                # E1.5: kill attention INTO this position before each band, keep MLPs
                for bname, band in ({} if a.skip_e15 else BANDS).items():
                    start = list(band)[0]
                    if start == 0:
                        continue
                    hooks = [(f"blocks.{L}.hook_attn_out",
                              (lambda p=p: (lambda x, hook: (x.__setitem__((slice(None), p, slice(None)), 0), x)[1]))())
                             for L in range(start)]
                    with m.hooks(fwd_hooks=hooks):
                        lg, ca = m.run_with_cache(toks)
                    met = final_metrics(lg, gid)
                    loc = local_assoc(ca, p, bank, ctrl)
                    rows.append(dict(sid=s["sample_id"], cond=cond, position=pname,
                                     band=bname, mode="attnoff_before", layer_n=start,
                                     **met, local=loc,
                                     d_gold_logit=met["gold_logit"] - base["gold_logit"],
                                     d_margin=met["margin"] - base["margin"],
                                     d_local=loc - base_loc))
                    del ca
            del base_cache, cached_mlp
    if (si + 1) % 20 == 0:
        print(f"  [{a.model}] {si+1}/{len(S)}  {time.time()-t0:.0f}s", flush=True)

out = ROOT / "results" / (f"E1_causal_{a.model}.json" if not a.shard else f"E1_causal_{a.model}.shard{SK}of{SN}.json")
json.dump(dict(model=a.model, n_samples=len(S), rows=rows), out.open("w"))
print(f"[{a.model}] {len(rows)} rows in {time.time()-t0:.0f}s -> {out}", flush=True)
