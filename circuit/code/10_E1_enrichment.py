"""E1.1 + E1.3 — where does association information enter, and at which positions?

Two readouts per (layer, position), deliberately both:

 (A) GEVA-STYLE SUBLAYER PROJECTION  (Geva et al. 2023, "Dissecting Recall of Factual
     Associations", sec. 3). Project the MLP sublayer output itself to vocabulary through the
     unembedding, take the top-k tokens, and report the ATTRIBUTES RATE = the fraction of
     those top-k that fall in a pre-registered attribute set. Ranking is invariant to any
     positive rescaling, so the frozen final-LN scale used here does not affect top-k or the
     rate; it only puts the magnitudes on a comparable scale.

 (B) DIFFERENCE OF LENS READINGS, post - pre, the more conservative additive diagnostic:
     it never applies a nonlinear norm to an isolated vector.

Architecture note: pythia-6.9b has parallel_attn_mlp=True, so attention and MLP both read the
SAME layer input and there is no resid_mid. We therefore define pre := resid_post - mlp_out,
which is exact for both sequential (Llama, GPT-2) and parallel (pythia) blocks. In a parallel
block the layer-L MLP cannot read layer-L attention at all -- relevant for E1.5.

Positions
    subj  last token of the query's subject        -> E1.1 (subject enrichment)
    dstr  the distractor token inside the prefix   -> E1.3 (does the MLP enrich an
                                                            IRRELEVANT source token?)
    end   last prompt position

Readout banks, all fixed before the run and disjoint (see assoc_vocab.py):
    gold   the exact gold token            ident  the distractor token ITSELF (source identity)
    A_s    subject attributes (LRE)        A_d    distractor's category, d removed
    C_r    query-category words            B_r    other valid answers of the query relation
    ctrl   norm-matched unrelated tokens
"""
import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load
from lre_pool import load_relations
from assoc_vocab import (subject_attr_cloud, category_cloud, query_category_words,
                         other_valid_answers, wiki_attr_cloud)

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--pairs", default="pairs_v1.json")
ap.add_argument("--conditions", default="nctx,golden,far_irr,long_irr")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--topk", type=int, default=20, help="k for the Geva attributes rate")
ap.add_argument("--shard", default="", help="k/n: only samples with index %% n == k; controls keep global indices")
ap.add_argument("--dump-tokens", type=int, default=3,
                help="samples for which to save decoded top-k tokens for manual audit")
a = ap.parse_args()

m = load(a.model, a.device)
R = load_relations()
S = json.load((ROOT / "data" / a.pairs).open())["samples"]
if a.limit:
    S = S[: a.limit]
CONDS = a.conditions.split(",")
nL, d_model = m.cfg.n_layers, m.cfg.d_model
W_U = m.W_U                                            # [d_model, vocab]
uni_norm = W_U.norm(dim=0).float()                     # per-token unembedding norm

def first_id(w):
    ids = m.tokenizer.encode(" " + str(w), add_special_tokens=False)
    return ids[0] if ids else None

def ids_of(words, cap=64):
    out = []
    for w in words:
        i = first_id(w)
        if i is not None:
            out.append(i)
    return sorted(set(out))[:cap]

# norm-matched control tokens: alphabetic, unrelated, matched on ||W_U[:,t]||
alpha_ids = [i for i in range(0, m.cfg.d_vocab, 7)
             if m.tokenizer.decode([i]).strip().isalpha()
             and len(m.tokenizer.decode([i]).strip()) > 3][:4000]
alpha_norm = uni_norm[alpha_ids]

def ctrl_ids(target_ids, k=40):
    """Tokens whose unembedding norm matches the bank's, but semantically unrelated."""
    if not target_ids:
        return []
    tgt = uni_norm[torch.tensor(target_ids, device=uni_norm.device)].mean()
    d = (alpha_norm - tgt).abs()
    order = torch.argsort(d)
    banned = set(target_ids)
    out = []
    for j in order.tolist():
        if alpha_ids[j] in banned:
            continue
        out.append(alpha_ids[j])
        if len(out) >= k:
            break
    return out

def char_to_tok(prompt, char_end):
    return m.to_tokens(prompt[:char_end]).shape[1] - 1

@torch.no_grad()
def lens(state, scale):
    """Logit lens with the model's final LN. `state` [d_model] -> [vocab] float32."""
    x = state.float()
    if m.cfg.normalization_type in ("LN", "LNPre"):
        x = x - x.mean(-1, keepdim=True)
    x = x / scale
    w = getattr(m.ln_final, "w", None)
    if w is not None:
        x = x * w.float()
    return x @ W_U.float()

# A Wikipedia summary is ~half encyclopedic boilerplate ('country', 'officially', 'covers',
# 'area'), so a norm-matched control does not rule out the MLP simply writing generic
# encyclopedia vocabulary. The sharper control is ANOTHER entity's wiki cloud: same
# boilerplate, different attributes. A_sO / A_dO below.
rows = []
t0 = time.time()
SK, SN = map(int, a.shard.split("/")) if a.shard else (0, 1)
for si, s in enumerate(S):
    if si % SN != SK:
        continue
    other = S[(si + 53) % len(S)]
    rel = R[s["relation"]]
    gold_id = first_id(s["gold"])
    banks_base = {
        "gold": [gold_id] if gold_id is not None else [],
        # A_s   : LRE cross-relation attributes (small, exact)
        # A_sW  : Wikipedia-summary vocabulary (large -> makes Geva's top-k rate usable)
        "A_s": ids_of(subject_attr_cloud(s["subject"], exclude=[s["gold"]])),
        # like-for-like control: ANOTHER subject's LRE attributes. Same kind of word
        # (common content words), different entity. The norm-matched `ctrl` bank is made of
        # rare alphabetic tokens, so it controls for norm but NOT for word frequency/kind.
        "A_sX": ids_of(subject_attr_cloud(other["subject"], exclude=[s["gold"]])),
        "A_sW": ids_of(wiki_attr_cloud(s["subject"], exclude=[s["gold"]]), cap=150),
        "A_sO": ids_of(wiki_attr_cloud(other["subject"], exclude=[s["gold"]]), cap=150),
        "C_r": ids_of(query_category_words(s["relation"])),
        "B_r": ids_of(other_valid_answers(s["relation"], s["gold"])),
    }
    for cond in CONDS:
        variants = [None] if cond == "nctx" else s["variants"].get(cond, [])[:1]
        for v in variants:
            prompt = s["query"] if v is None else f"{v['context']} {s['query']}"
            toks = m.to_tokens(prompt)
            END = toks.shape[1] - 1
            pos = {"end": END}
            cpos = prompt.rfind(s["subject"])
            if cpos >= 0:
                pos["subj"] = min(char_to_tok(prompt, cpos + len(s["subject"])), END)
            banks = dict(banks_base)
            if v is not None and v.get("distractor"):
                d = v["distractor"]
                dpos = v["context"].rfind(d)
                if dpos >= 0:
                    pos["dstr"] = min(char_to_tok(prompt, dpos + len(d)), END)
                crel = R.get(v["context_relation"])
                banks["ident"] = [first_id(d)] if first_id(d) else []
                banks["A_d"] = ids_of(category_cloud(d, crel["range"], exclude=[s["gold"]])) if crel else []
                banks["A_dW"] = ids_of(wiki_attr_cloud(d, exclude=[s["gold"]]), cap=150)
                od = None
                for ov in other["variants"].get("far_irr", []):
                    if ov.get("distractor") and ov["distractor"] != d:
                        od = ov["distractor"]; break
                banks["A_dO"] = ids_of(wiki_attr_cloud(od, exclude=[s["gold"]]), cap=150) if od else []
                ocrel = R.get(ov["context_relation"]) if od else None
                banks["A_dX"] = ids_of(category_cloud(od, ocrel["range"], exclude=[s["gold"]])) \
                    if (od and ocrel) else []
            banks["ctrl"] = ctrl_ids(sum((banks.get(k, []) for k in ("A_s", "A_d", "gold")), []))
            # control matched to the LARGE wiki banks separately (different norm profile)
            banks["ctrlW"] = ctrl_ids(sum((banks.get(k, []) for k in ("A_sW", "A_dW")), []), k=150)
            bank_ids = {k: torch.tensor(ids, device=W_U.device) for k, ids in banks.items() if ids}

            _, cache = m.run_with_cache(
                toks, names_filter=lambda n: ("hook_mlp_out" in n or "resid_post" in n
                                              or "ln_final.hook_scale" in n))
            ln_scale = cache["ln_final.hook_scale"][0].float()   # [pos, 1] - PER POSITION
            bank_sets = {k: set(v) for k, v in banks.items() if v}
            for pname, p in pos.items():
                # freeze the baseline final-LN scale AT THIS POSITION. Using the END scale
                # everywhere rescales each position by a different constant and makes
                # cross-position magnitudes incomparable.
                scale_end = ln_scale[p]
                for L in range(nL):
                    mlp_out = cache["mlp_out", L][0, p]
                    post_state = cache["resid_post", L][0, p]
                    pre_state = post_state - mlp_out          # exact for parallel and serial
                    pre = lens(pre_state, scale_end)
                    post = lens(post_state, scale_end)
                    gv = lens(mlp_out, scale_end)             # (A) Geva sublayer projection
                    topk = torch.topk(gv, a.topk).indices.tolist()
                    tset = set(topk)
                    rec = dict(sid=s["sample_id"], relation=s["relation"], cond=cond,
                               position=pname, layer=L)
                    for bk, bids in bank_ids.items():
                        rec[f"dmlp_{bk}"] = float(post[bids].mean() - pre[bids].mean())   # (B)
                        rec[f"proj_{bk}"] = float(gv[bids].mean())                        # (A) mag
                        rec[f"rate_{bk}"] = len(tset & bank_sets[bk]) / a.topk             # (A) rate
                    if si < a.dump_tokens:
                        rec["top_tokens"] = [m.tokenizer.decode([t]) for t in topk[:10]]
                    rows.append(rec)
            del cache
    if (si + 1) % 40 == 0:
        print(f"  [{a.model}] {si+1}/{len(S)} samples  {time.time()-t0:.0f}s", flush=True)

out = ROOT / "results" / (f"E1_enrich_{a.model}.json" if not a.shard else f"E1_enrich_{a.model}.shard{SK}of{SN}.json")
json.dump(dict(model=a.model, n_samples=len(S), conditions=CONDS, rows=rows), out.open("w"))
print(f"[{a.model}] {len(rows)} rows in {time.time()-t0:.0f}s -> {out}", flush=True)
