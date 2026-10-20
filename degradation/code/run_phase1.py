"""Phase-1 behavioural degradation metrics on the canonical LRE dataset (Appendix D, Table D.1).

For every sample and context class (golden / related / counterfactual / irrelevant / random) computes
final-layer scalar metrics at the last position — Δ_g (gold logit shift), Δ_d (distractor logit shift),
Δ_m, KL(nctx || wctx), gold/distractor ranks, argmax flip — plus a logit-lens trajectory every
`--lens-every` samples. The long-irrelevant class is handled by run_phase1_long_irrelevant.py, which
reuses the helpers defined here.

Results are written to degradation/results/<model>/phase1.json.
Supports model parallelism via --n-devices for models that do not fit on one GPU.

Usage:
    python run_phase1.py --model meta-llama/Llama-3.1-8B --dataset samples_5x250 \\
        --device cuda:0 --tag smoke --n-samples 10
"""

import json, time, argparse, string
from pathlib import Path

import torch

from transformer_lens import HookedTransformer

HERE = Path(__file__).resolve().parent
V2DATA = HERE.parent.parent / "lre_dataset" / "data"   # samples_5x250.json
OUT_ROOT = HERE.parent / "results"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

CLASSES = ["golden", "related", "counterfactual", "irrelevant", "random"]


STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "and", "or", "but", "not", "no",
    "this", "that", "these", "those", "it", "its",
    "their", "they", "them",
    "as", "so", "too", "very", "much", "many",
    "all", "any", "some", "one", "two", "three",
    "i", "you", "we", "he", "she",
    "into", "under", "over", "about",
    "also", "just", "only", "often", "always", "usually", "almost", "typically", "traditionally", "primarily",
    "several", "many", "few", "deep", "immense", "eight", "six", "twelve", "-eight",
    "long",
    "s",
}


def short(m): return m.split("/")[-1]


# ─────────────────────────────────────────────────────────────
# Phase 1: scalar metrics + logit lens
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def last_logits(model, prompt):
    tokens = model.to_tokens(prompt).to(next(model.parameters()).device)
    logits = model(tokens)[0, -1, :].float()
    probs  = torch.softmax(logits, dim=-1)
    return logits, probs, tokens


@torch.no_grad()
def lens_trajectory(model, tokens, gold_id, dstr_id):
    """MP-safe logit lens: iterate layers manually, moving each residual to W_U.device."""
    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: "resid_post" in n or n == "blocks.0.hook_resid_pre",
    )
    unembed_device = model.W_U.device
    u_g = model.W_U[:, gold_id].float()
    u_d = model.W_U[:, dstr_id].float()
    g_out = []; d_out = []
    # layer 0 input (embed only)
    if "blocks.0.hook_resid_pre" in cache.cache_dict:
        r0 = cache["blocks.0.hook_resid_pre"][0, -1, :].float().to(unembed_device)
        r0_ln = model.ln_final(r0.unsqueeze(0).unsqueeze(0))[0, 0, :]
        g_out.append(float((u_g * r0_ln).sum().item()))
        d_out.append(float((u_d * r0_ln).sum().item()))
    for L in range(model.cfg.n_layers):
        name = f"blocks.{L}.hook_resid_post"
        if name not in cache.cache_dict:
            continue
        r = cache[name][0, -1, :].float().to(unembed_device)
        r_ln = model.ln_final(r.unsqueeze(0).unsqueeze(0))[0, 0, :]
        g_out.append(float((u_g * r_ln).sum().item()))
        d_out.append(float((u_d * r_ln).sum().item()))
    del cache; torch.cuda.empty_cache()
    return g_out, d_out


def extract_content_token_ids(tokenizer, context, query):
    out = []; seen = set()
    ctx = context.rstrip(".")
    query_ids = set(tokenizer.encode(query, add_special_tokens=False))
    for word in ctx.split():
        w = word.strip(string.punctuation + "'").lower()
        if not w or w in STOPWORDS or not w.isalpha() or len(w) < 3:
            continue
        original = word.strip(string.punctuation + "'")
        for cand in [" " + original, " " + original.lower(), " " + original.capitalize()]:
            ids = tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in query_ids and ids[0] not in seen:
                out.append((ids[0], cand.strip())); seen.add(ids[0]); break
    return out


def rank_of(logits, tok_id):
    return int((torch.argsort(logits, descending=True) == tok_id).nonzero(as_tuple=True)[0].item()) + 1


@torch.no_grad()
def phase1_one_pair(model, wctx, nctx, gold_id, dstr_id, do_lens=False):
    w_l, w_p, w_t = last_logits(model, wctx)
    n_l, n_p, n_t = last_logits(model, nctx)

    lg_w, lg_n = w_l[gold_id].item(), n_l[gold_id].item()
    ld_w, ld_n = w_l[dstr_id].item(), n_l[dstr_id].item()
    pg_w, pg_n = w_p[gold_id].item(), n_p[gold_id].item()
    pd_w, pd_n = w_p[dstr_id].item(), n_p[dstr_id].item()
    rg_w = rank_of(w_l, gold_id); rg_n = rank_of(n_l, gold_id)
    rd_w = rank_of(w_l, dstr_id); rd_n = rank_of(n_l, dstr_id)
    kl = torch.nn.functional.kl_div(torch.log(w_p + 1e-30), n_p, reduction="sum").item()
    am_w = int(w_l.argmax().item()); am_n = int(n_l.argmax().item())
    cn = int(am_n == gold_id); cw = int(am_w == gold_id)

    rec = {
        "lg_wctx": lg_w, "lg_nctx": lg_n, "ld_wctx": ld_w, "ld_nctx": ld_n,
        "delta_gold":   lg_w - lg_n,
        "delta_dstr":   ld_w - ld_n,
        "delta_margin": (lg_w - ld_w) - (lg_n - ld_n),
        "pg_wctx": pg_w, "pg_nctx": pg_n, "pd_wctx": pd_w, "pd_nctx": pd_n,
        "rank_gold_wctx": rg_w, "rank_gold_nctx": rg_n,
        "rank_dstr_wctx": rd_w, "rank_dstr_nctx": rd_n,
        "kl_nctx_to_wctx": kl,
        "argmax_wctx_id": am_w, "argmax_nctx_id": am_n,
        "correct_nctx": cn, "correct_wctx": cw,
        "acc_flip_wctx_lost": int(cn and not cw),
    }
    if do_lens:
        ll_wg, ll_wd = lens_trajectory(model, w_t, gold_id, dstr_id)
        ll_ng, ll_nd = lens_trajectory(model, n_t, gold_id, dstr_id)
        rec.update({"lens_wctx_gold": ll_wg, "lens_wctx_dstr": ll_wd,
                    "lens_nctx_gold": ll_ng, "lens_nctx_dstr": ll_nd})
    return rec


@torch.no_grad()
def phase1_irrelevant(model, query, nctx_prompt, context, gold_id, device, do_lens=False):
    tok = model.tokenizer
    wctx_prompt = f"{context} {query}"
    w_l, w_p, w_t = last_logits(model, wctx_prompt)
    n_l, n_p, _ = last_logits(model, nctx_prompt)
    cands = extract_content_token_ids(tok, context, query)
    profile = []
    for tid, tstr in cands:
        dl = (w_l[tid] - n_l[tid]).item()
        dp = (w_p[tid] - n_p[tid]).item()
        profile.append({"token_id": tid, "token": tstr, "delta_logit": dl, "delta_prob": dp})
    profile.sort(key=lambda x: x["delta_logit"], reverse=True)
    if profile:
        auto = profile[0]
        auto_id = auto["token_id"]; auto_tok = auto["token"]
    else:
        auto_id = int(w_l.argmax().item()); auto_tok = tok.decode([auto_id])
    full = phase1_one_pair(model, wctx_prompt, nctx_prompt, gold_id, auto_id, do_lens=do_lens)
    full["context"] = context
    full["auto_distractor_id"] = auto_id
    full["auto_distractor_tok"] = auto_tok
    full["ctx_entrainment_profile"] = profile[:20]
    full["n_ctx_tokens_entrained_above_1"] = sum(1 for p in profile if p["delta_logit"] > 1.0)
    full["sum_ctx_entrainment"] = sum(p["delta_logit"] for p in profile)
    return full


def run_phase1(model, samples, lens_every=None, verbose=True):
    """Returns list of records. lens_every: compute lens trajectories for every kth sample per class."""
    tok = model.tokenizer
    records = []
    t0 = time.time()
    for i, samp in enumerate(samples):
        query = samp["query"]; gold = samp["gold"]
        gold_id = tok.encode(" " + gold, add_special_tokens=False)[0]
        nctx = query
        do_lens = (lens_every is not None and (i % lens_every) == 0)
        for cls in CLASSES:
            v = samp["variants"][cls]
            context = v["context"]; dstr_spec = v["distractor"]
            wctx = f"{context} {query}"
            if dstr_spec == "__auto__":
                out = phase1_irrelevant(model, query, nctx, context, gold_id,
                                         device=next(model.parameters()).device,
                                         do_lens=do_lens)
                out.update({
                    "sample_id": samp["sample_id"], "relation": samp.get("relation"),
                    "class": cls, "query": query,
                    "gold": gold, "gold_id": gold_id,
                    "distractor": out["auto_distractor_tok"], "distractor_id": out["auto_distractor_id"],
                    "wctx_prompt": wctx, "nctx_prompt": nctx,
                    "auto_distractor": True,
                })
            else:
                dstr_id = tok.encode(" " + dstr_spec, add_special_tokens=False)[0]
                out = phase1_one_pair(model, wctx, nctx, gold_id, dstr_id, do_lens=do_lens)
                out.update({
                    "sample_id": samp["sample_id"], "relation": samp.get("relation"),
                    "class": cls, "context": context, "query": query,
                    "gold": gold, "gold_id": gold_id,
                    "distractor": dstr_spec, "distractor_id": dstr_id,
                    "wctx_prompt": wctx, "nctx_prompt": nctx,
                    "auto_distractor": False,
                })
            records.append(out)
        if verbose and (i % max(1, len(samples) // 20) == 0):
            print(f"[phase1] {i+1}/{len(samples)}  elapsed={time.time()-t0:.0f}s", flush=True)
    return records


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="samples_5x250", help="dataset name in lre_dataset/data/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-devices", type=int, default=1)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n-samples", type=int, default=None, help="subset for smoke tests")
    ap.add_argument("--lens-every", type=int, default=50,
                    help="lens trajectory every kth sample (phase1)")
    args = ap.parse_args()

    mname = short(args.model)
    out_dir = OUT_ROOT / mname
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log_hdr = f"[{mname}|{args.tag}]"

    # ── Load data ──
    samples_path = V2DATA / f"{args.dataset}.json"
    data = json.load(open(samples_path))
    samples = data["samples"]
    if args.n_samples:
        samples = samples[:args.n_samples]
    print(f"{log_hdr} loaded {len(samples)} samples from {samples_path.name}", flush=True)

    # ── Load model ──
    print(f"{log_hdr} loading model  n_devices={args.n_devices}", flush=True)
    t0 = time.time()
    load_kwargs = dict(dtype=torch.float16, trust_remote_code=True)
    if args.n_devices > 1:
        load_kwargs["n_devices"] = args.n_devices
        # device argument is inferred from n_devices when > 1
    else:
        load_kwargs["device"] = args.device
    model = HookedTransformer.from_pretrained(args.model, **load_kwargs)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    print(f"{log_hdr} loaded in {time.time()-t0:.0f}s  L={model.cfg.n_layers} H={model.cfg.n_heads} d={model.cfg.d_model}",
          flush=True)

    results_meta = {
        "model": args.model, "model_short": mname,
        "dataset": args.dataset, "n_samples": len(samples),
        "tag": args.tag,
        "n_layers": model.cfg.n_layers, "n_heads": model.cfg.n_heads, "d_model": model.cfg.d_model,
        "n_devices": args.n_devices,
    }

    # ── Phase 1 ──
    print(f"{log_hdr} === PHASE 1 ===", flush=True)
    t = time.time()
    phase1_recs = run_phase1(model, samples, lens_every=args.lens_every, verbose=True)
    payload = {**results_meta, "records": phase1_recs, "elapsed_sec": time.time() - t}
    (out_dir / "phase1.json").write_text(json.dumps(payload, indent=2))
    print(f"{log_hdr} phase1 done in {(time.time()-t)/60:.1f} min; wrote phase1.json "
          f"({len(phase1_recs)} records)", flush=True)

    total = (time.time() - t_start) / 60
    print(f"\n{log_hdr} ALL DONE in {total:.1f} min  -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
