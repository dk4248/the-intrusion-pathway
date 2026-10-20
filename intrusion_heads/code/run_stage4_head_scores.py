"""Stage 4 — per-head scores under 6 conditions.

For each (sample, condition), capture per-(layer, head) statistics at END:
  * att_to_ctx  : sum of attention from END to prefix (context) token positions
  * att_to_gold : attention from END to the position where the gold token first appears
                  in the prompt (only meaningful when prefix contains gold; computed under
                  every condition where the gold-token id appears in the prompt tokens).
  * att_to_dstr : attention from END to the position where the distractor token first
                  appears in the prompt (under cf/related conditions; '__auto__' for irr).
  * att_max     : max attention from END to any non-BOS, non-END token
  * att_entropy : entropy of the attention distribution at END over all positions
  * dla_gold    : head's direct logit attribution to gold token at END
                  = (z[h, END] @ W_O[h]) @ W_U[:, gold_id]
  * dla_dstr    : same for distractor (where available)
  * z_norm      : ||z[h, END, :]||_2

Output: results/<model>__head_scores.npz   — scores [n_samples, n_conditions, n_layers, n_heads, n_stats]
        results/<model>__head_scores.json  — metadata + records: [{ sample_id, condition, seq_len,
          end_pos, pos_gold, pos_dstr, ctx_token_len, gold_id, dstr_id }, ...]
"""
import json, time, argparse, math
from pathlib import Path


import torch
import numpy as np

from transformer_lens import HookedTransformer, utils
REPO = Path(__file__).resolve().parents[2]  # repository root

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAMPLES_PATH = REPO / "lre_dataset/data/samples_6x250.json"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    ("nctx",            None),
    ("golden",          "golden"),
    ("related",         "related"),
    ("counterfactual",  "counterfactual"),
    ("short_irr",       "irrelevant"),
    ("long_irr",        "long_irrelevant"),
]


def short(m): return m.split("/")[-1]


def safe_first_pos(tokens_1d, target_id):
    """Return first position in tokens_1d matching target_id, or -1."""
    matches = (tokens_1d == target_id).nonzero(as_tuple=True)[0]
    return int(matches[0].item()) if len(matches) > 0 else -1


@torch.no_grad()
def run_one(model, prompt, gold_id, dstr_id, n_layers, n_heads, d_head, W_U):
    """Run one forward pass; return compact per-head stats.

    Returns dict with:
      n_ctx_toks (we don't know which positions are 'ctx' from this fn alone — caller passes ctx_len)
      per_layer_head: [n_layers, n_heads, n_stats]  (float32 numpy)
    """
    device = next(model.parameters()).device
    tokens = model.to_tokens(prompt).to(device)
    seq_len = tokens.shape[1]
    end_pos = seq_len - 1

    # Hook every layer's attention pattern + z output at END.
    names_pattern = [utils.get_act_name("pattern", L) for L in range(n_layers)]
    names_z       = [utils.get_act_name("z",       L) for L in range(n_layers)]
    keep = set(names_pattern + names_z)
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in keep)

    # Positions of gold/dstr tokens in prompt (first occurrence in non-END positions)
    pos_gold = safe_first_pos(tokens[0, :-1], gold_id) if gold_id is not None else -1
    pos_dstr = safe_first_pos(tokens[0, :-1], dstr_id) if dstr_id is not None else -1

    # Stats order:
    # 0: att_to_ctx_excl (later — we don't know ctx length here; just sum non-END, non-pos-0)
    # 1: att_to_gold
    # 2: att_to_dstr
    # 3: att_max
    # 4: att_entropy
    # 5: dla_gold
    # 6: dla_dstr
    # 7: z_norm
    n_stats = 8
    out = np.zeros((n_layers, n_heads, n_stats), dtype=np.float32)

    for L in range(n_layers):
        pat = cache[utils.get_act_name("pattern", L)][0]  # [n_heads, seq, seq]
        z   = cache[utils.get_act_name("z",       L)][0]  # [seq, n_heads, d_head] in TL

        # Attention from END across all key positions: shape [n_heads, seq]
        att_end = pat[:, end_pos, :].float()

        # Attention to all non-BOS, non-END positions
        non_be_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        non_be_mask[0] = False; non_be_mask[end_pos] = False
        att_non_be = att_end[:, non_be_mask]
        out[L, :, 0] = att_non_be.sum(dim=-1).cpu().numpy()
        out[L, :, 3] = att_non_be.max(dim=-1).values.cpu().numpy()

        # entropy
        eps = 1e-12
        p = att_end.clamp_min(eps)
        ent = -(p * torch.log(p)).sum(dim=-1)
        out[L, :, 4] = ent.cpu().numpy()

        # Attention to specific positions
        if 0 <= pos_gold < seq_len:
            out[L, :, 1] = att_end[:, pos_gold].cpu().numpy()
        if 0 <= pos_dstr < seq_len:
            out[L, :, 2] = att_end[:, pos_dstr].cpu().numpy()

        # z at END: [n_heads, d_head]
        z_end = z[end_pos, :, :].float()    # [n_heads, d_head]
        out[L, :, 7] = z_end.norm(dim=-1).cpu().numpy()

        # DLA: head's residual contribution = z[h] @ W_O[L, h, :, :]   shape [d_model]
        # Then DLA on token t = (z[h] @ W_O[L, h]) @ W_U[:, t]
        # W_O is [n_layers, n_heads, d_head, d_model] in TL
        W_O = model.W_O[L]  # [n_heads, d_head, d_model]
        # residual contribution per head: einsum n,nh,nhd->nd  but actually
        # for each head h: r_h = z_end[h] @ W_O[h]    [d_model]
        r_per_head = torch.einsum("hd,hdm->hm", z_end, W_O.float())  # [n_heads, d_model]
        # DLA on gold/dstr
        if gold_id is not None:
            dla_gold = (r_per_head @ W_U[:, gold_id].float())  # [n_heads]
            out[L, :, 5] = dla_gold.cpu().numpy()
        if dstr_id is not None and dstr_id != gold_id:
            dla_dstr = (r_per_head @ W_U[:, dstr_id].float())  # [n_heads]
            out[L, :, 6] = dla_dstr.cpu().numpy()

    return {
        "seq_len": seq_len,
        "end_pos": end_pos,
        "pos_gold": pos_gold,
        "pos_dstr": pos_dstr,
        "scores": out.tolist(),  # nested list for JSON
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-samples", type=int, default=None)
    args = ap.parse_args()

    mname = short(args.model)
    out_path = OUT / f"{mname}__head_scores.json"
    t0 = time.time()
    print(f"[{mname}] loading...", flush=True)
    model = HookedTransformer.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True, device=args.device,
    )
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    nL = model.cfg.n_layers
    nH = model.cfg.n_heads
    dh = model.cfg.d_head
    dm = model.cfg.d_model
    W_U = model.W_U  # [d_model, vocab]
    print(f"[{mname}] nL={nL} nH={nH} dh={dh} dm={dm}", flush=True)

    data = json.load(open(SAMPLES_PATH))
    samples = data["samples"]
    if args.n_samples:
        samples = samples[: args.n_samples]
    N = len(samples)
    print(f"[{mname}] {N} samples × {len(CONDITIONS)} conds = {N*len(CONDITIONS)} forwards", flush=True)

    tok = model.tokenizer
    C = len(CONDITIONS)
    n_stats = 8
    scores_all = np.zeros((N, C, nL, nH, n_stats), dtype=np.float32)
    metadata = {
        "model": args.model, "model_short": mname,
        "n_layers": nL, "n_heads": nH, "d_head": dh, "d_model": dm,
        "stat_names": ["att_to_ctx_excl", "att_to_gold", "att_to_dstr",
                       "att_max", "att_entropy", "dla_gold", "dla_dstr", "z_norm"],
        "conditions": [c[0] for c in CONDITIONS],
        "n_samples": N,
        "records": [],   # per (sample, cond): seq_len, end_pos, pos_gold, pos_dstr, ctx_token_len
    }

    for i, samp in enumerate(samples):
        sid = samp["sample_id"]
        query = samp["query"]
        gold = samp["gold"]
        gold_id = tok.encode(" " + gold, add_special_tokens=False)[0]
        for ci, (cond_name, vkey) in enumerate(CONDITIONS):
            if vkey is None:
                prompt = query
                dstr_id = None
                ctx_len = 0
            else:
                v = samp["variants"][vkey]
                ctx = v["context"]
                prompt = f"{ctx} {query}"
                ctx_len = len(tok.encode(ctx, add_special_tokens=False))
                dstr = v.get("distractor")
                if dstr is not None and dstr != "__auto__":
                    dstr_id = tok.encode(" " + dstr, add_special_tokens=False)[0]
                else:
                    dstr_id = None
            r = run_one(model, prompt, gold_id, dstr_id, nL, nH, dh, W_U)
            # Convert scores back to numpy array (currently nested list)
            scores_all[i, ci, :, :, :] = np.asarray(r["scores"], dtype=np.float32)
            metadata["records"].append({
                "sample_id": sid, "condition": cond_name,
                "seq_len": r["seq_len"], "end_pos": r["end_pos"],
                "pos_gold": r["pos_gold"], "pos_dstr": r["pos_dstr"],
                "ctx_token_len": ctx_len, "gold_id": int(gold_id),
                "dstr_id": int(dstr_id) if dstr_id is not None else -1,
            })
        if i % max(1, N//20) == 0:
            print(f"[{mname}] {i+1}/{N}  elapsed={time.time()-t0:.0f}s", flush=True)

    metadata["elapsed_sec"] = time.time() - t0
    # Save scores tensor in compressed NPZ, metadata in JSON
    npz_path = OUT / f"{mname}__head_scores.npz"
    json_path = OUT / f"{mname}__head_scores.json"
    np.savez_compressed(npz_path, scores=scores_all)
    json_path.write_text(json.dumps(metadata))
    print(f"[{mname}] DONE in {(time.time()-t0)/60:.1f} min -> {npz_path} + {json_path}", flush=True)


if __name__ == "__main__":
    main()
