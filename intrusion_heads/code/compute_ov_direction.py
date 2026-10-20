"""Compute per-(layer, head, condition) average OV write direction.

For each (sample, condition):
  forward, cache z[layer, end_pos, head, d_head]
  per (layer, head):
    r_h = z_h @ W_O[layer, head]   : [d_model]   (head's contribution to END residual)
    accumulate per (cond, layer, head): mean_write[cond, layer, head] += r_h
    per-sample save:
      dla_gold[sample, cond, layer, head]  = r_h . W_U[:, gold_id]
      prefix_writer[sample, cond, layer, head]
        = mean over prefix positions p of (r_h . W_U[:, token_at_pos_p])
        = (mean over p of W_U[:, token_id_p])   .   r_h
        (computed efficiently per sample)

Saves:
  results/<model>__ovdir_mean.npz   : mean_write [C, L, H, d_model]
  results/<model>__ovdir_scalar.npz : dla_gold [N, C, L, H], prefix_writer [N, C, L, H]
  results/<model>__ovdir.json       : metadata
"""
import json, time, argparse
from pathlib import Path


import torch
import numpy as np

from transformer_lens import HookedTransformer, utils
REPO = Path(__file__).resolve().parents[2]  # repository root

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAMPLES_PATH = REPO / "lre_dataset/data/samples_6x250.json"
OUT = ROOT / "results"

CONDITIONS = [
    ("nctx",            None),
    ("golden",          "golden"),
    ("related",         "related"),
    ("counterfactual",  "counterfactual"),
    ("short_irr",       "irrelevant"),
    ("long_irr",        "long_irrelevant"),
]


def short(m): return m.split("/")[-1]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-samples", type=int, default=None)
    args = ap.parse_args()

    mname = short(args.model)
    out_mean_npz = OUT / f"{mname}__ovdir_mean.npz"
    out_sc_npz   = OUT / f"{mname}__ovdir_scalar.npz"
    out_json     = OUT / f"{mname}__ovdir.json"
    t0 = time.time()
    print(f"[{mname}] loading...", flush=True)
    model = HookedTransformer.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True, device=args.device,
    )
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    nL = model.cfg.n_layers
    nH = model.cfg.n_heads
    dh = model.cfg.d_head
    dm = model.cfg.d_model
    print(f"[{mname}] L={nL} H={nH} dh={dh} dm={dm}", flush=True)

    data = json.load(open(SAMPLES_PATH))
    samples = data["samples"]
    if args.n_samples:
        samples = samples[: args.n_samples]
    N = len(samples); C = len(CONDITIONS)
    print(f"[{mname}] N={N} C={C}", flush=True)

    # Allocate
    mean_write = torch.zeros((C, nL, nH, dm), dtype=torch.float32, device=args.device)
    cond_count = np.zeros(C, dtype=np.int64)
    dla_gold = np.zeros((N, C, nL, nH), dtype=np.float32)
    prefix_writer = np.zeros((N, C, nL, nH), dtype=np.float32)

    tok = model.tokenizer
    W_U = model.W_U.float()  # [dm, vocab]
    # W_O per-layer accessed inside loop

    z_names = [utils.get_act_name("z", L) for L in range(nL)]
    keep = set(z_names)

    for i, samp in enumerate(samples):
        sid = samp["sample_id"]
        query = samp["query"]
        gold = samp["gold"]
        gold_id = tok.encode(" " + gold, add_special_tokens=False)[0]

        for ci, (cn, vkey) in enumerate(CONDITIONS):
            if vkey is None:
                prompt = query
                ctx_start, ctx_end = 0, 0
            else:
                ctx = samp["variants"][vkey]["context"]
                prompt = f"{ctx} {query}"
                ctx_only = tok.encode(ctx, add_special_tokens=False)
                full = model.to_tokens(prompt)[0].tolist()
                # ctx slice in full tokens: skip BOS if present
                ctx_start = 1 if len(full) > 1 and full[0] != ctx_only[0] else 0
                ctx_end = ctx_start + len(ctx_only)

            tokens = model.to_tokens(prompt).to(args.device)
            seq_len = tokens.shape[1]
            end_pos = seq_len - 1
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in keep)

            # Prefix token ids (in the full tokenization)
            if ctx_end > ctx_start:
                prefix_ids = tokens[0, ctx_start:ctx_end]  # [n_prefix]
                # W_U[:, prefix_ids] : [dm, n_prefix]
                U_pref = W_U[:, prefix_ids]
                U_pref_mean = U_pref.mean(dim=-1)  # [dm]
            else:
                U_pref_mean = None

            for L in range(nL):
                z = cache[utils.get_act_name("z", L)][0, end_pos, :, :].float()  # [n_heads, d_head]
                # r_h = z_h @ W_O[L, h]   for each head h
                W_O = model.W_O[L].float()  # [n_heads, d_head, d_model]
                r = torch.einsum("hd,hdm->hm", z, W_O)  # [n_heads, d_model]
                # accumulate
                mean_write[ci, L, :, :] += r
                # per-sample stats
                dg = (r * W_U[:, gold_id]).sum(dim=-1)  # [n_heads]
                dla_gold[i, ci, L, :] = dg.cpu().numpy()
                if U_pref_mean is not None:
                    pw = (r * U_pref_mean).sum(dim=-1)  # [n_heads]
                    prefix_writer[i, ci, L, :] = pw.cpu().numpy()
                # else: stays 0
            cond_count[ci] += 1
            del cache; torch.cuda.empty_cache()

        if i % max(1, N // 20) == 0:
            print(f"[{mname}] {i+1}/{N}  elapsed={time.time()-t0:.0f}s", flush=True)

    # Finalize mean_write
    for ci in range(C):
        if cond_count[ci] > 0:
            mean_write[ci] /= cond_count[ci]

    np.savez_compressed(out_mean_npz, mean_write=mean_write.cpu().numpy(),
                        conditions=np.array([c[0] for c in CONDITIONS]))
    np.savez_compressed(out_sc_npz, dla_gold=dla_gold, prefix_writer=prefix_writer)
    meta = {
        "model": args.model, "model_short": mname,
        "n_layers": nL, "n_heads": nH, "d_head": dh, "d_model": dm,
        "n_samples": N, "conditions": [c[0] for c in CONDITIONS],
        "elapsed_sec": time.time() - t0,
    }
    out_json.write_text(json.dumps(meta, indent=2))
    print(f"[{mname}] DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
