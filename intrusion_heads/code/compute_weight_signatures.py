"""Compute per-(layer, head) weight-level signatures for each model. For speed,
project to a SAMPLED vocab subset (not full vocab), and batch all heads of a layer
in one matmul.

For each head h:
  1. ov_copy_score      : top-K (K=5) accuracy of recovering the input token from
                          W_E[t] @ W_OV[h] @ W_U,  over a sampled vocab subset.
  2. ov_expected_logit  : mean of W_E[t]^T W_OV[h] W_U[:, t]   over sampled t.
                          (Positive = head writes input token to output direction.)
  3. ov_frob             : ||W_O[h] W_V[h]||_F
  4. ov_effective_rank   : stable rank ||W_OV||_F^2 / ||W_OV||_2^2 (spectral norm by power iteration).
  5. qk_copy_score        : Olsson-style. Fraction of sampled t where W_E[t] @ W_QK[h] @
                          W_E[t]^T is in top-K row of [W_E @ W_QK @ W_E^T]_t.
"""
import json, time, argparse
from pathlib import Path


import torch
import numpy as np

from transformer_lens import HookedTransformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results"

N_TOK = 512      # sampled vocab subset size
TOP_K = 5


def short(m): return m.split("/")[-1]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    mname = short(args.model)
    out_npz = OUT / f"{mname}__weight_sigs.npz"
    out_json = OUT / f"{mname}__weight_sigs.json"
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
    V  = model.tokenizer.vocab_size
    print(f"[{mname}] L={nL} H={nH} dh={dh} dm={dm} V={V}", flush=True)

    rng = np.random.default_rng(0)
    tok_ids = rng.choice(V, size=min(N_TOK, V), replace=False)
    tok_ids_t = torch.as_tensor(tok_ids, device=args.device, dtype=torch.long)

    W_E = model.W_E.float()           # [V, d_m]
    W_U = model.W_U.float()           # [d_m, V]
    E_sample = W_E[tok_ids_t]         # [N, d_m]
    U_sample = W_U[:, tok_ids_t]      # [d_m, N]
    U_sample_T = U_sample.T           # [N, d_m]

    ov_copy = np.zeros((nL, nH), dtype=np.float32)
    ov_exp_logit = np.zeros((nL, nH), dtype=np.float32)
    ov_frob = np.zeros((nL, nH), dtype=np.float32)
    ov_eff_rank = np.zeros((nL, nH), dtype=np.float32)
    qk_copy = np.zeros((nL, nH), dtype=np.float32)

    for L in range(nL):
        # Build batched W_OV[h] = W_V[h] @ W_O[h] : [n_heads, d_m, d_m]
        # W_O has shape [n_heads, d_head, d_model]; W_V has shape [n_heads, d_model, d_head]
        W_O = model.W_O[L].float()  # [H, dh, dm]
        W_V = model.W_V[L].float()  # [H, dm, dh]
        W_Q = model.W_Q[L].float()  # [H, dm, dh]
        W_K = model.W_K[L].float()  # [H, dm, dh]
        # batched W_OV : [H, dm, dm]
        W_OV = torch.einsum("hdm,hef->hde", W_V, W_O)  # overwritten by the einsum below
        # W_V[h] is [dm, dh], W_O[h] is [dh, dm] in TL, so W_OV[h] = W_V[h] @ W_O[h]
        # is [dm, dm]. Vectorize: torch.einsum('hij,hjk->hik', W_V, W_O)
        W_OV = torch.einsum("hij,hjk->hik", W_V, W_O)  # [H, dm, dm]

        # OV-copy logits over sampled vocab:
        # E_sample: [N, dm]
        # W_OV: [H, dm, dm]
        # For each head: E_sample @ W_OV[h] @ W_U[:, sample] = [N, N]
        # Batched: [N, dm] @ [H, dm, dm] = [H, N, dm]; then @ U_sample = [H, N, N]
        # E_sample has shape (N, dm); we want (H, N, dm)
        eov = torch.einsum("nd,hdm->hnm", E_sample, W_OV)  # [H, N, dm]
        logits = torch.einsum("hnm,me->hne", eov, U_sample)  # [H, N, N]   m≡dm, e≡N
        # diag of logits[h, i, i] vs top-K of logits[h, i, :]
        N = logits.shape[1]
        diag = logits.diagonal(dim1=-2, dim2=-1)  # [H, N]
        topk_idx = logits.topk(TOP_K, dim=-1).indices  # [H, N, K]
        diag_pos = torch.arange(N, device=args.device).unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        in_topk = (topk_idx == diag_pos).any(dim=-1).float()  # [H, N]
        ov_copy[L, :] = in_topk.mean(dim=-1).cpu().numpy()
        ov_exp_logit[L, :] = diag.mean(dim=-1).cpu().numpy()

        # Frobenius norm per head (batched)
        ov_frob[L, :] = W_OV.flatten(start_dim=-2).norm(dim=-1).cpu().numpy()
        # Stored as ov_effective_rank: stable_rank = ||M||_F^2 / ||M||_2^2 (avoids a per-head SVD).
        # Estimate spectral norm via power iteration (5 iters).
        v = torch.randn(nH, dm, device=W_OV.device, dtype=W_OV.dtype)
        v = v / v.norm(dim=-1, keepdim=True)
        for _ in range(5):
            v = torch.einsum("hij,hj->hi", W_OV, v)
            v = torch.einsum("hji,hj->hi", W_OV, v)
            v = v / (v.norm(dim=-1, keepdim=True) + 1e-12)
        # spectral norm estimate
        sn = torch.einsum("hij,hj->hi", W_OV, v).norm(dim=-1)
        stable_rank = (ov_frob[L, :] / (sn.cpu().numpy() + 1e-12)) ** 2
        ov_eff_rank[L, :] = stable_rank

        # QK-copy: W_QK[h] = W_Q[h] @ W_K[h]^T  : [dm, dm]
        # scores[h, n, n'] = E_sample[n] @ W_QK[h] @ E_sample[n']
        # use [N, dm] @ [H, dm, dm] = [H, N, dm], then @ E_sample.T = [H, N, N]
        W_QK = torch.einsum("hij,hkj->hik", W_Q, W_K)   # [H, dm, dm]   (Q @ K^T)
        eqk = torch.einsum("nd,hdm->hnm", E_sample, W_QK)  # [H, N, dm]
        scores = torch.einsum("hnm,em->hne", eqk, E_sample)  # [H, N, N]
        diag_qk = scores.diagonal(dim1=-2, dim2=-1)  # [H, N]
        topk_q = scores.topk(TOP_K, dim=-1).indices  # [H, N, K]
        in_topk_qk = (topk_q == diag_pos).any(dim=-1).float()  # [H, N]
        qk_copy[L, :] = in_topk_qk.mean(dim=-1).cpu().numpy()

        if L % 4 == 0 or L == nL - 1:
            print(f"[{mname}] L={L+1}/{nL}  elapsed={time.time()-t0:.0f}s", flush=True)
        del W_OV, eov, logits, W_QK, eqk, scores
        torch.cuda.empty_cache()

    np.savez_compressed(out_npz,
                        ov_copy=ov_copy, ov_exp_logit=ov_exp_logit,
                        ov_frob=ov_frob, ov_effective_rank=ov_eff_rank,
                        qk_copy=qk_copy)
    meta = {
        "model": args.model, "model_short": mname,
        "n_layers": nL, "n_heads": nH, "d_head": dh, "d_model": dm,
        "n_tokens_sampled": int(len(tok_ids)),
        "top_k": TOP_K,
        "elapsed_sec": time.time() - t0,
    }
    out_json.write_text(json.dumps(meta, indent=2))

    print(f"\n[{mname}] top-10 OV-copy heads:")
    flat = ov_copy.flatten()
    for idx in flat.argsort()[-10:][::-1]:
        L = int(idx // nH); H = int(idx % nH)
        print(f"   L{L:>2}H{H:>2}  ov_copy={ov_copy[L,H]:.3f}  qk_copy={qk_copy[L,H]:.3f}  "
              f"ov_exp={ov_exp_logit[L,H]:+.3f}  ov_frob={ov_frob[L,H]:.2f}")

    print(f"\n[{mname}] top-10 QK-copy heads:")
    flat = qk_copy.flatten()
    for idx in flat.argsort()[-10:][::-1]:
        L = int(idx // nH); H = int(idx % nH)
        print(f"   L{L:>2}H{H:>2}  ov_copy={ov_copy[L,H]:.3f}  qk_copy={qk_copy[L,H]:.3f}")

    print(f"\n[{mname}] DONE in {(time.time()-t0)/60:.1f} min -> {out_npz}", flush=True)


if __name__ == "__main__":
    main()
