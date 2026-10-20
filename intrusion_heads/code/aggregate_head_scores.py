"""Aggregate Stage 4 per-(sample, condition) head scores into per-(model, layer, head) summaries.

Computes:
  retrieval_score[h]:
    Under GOLDEN, heads that (a) attend strongly to the position of the gold token, and
    (b) write a positive DLA to gold. Defined as:
        rs(h) = mean_i  att_to_gold(h, i, golden)  *  dla_gold(h, i, golden)
    Higher = more retrieval-like.

  intrusion_score[h]:
    Under irrelevant context, heads that (a) attend to the irrelevant prefix, and (b)
    suppress the gold logit relative to nctx. Defined as:
        is(h) = mean_i  att_to_ctx(h, i, irr)  *  max(0, −Δdla_gold(h, i; irr−nctx))
    Where Δdla_gold = dla_gold(h, irr) − dla_gold(h, nctx); the score is averaged over the
    short- and long-irrelevant conditions.
    Higher = more intrusion-like.

  Auxiliary per-(model, head) statistics under each condition:
    att_to_ctx_mean, att_entropy_mean, dla_gold_mean, z_norm_mean.

Outputs:
  results/AGGREGATE_head_scores.json   — per-(model, head) stats and top-K sets
  results/HEAD_SETS.json               — per-model retrieval_set (R), intrusion_set (I),
                                         clean_intrusion_set (I*), overlap (head index tuples)
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

MODELS = ["Llama-3.1-8B", "Mistral-7B-v0.1", "Qwen2.5-7B", "pythia-6.9b"]
CONDITIONS = ["nctx", "golden", "related", "counterfactual", "short_irr", "long_irr"]
TOP_K_RET = 20         # top-K heads to call "retrieval"
TOP_K_INTR = 20        # top-K heads to call "intrusion"
STAT_NAMES = ["att_to_ctx_excl", "att_to_gold", "att_to_dstr", "att_max",
              "att_entropy", "dla_gold", "dla_dstr", "z_norm"]
SIDX = {s: i for i, s in enumerate(STAT_NAMES)}


def load_model(mname):
    npz = RES / f"{mname}__head_scores.npz"
    js  = RES / f"{mname}__head_scores.json"
    if not (npz.exists() and js.exists()):
        return None, None
    arr = np.load(npz)["scores"]  # [N, C, L, H, S]
    meta = json.load(js.open())
    return arr, meta


def main():
    out = {"by_model": {}, "models_present": []}
    head_sets = {}

    for m in MODELS:
        arr, meta = load_model(m)
        if arr is None:
            print(f"[skip] {m}")
            continue
        N, C, L, H, S = arr.shape
        conds = meta["conditions"]
        cidx = {c: i for i, c in enumerate(conds)}
        # Average across samples → [C, L, H, S]
        mean_cls = arr.mean(axis=0)  # [C, L, H, S]

        # Retrieval score per (L,H): att_to_gold × dla_gold under golden
        gi = cidx["golden"]
        s_att_g = arr[:, gi, :, :, SIDX["att_to_gold"]]
        s_dla_g = arr[:, gi, :, :, SIDX["dla_gold"]]
        ret_score = (s_att_g * s_dla_g).mean(axis=0)  # [L, H]

        # Intrusion score: under irrelevant short, average over samples of
        #   att_to_ctx × max(0, dla_gold_nctx - dla_gold_irr)
        # i.e. heads that attend to irrelevant context AND drop more gold than they did under nctx
        ii_s = cidx["short_irr"]
        ii_l = cidx["long_irr"]
        ni = cidx["nctx"]
        s_att_irr_s = arr[:, ii_s, :, :, SIDX["att_to_ctx_excl"]]
        s_att_irr_l = arr[:, ii_l, :, :, SIDX["att_to_ctx_excl"]]
        s_dla_irr_s = arr[:, ii_s, :, :, SIDX["dla_gold"]]
        s_dla_irr_l = arr[:, ii_l, :, :, SIDX["dla_gold"]]
        s_dla_nctx  = arr[:, ni,   :, :, SIDX["dla_gold"]]
        # per-sample increased gold-suppression: drop = nctx - irr (positive if irr suppresses)
        drop_s = np.clip(s_dla_nctx - s_dla_irr_s, 0, None)
        drop_l = np.clip(s_dla_nctx - s_dla_irr_l, 0, None)
        intr_short = (s_att_irr_s * drop_s).mean(axis=0)  # [L, H]
        intr_long  = (s_att_irr_l * drop_l).mean(axis=0)  # [L, H]
        intr_score = 0.5 * (intr_short + intr_long)       # [L, H]

        # Top-K head sets
        flat_ret = ret_score.flatten()
        flat_intr = intr_score.flatten()
        top_ret = flat_ret.argsort()[-TOP_K_RET:][::-1]
        top_intr = flat_intr.argsort()[-TOP_K_INTR:][::-1]
        retrieval_set = [(int(idx // H), int(idx % H), float(flat_ret[idx]))
                         for idx in top_ret]
        intrusion_set = [(int(idx // H), int(idx % H), float(flat_intr[idx]))
                          for idx in top_intr]
        ret_lh = set((l, h) for l, h, _ in retrieval_set)
        intr_lh = set((l, h) for l, h, _ in intrusion_set)
        overlap = ret_lh & intr_lh

        # CLEAN INTRUSION SET: top-K by intr_score that are NOT in retrieval top-2K
        # (i.e., strictly disjoint from the top-2K retrieval heads).
        top_ret_wide = set((int(idx // H), int(idx % H))
                           for idx in flat_ret.argsort()[-2*TOP_K_RET:][::-1])
        clean_intrusion_set = []
        for idx in top_intr.tolist() + flat_intr.argsort()[::-1].tolist():
            l, h = int(idx // H), int(idx % H)
            if (l, h) in top_ret_wide:
                continue
            if (l, h) in {(ll, hh) for ll, hh, _ in clean_intrusion_set}:
                continue
            clean_intrusion_set.append((l, h, float(flat_intr[idx])))
            if len(clean_intrusion_set) >= TOP_K_INTR:
                break

        head_sets[m] = {
            "retrieval_set":         retrieval_set,
            "intrusion_set":         intrusion_set,
            "clean_intrusion_set":   clean_intrusion_set,
            "overlap":               [(l, h) for (l, h) in sorted(overlap)],
            "overlap_n":             len(overlap),
        }

        # Save per-(L,H) score matrices for plotting
        out["by_model"][m] = {
            "n_layers": L, "n_heads": H, "n_samples": N,
            "ret_score": ret_score.tolist(),
            "intr_score_short": intr_short.tolist(),
            "intr_score_long":  intr_long.tolist(),
            "intr_score_mean":  intr_score.tolist(),
            "mean_cls": mean_cls.tolist(),  # [C, L, H, S]
            "stat_names": STAT_NAMES,
            "conditions": conds,
            "head_sets": head_sets[m],
        }
        out["models_present"].append(m)

        # Print summary
        print(f"\n=== {m}  (L={L}, H={H}, N={N}) ===")
        print(f"  Top {TOP_K_RET} retrieval heads:")
        for l, h, sc in retrieval_set[:10]:
            print(f"     L{l:>2}H{h:>2}  ret={sc:+.3f}")
        print(f"  Top {TOP_K_INTR} intrusion heads:")
        for l, h, sc in intrusion_set[:10]:
            print(f"     L{l:>2}H{h:>2}  intr={sc:+.3f}")
        print(f"  Overlap |R ∩ I|={len(overlap)}/{min(TOP_K_RET, TOP_K_INTR)}")
        if overlap:
            print(f"     {sorted(overlap)}")

    (RES / "AGGREGATE_head_scores.json").write_text(json.dumps(out))
    (RES / "HEAD_SETS.json").write_text(json.dumps(head_sets, indent=2))
    print(f"\nWROTE AGGREGATE_head_scores.json + HEAD_SETS.json")


if __name__ == "__main__":
    main()
