"""Aggregate the long_irrelevant results into a per-model degradation table (Appendix D,
Table D.1), joining against the 5-class Phase 1 numbers from
degradation/results/<model>/phase1.json (written by run_phase1.py).

Produces:
  degradation/results/AGGREGATE_degradation.txt   — human-readable table
  degradation/results/AGGREGATE_degradation.json  — machine-readable

Columns per (model, class):
  n              — sample count
  mean Δ_g       — gold suppression
  mean Δ_d       — distractor entrainment
  median Δ_g     — robust center
  mean rank_g_wctx — how far gold drops in rank
  flip_rate      — fraction of (clean-correct → distract-wrong)
  mean KL        — wctx vs nctx softmax divergence
  prefix tokens  — length stat (for irrelevant variants only)
"""
import json
from pathlib import Path
from collections import defaultdict
import statistics
REPO = Path(__file__).resolve().parents[2]  # repository root

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
EXISTING_SCALING = REPO / "degradation/results"
MODELS = ["Llama-3.1-8B", "Mistral-7B-v0.1", "Qwen2.5-7B", "pythia-6.9b"]
COMPARISON_CLASSES = ["golden", "related", "counterfactual", "irrelevant", "random"]
NEW_CLASS = "long_irrelevant"


def summarize(records):
    if not records:
        return None
    dg = [r["delta_gold"] for r in records]
    dd = [r["delta_dstr"] for r in records]
    kl = [r["kl_nctx_to_wctx"] for r in records]
    rg = [r["rank_gold_wctx"] for r in records]
    flips = [r["acc_flip_wctx_lost"] for r in records]
    correct_nctx = [r["correct_nctx"] for r in records]
    correct_wctx = [r["correct_wctx"] for r in records]
    out = {
        "n": len(records),
        "mean_delta_gold": sum(dg) / len(dg),
        "median_delta_gold": statistics.median(dg),
        "mean_delta_dstr": sum(dd) / len(dd),
        "median_delta_dstr": statistics.median(dd),
        "mean_kl": sum(kl) / len(kl),
        "mean_rank_gold_wctx": sum(rg) / len(rg),
        "median_rank_gold_wctx": statistics.median(rg),
        "frac_correct_nctx": sum(correct_nctx) / len(correct_nctx),
        "frac_correct_wctx": sum(correct_wctx) / len(correct_wctx),
        "flip_rate_overall": sum(flips) / len(flips),
        # conditional flip rate: flip-rate among samples where nctx was correct
        "flip_rate_among_clean_correct": (
            sum(flips) / max(1, sum(correct_nctx))
        ),
    }
    # prefix length stats (only meaningful for long_irrelevant)
    n_toks = [r.get("n_tokens_ws") for r in records if r.get("n_tokens_ws")]
    if n_toks:
        out["mean_prefix_ws_tokens"] = sum(n_toks) / len(n_toks)
        out["median_prefix_ws_tokens"] = statistics.median(n_toks)
    return out


def main():
    out_table = {}
    for model in MODELS:
        out_table[model] = {}
        # --- 5 classes from run_phase1.py (phase1.json) ---
        existing_path = EXISTING_SCALING / model / "phase1.json"
        if existing_path.exists():
            existing = json.load(existing_path.open())
            by_class = defaultdict(list)
            for r in existing["records"]:
                by_class[r["class"]].append(r)
            for cls in COMPARISON_CLASSES:
                if cls in by_class:
                    out_table[model][cls] = summarize(by_class[cls])
        else:
            print(f"[warn] missing 5-class phase1 results for {model}: {existing_path}")
        # --- long_irrelevant (run_phase1_long_irrelevant.py) ---
        new_path = RESULTS / model / "phase1_long_irrelevant_full.json"
        if new_path.exists():
            new = json.load(new_path.open())
            out_table[model][NEW_CLASS] = summarize(new["records"])
        else:
            print(f"[warn] missing long_irrelevant for {model}: {new_path}")

    # --- write JSON ---
    (RESULTS / "AGGREGATE_degradation.json").write_text(
        json.dumps(out_table, indent=2)
    )

    # --- pretty txt table ---
    lines = []
    lines.append("=" * 120)
    lines.append("PERFORMANCE DEGRADATION TABLE — short_irrelevant / long_irrelevant / random / counterfactual / related / golden")
    lines.append("=" * 120)
    lines.append("All numbers are means over 250 samples unless n != 250.")
    lines.append("dg = delta_gold (wctx - nctx, gold logit shift; negative = gold suppressed)")
    lines.append("dd = delta_dstr (distractor logit boost; positive = entrained)")
    lines.append("flip% = fraction of (clean-correct -> distract-wrong) among ALL samples")
    lines.append("rank_g = mean rank of gold token under wctx (1 = top)")
    lines.append("KL = KL(softmax(nctx) || softmax(wctx))")
    lines.append("prefix_toks = mean whitespace-tokens in the prefix (irrelevant variants only)")
    lines.append("")

    for model in MODELS:
        lines.append("-" * 120)
        lines.append(f"MODEL: {model}")
        lines.append("-" * 120)
        lines.append(f"  {'class':<22} {'n':>5} {'dg_mean':>10} {'dg_med':>10} {'dd_mean':>10} {'flip%':>8} {'rank_g':>10} {'KL':>8} {'prefix_toks':>12}")
        for cls in ["golden", "related", "counterfactual", "irrelevant", "long_irrelevant", "random"]:
            d = out_table[model].get(cls)
            if not d:
                lines.append(f"  {cls:<22} {'(missing)':>5}")
                continue
            ptoks = d.get("mean_prefix_ws_tokens", "")
            ptoks_str = f"{ptoks:>10.1f}" if isinstance(ptoks, (int, float)) else f"{'-':>12}"
            lines.append(
                f"  {cls:<22} {d['n']:>5} "
                f"{d['mean_delta_gold']:>+10.3f} "
                f"{d['median_delta_gold']:>+10.3f} "
                f"{d['mean_delta_dstr']:>+10.3f} "
                f"{100*d['flip_rate_overall']:>7.1f}% "
                f"{d['mean_rank_gold_wctx']:>10.2f} "
                f"{d['mean_kl']:>8.3f} "
                f"{ptoks_str}"
            )
        lines.append("")

    # --- cross-model summary table: short vs long vs random Δg ---
    lines.append("=" * 120)
    lines.append("CROSS-MODEL Δ_g (gold-logit suppression, negative = damage)")
    lines.append("=" * 120)
    lines.append(f"  {'model':<20} {'short_irr':>12} {'long_irr':>12} {'random':>12} {'cf (ref)':>12} {'gold (ref)':>12}")
    for model in MODELS:
        d = out_table[model]
        def cell(cls):
            return f"{d[cls]['mean_delta_gold']:>+12.3f}" if cls in d and d[cls] else f"{'(miss)':>12}"
        lines.append(
            f"  {model:<20} "
            f"{cell('irrelevant')} "
            f"{cell('long_irrelevant')} "
            f"{cell('random')} "
            f"{cell('counterfactual')} "
            f"{cell('golden')}"
        )
    lines.append("")

    lines.append("=" * 120)
    lines.append("CROSS-MODEL flip-rate % (clean-correct -> distract-wrong, ALL samples)")
    lines.append("=" * 120)
    lines.append(f"  {'model':<20} {'short_irr':>12} {'long_irr':>12} {'random':>12} {'cf (ref)':>12} {'gold (ref)':>12}")
    for model in MODELS:
        d = out_table[model]
        def cell(cls):
            return f"{100*d[cls]['flip_rate_overall']:>11.1f}%" if cls in d and d[cls] else f"{'(miss)':>12}"
        lines.append(
            f"  {model:<20} "
            f"{cell('irrelevant')} "
            f"{cell('long_irrelevant')} "
            f"{cell('random')} "
            f"{cell('counterfactual')} "
            f"{cell('golden')}"
        )
    lines.append("")

    lines.append("=" * 120)
    lines.append("CROSS-MODEL Δ_d (distractor entrainment, positive = boosted)")
    lines.append("=" * 120)
    lines.append(f"  {'model':<20} {'short_irr':>12} {'long_irr':>12} {'random':>12} {'cf (ref)':>12} {'gold (ref)':>12}")
    for model in MODELS:
        d = out_table[model]
        def cell(cls):
            return f"{d[cls]['mean_delta_dstr']:>+12.3f}" if cls in d and d[cls] else f"{'(miss)':>12}"
        lines.append(
            f"  {model:<20} "
            f"{cell('irrelevant')} "
            f"{cell('long_irrelevant')} "
            f"{cell('random')} "
            f"{cell('counterfactual')} "
            f"{cell('golden')}"
        )
    lines.append("")

    out_txt = "\n".join(lines)
    (RESULTS / "AGGREGATE_degradation.txt").write_text(out_txt)
    print(out_txt)
    print(f"\nWROTE: {RESULTS / 'AGGREGATE_degradation.txt'}")
    print(f"WROTE: {RESULTS / 'AGGREGATE_degradation.json'}")


if __name__ == "__main__":
    main()
