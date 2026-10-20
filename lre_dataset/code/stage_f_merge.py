"""
Stage F — final merge.

Reads:
  - variants_det.json                     (golden, related, counterfactual)
  - irrelevant_screen_{model}.json x 4    (per-model Δ_gold for 60 candidates)
Writes:
  - lre_dataset/data/samples_5x250.json   (final dataset)

For irrelevant: pick candidate context with MOST-NEGATIVE mean Δ_gold
                across 4 models (largest damage to gold).
For random:    fixed context = "Dog." with distractor = "Dog".
"""
import json
from pathlib import Path
from collections import defaultdict
REPO = Path(__file__).resolve().parents[2]  # repository root

BUILD = REPO / "lre_dataset/data/build"
OUT   = REPO / "lre_dataset/data/samples_5x250.json"
MODELS = ["Llama-3.1-8B", "Qwen2.5-7B", "pythia-6.9b", "Mistral-7B-v0.1"]

def main():
    det = json.load(open(BUILD / "variants_det.json"))
    cherry = json.load(open(BUILD / "cherry_250.json"))
    cherry_by_id = {s["sample_id"]: s for s in cherry["samples"]}

    per_model = {}
    for m in MODELS:
        fp = BUILD / f"irrelevant_screen_{m}.json"
        per_model[m] = json.load(open(fp))

    # For each sample, collect per-model per-candidate Δ_gold, average, pick argmin
    final = []
    missing = 0
    for s in det["samples"]:
        sid = s["sample_id"]
        # stack Δ_gold (60,) per model
        per_cand = defaultdict(list)  # idx -> [Δ_gold_model1, ..., Δ_gold_model4]
        for m in MODELS:
            # find this sample in per_model[m]
            entries = [x for x in per_model[m]["samples"] if x["sample_id"] == sid]
            if not entries or entries[0].get("skipped"):
                continue
            for c in entries[0]["irrelevant_candidates"]:
                per_cand[c["idx"]].append(c["delta_gold"])
        if not per_cand:
            missing += 1
            continue
        # mean Δ_gold across models
        mean_delta = {i: sum(v)/len(v) for i, v in per_cand.items() if len(v) == 4}
        if not mean_delta:
            missing += 1; continue
        worst_idx = min(mean_delta, key=lambda i: mean_delta[i])
        winner_ctx = [x for x in per_model[MODELS[0]]["samples"] if x["sample_id"] == sid][0]["irrelevant_candidates"][worst_idx]["context"]
        winner_delta = mean_delta[worst_idx]

        variants = dict(s["variants"])
        variants["irrelevant"] = {
            "context":      winner_ctx,
            "distractor":   "__auto__",
            "mean_delta_gold_at_pick": round(winner_delta, 3),
            "candidate_idx": int(worst_idx),
        }
        variants["random"] = {
            "context":    "Dog.",
            "distractor": "Dog",
        }
        final.append({
            "sample_id": sid,
            "relation":  s["relation"],
            "subject":   s["subject"],
            "gold":      s["gold"],
            "query":     cherry_by_id[sid]["query_zs"],
            "parametric_ranks":   cherry_by_id[sid]["gold_ranks"],
            "parametric_margins": cherry_by_id[sid]["gold_margins"],
            "variants":  variants,
        })

    out = {
        "description": (
            f"Final dataset: {len(final)} samples x 5 context classes. "
            f"250 triples cherry-picked from 6 LRE relations (country_capital_city, "
            f"city_in_country, country_language, landmark_in_country, "
            f"person_plays_instrument, object_superclass). "
            f"Filter: >=3/4 models rank-1, 4th rank<=500, 4th margin>=-10.5. "
            f"Irrelevant winner picked per triple from 60-sentence pool by "
            f"most-negative mean Δ_gold across 4 models. Random fixed to 'Dog.'"),
        "n_samples":    len(final),
        "n_missing":    missing,
        "models":       MODELS,
        "classes":      ["golden", "related", "counterfactual", "irrelevant", "random"],
        "samples":      final,
    }
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"Final samples: {len(final)}   missing: {missing}")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
