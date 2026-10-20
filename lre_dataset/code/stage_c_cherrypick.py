"""
Stage C — cross-model intersection + stratified cherry-pick.

For each relation, keep triples where:
  - >= MIN_RANK1_MODELS (3) of the 4 models have gold_rank == 1
  - worst gold rank across the 4 models <= MAX_RANK_WEAK (500)
  - min margin across the 4 models >= MIN_MARGIN_WEAK (-10.5)

Then cherry-pick stratified 250 total samples using a per-relation quota.
Within each relation, sort by (n_rank1, min margin, mean margin) (descending) and keep the top N.

Output: lre_dataset/data/build/cherry_250.json
"""
import json
from pathlib import Path
from collections import defaultdict
REPO = Path(__file__).resolve().parents[2]  # repository root

BUILD = REPO / "lre_dataset/data/build"
MODELS = ["Llama-3.1-8B", "Qwen2.5-7B", "pythia-6.9b", "Mistral-7B-v0.1"]

# Focus on 6 strong relations where at least 3/4 models get rank-1.
# Dropped: country_currency, person_native_language, landmark_on_continent, fruit_outside_color
# (Mistral parametrically too weak / templates awkward)
QUOTA = {
    "country_capital_city":     8,   # take all (8 pass filter)
    "city_in_country":         11,   # take all
    "country_language":         5,   # take all — few survive Mistral weakness
    "landmark_in_country":    109,   # top 109 of 128 by (n_rank1, min_margin)
    "person_plays_instrument": 91,   # take all 91
    "object_superclass":       26,   # take all 26
    # Dropped — zero survive 3-of-4 filter:
    "landmark_on_continent":    0,
    "person_native_language":   0,
    "fruit_outside_color":      0,
    "country_currency":         0,
}
assert sum(QUOTA.values()) == 250, sum(QUOTA.values())

MIN_RANK1_MODELS = 3       # >=3 of 4 models must rank gold #1
MAX_RANK_WEAK   = 500      # 4th model: gold in top-500
MIN_MARGIN_WEAK = -10.5    # 4th model: logit gap within ~e^10.5

def load_per_model():
    per = {}
    for m in MODELS:
        fp = BUILD / f"parametric_{m}.json"
        per[m] = json.load(open(fp))
    return per

def main():
    per = load_per_model()

    # Build a dict keyed by (relation, subject) across all 4 models
    table = defaultdict(dict)     # (rel, subj) -> {model: entry}
    golds = {}                     # (rel, subj) -> gold
    templates = {}                 # rel -> {'zs': ..., 'wctx': ...}
    for m in MODELS:
        for rel in per[m]["relations"]:
            rname = rel["name"]
            for e in rel["kept"]:
                key = (rname, e["subject"])
                table[key][m] = e
                golds[key] = e["gold"]

    # relation templates — take from candidates.json (model-independent)
    cand = json.load(open(BUILD / "candidates.json"))
    for r in cand["relations"]:
        templates[r["name"]] = {"zs": r["template_zs"], "wctx": r["template_wctx"], "range": r["range_name"]}

    # Filter: all 4 present; >=MIN_RANK1_MODELS rank 1; worst rank <= MAX_RANK_WEAK;
    # worst margin >= MIN_MARGIN_WEAK.
    passed = {}
    for key, by_model in table.items():
        if len(by_model) != 4: continue
        ranks = [by_model[m]["gold_rank"] for m in MODELS]
        margins = [by_model[m]["margin_vs_2nd"] for m in MODELS]
        n_rank1 = sum(1 for r in ranks if r == 1)
        if n_rank1 < MIN_RANK1_MODELS: continue
        if max(ranks) > MAX_RANK_WEAK: continue
        if min(margins) < MIN_MARGIN_WEAK: continue
        passed[key] = {
            "ranks":   ranks,
            "margins": margins,
            "n_rank1":    n_rank1,
            "min_margin": min(margins),
            "mean_margin": sum(margins) / 4,
        }

    # Group by relation and rank by (n_rank1 desc, min_margin desc, mean_margin desc)
    # This prioritises: all-4 rank-1 > 3-of-4 rank-1; within tier, strongest floor first.
    by_rel = defaultdict(list)
    for (rel, subj), meta in passed.items():
        by_rel[rel].append((subj, golds[(rel, subj)], meta))
    for rel in by_rel:
        by_rel[rel].sort(key=lambda t: (-t[2]["n_rank1"], -t[2]["min_margin"], -t[2]["mean_margin"]))

    # Report
    print(f"{'Relation':32s}  passed  quota  top_mean_margin  bottom_mean_margin")
    total_selected = 0
    selected = {}
    short_quota = {}
    for rel, want in QUOTA.items():
        pool = by_rel[rel]
        take = min(want, len(pool))
        sel = pool[:take]
        selected[rel] = sel
        if take < want: short_quota[rel] = (take, want)
        tm = f"{sel[0][2]['mean_margin']:+.2f}" if sel else "—"
        bm = f"{sel[-1][2]['mean_margin']:+.2f}" if sel else "—"
        print(f"  {rel:30s}  {len(pool):5d}   {take:3d}   {tm:>14s}   {bm}")
        total_selected += take

    print(f"\nTotal selected: {total_selected} / 250")
    if short_quota:
        print("SHORT relations:", short_quota)

    # Emit final
    out = {
        "description": f"Cherry-picked {total_selected} triples. "
                       f"Filter: >={MIN_RANK1_MODELS}/4 models rank-1, max rank <= {MAX_RANK_WEAK}, min margin >= {MIN_MARGIN_WEAK}.",
        "filter":       {"min_rank1_models": MIN_RANK1_MODELS,
                         "max_rank_weak": MAX_RANK_WEAK,
                         "min_margin_weak": MIN_MARGIN_WEAK},
        "per_relation_quota": QUOTA,
        "templates":  templates,
        "samples":    [],
    }
    sid = 0
    for rel, sel in selected.items():
        tpl = templates[rel]
        for subj, gold, meta in sel:
            sid += 1
            out["samples"].append({
                "sample_id":  f"{rel[:3]}_{sid:03d}",
                "relation":   rel,
                "subject":    subj,
                "gold":       gold,
                "query_zs":   tpl["zs"].format(subj),
                "wctx_template": tpl["wctx"],
                "range_name": tpl["range"],
                "gold_ranks":   meta["ranks"],
                "gold_margins": [round(x, 3) for x in meta["margins"]],
            })

    json.dump(out, open(BUILD / "cherry_250.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nWrote {BUILD/'cherry_250.json'}  ({len(out['samples'])} samples)")

if __name__ == "__main__":
    main()
