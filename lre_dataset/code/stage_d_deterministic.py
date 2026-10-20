"""
Stage D — construct the 3 deterministic class variants per sample:
  - golden         : context affirms the gold     (distractor = gold)
  - related        : context from same relation, DIFFERENT subject/answer
  - counterfactual : context claims gold's subject has a DIFFERENT answer

For `related`, pick a paired (S', G') from the same relation, preferring
one with a single-token G' that's popular across all 4 tokenizers (safe).
For `counterfactual`, pick G_fake from the same relation's answer pool,
constrained to G_fake != gold.

Writes: lre_dataset/data/build/variants_det.json   — 250 × 3 variants.
Stage E (irrelevant/random) runs separately (needs models).
"""
import json, random
from pathlib import Path
from collections import defaultdict
REPO = Path(__file__).resolve().parents[2]  # repository root

random.seed(42)
BUILD = REPO / "lre_dataset/data/build"
def pick_related_pair(rel_samples, subject, gold, rng):
    """Pick a different (subject', gold') where gold' != gold."""
    candidates = [(s,g) for s,g in rel_samples if s != subject and g != gold]
    if not candidates:
        # fallback: same gold allowed (unavoidable for small pools like colors)
        candidates = [(s,g) for s,g in rel_samples if s != subject]
    return rng.choice(candidates) if candidates else None

def pick_cf_answer(rel_gold_pool, gold, rng):
    """Pick G_fake != gold from relation's gold pool."""
    pool = [g for g in rel_gold_pool if g != gold]
    return rng.choice(pool) if pool else None

def main():
    cherry = json.load(open(BUILD / "cherry_250.json"))
    samples = cherry["samples"]

    # Build per-relation (subject, gold) list and gold pool
    rel_pairs = defaultdict(list)
    rel_golds = defaultdict(list)
    for s in samples:
        rel_pairs[s["relation"]].append((s["subject"], s["gold"]))
        rel_golds[s["relation"]].append(s["gold"])
    # Deduplicate gold pools (preserve order by set semantics)
    for r in rel_golds:
        seen=set(); uniq=[]
        for g in rel_golds[r]:
            if g in seen: continue
            seen.add(g); uniq.append(g)
        rel_golds[r] = uniq

    rng = random.Random(42)
    out_samples = []
    for s in samples:
        rel = s["relation"]
        subj, gold = s["subject"], s["gold"]
        tpl_wctx = s["wctx_template"]       # "The capital city of {} is"
        query_zs = s["query_zs"]            # already filled

        # GOLDEN — context asserts the truth.
        golden_ctx = f"{tpl_wctx.format(subj)} {gold}."

        # RELATED — pair from same relation
        pair = pick_related_pair(rel_pairs[rel], subj, gold, rng)
        if pair is None:
            # Shouldn't happen for n>=2 cherrypicks, but guard.
            rel_ctx = None; rel_dstr = None
        else:
            rel_subj, rel_gold = pair
            rel_ctx  = f"{tpl_wctx.format(rel_subj)} {rel_gold}."
            rel_dstr = rel_gold

        # COUNTERFACTUAL — different gold for this subject
        cf_gold = pick_cf_answer(rel_golds[rel], gold, rng)
        if cf_gold is None:
            cf_ctx = None; cf_dstr = None
        else:
            cf_ctx  = f"{tpl_wctx.format(subj)} {cf_gold}."
            cf_dstr = cf_gold

        out_samples.append({
            "sample_id": s["sample_id"],
            "relation":  rel,
            "subject":   subj,
            "gold":      gold,
            "query":     query_zs,
            "variants": {
                "golden":         {"context": golden_ctx, "distractor": gold},
                "related":        {"context": rel_ctx,    "distractor": rel_dstr},
                "counterfactual": {"context": cf_ctx,     "distractor": cf_dstr},
            },
        })

    # Summary
    missing_related = sum(1 for s in out_samples if s["variants"]["related"]["context"] is None)
    missing_cf      = sum(1 for s in out_samples if s["variants"]["counterfactual"]["context"] is None)
    print(f"Samples: {len(out_samples)}")
    print(f"Missing related: {missing_related}")
    print(f"Missing counterfactual: {missing_cf}")

    out = {
        "description": "Deterministic class variants (golden, related, counterfactual) for 250 cherry-picked triples.",
        "samples": out_samples,
    }
    json.dump(out, open(BUILD / "variants_det.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nWrote {BUILD/'variants_det.json'}")

if __name__ == "__main__":
    main()
