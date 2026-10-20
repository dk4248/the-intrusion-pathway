"""
Stage A — pure-python: load 10 chosen LRE relations, emit per-relation
candidate subjects with their templates. No model loading.

Output: lre_dataset/data/build/candidates.json
"""
import json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]  # repository root

LRE_ROOT = REPO / "external/lre_dataset"
OUT = REPO / "lre_dataset/data/build/candidates.json"
RELATION_PATHS = [
    ("factual/country_capital_city.json",     "country_capital_city",     "city"),
    ("factual/city_in_country.json",          "city_in_country",          "country"),
    ("factual/country_currency.json",         "country_currency",         "currency"),
    ("factual/country_language.json",         "country_language",         "language"),
    ("factual/landmark_in_country.json",      "landmark_in_country",      "country"),
    ("factual/landmark_on_continent.json",    "landmark_on_continent",    "continent"),
    ("factual/person_native_language.json",   "person_native_language",   "language"),
    ("factual/person_plays_instrument.json",  "person_plays_instrument",  "instrument"),
    ("commonsense/fruit_outside_color.json",  "fruit_outside_color",      "color"),
    ("commonsense/object_superclass.json",    "object_superclass",        "superclass"),
]

def main():
    out = {"relations": [], "description": "Stage A candidates: raw LRE triples, 10 relations."}
    for rel_path, name, range_name in RELATION_PATHS:
        fp = LRE_ROOT / rel_path
        d = json.load(open(fp))
        samples = d["samples"]
        tpl_wctx = d["prompt_templates"][0]      # used for context: "The capital of {} is"
        tpl_zs   = d.get("prompt_templates_zs", [d["prompt_templates"][0]])[0]  # used for zero-shot query
        # Deduplicate by subject (object_superclass has repeats)
        seen = set(); uniq = []
        for s in samples:
            if s["subject"] in seen: continue
            seen.add(s["subject"])
            uniq.append(s)
        out["relations"].append({
            "name": name,
            "range_name": range_name,
            "template_wctx": tpl_wctx,      # e.g. "The capital city of {} is"
            "template_zs":   tpl_zs,
            "properties": d.get("properties", {}),
            "candidates": [
                {"subject": s["subject"], "gold": s["object"]}
                for s in uniq
            ],
        })
        print(f"  {name:30s}  n_candidates={len(uniq)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT}  ({sum(len(r['candidates']) for r in out['relations'])} total candidates)")

if __name__ == "__main__":
    main()
