"""Build the 500-problem GSM-IC evaluation subset used in Appendix D (Tables D.2, D.3).

Source: GSM-IC 2-step split (Shi et al., 2023), `GSM-IC_2step.json` from
https://github.com/google-research-datasets/GSM-IC, saved as gsm_ic/data/gsm-ic_2step.json.

Subset: problems whose inserted sentence is labelled `in_topic` (15,404 of 34,220), shuffled with
random.Random(42), first 500 kept.

Output: gsm_ic/data/gsm_ic_500.json

Usage:
    python build_gsm_ic_500.py [--src ../data/gsm-ic_2step.json] [--out ../data/gsm_ic_500.json]
"""
import argparse
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(DATA / "gsm-ic_2step.json"))
ap.add_argument("--out", default=str(DATA / "gsm_ic_500.json"))
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

problems = json.load(open(args.src))
pool = [p for p in problems if p["sentence_label"] == "in_topic"]
random.Random(args.seed).shuffle(pool)
subset = pool[:args.n]

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(subset, open(args.out, "w"), indent=2)
print(f"{len(problems)} problems, {len(pool)} in_topic -> {len(subset)} written to {args.out}")
