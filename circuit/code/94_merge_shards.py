"""Merge sharded E1 outputs into the single file 13_C1_analyze.py reads, in the unsharded row order.

usage: python code/94_merge_shards.py <E1_enrich|E1_causal> <model> <n_shards>
Shards come from `--shard k/n` (samples with index % n == k, global indices kept for donors/controls), so
concatenating and re-sorting by the sample order of data/pairs_v1.json reproduces an unsharded run.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
kind, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
order = {s["sample_id"]: i for i, s in enumerate(json.load((ROOT / "data" / "pairs_v1.json").open())["samples"])}
parts = [json.load((ROOT / "results" / f"{kind}_{model}.shard{k}of{n}.json").open()) for k in range(n)]
rows = [r for p in parts for r in p["rows"]]
rows.sort(key=lambda r: order[r["sid"]])                     # stable: keeps each sample's internal row order
out = {k: v for k, v in parts[0].items() if k != "rows"}; out["rows"] = rows
json.dump(out, (ROOT / "results" / f"{kind}_{model}.json").open("w"))
print(f"{kind} {model}: {len(rows)} rows from {n} shards, {len({r['sid'] for r in rows})} samples")
