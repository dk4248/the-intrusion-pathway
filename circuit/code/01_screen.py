"""Stage 1 — parametric screen. Which facts does each model already know with NO context?

For every (relation, subject, gold): bare query -> gold first-token rank, logit, margin
vs best competitor, argmax, and the teacher-forced log P of the whole gold string.
"""
import argparse, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load, last_token_logits, answer_logprob, answer_ids
from lre_pool import load_relations, query_prompt

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--batch", type=int, default=32)
a = ap.parse_args()

t0 = time.time()
model = load(a.model, a.device)
R = load_relations()

prompts, meta = [], []
for rname, rel in R.items():
    for subj, gold in rel["facts"]:
        prompts.append(query_prompt(rel, subj))
        meta.append((rname, subj, gold))
print(f"[{a.model}] {len(prompts)} facts, {len(R)} relations", flush=True)

logits = last_token_logits(model, prompts, a.batch)
lp = answer_logprob(model, prompts, [m[2] for m in meta], batch=a.batch // 2)

recs = []
for i, (rname, subj, gold) in enumerate(meta):
    ids = answer_ids(model, gold)
    gid = ids[0]
    row = logits[i]
    order = torch.argsort(row, descending=True)
    rank = int((order == gid).nonzero()[0, 0]) + 1
    gold_logit = float(row[gid])
    mask = torch.ones_like(row, dtype=torch.bool); mask[gid] = False
    recs.append(dict(
        relation=rname, subject=subj, gold=gold,
        gold_first_id=int(gid), n_gold_tokens=len(ids), single_token=len(ids) == 1,
        rank=rank, gold_logit=gold_logit,
        margin=gold_logit - float(row[mask].max()),
        gold_logprob_full=lp[i],
        argmax=model.tokenizer.decode([int(order[0])]),
    ))

json.dump({"model": a.model, "n": len(recs), "records": recs},
          (OUT / f"screen_{a.model}.json").open("w"), indent=1)
print(f"[{a.model}] done in {time.time()-t0:.0f}s -> screen_{a.model}.json", flush=True)
