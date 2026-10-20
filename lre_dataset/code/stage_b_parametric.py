"""
Stage B — per-model parametric screen.

For each candidate (relation, subject, gold), build the zero-shot query
template and compute:
  - single_token_ok  : 1st token of " {gold}" is a full gold (i.e. space-prefix
                       encodes to exactly 1 sub-word)
  - gold_rank        : rank of gold's 1st-token id in the last-token logits
  - gold_margin      : logit(gold_id) - logit(argmax_non_gold)

Save per-model JSON:  lre_dataset/data/build/parametric_<short_name>.json

Usage:
    python stage_b_parametric.py --model meta-llama/Llama-3.1-8B --device cuda:0
"""
import argparse, json, time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
REPO = Path(__file__).resolve().parents[2]  # repository root

CAND = REPO / "lre_dataset/data/build/candidates.json"
def short(m): return m.split("/")[-1]

def first_token_id(tok, word):
    ids = tok.encode(" " + word, add_special_tokens=False)
    return ids[0] if ids else None

def is_single_token(tok, word):
    ids = tok.encode(" " + word, add_special_tokens=False)
    return len(ids) == 1

@torch.no_grad()
def batched_last_logits(model, tok, prompts, device, batch=16):
    out = []
    for i in range(0, len(prompts), batch):
        ch = prompts[i:i+batch]
        enc = tok(ch, return_tensors="pt", padding=True, truncation=True).to(device)
        logits = model(**enc).logits.float()
        lengths = enc["attention_mask"].sum(dim=1)
        bs = torch.arange(len(ch))
        last = logits[bs, lengths-1, :].cpu()
        out.append(last)
    return torch.cat(out, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[{short(args.model)}] loading on {args.device}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device, low_cpu_mem_usage=True
    )
    model.eval()

    cand = json.load(open(CAND))
    out_relations = []
    total_kept = 0
    for rel in cand["relations"]:
        rel_name = rel["name"]
        tpl_zs = rel["template_zs"]
        cands = rel["candidates"]
        # stage-b prompt: just the zero-shot template — no extra suffix
        prompts = [tpl_zs.format(c["subject"]) for c in cands]
        # Pre-filter: gold must be single-token under THIS tokenizer
        kept_idx = [i for i, c in enumerate(cands) if is_single_token(tok, c["gold"])]
        if not kept_idx:
            out_relations.append({"name": rel_name, "kept": []})
            print(f"[{short(args.model)}] {rel_name:30s}  0/{len(cands)} single-token golds → skip")
            continue

        sub_prompts = [prompts[i] for i in kept_idx]
        sub_cands   = [cands[i]   for i in kept_idx]
        gold_ids = [first_token_id(tok, c["gold"]) for c in sub_cands]

        print(f"[{short(args.model)}] {rel_name:30s}  {len(sub_cands)}/{len(cands)} single-tok  running...", flush=True)
        last = batched_last_logits(model, tok, sub_prompts, args.device, args.batch)
        # rank + margin
        entries = []
        for idx, (c, gid) in enumerate(zip(sub_cands, gold_ids)):
            row = last[idx]
            # rank of gid (1 = top)
            sorted_vals, sorted_ids = torch.sort(row, descending=True)
            rank = int((sorted_ids == gid).nonzero(as_tuple=True)[0][0].item()) + 1
            gold_logit = float(row[gid].item())
            # 2nd-best non-gold max
            mask = torch.ones_like(row, dtype=torch.bool); mask[gid] = False
            second = float(row[mask].max().item())
            margin = gold_logit - second
            entries.append({
                "subject":     c["subject"],
                "gold":        c["gold"],
                "gold_token_id": int(gid),
                "zs_prompt":   sub_prompts[idx],
                "gold_rank":   rank,
                "gold_logit":  gold_logit,
                "margin_vs_2nd": margin,
            })
        out_relations.append({"name": rel_name, "kept": entries})
        total_kept += len(entries)

    out = {
        "model":    args.model,
        "short":    short(args.model),
        "n_kept":   total_kept,
        "relations": out_relations,
    }
    out_fp = CAND.parent / f"parametric_{short(args.model)}.json"
    json.dump(out, open(out_fp, "w"), indent=2, ensure_ascii=False)
    print(f"[{short(args.model)}] DONE in {time.time()-t0:.1f}s  total_kept={total_kept}  → {out_fp}")

if __name__ == "__main__":
    main()
