"""
Stage E — per-model: for each of 250 triples, run all 60 irrelevant
candidate contexts paired with the query. Record:
    Δ_gold = logit_wctx(gold_first_token) - logit_nctx(gold_first_token)

The selector (stage F, stage_f_merge.py) picks the context with the MOST NEGATIVE mean Δ_gold
across 4 models — the one that damages gold the most.

Usage:
    python stage_e_irrelevant_screen.py --model meta-llama/Llama-3.1-8B --device cuda:0
"""
import argparse, json, time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from irrelevant_pool import IRRELEVANT_POOL
REPO = Path(__file__).resolve().parents[2]  # repository root

BUILD = REPO / "lre_dataset/data/build"
def short(m): return m.split("/")[-1]

def first_token_id(tok, word):
    ids = tok.encode(" " + word, add_special_tokens=False)
    return ids[0] if ids else None

@torch.no_grad()
def batched_last_logits_at(model, tok, prompts, token_ids, device, batch=16):
    """Return list of logit(token_id) at last position, one per prompt."""
    out = []
    for i in range(0, len(prompts), batch):
        ch_p = prompts[i:i+batch]
        ch_ids = token_ids[i:i+batch]
        enc = tok(ch_p, return_tensors="pt", padding=True, truncation=True).to(device)
        logits = model(**enc).logits.float()
        lengths = enc["attention_mask"].sum(dim=1)
        bs = torch.arange(len(ch_p))
        last = logits[bs, lengths-1, :]      # (B, V)
        for j in range(len(ch_p)):
            out.append(float(last[j, ch_ids[j]].item()))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[{short(args.model)}] loading...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device, low_cpu_mem_usage=True
    )
    model.eval()

    cherry = json.load(open(BUILD / "cherry_250.json"))
    det = json.load(open(BUILD / "variants_det.json"))
    samples = det["samples"]
    # mapping from sample_id -> cherry (has query)
    cherry_by_id = {s["sample_id"]: s for s in cherry["samples"]}

    # Build batch of queries: [nctx] for each sample, then [wctx_cand] for each (sample, cand)
    nctx_prompts = []
    nctx_gold_ids = []
    wctx_prompts = []
    wctx_gold_ids = []
    wctx_map = []     # (sample_idx, cand_idx)
    for si, s in enumerate(samples):
        query = cherry_by_id[s["sample_id"]]["query_zs"]
        gid = first_token_id(tok, s["gold"])
        if gid is None: continue
        nctx_prompts.append(query)
        nctx_gold_ids.append(gid)
        for ci, ctx in enumerate(IRRELEVANT_POOL):
            wctx_prompts.append(f"{ctx} {query}")
            wctx_gold_ids.append(gid)
            wctx_map.append((si, ci))

    print(f"[{short(args.model)}] n_samples={len(samples)}  n_irrelevant={len(IRRELEVANT_POOL)}  "
          f"total_wctx_passes={len(wctx_prompts)}", flush=True)

    # Run no-context batch
    print(f"[{short(args.model)}] running nctx batch ({len(nctx_prompts)})...", flush=True)
    nctx_logits = batched_last_logits_at(model, tok, nctx_prompts, nctx_gold_ids, args.device, args.batch)

    print(f"[{short(args.model)}] running wctx batches ({len(wctx_prompts)}) ...", flush=True)
    wctx_logits = []
    n_done = 0
    B = args.batch
    for start in tqdm(range(0, len(wctx_prompts), B), desc="wctx"):
        ch_p = wctx_prompts[start:start+B]
        ch_i = wctx_gold_ids[start:start+B]
        res = batched_last_logits_at(model, tok, ch_p, ch_i, args.device, B)
        wctx_logits.extend(res)

    # Build output
    out_samples = []
    k = 0
    for si, s in enumerate(samples):
        gid = first_token_id(tok, s["gold"])
        if gid is None:
            out_samples.append({"sample_id": s["sample_id"], "skipped": True})
            continue
        nctx = nctx_logits[si]
        # The next 60 wctx belong to this sample
        rows = []
        for ci, ctx in enumerate(IRRELEVANT_POOL):
            wctx = wctx_logits[k]; k += 1
            rows.append({
                "idx":  ci,
                "context":  ctx,
                "wctx_gold_logit": wctx,
                "delta_gold":      wctx - nctx,
            })
        out_samples.append({
            "sample_id": s["sample_id"],
            "nctx_gold_logit": nctx,
            "irrelevant_candidates": rows,
        })

    out = {"model": args.model, "short": short(args.model), "samples": out_samples}
    fp = BUILD / f"irrelevant_screen_{short(args.model)}.json"
    json.dump(out, open(fp, "w"), indent=2, ensure_ascii=False)
    print(f"[{short(args.model)}] DONE in {time.time()-t0:.1f}s → {fp}")

if __name__ == "__main__":
    main()
