"""Phase 1 runner restricted to the `long_irrelevant` variant.

For each sample:
  - run nctx forward (bare query)
  - run wctx forward ({wiki paragraph} {query})
  - compute Δ_g, Δ_d, Δ_m, KL, ranks, argmax, flip
  - auto-discover distractor token (same procedure as the short irrelevant class in
    run_phase1.py: pick the prefix content token with largest Δℓ)

Outputs:
  degradation/results/<model_short>/phase1_long_irrelevant_<out_tag>.json

Usage:
  python run_phase1_long_irrelevant.py --model meta-llama/Llama-3.1-8B --device cuda:0
"""


import json, time, argparse, string
from pathlib import Path

import torch

from transformer_lens import HookedTransformer
import run_phase1  # reuse helper fns

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT.parent / "lre_dataset" / "data" / "samples_6x250.json"
OUT_ROOT = ROOT / "results"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def short(m): return m.split("/")[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--n-samples", type=int, default=None,
                   help="cap (smoke test); None = all 250")
    p.add_argument("--out-tag", default="full")
    p.add_argument("--include-short-irr-recompute", action="store_true",
                   help="also recompute short_irrelevant + random + counterfactual for sanity check")
    args = p.parse_args()

    model_short = short(args.model)
    out_dir = OUT_ROOT / model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase1_long_irrelevant_{args.out_tag}.json"
    print(f"[{model_short}] loading model on {args.device}  dtype={args.dtype}", flush=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    t0 = time.time()
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, dtype=dtype, device=args.device,
    )
    model.eval()
    print(f"[{model_short}] loaded in {time.time()-t0:.0f}s  "
          f"n_layers={model.cfg.n_layers}  n_heads={model.cfg.n_heads}", flush=True)

    src = json.load(DATA.open())
    samples = src["samples"]
    if args.n_samples:
        samples = samples[:args.n_samples]
    print(f"[{model_short}] {len(samples)} samples loaded", flush=True)

    tok = model.tokenizer
    records = []
    t_start = time.time()

    # Decide which classes to run
    classes_to_run = ["long_irrelevant"]
    if args.include_short_irr_recompute:
        classes_to_run = ["long_irrelevant", "irrelevant", "random", "counterfactual"]

    n_total = len(samples) * len(classes_to_run)
    n_done = 0

    for i, samp in enumerate(samples):
        query = samp["query"]
        gold = samp["gold"]
        gold_id = tok.encode(" " + gold, add_special_tokens=False)[0]
        nctx = query

        for cls in classes_to_run:
            if cls not in samp["variants"]:
                continue
            v = samp["variants"][cls]
            context = v["context"]
            dstr_spec = v["distractor"]
            wctx = f"{context} {query}"
            try:
                if dstr_spec == "__auto__":
                    out = run_phase1.phase1_irrelevant(
                        model, query, nctx, context, gold_id,
                        device=args.device, do_lens=False)
                    out.update({
                        "sample_id": samp["sample_id"],
                        "relation": samp.get("relation"),
                        "class": cls,
                        "query": query,
                        "gold": gold, "gold_id": gold_id,
                        "distractor": out["auto_distractor_tok"],
                        "distractor_id": out["auto_distractor_id"],
                        "wctx_prompt": wctx, "nctx_prompt": nctx,
                        "auto_distractor": True,
                        "wiki_chunk_id": v.get("wiki_chunk_id"),
                        "n_tokens_ws": v.get("n_tokens_ws"),
                    })
                else:
                    dstr_id = tok.encode(" " + dstr_spec, add_special_tokens=False)[0]
                    out = run_phase1.phase1_one_pair(
                        model, wctx, nctx, gold_id, dstr_id, do_lens=False)
                    out.update({
                        "sample_id": samp["sample_id"],
                        "relation": samp.get("relation"),
                        "class": cls, "context": context, "query": query,
                        "gold": gold, "gold_id": gold_id,
                        "distractor": dstr_spec, "distractor_id": dstr_id,
                        "wctx_prompt": wctx, "nctx_prompt": nctx,
                        "auto_distractor": False,
                    })
                records.append(out)
            except Exception as e:
                print(f"[{model_short}] ERROR sample={samp['sample_id']} class={cls}: {e}",
                      flush=True)
            n_done += 1
            if n_done % max(1, n_total // 20) == 0:
                el = time.time() - t_start
                eta = el / n_done * (n_total - n_done)
                print(f"[{model_short}] {n_done}/{n_total}  elapsed={el:.0f}s  eta={eta:.0f}s",
                      flush=True)

    payload = {
        "model": args.model,
        "model_short": model_short,
        "dataset": str(DATA),
        "n_samples": len(samples),
        "classes_run": classes_to_run,
        "elapsed_sec": time.time() - t_start,
        "records": records,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[{model_short}] WROTE {out_path}  records={len(records)}  total_sec={time.time()-t0:.0f}",
          flush=True)


if __name__ == "__main__":
    main()
