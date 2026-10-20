"""Stage 4 defense: class-conditional attention knockout on intrusion heads.

For each (sample, condition):
  baseline forward → lg_gold, ld_dstr, rank_gold
  knockout forward (hook attn_scores at intrusion heads, set scores from END to
    prefix-context positions to -1e9 before softmax) → lg_gold_ko, etc.

Intrusion heads loaded from results/HEAD_SETS.json (per model).

Output: results/<model>__knockout.json
  records: per (sample, condition): { lg_base, lg_ko, ld_base, ld_ko, rank_base, rank_ko }
"""
import sys, json, time, argparse
from pathlib import Path
from collections import defaultdict


import torch
import numpy as np

from transformer_lens import HookedTransformer, utils
REPO = Path(__file__).resolve().parents[2]  # repository root

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAMPLES_PATH = REPO / "lre_dataset/data/samples_6x250.json"
OUT = ROOT / "results"

CONDITIONS = [
    ("nctx",            None),
    ("golden",          "golden"),
    ("related",         "related"),
    ("counterfactual",  "counterfactual"),
    ("short_irr",       "irrelevant"),
    ("long_irr",        "long_irrelevant"),
]


def short(m): return m.split("/")[-1]


def rank_of(logits1d, tok_id):
    return int((torch.argsort(logits1d, descending=True) == tok_id).nonzero(as_tuple=True)[0].item()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=10,
                    help="Number of top intrusion heads to knock out")
    ap.add_argument("--set-name", default="clean_intrusion_set",
                    help="Which head-set to use: intrusion_set or clean_intrusion_set")
    ap.add_argument("--output-path", default=None,
                    help="Optional explicit output path; overrides default <model>__knockout.json")
    args = ap.parse_args()

    mname = short(args.model)
    out_path = Path(args.output_path) if args.output_path else (OUT / f"{mname}__knockout.json")

    # Load head sets
    hs_path = OUT / "HEAD_SETS.json"
    if not hs_path.exists():
        print(f"ERR: {hs_path} missing. Run aggregate_head_scores.py first.")
        sys.exit(1)
    head_sets = json.load(hs_path.open())
    if mname not in head_sets:
        print(f"ERR: {mname} not in HEAD_SETS.json")
        sys.exit(1)
    if args.set_name not in head_sets[mname]:
        print(f"ERR: set_name '{args.set_name}' not present; available: {list(head_sets[mname].keys())}")
        sys.exit(1)
    intrusion = head_sets[mname][args.set_name][: args.top_k]
    print(f"[{mname}] using {args.set_name}, knocking out {len(intrusion)} heads:")
    for L, H, sc in intrusion:
        print(f"   L{L}H{H}  intr={sc:+.3f}")

    # Group heads by layer for hook efficiency
    heads_by_layer = defaultdict(list)
    for L, H, _ in intrusion:
        heads_by_layer[L].append(H)

    t0 = time.time()
    print(f"[{mname}] loading...", flush=True)
    model = HookedTransformer.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True, device=args.device,
    )
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    print(f"[{mname}] L={model.cfg.n_layers} H={model.cfg.n_heads}", flush=True)

    data = json.load(open(SAMPLES_PATH))
    samples = data["samples"]
    if args.n_samples:
        samples = samples[: args.n_samples]

    tok = model.tokenizer

    def make_score_hook(layer_idx, end_pos, ctx_start, ctx_end, heads):
        def fn(scores, hook):
            # scores: [batch, n_heads, q_pos, k_pos]
            for h in heads:
                scores[0, h, end_pos, ctx_start:ctx_end] = -1e9
            return scores
        return fn

    @torch.no_grad()
    def forward(prompt, hooks=None):
        tokens = model.to_tokens(prompt).to(model.cfg.device)
        if hooks:
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        else:
            logits = model(tokens)
        last = logits[0, -1, :].float()
        return last, tokens.shape[1]

    records = []
    for i, samp in enumerate(samples):
        sid = samp["sample_id"]
        query = samp["query"]
        gold = samp["gold"]
        gold_id = tok.encode(" " + gold, add_special_tokens=False)[0]

        # Pre-compute ctx-token slice once per (sample, cond) by tokenizing the prefix
        for cond_name, vkey in CONDITIONS:
            if vkey is None:
                prompt = query
                ctx_start, ctx_end = 0, 0
                dstr_id = None
            else:
                v = samp["variants"][vkey]
                ctx = v["context"]
                prompt = f"{ctx} {query}"
                # Token positions of ctx: full_prompt starts with BOS (pos 0) then "ctx" then " query"
                # We use tokenizer to find the exact ctx slice.
                bos_offset = 1 if model.tokenizer.bos_token_id is not None and \
                    tok.encode(prompt, add_special_tokens=False)[0] != \
                    tok.encode(prompt, add_special_tokens=True)[0] else 0
                # safer: tokenize ctx alone (no bos) and use its length
                ctx_only_tokens = tok.encode(ctx, add_special_tokens=False)
                # full prompt tokens (with BOS depending on model):
                full = model.to_tokens(prompt)[0].tolist()  # 1-d list incl BOS if applicable
                # ctx starts after BOS (if any), spans ctx_only_tokens length
                ctx_start = (1 if len(full) > 1 and full[0] != ctx_only_tokens[0] else 0)
                ctx_end   = ctx_start + len(ctx_only_tokens)
                dstr = v.get("distractor")
                if dstr is not None and dstr != "__auto__":
                    dstr_id = tok.encode(" " + dstr, add_special_tokens=False)[0]
                else:
                    dstr_id = None

            # Get seq_len for hooks
            full_tokens = model.to_tokens(prompt)
            seq_len = full_tokens.shape[1]
            end_pos = seq_len - 1

            # Baseline forward
            base_last, _ = forward(prompt)
            lg_b = float(base_last[gold_id].item())
            ld_b = float(base_last[dstr_id].item()) if dstr_id is not None else float("nan")
            rk_b = rank_of(base_last, gold_id)

            # Knockout forward
            hooks = []
            for L, heads in heads_by_layer.items():
                if ctx_end > ctx_start:
                    hooks.append((utils.get_act_name("attn_scores", L),
                                  make_score_hook(L, end_pos, ctx_start, ctx_end, heads)))
            if hooks:
                ko_last, _ = forward(prompt, hooks=hooks)
            else:
                ko_last = base_last  # nothing to knock out (no ctx)
            lg_k = float(ko_last[gold_id].item())
            ld_k = float(ko_last[dstr_id].item()) if dstr_id is not None else float("nan")
            rk_k = rank_of(ko_last, gold_id)

            records.append({
                "sample_id": sid, "condition": cond_name,
                "ctx_start": ctx_start, "ctx_end": ctx_end, "seq_len": seq_len,
                "lg_base": lg_b, "lg_ko": lg_k,
                "ld_base": ld_b, "ld_ko": ld_k,
                "rank_base": rk_b, "rank_ko": rk_k,
                "delta_lg": lg_k - lg_b,
            })
        if i % max(1, len(samples)//20) == 0:
            print(f"[{mname}] {i+1}/{len(samples)}  elapsed={time.time()-t0:.0f}s", flush=True)

    out = {
        "model": args.model, "model_short": mname,
        "n_top_intrusion": args.top_k,
        "intrusion_set": intrusion,
        "records": records,
        "elapsed_sec": time.time() - t0,
    }
    out_path.write_text(json.dumps(out))
    print(f"[{mname}] DONE in {(time.time()-t0)/60:.1f} min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
