"""GSM-IC defense experiment.

For each math problem in GSM-IC:
  - nctx prompt: "Q: {original_question}\nA:" (clean, no distractor)
  - wctx prompt: "Q: {new_question}\nA:" (with distractor sentence)
  - gold answer: numeric

Method:
  1. Run nctx forward → cache resid_pre[L][last_pos] for L in sweep
  2. Run wctx forward → record argmax & gold-token logit
  3. For each L: patch resid_pre[L][last_pos_wctx] = cached nctx; record argmax & gold-token logit

Metrics:
  - Single-token argmax accuracy on the FIRST token of the gold answer.
    (Robust across tokenizers since some split 72 → "7","2" and some keep "72".)
  - delta_gold logit at position last_pos.

Usage:
    python run_gsm_ic_defense.py --model meta-llama/Llama-3.1-8B \\
        --device cuda:0 --n-samples 500
"""

import json, time, argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

from transformer_lens import HookedTransformer, utils

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "gsm_ic_500.json"
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def short(m): return m.split("/")[-1]


def make_prompt(question):
    # Trailing space forces the model's next-token prediction to BE the answer digit
    # rather than the leading space (which it would otherwise predict trivially).
    return f"Q: {question}\nA: "


def gold_first_token_id(tokenizer, answer):
    """Get the first non-space token of the answer.
    With prompt ending in 'A: ', model's next-token prediction is the answer digit/chunk."""
    # Try multiple encodings to find the first content token
    for encoding_input in [answer, " " + answer]:
        ids = tokenizer.encode(encoding_input, add_special_tokens=False)
        # Skip leading space-only tokens
        for tid in ids:
            decoded = tokenizer.decode([tid]).strip()
            if decoded:  # non-empty after strip = real content
                return tid
    return None


@torch.no_grad()
def run_one_problem(model, nctx, wctx, gold_first_id, layers):
    """Run nctx + wctx + patched at each layer L, return per-layer metrics."""
    device = next(model.parameters()).device
    # 1) nctx forward + cache
    n_tokens = model.to_tokens(nctx).to(device)
    n_logits, n_cache = model.run_with_cache(
        n_tokens,
        names_filter=lambda n: any(n == utils.get_act_name("resid_pre", L) for L in layers),
    )
    n_last = n_logits[0, -1, :].float()
    n_argmax = int(n_last.argmax().item())
    n_correct = int(n_argmax == gold_first_id)
    n_gold_logit = float(n_last[gold_first_id].item())
    nctx_resid = {}
    for L in layers:
        name = utils.get_act_name("resid_pre", L)
        if name in n_cache.cache_dict:
            nctx_resid[L] = n_cache[name][0, -1, :].detach().clone()
    del n_cache; torch.cuda.empty_cache()

    # 2) wctx forward
    w_tokens = model.to_tokens(wctx).to(device)
    last_pos_wctx = w_tokens.shape[1] - 1
    w_logits = model(w_tokens)
    w_last = w_logits[0, -1, :].float()
    w_argmax = int(w_last.argmax().item())
    w_correct = int(w_argmax == gold_first_id)
    w_gold_logit = float(w_last[gold_first_id].item())

    # 3) Patched runs per layer
    per_layer = {}
    for L in layers:
        if L not in nctx_resid: continue
        replacement = nctx_resid[L]
        hook_name = utils.get_act_name("resid_pre", L)
        def make_fn(pos, repl):
            def fn(resid, hook):
                resid[0, pos, :] = repl.to(resid.device).to(resid.dtype)
                return resid
            return fn
        patched_logits = model.run_with_hooks(
            w_tokens, fwd_hooks=[(hook_name, make_fn(last_pos_wctx, replacement))],
        )
        p_last = patched_logits[0, -1, :].float()
        p_argmax = int(p_last.argmax().item())
        p_correct = int(p_argmax == gold_first_id)
        p_gold_logit = float(p_last[gold_first_id].item())
        per_layer[L] = {
            "argmax": p_argmax, "correct": p_correct, "gold_logit": p_gold_logit,
        }
    torch.cuda.empty_cache()
    return {
        "n_argmax": n_argmax, "n_correct": n_correct, "n_gold_logit": n_gold_logit,
        "w_argmax": w_argmax, "w_correct": w_correct, "w_gold_logit": w_gold_logit,
        "per_layer": per_layer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-devices", type=int, default=1)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--data-path", default=str(DATA_PATH))
    args = ap.parse_args()

    mname = short(args.model)
    out_path = OUT / f"{mname}.json"
    t0 = time.time()
    print(f"[{mname}|{args.tag}] loading model n_devices={args.n_devices}", flush=True)
    load_kwargs = dict(dtype=torch.float16, trust_remote_code=True)
    if args.n_devices > 1:
        load_kwargs["n_devices"] = args.n_devices
    else:
        load_kwargs["device"] = args.device
    model = HookedTransformer.from_pretrained(args.model, **load_kwargs)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    nL = model.cfg.n_layers
    print(f"[{mname}] loaded in {time.time()-t0:.0f}s  L={nL}", flush=True)

    # Patch layers: 50%, 75%, 85%, 95% depth and the last layer
    layers = sorted(set([
        nL // 2,
        int(0.75 * nL),
        int(0.85 * nL),
        int(0.95 * nL),
        nL - 1,
    ]))
    print(f"[{mname}] sweep over layers {layers}  (depth fractions: {[round(L/nL, 2) for L in layers]})", flush=True)

    # Load GSM-IC subset
    data = json.load(open(args.data_path))
    if args.n_samples and args.n_samples < len(data):
        data = data[:args.n_samples]
    print(f"[{mname}] {len(data)} GSM-IC problems", flush=True)

    tok = model.tokenizer
    out = {
        "model": args.model, "model_short": mname,
        "n_layers": nL, "n_heads": model.cfg.n_heads,
        "layers_swept": layers, "n_problems": len(data),
        "records": [],
    }

    n_processed = 0
    for i, prob in enumerate(data):
        try:
            orig_q = prob["original_question"]
            new_q = prob["new_question"]
            answer = prob["answer"]
            nctx = make_prompt(orig_q)
            wctx = make_prompt(new_q)
            gold_id = gold_first_token_id(tok, answer)
            if gold_id is None: continue
            r = run_one_problem(model, nctx, wctx, gold_id, layers)
            r["sample_idx"] = i
            r["answer"] = answer
            r["gold_first_id"] = gold_id
            r["n_steps"] = prob.get("n_steps")
            r["sentence_label"] = prob.get("sentence_label")
            r["number_label"] = prob.get("number_label")
            out["records"].append(r)
            n_processed += 1
        except Exception as e:
            print(f"[{mname}] problem {i} failed: {e}", flush=True)
        if (i + 1) % max(1, len(data) // 20) == 0:
            print(f"[{mname}] {i+1}/{len(data)}  elapsed={time.time()-t0:.0f}s", flush=True)

    out["elapsed_sec"] = time.time() - t0
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{mname}] DONE in {(time.time()-t0)/60:.1f} min  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
