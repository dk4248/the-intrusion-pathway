"""GSM-IC CoT defense with PER-STEP residual patching.

At every generation step k, the wctx residual at last_pos is replaced
with the nctx residual at the corresponding position from a parallel
nctx forward pass that has been extended with the same generated
tokens so far.

Algorithm per problem:
  1. nctx_prompt = "Q: <orig_q>\nA: Let's think step by step."
  2. wctx_prompt = "Q: <new_q>\nA: Let's think step by step."
  3. For step k = 1..max_new:
       a) Forward nctx_prompt + generated_so_far → cache resid_pre[L][last_pos]
       b) Forward wctx_prompt + generated_so_far with hook patching
          resid_pre[L][last_pos_wctx] ← cached nctx residual
       c) Take argmax of last logit, append to generated_so_far
       d) Stop if EOS or stop pattern hit

Compare to:
  - clean: nctx alone, normal greedy generation (no hook)
  - distract: wctx alone, normal greedy generation (no hook)
  - patched: wctx with per-step hook (this method)

Cost: 2 forward passes per generation step (one nctx, one wctx-patched)
plus standalone clean and distract generations.

Usage:
    python run_gsm_ic_cot_perstep.py --model meta-llama/Llama-3.1-8B \\
        --device cuda:0 --n-samples 10
"""

import json, time, argparse, re
from pathlib import Path

import torch

from transformer_lens import HookedTransformer, utils

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "gsm_ic_500.json"
OUT = HERE.parent / "results_cot_perstep"
OUT.mkdir(parents=True, exist_ok=True)


def short(m): return m.split("/")[-1]


FEWSHOT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: Let's think step by step. There are 15 trees originally. After planting, there are 21 trees. So the workers planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. Together they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

"""


def make_prompt(question):
    return FEWSHOT + f"Q: {question}\nA: Let's think step by step."


STOP_PATTERN = "\n\nQ:"


def extract_answer(text):
    if STOP_PATTERN in text:
        text = text.split(STOP_PATTERN)[0]
    m = re.search(r"[Aa]nswer\s+is\s*\$?(-?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"(-?\d[\d,]*(?:\.\d+)?)", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


@torch.no_grad()
def greedy_generate_no_hook(model, tokens, max_new=120, stop_text=STOP_PATTERN):
    """Plain greedy generation, no hook."""
    device = next(model.parameters()).device
    cur = tokens.clone().to(device)
    generated = []
    tok = model.tokenizer
    for _ in range(max_new):
        out = model(cur)
        nxt = int(out[0, -1, :].argmax().item())
        generated.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        # quick stop check on decoded tail
        if len(generated) >= 4:
            tail = tok.decode(generated[-8:])
            if stop_text in tail:
                break
    return generated


@torch.no_grad()
def greedy_generate_perstep_patch(model, nctx_tokens, wctx_tokens, layer, max_new=120,
                                   stop_text=STOP_PATTERN):
    """Per-step patching: at every generation step, patch wctx's last-pos resid
    at layer L with nctx's last-pos resid (computed from nctx_tokens + same
    generated suffix)."""
    device = next(model.parameters()).device
    nctx_cur = nctx_tokens.clone().to(device)
    wctx_cur = wctx_tokens.clone().to(device)
    name = utils.get_act_name("resid_pre", layer)
    generated = []
    tok = model.tokenizer
    for _ in range(max_new):
        # Forward on nctx side; cache resid_pre[L] at last position
        _, n_cache = model.run_with_cache(nctx_cur, names_filter=lambda n: n == name)
        if name not in n_cache.cache_dict:
            # safety fallback
            del n_cache; torch.cuda.empty_cache()
            return generated
        nctx_resid = n_cache[name][0, -1, :].detach().clone()
        del n_cache

        # Forward on wctx side with hook
        last_pos_wctx = wctx_cur.shape[1] - 1
        repl = nctx_resid
        def fn(resid, hook):
            resid[0, last_pos_wctx, :] = repl.to(resid.device).to(resid.dtype)
            return resid
        out = model.run_with_hooks(wctx_cur, fwd_hooks=[(name, fn)])
        nxt = int(out[0, -1, :].argmax().item())
        generated.append(nxt)
        # Append to BOTH so the next step's contexts grow in lockstep
        nxt_t = torch.tensor([[nxt]], device=device)
        nctx_cur = torch.cat([nctx_cur, nxt_t], dim=1)
        wctx_cur = torch.cat([wctx_cur, nxt_t], dim=1)
        # stop check
        if len(generated) >= 4:
            tail = tok.decode(generated[-8:])
            if stop_text in tail:
                break
    return generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-devices", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--data-path", default=str(DATA_PATH))
    ap.add_argument("--patch-depth", type=float, default=0.95)
    args = ap.parse_args()

    mname = short(args.model)
    out_path = OUT / f"{mname}.json"
    t0 = time.time()
    print(f"[{mname}|cot-perstep] loading model n_devices={args.n_devices}", flush=True)
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
    patch_layer = int(args.patch_depth * nL)
    if patch_layer >= nL: patch_layer = nL - 1
    print(f"[{mname}] patch_layer = {patch_layer} (depth {patch_layer/nL:.2f})", flush=True)

    tok = model.tokenizer

    data = json.load(open(args.data_path))
    if args.n_samples and args.n_samples < len(data):
        data = data[:args.n_samples]
    print(f"[{mname}] {len(data)} problems × 3 conditions × ~{args.max_new_tokens} steps "
          f"(per-step costs 2 forwards in patched condition)", flush=True)

    out = {
        "model": args.model, "model_short": mname,
        "n_layers": nL, "patch_layer": patch_layer,
        "n_problems": len(data), "max_new_tokens": args.max_new_tokens,
        "method": "per-step patching",
        "records": [],
    }

    n_clean = 0; n_distract = 0; n_patched = 0
    for i, prob in enumerate(data):
        try:
            orig_q = prob["original_question"]
            new_q = prob["new_question"]
            answer = str(prob["answer"]).strip()

            nctx = make_prompt(orig_q)
            wctx = make_prompt(new_q)
            n_tokens = model.to_tokens(nctx).to(next(model.parameters()).device)
            w_tokens = model.to_tokens(wctx).to(next(model.parameters()).device)

            # Clean: greedy on nctx
            n_gen_ids = greedy_generate_no_hook(model, n_tokens, max_new=args.max_new_tokens)
            n_gen = tok.decode(n_gen_ids)

            # Distractor: greedy on wctx
            w_gen_ids = greedy_generate_no_hook(model, w_tokens, max_new=args.max_new_tokens)
            w_gen = tok.decode(w_gen_ids)

            # Patched: per-step patching
            p_gen_ids = greedy_generate_perstep_patch(
                model, n_tokens, w_tokens, patch_layer,
                max_new=args.max_new_tokens,
            )
            p_gen = tok.decode(p_gen_ids)

            n_ans = extract_answer(n_gen)
            w_ans = extract_answer(w_gen)
            p_ans = extract_answer(p_gen)
            n_correct = (n_ans == answer)
            w_correct = (w_ans == answer)
            p_correct = (p_ans == answer)

            n_clean += n_correct
            n_distract += w_correct
            n_patched += p_correct

            out["records"].append({
                "sample_idx": i, "answer": answer,
                "n_gen": n_gen, "n_ans": n_ans, "n_correct": int(n_correct),
                "w_gen": w_gen, "w_ans": w_ans, "w_correct": int(w_correct),
                "p_gen": p_gen, "p_ans": p_ans, "p_correct": int(p_correct),
            })
        except Exception as e:
            print(f"[{mname}] problem {i} failed: {e}", flush=True)

        n = i + 1
        print(f"[{mname}] {n}/{len(data)}  elapsed={time.time()-t0:.0f}s  "
              f"clean={n_clean/n:.3f}  distract={n_distract/n:.3f}  "
              f"patched={n_patched/n:.3f}", flush=True)

    out["elapsed_sec"] = time.time() - t0
    n_done = max(len(out["records"]), 1)
    out["clean_acc"] = n_clean / n_done
    out["distract_acc"] = n_distract / n_done
    out["patched_acc"] = n_patched / n_done
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{mname}] DONE in {(time.time()-t0)/60:.1f} min  -> {out_path}", flush=True)
    print(f"[{mname}] FINAL clean={out['clean_acc']:.3f}  distract={out['distract_acc']:.3f}  "
          f"patched={out['patched_acc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
