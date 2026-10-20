"""Per-step residual-patching defense on 3 downstream tasks with k-sweep.

Tasks (auto-detected from jsonl filename):
  hotpotqa_k{0,2,4,6,8}.jsonl   — open-domain multi-hop, free-form answer
  pubmedqa_k{0,1,2,4}.jsonl     — biomedical, yes/no/maybe answer
  gsmdc_irr{0..17}.jsonl        — graph-structured math, integer answer
                                  (clean/distract pair from rebuild_gsmdc.py)

For each item (HotpotQA, PubMedQA, GSM-DC):
  clean    = fewshot + gold_context + question -> greedy 40 tokens
  distract = fewshot + cf_context  + question -> greedy 40 tokens
  patched  = wctx with per-step resid_pre[L*][last_pos] replaced from nctx
             (L* = patch_depth × n_layers, default 0.95)

Task files without gold_context (e.g. GSM-DC as written by prep_downstream.py):
  distract only (no defense possible without a clean/distract pair).
  For GSM-DC, accuracy is reported as a function of irrelevant_nodes_count k.

Outputs:
  downstream/results/<model_short>__<task_jsonl_stem>.json
"""

import json, time, argparse, re
from pathlib import Path

import torch

from transformer_lens import HookedTransformer, utils

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)


# ---------- Per-task few-shot priming ----------
FEWSHOT_QA = """Q: What is the capital of France?
A: Paris.

Q: Who wrote the play 'Romeo and Juliet'?
A: William Shakespeare.

Q: In what year did World War II end?
A: 1945.

"""

FEWSHOT_PUBMEDQA = """Q: Does aspirin reduce the risk of heart attack?
A: yes.

Q: Is exposure to sunlight harmful to all skin types equally?
A: no.

Q: Does coffee consumption affect cardiovascular health in older adults?
A: maybe.

"""

FEWSHOT_GSMDC = """Q: The number of A is 2. The number of B equals A. How much is B?
A: 2.

Q: The number of X equals 5. The number of Y is the sum of X and 3. How much is Y?
A: 8.

Q: A = 4. B = 2 * A. C = B - 3. How much is C?
A: 5.

"""

STOP_PATTERN = "\n\nQ:"


def short(m): return m.split("/")[-1]


def fewshot_for(family):
    return {
        "hotpotqa": FEWSHOT_QA,
        "pubmedqa": FEWSHOT_PUBMEDQA,
        "gsmdc":    FEWSHOT_GSMDC,
    }.get(family, FEWSHOT_QA)


def make_nctx(family, question, gold_context, max_ctx_chars=3000):
    fs = fewshot_for(family)
    if gold_context:
        return fs + f"Context: {gold_context[:max_ctx_chars]}\n\nQ: {question}\nA:"
    return fs + f"Q: {question}\nA:"


def make_wctx(family, question, cf_context, max_ctx_chars=3000):
    fs = fewshot_for(family)
    return fs + f"Context: {cf_context[:max_ctx_chars]}\n\nQ: {question}\nA:"


def normalize(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def first_segment(gen):
    return gen.split(STOP_PATTERN)[0]


def score_hotpotqa(gen, target):
    return normalize(target) in normalize(first_segment(gen))


def score_pubmedqa(gen, target):
    # answer is "yes"/"no"/"maybe"; first non-space word match
    seg = normalize(first_segment(gen)).lstrip(" .,:;-")
    t = normalize(target)
    if not t: return False
    return seg.startswith(t) or (" " + t + " ") in (" " + seg + " ")


def score_gsmdc(gen, target):
    # extract first integer from generation, compare numerically
    seg = first_segment(gen)
    m = re.search(r"-?\d+(?:\.\d+)?", seg)
    if not m: return False
    try:
        return float(m.group(0)) == float(target)
    except Exception:
        return False


SCORERS = {
    "hotpotqa": score_hotpotqa,
    "pubmedqa": score_pubmedqa,
    "gsmdc":    score_gsmdc,
}


@torch.no_grad()
def greedy(model, tokens, max_new=40, stop=STOP_PATTERN):
    device = next(model.parameters()).device
    cur = tokens.clone().to(device)
    gen = []
    tok = model.tokenizer
    for _ in range(max_new):
        out = model(cur)
        nxt = int(out[0, -1, :].argmax().item())
        gen.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        if len(gen) >= 4:
            tail = tok.decode(gen[-8:])
            if stop in tail:
                break
    return gen


@torch.no_grad()
def greedy_perstep_patch(model, n_tok, w_tok, layer, max_new=40, stop=STOP_PATTERN):
    device = next(model.parameters()).device
    n_cur = n_tok.clone().to(device)
    w_cur = w_tok.clone().to(device)
    name = utils.get_act_name("resid_pre", layer)
    gen = []
    tok = model.tokenizer
    for _ in range(max_new):
        _, n_cache = model.run_with_cache(n_cur, names_filter=lambda nn: nn == name)
        if name not in n_cache.cache_dict:
            del n_cache; torch.cuda.empty_cache(); return gen
        n_resid = n_cache[name][0, -1, :].detach().clone()
        del n_cache
        last_pos = w_cur.shape[1] - 1
        repl = n_resid
        def fn(resid, hook):
            resid[0, last_pos, :] = repl.to(resid.device).to(resid.dtype)
            return resid
        out = model.run_with_hooks(w_cur, fwd_hooks=[(name, fn)])
        nxt = int(out[0, -1, :].argmax().item())
        gen.append(nxt)
        nxt_t = torch.tensor([[nxt]], device=device)
        n_cur = torch.cat([n_cur, nxt_t], dim=1)
        w_cur = torch.cat([w_cur, nxt_t], dim=1)
        if len(gen) >= 4:
            tail = tok.decode(gen[-8:])
            if stop in tail:
                break
    return gen


def run_task(model, jsonl_path, patch_layer, max_new=40, max_ctx_chars=3000, log_prefix=""):
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not rows:
        return None
    family = rows[0].get("task_family", "hotpotqa")
    scorer = SCORERS[family]
    # gsmdc has gold_context populated by rebuild_gsmdc.py (relevant statements only): run clean/distract/patched
    distract_only = False
    # safety: if any item lacks gold_context, fall back to distract-only
    if not any(r.get("gold_context") for r in rows):
        distract_only = True
    tok = model.tokenizer

    out = {
        "task_file": jsonl_path.name,
        "task_family": family,
        "n_problems": len(rows),
        "max_new_tokens": max_new,
        "patch_layer": patch_layer,
        "distract_only": distract_only,
        "records": [],
    }

    n_c = n_w = n_p = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        q = r["question"]
        param = r["parametric_answer"]
        cf_ctx = r["cf_context"]
        try:
            w_prompt = make_wctx(family, q, cf_ctx, max_ctx_chars=max_ctx_chars)
            w_tok = model.to_tokens(w_prompt).to(next(model.parameters()).device)
            w_gen = tok.decode(greedy(model, w_tok, max_new=max_new))
            w_ok = scorer(w_gen, param)
            rec = {"id": r["id"], "param": param, "k": r.get("n_distractors"),
                   "w_gen": w_gen[:300], "w_ok": int(w_ok)}
            if distract_only:
                # No clean pair (no gold_context). Report w_ok only.
                rec.update({"n_gen": None, "p_gen": None, "n_ok": None, "p_ok": None})
                n_w += int(w_ok)
            else:
                gold_ctx = r.get("gold_context", "") or ""
                n_prompt = make_nctx(family, q, gold_ctx, max_ctx_chars=max_ctx_chars)
                n_tok = model.to_tokens(n_prompt).to(next(model.parameters()).device)
                n_gen = tok.decode(greedy(model, n_tok, max_new=max_new))
                n_ok = scorer(n_gen, param)
                # Per-step patched only if k > 0 AND there's something to patch from
                if r.get("n_distractors", 0) > 0 and gold_ctx:
                    p_gen = tok.decode(greedy_perstep_patch(model, n_tok, w_tok, patch_layer,
                                                             max_new=max_new))
                else:
                    p_gen = w_gen
                p_ok = scorer(p_gen, param)
                rec.update({"n_gen": n_gen[:300], "p_gen": p_gen[:300],
                            "n_ok": int(n_ok), "p_ok": int(p_ok)})
                n_c += int(n_ok); n_w += int(w_ok); n_p += int(p_ok)
            out["records"].append(rec)
        except Exception as e:
            print(f"{log_prefix} item {i} failed: {e}", flush=True)
        n = i + 1
        if n % 10 == 0 or n == len(rows):
            if distract_only:
                print(f"{log_prefix} {n}/{len(rows)}  elapsed={time.time()-t0:.0f}s  "
                      f"w_acc={n_w/n:.3f}", flush=True)
            else:
                print(f"{log_prefix} {n}/{len(rows)}  elapsed={time.time()-t0:.0f}s  "
                      f"clean={n_c/n:.3f}  distract={n_w/n:.3f}  patched={n_p/n:.3f}",
                      flush=True)
    nd = max(len(out["records"]), 1)
    out["w_acc"] = n_w / nd
    if not distract_only:
        out["clean_acc"] = n_c / nd
        out["patched_acc"] = n_p / nd
    out["elapsed_sec"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="Specific jsonl stems to run (e.g., 'hotpotqa_k2 pubmedqa_k1'). "
                         "Default: all jsonls in downstream/data/")
    ap.add_argument("--patch-depth", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--max-ctx-chars", type=int, default=3000)
    args = ap.parse_args()

    mname = short(args.model)
    log_prefix = f"[{mname}]"

    print(f"{log_prefix} loading model on {args.device}  dtype={args.dtype}", flush=True)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    t0 = time.time()
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, dtype=dtype, device=args.device,
    )
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    nL = model.cfg.n_layers
    patch_layer = int(args.patch_depth * nL)
    if patch_layer >= nL: patch_layer = nL - 1
    print(f"{log_prefix} loaded in {time.time()-t0:.0f}s  L={nL}  patch_L={patch_layer}", flush=True)

    if args.tasks:
        task_files = [DATA / f"{t}.jsonl" for t in args.tasks]
    else:
        task_files = sorted(DATA.glob("*.jsonl"))
    print(f"{log_prefix} {len(task_files)} task files to run", flush=True)

    for tp in task_files:
        if not tp.exists():
            print(f"{log_prefix} SKIP missing {tp}", flush=True)
            continue
        sub_prefix = f"{log_prefix}[{tp.stem}]"
        print(f"\n{sub_prefix} starting", flush=True)
        res = run_task(model, tp, patch_layer, max_new=args.max_new_tokens,
                       max_ctx_chars=args.max_ctx_chars, log_prefix=sub_prefix)
        if res is None:
            continue
        res["model"] = args.model
        res["model_short"] = mname
        res["n_layers"] = nL
        out_path = OUT / f"{mname}__{tp.stem}.json"
        out_path.write_text(json.dumps(res, indent=2))
        print(f"{sub_prefix} WROTE {out_path}", flush=True)

    print(f"\n{log_prefix} ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
