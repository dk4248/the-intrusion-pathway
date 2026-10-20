"""Per-step HEAD-LEVEL attention knockout defense on the downstream tasks.

Same iteration loop and task set as run_downstream.py, but instead of replacing
resid_pre at a layer, we hook attn_scores at the intrusion-head layers and set
scores[h, GEN_POS, ctx_positions] = -1e9 (so attention from the position being
generated to the irrelevant-context tokens is zero for those heads).

Identifies ctx_positions by locating the "Context: ... \\n\\nQ:" span in the wctx
prompt at tokenization time.

Outputs results/<model>__<task>__hko_K{K}_{clean|full}.json with clean_acc and
distract_acc as in run_downstream.py, plus "hko_acc" (head-knockout accuracy)
in place of the patched accuracy.

Usage:
  python run_downstream_head_ko.py --model meta-llama/Llama-3.1-8B --tasks gsmdc_irr4
    --top-k 10 --set-name clean_intrusion_set
"""

import json, time, argparse, re
from pathlib import Path
from collections import defaultdict

import torch

from transformer_lens import HookedTransformer, utils
REPO = Path(__file__).resolve().parents[2]  # repository root

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "results"
HEAD_SETS_PATH = REPO / "intrusion_heads/results/HEAD_SETS.json"
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


def make_nctx(family, q, gold_context, max_ctx_chars=3000):
    fs = fewshot_for(family)
    if gold_context:
        return fs + f"Context: {gold_context[:max_ctx_chars]}\n\nQ: {q}\nA:"
    return fs + f"Q: {q}\nA:"


def make_wctx(family, q, cf_context, max_ctx_chars=3000):
    fs = fewshot_for(family)
    return fs + f"Context: {cf_context[:max_ctx_chars]}\n\nQ: {q}\nA:"


def normalize(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def first_segment(gen):
    return gen.split(STOP_PATTERN)[0]


def score_hotpotqa(gen, target):
    return normalize(target) in normalize(first_segment(gen))


def score_pubmedqa(gen, target):
    seg = normalize(first_segment(gen)).lstrip(" .,:;-")
    t = normalize(target)
    return seg.startswith(t) if t else False


def score_gsmdc(gen, target):
    seg = first_segment(gen)
    m = re.search(r"-?\d+(?:\.\d+)?", seg)
    if not m: return False
    try:
        return float(m.group(0)) == float(target)
    except Exception:
        return False


SCORERS = {"hotpotqa": score_hotpotqa, "pubmedqa": score_pubmedqa, "gsmdc": score_gsmdc}


def find_ctx_span_in_tokens(model, family, cf_context, q, max_ctx_chars=3000):
    """Return (ctx_start, ctx_end) token positions of the irrelevant-content span
    within the tokenized wctx prompt. ctx_end is exclusive.

    Strategy: tokenize the prompt up to (and not including) the cf_context, then
    tokenize again with the cf_context included; the difference is the cf-tokens.
    """
    fs = fewshot_for(family)
    pre = fs + "Context: "
    post = f"\n\nQ: {q}\nA:"
    cf_chunk = cf_context[:max_ctx_chars]
    pre_tokens = model.to_tokens(pre)[0]
    full_tokens = model.to_tokens(pre + cf_chunk + post)[0]
    pre_then_cf = model.to_tokens(pre + cf_chunk)[0]
    ctx_start = len(pre_tokens)
    ctx_end   = len(pre_then_cf)
    # Sanity: ctx_end must be <= full_tokens length
    seq_len = len(full_tokens)
    if not (0 <= ctx_start < ctx_end <= seq_len):
        return None, None
    return ctx_start, ctx_end


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


def make_score_hook(layer_idx, ctx_start, ctx_end, heads_in_layer):
    """Hook on attn_scores: for each gen step, set scores at GEN_POS row, ctx col range
    to -1e9 for the given heads."""
    def fn(scores, hook):
        # scores: [batch, n_heads, q_pos, k_pos]
        seq_len = scores.shape[-1]
        gen_pos = seq_len - 1
        for h in heads_in_layer:
            scores[0, h, gen_pos, ctx_start:min(ctx_end, seq_len)] = -1e9
        return scores
    return fn


@torch.no_grad()
def greedy_head_ko(model, w_tok, ctx_start, ctx_end, intrusion_heads_by_layer,
                    max_new=40, stop=STOP_PATTERN):
    device = next(model.parameters()).device
    cur = w_tok.clone().to(device)
    gen = []
    tok = model.tokenizer
    hooks_by_layer = {L: heads for L, heads in intrusion_heads_by_layer.items()}
    for _ in range(max_new):
        # Build hooks dynamic to current seq_len
        fwd_hooks = []
        for L, heads in hooks_by_layer.items():
            fwd_hooks.append((utils.get_act_name("attn_scores", L),
                              make_score_hook(L, ctx_start, ctx_end, heads)))
        out = model.run_with_hooks(cur, fwd_hooks=fwd_hooks)
        nxt = int(out[0, -1, :].argmax().item())
        gen.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        if len(gen) >= 4:
            tail = tok.decode(gen[-8:])
            if stop in tail:
                break
    return gen


def run_task(model, jsonl_path, intrusion_heads_by_layer, max_new=40,
             max_ctx_chars=3000, log_prefix=""):
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    if not rows: return None
    family = rows[0].get("task_family", "hotpotqa")
    scorer = SCORERS[family]
    distract_only = not any(r.get("gold_context") for r in rows)
    tok = model.tokenizer
    out = {
        "task_file": jsonl_path.name,
        "task_family": family,
        "n_problems": len(rows),
        "max_new_tokens": max_new,
        "distract_only": distract_only,
        "records": [],
    }
    n_c = n_w = n_h = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        q = r["question"]
        param = r["parametric_answer"]
        cf_ctx = r["cf_context"]
        try:
            w_prompt = make_wctx(family, q, cf_ctx, max_ctx_chars=max_ctx_chars)
            w_tok = model.to_tokens(w_prompt).to(next(model.parameters()).device)
            ctx_start, ctx_end = find_ctx_span_in_tokens(
                model, family, cf_ctx, q, max_ctx_chars)
            if ctx_start is None:
                # fallback: no knockout possible; head-KO == w_gen
                w_gen = tok.decode(greedy(model, w_tok, max_new=max_new))
                h_gen = w_gen
                w_ok = scorer(w_gen, param); h_ok = w_ok
            else:
                w_gen = tok.decode(greedy(model, w_tok, max_new=max_new))
                h_gen = tok.decode(greedy_head_ko(model, w_tok, ctx_start, ctx_end,
                                                    intrusion_heads_by_layer, max_new=max_new))
                w_ok = scorer(w_gen, param)
                h_ok = scorer(h_gen, param)
            rec = {"id": r["id"], "param": param, "k": r.get("n_distractors"),
                   "w_gen": w_gen[:300], "h_gen": h_gen[:300],
                   "w_ok": int(w_ok), "h_ok": int(h_ok),
                   "ctx_start": ctx_start, "ctx_end": ctx_end}
            if distract_only:
                rec.update({"n_gen": None, "n_ok": None})
                n_w += int(w_ok); n_h += int(h_ok)
            else:
                gold_ctx = r.get("gold_context", "") or ""
                n_prompt = make_nctx(family, q, gold_ctx, max_ctx_chars=max_ctx_chars)
                n_tok = model.to_tokens(n_prompt).to(next(model.parameters()).device)
                n_gen = tok.decode(greedy(model, n_tok, max_new=max_new))
                n_ok = scorer(n_gen, param)
                rec.update({"n_gen": n_gen[:300], "n_ok": int(n_ok)})
                n_c += int(n_ok); n_w += int(w_ok); n_h += int(h_ok)
            out["records"].append(rec)
        except Exception as e:
            print(f"{log_prefix} item {i} failed: {e}", flush=True)
        n = i + 1
        if n % 10 == 0 or n == len(rows):
            if distract_only:
                print(f"{log_prefix} {jsonl_path.stem} {n}/{len(rows)} "
                      f"distract={n_w/n:.3f}  hko={n_h/n:.3f}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            else:
                print(f"{log_prefix} {jsonl_path.stem} {n}/{len(rows)} "
                      f"clean={n_c/n:.3f}  distract={n_w/n:.3f}  hko={n_h/n:.3f}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    nd = len([r for r in out["records"]])
    if nd:
        if not distract_only:
            out["clean_acc"] = n_c / nd
        out["distract_acc"] = n_w / nd
        out["hko_acc"] = n_h / nd
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="filename stems under data/, e.g. gsmdc_irr4 hotpotqa_k4 pubmedqa_k2")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--set-name", default="clean_intrusion_set",
                    help="clean_intrusion_set or intrusion_set")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--max-ctx-chars", type=int, default=3000)
    ap.add_argument("--label-suffix", default=None,
                    help="custom suffix; default = K{K}_{clean|full}")
    args = ap.parse_args()

    mname = short(args.model)
    log_prefix = f"[{mname}|hko]"

    head_sets = json.load(HEAD_SETS_PATH.open())
    if mname not in head_sets:
        raise RuntimeError(f"{mname} not in {HEAD_SETS_PATH}")
    if args.set_name not in head_sets[mname]:
        raise RuntimeError(f"set_name {args.set_name} not in head_sets[{mname}]")
    intrusion = head_sets[mname][args.set_name][:args.top_k]
    print(f"{log_prefix} loaded {len(intrusion)} intrusion heads (set={args.set_name}, top-{args.top_k})")
    heads_by_layer = defaultdict(list)
    for L, H, _ in intrusion:
        heads_by_layer[L].append(H)
    print(f"{log_prefix} heads_by_layer = {dict(heads_by_layer)}")

    print(f"{log_prefix} loading model on {args.device}", flush=True)
    t0 = time.time()
    model = HookedTransformer.from_pretrained(args.model,
        dtype=torch.float16, trust_remote_code=True, device=args.device)
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    nL = model.cfg.n_layers
    print(f"{log_prefix} loaded in {time.time()-t0:.0f}s L={nL}", flush=True)

    if args.tasks:
        task_paths = []
        for t in args.tasks:
            tp = DATA / f"{t}.jsonl"
            if tp.exists():
                task_paths.append(tp)
            else:
                print(f"{log_prefix} skip: {tp} missing")
    else:
        # Default: all gsmdc_irr*, hotpotqa_k*, pubmedqa_k* with >= 10 problems
        task_paths = [p for p in sorted(DATA.glob("*.jsonl"))
                       if len(p.read_text().splitlines()) >= 10]

    label = args.label_suffix or f"K{args.top_k}_{'clean' if 'clean' in args.set_name else 'full'}"
    for tp in task_paths:
        out_path = OUT / f"{mname}__{tp.stem}__hko_{label}.json"
        if out_path.exists():
            print(f"{log_prefix} SKIP {out_path.name} (exists)")
            continue
        res = run_task(model, tp, dict(heads_by_layer), max_new=args.max_new_tokens,
                        max_ctx_chars=args.max_ctx_chars, log_prefix=log_prefix)
        if res is not None:
            out_path.write_text(json.dumps(res))
            print(f"{log_prefix} WROTE {out_path}")


if __name__ == "__main__":
    main()
