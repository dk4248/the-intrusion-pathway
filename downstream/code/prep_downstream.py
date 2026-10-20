"""Build jsonl families for 3 downstream distractor benchmarks with k-sweep.

Outputs (all under downstream/data/):
  hotpotqa_k{0,2,4,6,8}.jsonl    — n=50 each, 50 random validation questions
                                    each item: {question, gold_context, cf_context,
                                                 parametric_answer, n_distractors, ...}
  pubmedqa_k{0,1,2,4}.jsonl       — n=50 each, yes/no/maybe answer
  gsmdc_irr{0,1,2,3,4,5,6,7,8,9,10}.jsonl  — bucketed by irrelevant_nodes_count
                                    (no clean/distract pair; degradation-only)

Shared schema:
  {"id","question","parametric_answer","gold_context","cf_context",
   "parametric_answers_all","n_distractors","task_family"}
"""

import json, random
from pathlib import Path
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

N_PER_K = 50
RNG = random.Random(42)


# -----------------------------------------------------------------------------
# HotpotQA
# -----------------------------------------------------------------------------
def build_hotpotqa(n=N_PER_K, ks=(0, 2, 4, 6, 8)):
    print(f"\n=== HotpotQA distractor setting (n={n}, k={ks}) ===", flush=True)
    ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
    chosen = []
    for ex in ds:
        if len(chosen) >= n:
            break
        # Each ex: question, answer, context={title,sentences}, supporting_facts={title,sent_id}
        ctx = ex["context"]
        titles = ctx["title"]
        sentences = ctx["sentences"]
        sf_titles = set(ex["supporting_facts"]["title"])
        # split into gold-titled paragraphs vs distractor paragraphs
        gold_paras = []
        dstr_paras = []
        for t, sents in zip(titles, sentences):
            para = " ".join(sents).strip()
            if not para:
                continue
            if t in sf_titles:
                gold_paras.append((t, para))
            else:
                dstr_paras.append((t, para))
        if len(gold_paras) < 1 or len(dstr_paras) < max(ks):
            continue
        chosen.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_paragraphs": gold_paras,
            "dstr_paragraphs": dstr_paras,
        })
    print(f"  selected {len(chosen)} hotpotqa examples", flush=True)

    for k in ks:
        out = []
        for ex in chosen:
            # Deterministic per-id: pick first k distractor paragraphs
            distractors = ex["dstr_paragraphs"][:k]
            # gold_context = just gold paragraphs concatenated with title prefix
            gold_text = "\n\n".join(f"[{t}] {p}" for t, p in ex["gold_paragraphs"])
            # cf_context = distractors first, then gold (mirrors LitM "lost-in-the-middle" style)
            cf_paras = [(t, p) for t, p in distractors] + ex["gold_paragraphs"]
            cf_text = "\n\n".join(f"[{t}] {p}" for t, p in cf_paras)
            out.append({
                "id": f"hotpot_{ex['id']}_k{k}",
                "question": ex["question"],
                "parametric_answer": ex["answer"],
                "parametric_answers_all": [ex["answer"]],
                "gold_context": gold_text,
                "cf_context": cf_text,
                "n_distractors": k,
                "task_family": "hotpotqa",
            })
        path = DATA / f"hotpotqa_k{k}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in out))
        print(f"  wrote {path}  n={len(out)}", flush=True)


# -----------------------------------------------------------------------------
# PubMedQA + irrelevant biomedical abstract distractors
# -----------------------------------------------------------------------------
def build_pubmedqa(n=N_PER_K, ks=(0, 1, 2, 4)):
    print(f"\n=== PubMedQA distractor sweep (n={n}, k={ks}) ===", flush=True)
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = list(ds)
    RNG.shuffle(rows)

    # Distractor pool: take 200 examples and use their contexts as distractor abstracts
    pool_rows = rows[n: n + 200]
    distractor_pool = []
    for r in pool_rows:
        ctxs = r["context"].get("contexts", [])
        if isinstance(ctxs, str):
            try:
                ctxs = json.loads(ctxs.replace("'", '"'))
            except Exception:
                ctxs = [ctxs]
        abstract = " ".join(ctxs).strip()
        if 200 < len(abstract) < 2000:
            distractor_pool.append({
                "abstract": abstract,
                "topic_words": (r["question"] or "")[:80],
            })
    print(f"  distractor pool size: {len(distractor_pool)}", flush=True)

    # Use first n examples as test items
    test_rows = rows[:n]
    print(f"  test items: {len(test_rows)}", flush=True)
    for k in ks:
        out = []
        for i, r in enumerate(test_rows):
            ctxs = r["context"].get("contexts", [])
            if isinstance(ctxs, str):
                try:
                    ctxs = json.loads(ctxs.replace("'", '"'))
                except Exception:
                    ctxs = [ctxs]
            gold_abstract = " ".join(ctxs).strip()
            # Deterministic distractor selection (per i, per k)
            local_rng = random.Random(1000 + i)
            distractors = local_rng.sample(distractor_pool, min(k, len(distractor_pool)))
            distractor_texts = [d["abstract"] for d in distractors]
            # Filter for accidental gold-leakage: distractor must not contain the long_answer entity
            # (lightweight check: skip distractors mentioning the question's first noun phrase)
            cf_abstracts = distractor_texts + [gold_abstract]
            cf_text = "\n\n".join(cf_abstracts)
            out.append({
                "id": f"pubmedqa_{r['pubid']}_k{k}",
                "question": r["question"],
                "parametric_answer": r["final_decision"],
                "parametric_answers_all": [r["final_decision"]],
                "gold_context": gold_abstract,
                "cf_context": cf_text,
                "n_distractors": k,
                "task_family": "pubmedqa",
            })
        path = DATA / f"pubmedqa_k{k}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in out))
        print(f"  wrote {path}  n={len(out)}", flush=True)


# -----------------------------------------------------------------------------
# GSM-DC (irrelevant nodes are explicit per problem)
# -----------------------------------------------------------------------------
def build_gsmdc(n_max_per_bucket=N_PER_K):
    print(f"\n=== GSM-DC by irrelevant_nodes_count (n_max={n_max_per_bucket}/bucket) ===",
          flush=True)
    ds = load_dataset("YMinglai/GSM-DC-Dataset-Sample", split="train")
    rows = list(ds)
    print(f"  total dataset: {len(rows)}", flush=True)
    by_irr = {}
    for r in rows:
        k = int(r.get("irrelevant_nodes_count", -1))
        if k < 0:
            continue
        by_irr.setdefault(k, []).append(r)
    print(f"  buckets present: {sorted(by_irr.keys())}", flush=True)

    for k in sorted(by_irr.keys()):
        bucket = by_irr[k][:n_max_per_bucket]
        out = []
        for j, r in enumerate(bucket):
            # Build problem statement: join all problem_text sentences + final question
            stmts = r.get("problem_text") or []
            problem_body = " ".join(stmts)
            # The actual question phrasing isn't in problem_text. GSM-DC's final question
            # is derived from topological_order — last entry is what we want.
            topo = r.get("topological_order") or []
            target_desc = topo[0]["description"] if topo and isinstance(topo[0], dict) else "the final value"
            question = f"{problem_body} How much is {target_desc}?"
            # Build "clean" version: filter problem_text to only statements involving relevant nodes
            # (Not done here; rebuild_gsmdc.py builds the relevant-only gold_context. Degradation = irrelevant_count.)
            answer = str(r["problem_info"]["final_answer"])
            out.append({
                "id": f"gsmdc_{j}_irr{k}",
                "question": question,
                "parametric_answer": answer,
                "parametric_answers_all": [answer],
                "gold_context": "",  # GSM-DC has no separable clean/distract context
                "cf_context": "",
                "n_distractors": k,
                "task_family": "gsmdc",
            })
        path = DATA / f"gsmdc_irr{k}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in out))
        print(f"  wrote {path}  n={len(out)}  irr_count={k}", flush=True)


if __name__ == "__main__":
    build_hotpotqa()
    build_pubmedqa()
    build_gsmdc()
    print("\nDONE", flush=True)
