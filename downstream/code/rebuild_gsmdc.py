"""Rebuild gsmdc_irr*.jsonl with gold_context (relevant-only) + cf_context (full)
fields populated, so the same clean/distract/patched protocol used for
HotpotQA/PubMedQA applies.

Filter rule:
  A problem_text statement is RELEVANT if it mentions any description in the
  problem's `topological_order`. The final question (statement that doesn't
  match "The number of each X equals ..." pattern) is ALWAYS kept.
"""
import json, re
from pathlib import Path
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

N_PER_K = 50


def is_definition(stmt):
    """A 'definition' statement has the form 'The number of each X equals ...'."""
    return stmt.strip().lower().startswith("the number of each")


def filter_relevant(problem_text, topo_descs):
    """Keep statements that either (a) define a node in topo_descs or
    (b) are not definitions (i.e., the final question)."""
    keep = []
    for s in problem_text:
        if not is_definition(s):
            keep.append(s)
            continue
        # extract LHS of "the number of each LHS equals ..."
        m = re.match(r"\s*The number of each (.+?) equals", s, flags=re.I)
        if m:
            lhs = m.group(1).strip()
            # match if topo_desc equals lhs OR is a substring of it
            if any(d == lhs or d in lhs or lhs in d for d in topo_descs):
                keep.append(s)
        # Statements without recognizable LHS: keep them (conservative)
        else:
            keep.append(s)
    return keep


def main():
    ds = load_dataset("YMinglai/GSM-DC-Dataset-Sample", split="train")
    rows = list(ds)
    print(f"loaded {len(rows)} GSM-DC problems", flush=True)
    by_irr = {}
    for r in rows:
        k = int(r.get("irrelevant_nodes_count", -1))
        if k < 0:
            continue
        by_irr.setdefault(k, []).append(r)

    diagnostics = []
    for k in sorted(by_irr.keys()):
        bucket = by_irr[k][:N_PER_K]
        out = []
        n_actually_filtered = 0
        for j, r in enumerate(bucket):
            stmts = r.get("problem_text") or []
            topo = r.get("topological_order") or []
            topo_descs = [t["description"] for t in topo if isinstance(t, dict)]
            relevant = filter_relevant(stmts, set(topo_descs))
            if len(relevant) < len(stmts):
                n_actually_filtered += 1
            # the final question is what the dataset uses as the question
            # (last definition's LHS becomes the target)
            target_desc = topo[0]["description"] if topo and isinstance(topo[0], dict) else "the final value"

            # gold_context = relevant statements only (joined)
            gold_body = " ".join(relevant)
            # cf_context = all statements
            cf_body = " ".join(stmts)
            # The "question" is a uniform prompt asking for the target
            question = f"How much is {target_desc}?"
            answer = str(r["problem_info"]["final_answer"])

            out.append({
                "id": f"gsmdc_{j}_irr{k}",
                "question": question,
                "parametric_answer": answer,
                "parametric_answers_all": [answer],
                "gold_context": gold_body,
                "cf_context": cf_body,
                "n_distractors": k,
                "task_family": "gsmdc",
                "n_statements_total": len(stmts),
                "n_statements_relevant": len(relevant),
            })
        diagnostics.append({"k": k, "n_problems": len(bucket),
                            "n_with_filtering": n_actually_filtered,
                            "mean_n_total": sum(o["n_statements_total"] for o in out) / max(1, len(out)),
                            "mean_n_relevant": sum(o["n_statements_relevant"] for o in out) / max(1, len(out))})
        path = DATA / f"gsmdc_irr{k}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in out))
        print(f"  k={k:>2}: n={len(out):>2}  mean_stmts={diagnostics[-1]['mean_n_total']:>5.1f} "
              f"-> {diagnostics[-1]['mean_n_relevant']:>5.1f} relevant  "
              f"({n_actually_filtered}/{len(bucket)} had filtering)  -> {path.name}", flush=True)

    print("\nDONE", flush=True)
    print("Diagnostics summary (per k):", flush=True)
    for d in diagnostics:
        ratio = d["mean_n_relevant"] / d["mean_n_total"]
        print(f"  k={d['k']:>2}  n_problems={d['n_problems']:>2}  "
              f"mean_stmts {d['mean_n_total']:>5.1f} -> {d['mean_n_relevant']:>5.1f}  "
              f"({ratio:.0%} retained)")


if __name__ == "__main__":
    main()
