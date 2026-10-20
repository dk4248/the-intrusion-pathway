"""Aggregate downstream-defense results across (model, task, k).

Produces:
  downstream/results/AGGREGATE_downstream.json
  downstream/results/AGGREGATE_downstream.txt
"""
import json, re
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"

MODELS = ["Llama-3.1-8B", "Mistral-7B-v0.1", "Qwen2.5-7B", "pythia-6.9b"]


def parse_file(p):
    if not p.exists():
        return None
    try:
        return json.load(p.open())
    except Exception:
        return None


def main():
    data = {}  # model -> task_family -> k -> {clean, distract, patched, n}
    for model in MODELS:
        data[model] = {"hotpotqa": {}, "pubmedqa": {}, "gsmdc": {}}
        for f in sorted(RESULTS.glob(f"{model}__*.json")):
            if "AGGREGATE" in f.name or "__hko_" in f.name:   # head-knockout runs are a separate intervention
                continue
            res = parse_file(f)
            if not res:
                continue
            stem = f.stem.replace(f"{model}__", "")
            family = res.get("task_family")
            if family is None:
                if stem.startswith("hotpot"):
                    family = "hotpotqa"
                elif stem.startswith("pubmed"):
                    family = "pubmedqa"
                elif stem.startswith("gsmdc"):
                    family = "gsmdc"
            m = re.search(r"(?:k|irr)(\d+)", stem)
            k = int(m.group(1)) if m else None
            cell = {
                "n": res["n_problems"],
                "w_acc": res.get("w_acc"),
            }
            if not res.get("distract_only"):
                cell["clean_acc"] = res.get("clean_acc")
                cell["patched_acc"] = res.get("patched_acc")
                cl = cell["clean_acc"]; di = cell["w_acc"]; pa = cell["patched_acc"]
                if cl is not None and di is not None and pa is not None and cl > di:
                    cell["recovery_pp_of_drop"] = (pa - di) / (cl - di) * 100.0
                else:
                    cell["recovery_pp_of_drop"] = None
            data[model][family][k] = cell

    (RESULTS / "AGGREGATE_downstream.json").write_text(json.dumps(data, indent=2))

    lines = []
    lines.append("=" * 120)
    lines.append("DOWNSTREAM DEFENSE RESULTS — HotpotQA + PubMedQA + GSM-DC")
    lines.append("=" * 120)
    lines.append("n=50 per (model, task, k) unless noted.")
    lines.append("clean = greedy with gold context  |  distract = with cf context  |  patched = per-step resid hook L*=0.95N")
    lines.append("recovery_% = (patched - distract) / (clean - distract) × 100, only when clean > distract.")
    lines.append("")

    # ---- HotpotQA dose-response ----
    lines.append("-" * 120)
    lines.append("HotpotQA (open-domain multi-hop, free-form answer; k = number of distractor paragraphs added to gold)")
    lines.append("-" * 120)
    header = f"  {'Model':<20} | " + "  ".join(f"{'k=':>1}{k:<2}{'cln':>4}/{'dis':>4}/{'pat':>4}/{'rec%':>5}" for k in [0,2,4,6,8])
    lines.append(header)
    for model in MODELS:
        row = f"  {model:<20} | "
        cells = []
        for k in [0, 2, 4, 6, 8]:
            d = data[model]["hotpotqa"].get(k)
            if d:
                cln = d.get("clean_acc"); dis = d.get("w_acc"); pat = d.get("patched_acc")
                rec = d.get("recovery_pp_of_drop")
                cells.append(f"k={k:<2} {cln:.2f}/{dis:.2f}/{pat:.2f}/{('--' if rec is None else f'{rec:>4.0f}%'):>5}")
            else:
                cells.append(f"k={k:<2} {'.':>20}")
        lines.append(row + "  ".join(cells))
    lines.append("")

    # ---- PubMedQA dose-response ----
    lines.append("-" * 120)
    lines.append("PubMedQA (yes/no/maybe; k = number of irrelevant biomedical abstracts prepended to gold abstract)")
    lines.append("-" * 120)
    header = f"  {'Model':<20} | " + "  ".join(f"{'k=':>1}{k:<2}{'cln':>4}/{'dis':>4}/{'pat':>4}/{'rec%':>5}" for k in [0,1,2,4])
    lines.append(header)
    for model in MODELS:
        row = f"  {model:<20} | "
        cells = []
        for k in [0, 1, 2, 4]:
            d = data[model]["pubmedqa"].get(k)
            if d:
                cln = d.get("clean_acc"); dis = d.get("w_acc"); pat = d.get("patched_acc")
                rec = d.get("recovery_pp_of_drop")
                cells.append(f"k={k:<2} {cln:.2f}/{dis:.2f}/{pat:.2f}/{('--' if rec is None else f'{rec:>4.0f}%'):>5}")
            else:
                cells.append(f"k={k:<2} {'.':>20}")
        lines.append(row + "  ".join(cells))
    lines.append("")

    # ---- GSM-DC dose-response (clean/distract/patched) ----
    lines.append("-" * 120)
    lines.append("GSM-DC (graph-structured math; k = irrelevant_nodes_count;")
    lines.append("        clean = problem with relevant-only statements; distract = full problem; patched = per-step hook)")
    lines.append("-" * 120)
    # Two slices: low-k (0..8) with full coverage, and high-k (9..17) with thinning samples
    for k_subset, label in [([0,1,2,3,4,5,6,7,8], "low-k (n=50 each)"),
                             ([9,10,11,12,13], "high-k (decreasing n)")]:
        lines.append(f"  -- {label} --")
        header = f"  {'Model':<20} | " + "  ".join(f"k={k:<2}{'cln':>5}/{'dis':>4}/{'pat':>4}/{'rec':>4}" for k in k_subset)
        lines.append(header)
        for model in MODELS:
            cells = []
            for k in k_subset:
                d = data[model]["gsmdc"].get(k)
                if d and "clean_acc" in d:
                    cln = d["clean_acc"]; dis = d["w_acc"]; pat = d["patched_acc"]
                    rec = d.get("recovery_pp_of_drop")
                    rec_s = "--" if rec is None else f"{rec:>3.0f}%"
                    cells.append(f"k={k:<2} {cln:.2f}/{dis:.2f}/{pat:.2f}/{rec_s:>4}")
                elif d:
                    cells.append(f"k={k:<2} {'-/' + str(round(d['w_acc'],2)) + '/-/-':>20}")
                else:
                    cells.append(f"k={k:<2} {'.':>20}")
            lines.append(f"  {model:<20} | " + "  ".join(cells))
        lines.append("")

    # Summary: dose-response in distract accuracy + defense recovery at heaviest k
    lines.append("=" * 120)
    lines.append("HEADLINE: defense recovery at heaviest k per task (per-model)")
    lines.append("=" * 120)
    heaviest = [("hotpotqa", 8), ("pubmedqa", 4), ("gsmdc", 8)]
    lines.append(f"  {'Model':<20}  " + "  ".join(f"{t}-k{k}  cln/dis/pat/rec" for t, k in heaviest))
    for model in MODELS:
        cells = []
        for t, k in heaviest:
            d = data[model][t].get(k)
            if d and "clean_acc" in d:
                cln = d["clean_acc"]; dis = d["w_acc"]; pat = d["patched_acc"]
                rec = d.get("recovery_pp_of_drop")
                rec_s = "--" if rec is None else f"{rec:>3.0f}%"
                cells.append(f"{cln:.2f}/{dis:.2f}/{pat:.2f}/{rec_s}")
            else:
                cells.append("missing")
        lines.append(f"  {model:<20}  " + "  ".join(f"{c:>22}" for c in cells))
    lines.append("")

    out_txt = "\n".join(lines)
    (RESULTS / "AGGREGATE_downstream.txt").write_text(out_txt)
    print(out_txt)
    print(f"\nWROTE: {RESULTS / 'AGGREGATE_downstream.txt'}")


if __name__ == "__main__":
    main()
