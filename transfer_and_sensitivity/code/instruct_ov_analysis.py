#!/usr/bin/env python
"""
Instruction-tuned transfer, part 2 — OV-copy (Table E.2) + write-direction (Table E.3) signatures,
computed IDENTICALLY for base and instruct, to show whether the retrieval-vs-intrusion
weight-level dissociation survives RLHF.

OV-copy: retrieval-only heads literally copy the gold token; intrusion-only heads
  have OV-copy indistinguishable from random. Report per-set mean + Mann-Whitney.
Write-direction: s_gold (gold-writer under golden) vs s_pfx (prefix-writer under
  short_irr). Retrieval-only lie on the gold axis, intrusion-only on the prefix axis.
"""
import json, numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
REPO = Path(__file__).resolve().parents[2]  # repository root

RES = REPO / "intrusion_heads/results"
OUT = REPO / "transfer_and_sensitivity/results"
MODELS = {"base": "Llama-3.1-8B", "instruct": "Llama-3.1-8B-Instruct"}
hs = json.load(open(RES / "HEAD_SETS.json"))


def sets_for(m, L, H):
    R = {(l, h) for l, h, _ in hs[m]["retrieval_set"]}
    I = {(l, h) for l, h, _ in hs[m]["intrusion_set"]}
    RandI = R & I
    Ronly, Ionly = R - I, I - R
    late = {(l, h) for l in range(L // 2, L) for h in range(H)}
    rand = sorted(late - R - I)
    return dict(R=R, I=I, dual=sorted(RandI), Ronly=sorted(Ronly),
                Ionly=sorted(Ionly), rand=rand)


def vals(mat, cells):
    return np.array([mat[l, h] for l, h in cells], float)


report = {}
for tag, m in MODELS.items():
    ws = np.load(RES / f"{m}__weight_sigs.npz")
    ovc = ws["ov_copy"]  # [L,H]
    L, H = ovc.shape
    sc = np.load(RES / f"{m}__ovdir_scalar.npz")
    conds = list(np.load(RES / f"{m}__ovdir_mean.npz", allow_pickle=True)["conditions"])
    gi = conds.index("golden") if "golden" in conds else 1
    si = conds.index("short_irr") if "short_irr" in conds else 4
    s_gold = sc["dla_gold"][:, gi, :, :].mean(0)          # [L,H] gold-writer under golden
    s_pfx = sc["prefix_writer"][:, si, :, :].mean(0)      # [L,H] prefix-writer under short_irr
    S = sets_for(m, L, H)

    ovc_Ronly, ovc_Ionly = vals(ovc, S["Ronly"]), vals(ovc, S["Ionly"])
    ovc_rand = vals(ovc, S["rand"])
    mw = mannwhitneyu(ovc_Ronly, ovc_Ionly, alternative="greater") if len(ovc_Ronly) and len(ovc_Ionly) else (np.nan, np.nan)

    report[tag] = dict(
        model=m, n_dual=len(S["dual"]), n_Ronly=len(S["Ronly"]), n_Ionly=len(S["Ionly"]),
        ov_copy=dict(
            retrieval_only_mean=round(float(ovc_Ronly.mean()), 4),
            intrusion_only_mean=round(float(ovc_Ionly.mean()), 4),
            dual_mean=round(float(vals(ovc, S["dual"]).mean()), 4) if S["dual"] else None,
            random_mean=round(float(ovc_rand.mean()), 4),
            ratio_ret_over_intr=round(float(ovc_Ronly.mean() / (ovc_Ionly.mean() + 1e-9)), 2),
            mw_p_ret_gt_intr=float(mw[1]),
        ),
        write_direction=dict(
            retrieval_only=dict(s_gold=round(float(vals(s_gold, S["Ronly"]).mean()), 3),
                                s_pfx=round(float(vals(s_pfx, S["Ronly"]).mean()), 3)),
            intrusion_only=dict(s_gold=round(float(vals(s_gold, S["Ionly"]).mean()), 3),
                                s_pfx=round(float(vals(s_pfx, S["Ionly"]).mean()), 3)),
        ),
    )

json.dump(report, open(OUT / "e5_ov_analysis.json", "w"), indent=2)

md = ["# OV-copy (Table E.2) + write-direction (Table E.3), base vs instruct\n",
      "Does the retrieval-vs-intrusion weight-level dissociation survive RLHF?\n",
      "## OV-copy score per head set (retrieval-only literally copies; intrusion-only ≈ random)",
      "| Checkpoint | retrieval-only | intrusion-only | ratio | random | MW p (ret>intr) |",
      "|---|---|---|---|---|---|"]
for tag in ("base", "instruct"):
    o = report[tag]["ov_copy"]
    md.append(f"| {tag} | {o['retrieval_only_mean']} | {o['intrusion_only_mean']} | "
              f"{o['ratio_ret_over_intr']}× | {o['random_mean']} | {o['mw_p_ret_gt_intr']:.1e} |")
md += ["\n## Write-direction (gold-writer s_gold under golden vs prefix-writer s_pfx under short_irr)",
       "| Checkpoint | set | s_gold | s_pfx |", "|---|---|---|---|"]
for tag in ("base", "instruct"):
    w = report[tag]["write_direction"]
    md.append(f"| {tag} | retrieval-only | {w['retrieval_only']['s_gold']} | {w['retrieval_only']['s_pfx']} |")
    md.append(f"| {tag} | intrusion-only | {w['intrusion_only']['s_gold']} | {w['intrusion_only']['s_pfx']} |")
md.append("\n**Reading:** if the ratio stays 3–5× and MW p stays small on instruct, and retrieval-only "
          "keeps high s_gold/low s_pfx while intrusion-only keeps low s_gold/high s_pfx, the weight-level "
          "dissociation is preserved under instruction tuning.")
open(OUT / "e5_ov_analysis.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
