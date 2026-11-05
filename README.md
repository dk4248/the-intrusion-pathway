# The Intrusion Pathway — code

Code for *"The Intrusion Pathway: How Non-Evidential Context Degrades Language Model Performance"*.
All experiments use open-weight models through
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens); no model is trained.

**Website:** <https://dk4248.github.io/the-intrusion-pathway/> · **Paper (PDF):** <https://dk4248.github.io/the-intrusion-pathway/intrusion-pathway.pdf>

## Layout

Each component has a `code/` directory; its scripts write their outputs next to it
(`data/`, `results/`, `logs/`, `figs/`). These folders ship empty; `external/` is where the
third-party data listed under *Setup* goes.

| Directory | Paper |
|---|---|
| `circuit/` | §2–§4 (Figs 4–6, data for Fig 3), App. A (176-fact factual-recall set), App. B (Tables B.1–B.8, data for Figs B.1–B.7), App. C (Tables C.1–C.10, data for Figs C.1–C.6) — Llama-3.1-8B, Pythia-6.9B, GPT-2 XL |
| `lre_dataset/` | Canonical 250-fact LRE dataset with golden / related / counterfactual / short-irrelevant / long-irrelevant / random contexts, used by `intrusion_heads/`, `degradation/`, `transfer_and_sensitivity/`; also builds the WikiText-103 paragraph pool used by `circuit/` |
| `intrusion_heads/` | Head scores, head sets (retrieval R, intrusion I, clean intrusion set I\*), class-conditional attention knockout and its K sweep (§5 K10/K20 sets, App. E, App. F) |
| `degradation/` | LRE degradation metrics (supporting analysis, not a numbered item in the paper) |
| `downstream/` | §5 and App. D: HotpotQA, PubMedQA, GSM-DC — END-residual patching and intrusion-head knockout (Table 2, Tables D.4–D.6, data for Figs 7 and D.1–D.3) |
| `gsm_ic/` | App. D: GSM-IC single-token (Table D.2) and per-step chain of thought (Table D.3) |
| `transfer_and_sensitivity/` | App. E (instruction-tuned transfer, Tables E.1–E.3), App. F (K / α sensitivity, data for Figs F.1–F.2), held-out selection stability (§5, App. F) |

## Setup

```bash
pip install -r requirements.txt
```

* **Models** (Hugging Face): `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.1-8B-Instruct`,
  `mistralai/Mistral-7B-v0.1`, `Qwen/Qwen2.5-7B`, `EleutherAI/pythia-6.9b`, `gpt2-xl`; for Table D.2
  additionally `meta-llama/Llama-3.1-70B`, `Qwen/Qwen2.5-32B`, `Qwen/Qwen2.5-72B`, `EleutherAI/pythia-12b`.
  Llama checkpoints are gated: accept the licence and log in (`huggingface-cli login`) or set `HF_TOKEN`.
  Set `HF_HOME` to choose the cache location.
* **Hardware**: every 7–8B experiment fits on one 80 GB GPU (most on 40 GB). The 32B–72B GSM-IC runs use
  `--n-devices 2`. Launch scripts assume two GPUs (`cuda:0`, `cuda:1`); edit the device ids if needed.
* **External data** (not redistributed here):
  * LRE relations (Hernandez et al., 2024): `git clone https://github.com/evandez/relations` and copy its
    `data/` directory to `external/lre_dataset/` (sub-folders `bias/ commonsense/ factual/ linguistic/`).
  * GSM-IC (Shi et al., 2023): download `GSM-IC_2step.json` from
    https://github.com/google-research-datasets/GSM-IC to `gsm_ic/data/gsm-ic_2step.json`.
  * Downloaded automatically through `datasets`: WikiText-103 (`wikitext`, `wikitext-103-raw-v1`),
    HotpotQA (`hotpot_qa`, distractor), PubMedQA (`qiaojin/PubMedQA`), GSM-DC
    (`YMinglai/GSM-DC-Dataset-Sample`). `circuit/code/09_fetch_wiki_attrs.py` queries the public Wikipedia
    REST API.

Scripts in `circuit/` take the short model keys defined in `circuit/code/common.py`
(`Llama-3.1-8B`, `pythia-6.9b`, `gpt2-xl`, `Mistral-7B-v0.1`, `Qwen2.5-7B`); all other components take
Hugging Face ids (e.g. `meta-llama/Llama-3.1-8B`).

## Reproduction

Run the components in the order below. Commands are given from the component's `code/` directory unless
stated otherwise; `$M` loops over the models listed for that step.

### 1. `lre_dataset/` — canonical 250-fact dataset

```bash
cd lre_dataset/code
python stage_a_candidates.py            # candidate facts from 10 LRE relations
bash   run_stage_b.sh                   # per-model parametric screen (4 base models)
python stage_c_cherrypick.py            # 250 facts known by >= 3 of 4 models
python stage_d_deterministic.py         # golden / related / counterfactual contexts (seed 42)
bash   run_stage_e.sh                   # per-model screen of the 60-sentence irrelevant pool
python stage_f_merge.py                 # -> data/samples_5x250.json
python stage_g_long_irrelevant.py       # -> data/wiki_pool.json, data/samples_6x250.json (+ long irrelevant)
```

### 2. `circuit/` — §2–§4, Appendices A–C

Run from `circuit/` (not `circuit/code/`). `M ∈ {Llama-3.1-8B, pythia-6.9b, gpt2-xl}` unless noted.

```bash
cd circuit
# data: 176 facts from 9 relations, known by >= 3 of 5 models (App. A)
for M in Llama-3.1-8B Mistral-7B-v0.1 Qwen2.5-7B pythia-6.9b gpt2-xl; do python code/01_screen.py --model $M; done
python code/02_build_pairs.py --k 5 --classes golden,far_irr,far_irr_long,long_irr --allow-multitoken-gold --out pairs_v1.json
python code/09_fetch_wiki_attrs.py      # Wikipedia association banks (App. B.1)
python code/41_E5_3_build_factorial.py  # golden+irrelevant and filler cells

# optional correctness gates for the source-accounting / patching machinery
python code/G0_1_gates.py --model $M ; python code/G0_2_copy_audit.py

# App. B.1: MLP enrichment and write cancellation (Table B.1, Figs B.1-B.2)
bash code/run_C1.sh && python code/13_C1_analyze.py

# App. B.2-B.4: copying heads, identity substitution, cue removal, multi-source copying, relational readers
python code/15_E2_3_copying.py --model $M          # selects the copying heads used below
python code/19_E2_2_identity_swap.py --model $M    # Fig B.3
python code/16_E2_1_cue_removal.py --model $M      # Fig B.4
python code/26_E4_3_granularity.py --model $M      # Table B.2
python code/17_E2_4_diffuse.py --model $M          # Table B.3
python code/21_E3_1_transport.py --model $M        # Table B.4 (association excess)
python code/23_E3_3_unchanged.py --model $M        # Table B.4 (profile alignment)
python code/22_E3_2_cloud_removal.py --model $M    # Table B.5
python code/28_M1_slot_gate.py --model $M          # Fig B.5

# App. B.5: restoring middle-block writes (Tables B.6-B.8, Figs B.6-B.7)
python code/31_window_restore.py --model $M --limit 200
python code/34_regeneration.py --model $M --limit 60

# §4 / App. C: intrusion heads
python code/37_E5_1_screen.py --model $M           # Eq. 3 screen; writes the 86/90 discovery/held-out split
python code/61_sep_scores.py --model $M            # retrieval score, R, I, I*
python code/43_E5_5_stability.py --model $M        # Table C.1 (inclusion, Jaccard)
python code/38_E5_1_validate.py --model $M         # Table C.4
python code/44_E5_5_dose.py --model $M             # Fig C.2
python code/49_E7_6_specificity.py --model $M      # Table C.2
python code/75_exp1_two_sided.py --model $M        # Table C.1 (attention shifts), Fig C.1, Table C.3, Fig C.4, Table C.6
python code/76_exp2_four_writes.py --model $M      # Table C.5, Figs C.3 and C.5, Table C.7
python code/77_exp3_route.py --model $M            # Fig C.6, Table C.8
python code/78_xp1_analyze.py ; python code/79_xp2_analyze.py ; python code/80_xp3_analyze.py
python code/62_sep_causal.py --model $M            # Table C.10
python code/67_stop_here.py --model $M --split all --limit 176 --stages mid_ko,mid_restore --tag _all176  # §3 masking vs restoring (data for Fig 3)
python code/63_niu_entrainment.py --model Llama-3.1-8B --epochs 500 --eval_every 10 --seed 0 --pad_bug   # Niu et al. (2025) entrainment heads
python code/64_IE_overlap.py                       # Table C.9

# figures
python code/86_fig_glyph_split.py Llama-3.1-8B     # Fig 4
python code/88_fig_mask_compact.py gpt2-xl 40 22   # Fig 5
python code/89_fig_L_gpt2.py                       # Fig 6
```

`10_E1_enrichment.py` and `12_E1_causal.py` accept `--shard k/n` to split a run; merge with
`python code/94_merge_shards.py <E1_enrich|E1_causal> <model> <n>`.

### 3. `intrusion_heads/` — head sets and LRE-level knockout

`M ∈ {meta-llama/Llama-3.1-8B, mistralai/Mistral-7B-v0.1, Qwen/Qwen2.5-7B, EleutherAI/pythia-6.9b}`.

```bash
cd intrusion_heads/code
python run_stage4_head_scores.py --model $M       # per-head attention / direct-logit scores, 6 conditions
python compute_weight_signatures.py --model $M    # OV-copy scores (supporting analysis)
python compute_ov_direction.py --model $M         # OV write direction (supporting analysis)
python aggregate_head_scores.py                   # -> results/HEAD_SETS.json (R, I, I*)
bash   launch_K_sweep.sh                          # knockout: K10 clean (I*), K20 full (I), K sweep
python aggregate_K_sweep.py                       # -> results/K_sweep_pareto.json
```

### 4. `degradation/` — supporting analysis

```bash
cd degradation/code
bash run_phase1.sh                                         # golden / related / counterfactual / short irrelevant / random
python run_phase1_long_irrelevant.py --model $M            # long irrelevant, for each of the 4 base models
python aggregate_degradation.py                            # -> results/AGGREGATE_degradation.{txt,json}
```

### 5. `downstream/` — §5, Tables 2, D.4–D.6 (and the data for Figs 7, D.1–D.3)

```bash
cd downstream/code
python prep_downstream.py        # HotpotQA (k=0..8), PubMedQA (k=0..4), GSM-DC (by injected-node count)
python rebuild_gsmdc.py          # GSM-DC clean condition (relevant statements only)
python run_downstream.py --model $M                     # clean / +irrelevant / +END-residual patch (L = 0.95 N)
bash   launch_hko_sweep.sh                              # intrusion-head knockout, K10 clean and K20 full
python run_downstream_head_ko.py --model $M --tasks <task files> --top-k 10 --set-name clean_intrusion_set   # further cells
python aggregate_downstream.py
```

Head knockout reads `intrusion_heads/results/HEAD_SETS.json` (step 3).

### 6. `gsm_ic/` — Tables D.2 and D.3

```bash
cd gsm_ic/code
python build_gsm_ic_500.py       # 500 in-topic problems from GSM-IC 2-step (seed 42)
bash run_all.sh                  # single-token, 7B models (layer sweep)
bash run_big.sh                  # single-token, 12B-72B models
bash run_cot_perstep.sh          # per-step chain of thought, n=100
python table_gsm_ic.py           # Tables D.2 and D.3
```

### 7. `transfer_and_sensitivity/` — Appendices E and F

Instruction-tuned transfer (App. E), with `MI=meta-llama/Llama-3.1-8B-Instruct`:

```bash
# in intrusion_heads/code
python run_stage4_head_scores.py --model $MI ; python compute_weight_signatures.py --model $MI ; python compute_ov_direction.py --model $MI
# in transfer_and_sensitivity/code
python instruct_head_sets.py      # Table E.1; adds the instruct sets to HEAD_SETS.json
python instruct_ov_analysis.py    # OV-copy and write-direction signatures (supporting analysis)
# in intrusion_heads/code
python run_attention_knockout.py --model $MI --top-k 10 --set-name clean_intrusion_set \
    --output-path ../../transfer_and_sensitivity/results/e5_knockout_instruct_K10_clean.json
python run_attention_knockout.py --model $MI --top-k 20 --set-name intrusion_set \
    --output-path ../../transfer_and_sensitivity/results/e5_knockout_instruct_K20_full.json
# in transfer_and_sensitivity/code
python instruct_knockout_table.py # Table E.2
# downstream + GSM-IC for base and instruct, then Table E.3
#   downstream/code: run_downstream.py and run_downstream_head_ko.py (K10 clean) --model $MI
#     on gsmdc_irr4 gsmdc_irr7 hotpotqa_k4 hotpotqa_k8 pubmedqa_k2 pubmedqa_k4
#   gsm_ic/code:     run_gsm_ic_defense.py --model $MI
python instruct_downstream_table.py
```

Sensitivity to K and α, and held-out stability (App. F; §5):

```bash
cd transfer_and_sensitivity/code
python heldout_stability.py                 # held-out Spearman / Jaccard / exclusion-radius statistics
python inject_heldout_sets.py Llama-3.1-8B  # fold-A sets for the disjoint-fold knockout
for F in relfoldA sampfoldA; do
  (cd ../../intrusion_heads/code && python run_attention_knockout.py --model meta-llama/Llama-3.1-8B --top-k 10 \
     --set-name Istar_$F --output-path ../../transfer_and_sensitivity/results/e1_ko_llama_$F.json)
done
python heldout_knockout_table.py
for M in Llama-3.1-8B Mistral-7B-v0.1 Qwen2.5-7B pythia-6.9b; do python alpha_sets.py $M; done   # I*_alpha sets
bash run_alpha_sweep.sh                     # alpha in {0.5, 1, 2}, K = 10
bash run_cross_sweep.sh                     # K x alpha grid (resumable chunks: run_cross_chunk.sh START COUNT)
bash run_fill_sweep.sh                      # clean K=20 and full K=10 cells
python table_K_alpha_sweep.py               # data for Fig F.1
python table_K_alpha_grid.py                # data for Fig F.2
```

## Paper item → script

Script docstrings and printed headings may still use earlier draft numbers; this table follows the paper.

| Item | Script(s) |
|---|---|
| Data for Fig 3 | `circuit/code/67_stop_here.py` |
| Fig 4 | `circuit/code/86_fig_glyph_split.py` |
| Fig 5 | `circuit/code/88_fig_mask_compact.py` |
| Fig 6, Table C.10 | `circuit/code/61_sep_scores.py`, `62_sep_causal.py`, `89_fig_L_gpt2.py` |
| Table B.1, Figs B.1, B.2 | `circuit/code/10_E1_enrichment.py`, `12_E1_causal.py`, `13_C1_analyze.py` |
| Fig B.3 / Fig B.4 / Table B.2 | `circuit/code/19_E2_2_identity_swap.py` / `16_E2_1_cue_removal.py` / `26_E4_3_granularity.py` |
| Table B.3 / Table B.4 / Table B.5 / Fig B.5 | `circuit/code/17_E2_4_diffuse.py` / `21_E3_1_transport.py`, `23_E3_3_unchanged.py` / `22_E3_2_cloud_removal.py` / `28_M1_slot_gate.py` |
| Tables B.6–B.8, Fig B.6 / Fig B.7 | `circuit/code/31_window_restore.py` / `34_regeneration.py` |
| Eq. 3, Table C.1 | `circuit/code/37_E5_1_screen.py`, `43_E5_5_stability.py`, `75_exp1_two_sided.py` |
| Fig C.1, Tables C.3, C.6, Fig C.4 / Table C.2 / Table C.4 / Fig C.2 | `75_exp1_two_sided.py` + `78_xp1_analyze.py` / `49_E7_6_specificity.py` / `38_E5_1_validate.py` / `44_E5_5_dose.py` |
| Table C.5, Figs C.3, C.5, Table C.7 / Fig C.6, Table C.8 | `76_exp2_four_writes.py` + `79_xp2_analyze.py` / `77_exp3_route.py` + `80_xp3_analyze.py` |
| Table C.9 | `circuit/code/63_niu_entrainment.py`, `64_IE_overlap.py` |
| Table 2, Tables D.4–D.6, data for Figs 7, D.1–D.3 | `downstream/code/run_downstream.py`, `run_downstream_head_ko.py`, `aggregate_downstream.py`; GSM-IC column: `gsm_ic/` |
| Tables D.2, D.3 | `gsm_ic/code/*` |
| Tables E.1 / E.2 / E.3 | `transfer_and_sensitivity/code/instruct_head_sets.py` / `instruct_knockout_table.py` / `instruct_downstream_table.py` |
| Data for Fig F.1 / Fig F.2 | `transfer_and_sensitivity/code/table_K_alpha_sweep.py` / `table_K_alpha_grid.py` |
| Held-out stability (§5, App. F) | `transfer_and_sensitivity/code/heldout_stability.py`, `heldout_knockout_table.py` |
| LRE degradation metrics; OV-copy and write-direction signatures (supporting, not numbered in the paper) | `degradation/code/*`; `intrusion_heads/code/compute_weight_signatures.py`, `compute_ov_direction.py`, `transfer_and_sensitivity/code/instruct_ov_analysis.py` |
