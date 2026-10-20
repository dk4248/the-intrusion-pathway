#!/bin/bash
# LRE-level class-conditional attention knockout for every (K, set) cell that aggregate_K_sweep.py
# reads: the main runs (K=10 clean set I*, K=20 full set I; Section 5, Tables E.4 and F.1) plus the K
# sweep K in {5, 8, 12, 15} for both sets (Table F.1). The two remaining cells (K=10 full, K=20 clean)
# are run by transfer_and_sensitivity/code/run_fill_sweep.sh.
set -e
cd "$(dirname "$0")"
OUTDIR=../results/knockout_K_sweep
LOGDIR=../logs
mkdir -p $OUTDIR $LOGDIR

MODELS_HF=("meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.1" "Qwen/Qwen2.5-7B" "EleutherAI/pythia-6.9b")
MODELS_SHORT=("Llama-3.1-8B" "Mistral-7B-v0.1" "Qwen2.5-7B" "pythia-6.9b")
K_VALUES=(5 8 10 12 15 20)
SETS=("clean_intrusion_set" "intrusion_set")

COUNT=0
for i in 0 1 2 3; do
  HF=${MODELS_HF[$i]}
  M=${MODELS_SHORT[$i]}
  for K in "${K_VALUES[@]}"; do
    for SET in "${SETS[@]}"; do
      LABEL="clean"; [ "$SET" = "intrusion_set" ] && LABEL="full"
      # K=10 is run for the clean set only and K=20 for the full set only
      if { [ "$K" = 10 ] && [ "$LABEL" = full ]; } || { [ "$K" = 20 ] && [ "$LABEL" = clean ]; }; then continue; fi
      OUT="$OUTDIR/${M}__K${K}_${LABEL}.json"
      if [ -f "$OUT" ]; then echo "[skip] $OUT"; continue; fi
      DEV="cuda:$((COUNT % 2))"
      echo "[run] $M K=$K $SET on $DEV -> $OUT"
      python run_attention_knockout.py --model "$HF" --device "$DEV" \
        --top-k "$K" --set-name "$SET" --output-path "$OUT" \
        > "$LOGDIR/sw_${M}_K${K}_${LABEL}.log" 2>&1 &
      COUNT=$((COUNT + 1))
      if [ $((COUNT % 4)) -eq 0 ]; then
        wait
      fi
    done
  done
done
wait
echo "ALL DONE"
ls "$OUTDIR" | wc -l
