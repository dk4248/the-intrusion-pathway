#!/bin/bash
set -e
# Downstream intrusion-head knockout (Section 5, Tables 3 / D.4 / D.5): K10 clean (I*) and K20 full sets.
# Other task cells (e.g. hotpotqa_k8, pubmedqa_k4) use the same runner with a different --tasks list.
cd "$(dirname "$0")"
LOGDIR=../logs
mkdir -p $LOGDIR

MODELS_HF=("meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.1" "Qwen/Qwen2.5-7B" "EleutherAI/pythia-6.9b")
MODELS_SHORT=("Llama-3.1-8B" "Mistral-7B-v0.1" "Qwen2.5-7B" "pythia-6.9b")
TASKS="gsmdc_irr4 gsmdc_irr8 hotpotqa_k4 pubmedqa_k2"
# Two K settings: 10 clean (surgical) and 20 full (aggressive)
SETTINGS=("10:clean_intrusion_set" "20:intrusion_set")

COUNT=0
for i in 0 1 2 3; do
  HF=${MODELS_HF[$i]}
  M=${MODELS_SHORT[$i]}
  for S in "${SETTINGS[@]}"; do
    K=${S%%:*}
    SET=${S##*:}
    DEV="cuda:$((COUNT % 2))"
    LABEL="K${K}_clean"; [ "$SET" = "intrusion_set" ] && LABEL="K${K}_full"
    LOG="$LOGDIR/hko_${M}_${LABEL}.log"
    echo "[launch] $M $LABEL on $DEV (tasks=$TASKS)"
    nohup python run_downstream_head_ko.py --model "$HF" --device "$DEV" \
        --tasks $TASKS --top-k "$K" --set-name "$SET" > "$LOG" 2>&1 &
    COUNT=$((COUNT + 1))
    if [ $((COUNT % 4)) -eq 0 ]; then
      wait
    fi
  done
done
wait
echo "ALL DONE"
ls ../results/*__hko_*.json | wc -l
