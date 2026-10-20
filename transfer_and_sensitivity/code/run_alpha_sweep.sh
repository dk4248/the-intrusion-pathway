#!/bin/bash
# Table F.2: knockout recovery for the alpha-penalised sets (alpha in {0.5,1.0,2.0}, K=10), 4 models.
# Run alpha_sets.py <model> first to add the Istar_alpha* sets to HEAD_SETS.json. 2 GPUs, 2 concurrent.
R="$(cd "$(dirname "$0")/.." && pwd)"          # transfer_and_sensitivity/
cd "$R/../intrusion_heads/code"               # knockout runner lives here
mkdir -p "$R/results" "$R/logs"
declare -A HFID=( [Llama-3.1-8B]=meta-llama/Llama-3.1-8B [Mistral-7B-v0.1]=mistralai/Mistral-7B-v0.1 [Qwen2.5-7B]=Qwen/Qwen2.5-7B [pythia-6.9b]=EleutherAI/pythia-6.9b )

run_cfg() {  # model alpha gpu
  python run_attention_knockout.py --model ${HFID[$1]} --device cuda:$3 --top-k 10 \
    --set-name Istar_alpha$2 --output-path $R/results/e1_alpha_$1_a$2.json \
    > $R/logs/e1_alpha_$1_a$2.log 2>&1
}

# build the 12-config list
CONFIGS=()
for m in Llama-3.1-8B Mistral-7B-v0.1 Qwen2.5-7B pythia-6.9b; do
  for a in 0.5 1.0 2.0; do CONFIGS+=("$m $a"); done
done

# process 2 at a time (cuda:0 and cuda:1)
n=${#CONFIGS[@]}
for ((i=0; i<n; i+=2)); do
  read m1 a1 <<< "${CONFIGS[$i]}"
  run_cfg "$m1" "$a1" 0 &
  pid0=$!
  if (( i+1 < n )); then
    read m2 a2 <<< "${CONFIGS[$((i+1))]}"
    run_cfg "$m2" "$a2" 1 &
    pid1=$!
  fi
  wait $pid0 2>/dev/null
  [ -n "$pid1" ] && wait $pid1 2>/dev/null
  echo "wave $((i/2+1)) done: ${CONFIGS[$i]} | ${CONFIGS[$((i+1))]:-}"
  unset pid1
done
echo "ALPHA_SWEEP_COMPLETE"
