#!/bin/bash
# Table F.1 cells not covered by the K sweep: clean K=20 (clean_intrusion_set[:20]) and full K=10 (intrusion_set[:10]).
R="$(cd "$(dirname "$0")/.." && pwd)"          # transfer_and_sensitivity/
cd "$R/../intrusion_heads/code"               # knockout runner lives here
mkdir -p "$R/results" "$R/logs"
declare -A HFID=( [Llama-3.1-8B]=meta-llama/Llama-3.1-8B [Mistral-7B-v0.1]=mistralai/Mistral-7B-v0.1 [Qwen2.5-7B]=Qwen/Qwen2.5-7B [pythia-6.9b]=EleutherAI/pythia-6.9b )

run_cfg() {  # model setname topk tag gpu
  python run_attention_knockout.py --model ${HFID[$1]} --device cuda:$5 --top-k $3 \
    --set-name $2 --output-path $R/results/e1_fill_$1_$4.json > $R/logs/e1_fill_$1_$4.log 2>&1
}

CONFIGS=()
for m in Llama-3.1-8B Mistral-7B-v0.1 Qwen2.5-7B pythia-6.9b; do
  CONFIGS+=("$m clean_intrusion_set 20 clean20")
  CONFIGS+=("$m intrusion_set 10 full10")
done

n=${#CONFIGS[@]}
for ((i=0; i<n; i+=2)); do
  read m1 s1 k1 t1 <<< "${CONFIGS[$i]}"; run_cfg "$m1" "$s1" "$k1" "$t1" 0 & pid0=$!
  if (( i+1 < n )); then read m2 s2 k2 t2 <<< "${CONFIGS[$((i+1))]}"; run_cfg "$m2" "$s2" "$k2" "$t2" 1 & pid1=$!; fi
  wait $pid0 2>/dev/null; [ -n "$pid1" ] && wait $pid1 2>/dev/null
  echo "wave $((i/2+1)) done"; unset pid1
done
echo "FILL_SWEEP_COMPLETE"
