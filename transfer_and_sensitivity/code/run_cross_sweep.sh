#!/bin/bash
# Tables F.3-F.6: K x alpha grid, alpha in {0.5,1.0,2.0} x K in {5,8,12,15,20}
# (the alpha=0 row comes from the K sweep in intrusion_heads/).
R="$(cd "$(dirname "$0")/.." && pwd)"          # transfer_and_sensitivity/
cd "$R/../intrusion_heads/code"               # knockout runner lives here
mkdir -p "$R/results" "$R/logs"
declare -A HFID=( [Llama-3.1-8B]=meta-llama/Llama-3.1-8B [Mistral-7B-v0.1]=mistralai/Mistral-7B-v0.1 [Qwen2.5-7B]=Qwen/Qwen2.5-7B [pythia-6.9b]=EleutherAI/pythia-6.9b )

run_cfg() {  # model alpha K gpu
  local out=$R/results/e1_cross_$1_a$2_K$3.json
  # resume: skip if a VALID (non-truncated) result already exists
  if python -c "import json,sys;d=json.load(open('$out'));sys.exit(0 if len(d.get('records',[]))>0 else 1)" 2>/dev/null; then
    echo "skip $1 a$2 K$3 (already done)"; return 0
  fi
  python run_attention_knockout.py --model ${HFID[$1]} --device cuda:$4 --top-k $3 \
    --set-name Istar_alpha$2 --output-path $out \
    > $R/logs/e1_cross_$1_a$2_K$3.log 2>&1
}

CONFIGS=()
for m in Llama-3.1-8B Mistral-7B-v0.1 Qwen2.5-7B pythia-6.9b; do
  for a in 0.5 1.0 2.0; do
    for K in 5 8 12 15 20; do CONFIGS+=("$m $a $K"); done
  done
done

n=${#CONFIGS[@]}
echo "total configs: $n"
for ((i=0; i<n; i+=2)); do
  read m1 a1 k1 <<< "${CONFIGS[$i]}"; run_cfg "$m1" "$a1" "$k1" 0 & pid0=$!
  if (( i+1 < n )); then read m2 a2 k2 <<< "${CONFIGS[$((i+1))]}"; run_cfg "$m2" "$a2" "$k2" 1 & pid1=$!; fi
  wait $pid0 2>/dev/null; [ -n "$pid1" ] && wait $pid1 2>/dev/null
  echo "wave $((i/2+1))/$(( (n+1)/2 )) done: ${CONFIGS[$i]} | ${CONFIGS[$((i+1))]:-}"
  unset pid1
done
echo "CROSS_SWEEP_COMPLETE"
