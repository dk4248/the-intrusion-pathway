#!/bin/bash
# GSM-IC per-step chain-of-thought residual patching (Table D.3), n=100 problems,
# Mistral-7B-v0.1 and Llama-3.1-8B.

cd "$(dirname "$0")"
mkdir -p ../logs_cot_perstep

N=100

# 7B baselines: Mistral-7B + Llama-8B in parallel on GPU0+1
echo "[$(date +%H:%M:%S)] Phase 1: Mistral-7B + Llama-3.1-8B (n=$N)" >&2
nohup python3 run_gsm_ic_cot_perstep.py --model mistralai/Mistral-7B-v0.1 --device cuda:0 --n-samples $N \
  > ../logs_cot_perstep/Mistral-7B-v0.1.log 2>&1 &
P1=$!
nohup python3 run_gsm_ic_cot_perstep.py --model meta-llama/Llama-3.1-8B   --device cuda:1 --n-samples $N \
  > ../logs_cot_perstep/Llama-3.1-8B.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[$(date +%H:%M:%S)] ALL DONE (n=$N)" >&2
