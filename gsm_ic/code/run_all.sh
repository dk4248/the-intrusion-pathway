#!/bin/bash
# GSM-IC single-token residual patching (Table D.2) on the four 7B base models.
# All 4 in parallel (2 per GPU).

cd "$(dirname "$0")"
mkdir -p ../logs

N=500

nohup python run_gsm_ic_defense.py --model EleutherAI/pythia-6.9b --device cuda:0 --n-samples $N \
  > ../logs/pythia-6.9b.log 2>&1 &
P1=$!
nohup python run_gsm_ic_defense.py --model Qwen/Qwen2.5-7B --device cuda:0 --n-samples $N \
  > ../logs/Qwen2.5-7B.log 2>&1 &
P2=$!
nohup python run_gsm_ic_defense.py --model mistralai/Mistral-7B-v0.1 --device cuda:1 --n-samples $N \
  > ../logs/Mistral-7B-v0.1.log 2>&1 &
P3=$!
nohup python run_gsm_ic_defense.py --model meta-llama/Llama-3.1-8B --device cuda:1 --n-samples $N \
  > ../logs/Llama-3.1-8B.log 2>&1 &
P4=$!

echo "[$(date +%H:%M:%S)] launched 4 GSM-IC defense PIDs $P1 $P2 $P3 $P4" >&2
wait $P1 $P2 $P3 $P4
echo "[$(date +%H:%M:%S)] ALL 4 GSM-IC RUNS DONE" >&2
