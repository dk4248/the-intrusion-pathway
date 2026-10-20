#!/bin/bash
# Table D.1 (short-irrelevant / random columns): phase-1 metrics for the four 7B base models on the
# 250-sample LRE dataset. Two rounds of 2-at-a-time (single GPU each).

cd "$(dirname "$0")"
mkdir -p ../logs

DATASET=samples_5x250

# Round A: Pythia-6.9B (cuda:0) + Mistral-7B (cuda:1)
echo "[$(date +%H:%M:%S)] Round A: Pythia-6.9B + Mistral-7B" >&2
python run_phase1.py \
  --model EleutherAI/pythia-6.9b --dataset $DATASET \
  --device cuda:0 --tag full \
  > ../logs/pythia-6.9b.log 2>&1 &
PA=$!
python run_phase1.py \
  --model mistralai/Mistral-7B-v0.1 --dataset $DATASET \
  --device cuda:1 --tag full \
  > ../logs/Mistral-7B-v0.1.log 2>&1 &
PB=$!
wait $PA $PB
echo "[$(date +%H:%M:%S)] Round A done" >&2

# Round B: Qwen2.5-7B (cuda:0) + Llama-3.1-8B (cuda:1)
echo "[$(date +%H:%M:%S)] Round B: Qwen2.5-7B + Llama-3.1-8B" >&2
python run_phase1.py \
  --model Qwen/Qwen2.5-7B --dataset $DATASET \
  --device cuda:0 --tag full \
  > ../logs/Qwen2.5-7B.log 2>&1 &
PC=$!
python run_phase1.py \
  --model meta-llama/Llama-3.1-8B --dataset $DATASET \
  --device cuda:1 --tag full \
  > ../logs/Llama-3.1-8B.log 2>&1 &
PD=$!
wait $PC $PD
echo "[$(date +%H:%M:%S)] Round B done" >&2

echo "[$(date +%H:%M:%S)] ALL DONE" >&2
