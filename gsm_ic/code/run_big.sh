#!/bin/bash
# GSM-IC single-token residual patching (Table D.2) on the larger models.
# Round 1 (parallel): Pythia-12B (cuda:0) + Qwen-32B (cuda:1)
# Round 2 (MP): Llama-3.1-70B
# Round 3 (MP): Qwen2.5-72B

cd "$(dirname "$0")"
mkdir -p ../logs

N=500

# ─── ROUND 1 ───
echo "[$(date +%H:%M:%S)] Round 1: Pythia-12B (cuda:0) + Qwen-32B (cuda:1)" >&2
nohup python run_gsm_ic_defense.py --model EleutherAI/pythia-12b --device cuda:0 --n-samples $N \
  > ../logs/pythia-12b.log 2>&1 &
P1=$!
nohup python run_gsm_ic_defense.py --model Qwen/Qwen2.5-32B --device cuda:1 --n-samples $N \
  > ../logs/Qwen2.5-32B.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[$(date +%H:%M:%S)] Round 1 done" >&2

# ─── ROUND 2 ───
echo "[$(date +%H:%M:%S)] Round 2: Llama-3.1-70B (MP)" >&2
python run_gsm_ic_defense.py --model meta-llama/Llama-3.1-70B --n-devices 2 --n-samples $N \
  > ../logs/Llama-3.1-70B.log 2>&1
echo "[$(date +%H:%M:%S)] Llama-70B done" >&2

# ─── ROUND 3 ───
echo "[$(date +%H:%M:%S)] Round 3: Qwen2.5-72B (MP)" >&2
python run_gsm_ic_defense.py --model Qwen/Qwen2.5-72B --n-devices 2 --n-samples $N \
  > ../logs/Qwen2.5-72B.log 2>&1
echo "[$(date +%H:%M:%S)] Qwen-72B done" >&2

echo "[$(date +%H:%M:%S)] ALL BIG GSM-IC RUNS DONE" >&2
