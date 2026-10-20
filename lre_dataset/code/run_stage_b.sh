#!/usr/bin/env bash
# Run parametric screen on all 4 models in parallel: 2 per GPU.
set -uo pipefail
cd "$(dirname "$0")"

LOG=../logs/stage_b
mkdir -p "$LOG"

launch() {
    local MODEL="$1" GPU="$2"
    local SHORT=$(echo "$MODEL" | rev | cut -d/ -f1 | rev)
    CUDA_VISIBLE_DEVICES=$GPU python stage_b_parametric.py --model "$MODEL" --device cuda:0 \
        > "$LOG/b_${SHORT}.log" 2>&1 &
    echo $!
}

echo "════════════════════════════════════════"
echo " Stage B — parametric screen (4 models)"
echo "════════════════════════════════════════"

P1=$(launch "meta-llama/Llama-3.1-8B"    0); echo "[PID $P1] Llama    GPU0"
P2=$(launch "Qwen/Qwen2.5-7B"            0); echo "[PID $P2] Qwen     GPU0"
P3=$(launch "EleutherAI/pythia-6.9b"     1); echo "[PID $P3] Pythia   GPU1"
P4=$(launch "mistralai/Mistral-7B-v0.1"  1); echo "[PID $P4] Mistral  GPU1"

for P in $P1 $P2 $P3 $P4; do wait $P || true; done

echo ""
echo "All 4 models done. Output files:"
ls -la ../data/build/parametric_*.json 2>/dev/null
