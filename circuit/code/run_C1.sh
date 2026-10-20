#!/bin/bash
# Appendix B.1 (MLP enrichment): measurement (E1.1/E1.3/E1.4) followed by the causal
# interventions (E1.2/E1.3-causal/E1.5), for all three models.
# Requires data/pairs_v1.json (02_build_pairs.py) and data/wiki_attrs.json (09_fetch_wiki_attrs.py).
cd "$(dirname "$0")/.."
python -c "import json;c=json.load(open('data/wiki_attrs.json'));print('resolved',sum(1 for v in c.values() if v.get('extract')),'/',len(c))"

echo "=== E1.1/E1.3 measurement ==="
for spec in "Llama-3.1-8B cuda:0" "pythia-6.9b cuda:1" "gpt2-xl cuda:1"; do
  set -- $spec
  ( sleep $((RANDOM%5)); python code/10_E1_enrichment.py --model $1 --device $2 2>&1 | grep -E "rows in|Error|Traceback" ) &
done
wait
echo "=== E1.2/E1.5 causal ==="
for spec in "Llama-3.1-8B cuda:0" "pythia-6.9b cuda:1" "gpt2-xl cuda:1"; do
  set -- $spec
  ( sleep $((RANDOM%5)); python code/12_E1_causal.py --model $1 --device $2 2>&1 | grep -E "rows in|Error|Traceback" ) &
done
wait
echo "C1_PIPELINE_DONE"
