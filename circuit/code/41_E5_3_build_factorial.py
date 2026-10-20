"""E5.3 — build the evidence x distraction factorial. The evidence+distraction cell does not
exist in pairs_v1 and has to be constructed.

Four cells, with the golden sentence ALWAYS in the same place relative to the query (immediately
before it) so evidence placement is matched across the cells that contain it:

  bare        [query]
  pad         [filler] [query]                   <- length control with no evidence, no distraction
  ev          [golden] [query]
  irr         [far_irr] [query]
  ev_irr      [far_irr] [golden] [query]          <- new

Adding text changes length, which is a confound for `ev` vs `ev_irr`. So a length-matched evidence
cell is also built, with the irrelevant sentence replaced by filler of the same token length drawn
from the same relation templates but carrying no entity (so it cannot itself distract):

  ev_pad      [filler] [golden] [query]

The interaction of interest is whether the SAME frozen head edit has a different effect with and
without evidence present: CE(ev_irr) - CE(ev), with CE(irr) - CE(bare) as the no-evidence baseline.
Filler length is matched per model tokenizer at measurement time, not here, so this file stores the
pieces and the scripts that read this file assemble them.
"""
import json, sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
S = json.load((ROOT / "data" / "pairs_v1.json").open())
rg = random.Random(4242)

# neutral filler: grammatical, entity-free, no relation content
FILLER_POOL = [
    # entity-free generic prose, spanning ~5-20 tokens so every irr_ctx length has near matches.
    # No proper nouns, no relation content, nothing a distractor cloud could be built from.
    "It was noted at the time.",
    "This happened more than once.",
    "The result was not surprising.",
    "Few people disagreed with it.",
    "Such cases are not unusual.",
    "The process took a while.",
    "Nothing further was said about it.",
    "It remained that way for some time.",
    "The matter was left there for now.",
    "Opinions on this have always varied.",
    "There was little reason to object.",
    "Most of it was settled quickly.",
    "The outcome was broadly as expected.",
    "Several accounts of it were recorded.",
    "Work continued without notable interruption.",
    "A summary was prepared in advance.",
    "The details were set out plainly.",
    "It has been described in similar terms.",
    "The arrangement stayed in place for years.",
    "Observers considered the outcome unremarkable.",
    "Much of the discussion went unrecorded.",
    "The question was raised again later that year.",
    "Very little changed in the period that followed.",
    "The explanation offered at the time was accepted.",
    "It is generally agreed that the effect is small.",
    "Reports from the period are largely consistent.",
    "The practice continued for a considerable while.",
    "No further action was thought to be necessary.",
    "The matter was discussed at some length beforehand.",
    "Those involved described the process as routine.",
    "It has been noted that the usual pattern held.",
    "The arrangement remained in place for a number of years.",
    "A brief account of the proceedings was circulated widely.",
    "The details were set out plainly so that no confusion arose.",
    "Several separate accounts of the events were later compared.",
    "Work continued steadily throughout the period without interruption.",
    "It has been noted that the process usually takes a considerable while.",
    "Observers described the eventual outcome as broadly consistent with expectations.",
    "The matter was discussed at some length before any decision was reached.",
    "A summary of the proceedings was prepared and circulated well in advance.",
    "Very little of substance changed during the years that immediately followed this.",
    "Those who were involved at the time have described the process as entirely routine.",
    "It is generally agreed by now that the overall effect was rather smaller than claimed.",
    "The explanation that was offered at the time seems to have been accepted without much argument.",
]

out = []
for s in S["samples"]:
    g = s["variants"].get("golden") or []
    f = s["variants"].get("far_irr") or []
    if not g or not f:
        continue
    gold_ctx = g[0]["context"]
    irr = f[0]
    out.append({
        "sample_id": s["sample_id"], "relation": s["relation"], "subject": s["subject"],
        "gold": s["gold"], "query": s["query"],
        "golden_ctx": gold_ctx,
        "irr_ctx": irr["context"],
        "irr_relation": irr.get("context_relation"),
        "irr_subject": irr.get("context_subject"),
        "distractor": irr.get("distractor"),
        "filler_pool": FILLER_POOL,
        "cells": {
            "bare":   "{query}",
            "pad":    "{filler} {query}",
            "ev":     "{golden_ctx} {query}",
            "irr":    "{irr_ctx} {query}",
            "ev_irr": "{irr_ctx} {golden_ctx} {query}",
            "ev_pad": "{filler} {golden_ctx} {query}",
        },
    })
meta = {"n": len(out), "source": "pairs_v1.json",
        "note": "ev_pad filler is chosen per sample at measurement time to match irr_ctx token length",
        "samples": out}
p = ROOT / "data" / "e53_factorial.json"
json.dump(meta, p.open("w"), indent=1)
print(f"built {len(out)} factorial samples -> {p}")
byrel = {}
for x in out: byrel[x["relation"]] = byrel.get(x["relation"], 0) + 1
for k, v in sorted(byrel.items(), key=lambda kv: -kv[1]): print(f"  {k:26s} {v}")
