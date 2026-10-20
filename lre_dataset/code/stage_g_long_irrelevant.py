"""Build samples_6x250.json: extends samples_5x250.json with a `long_irrelevant`
variant per sample, drawn from wikitext-103 Wikipedia paragraphs.

Selection rule:
  - target paragraph length: 90-180 whitespace tokens (Wikipedia-paragraph scale)
  - reject paragraphs containing the sample's subject or gold (any case)
  - reject paragraphs containing the relation's other subjects/golds
    (e.g., other capitals if relation is country_capital_city) to prevent
    accidental relational leakage
  - deterministic per sample_id: stable hash modulo filtered pool index

Output:
  lre_dataset/data/wiki_pool.json       — the cleaned 1500-paragraph pool
  lre_dataset/data/samples_6x250.json   — extended dataset with 6th variant
"""
import json, hashlib, re, random
from pathlib import Path
from collections import defaultdict
REPO = Path(__file__).resolve().parents[2]  # repository root

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

SRC_DATASET = REPO / "lre_dataset/data/samples_5x250.json"
WIKI_POOL_PATH = DATA / "wiki_pool.json"
OUT_DATASET   = DATA / "samples_6x250.json"

TARGET_MIN_TOKS = 90
TARGET_MAX_TOKS = 180
POOL_TARGET_SIZE = 1500   # filtered pool size (oversized so per-sample filters still leave many)


def load_wiki_paragraphs():
    """Stream wikitext-103 train split, extract clean paragraphs in length range."""
    from datasets import load_dataset
    print("loading wikitext-103 (streaming)...", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    paragraphs = []
    seen_first16 = set()
    for ex in ds:
        text = ex.get("text") or ""
        if not text or text.startswith(" =") or text.startswith("="):
            # skip section headings
            continue
        text = text.strip()
        if not text:
            continue
        # split on double-newline to get paragraphs; wikitext uses single \n though
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            n_toks = len(para.split())
            if n_toks < TARGET_MIN_TOKS or n_toks > TARGET_MAX_TOKS:
                continue
            # quality filters: must end with period; mostly alpha
            if not para.endswith((".", '."', '?', '!')):
                continue
            n_alpha = sum(c.isalpha() for c in para)
            if n_alpha / max(1, len(para)) < 0.6:
                continue
            # dedupe by first 16 words
            sig = " ".join(para.split()[:16]).lower()
            if sig in seen_first16:
                continue
            seen_first16.add(sig)
            paragraphs.append(para)
            if len(paragraphs) >= 50000:
                return paragraphs
    return paragraphs


def collect_relation_leaks(samples):
    """For each (subject, gold) tuple, collect the set of forbidden words so we
    can filter paragraphs that accidentally mention any sample's gold/subject.
    Per-relation forbid-list: any subject or gold of any sample in that relation."""
    by_rel = defaultdict(set)
    for s in samples:
        rel = s["relation"]
        for w in (s["subject"], s["gold"]):
            for token in re.split(r"\s+", w):
                t = token.strip().lower()
                if len(t) >= 3:
                    by_rel[rel].add(t)
    return dict(by_rel)


def filter_paragraph_for_sample(para, sample, rel_leak_set):
    """Strict per-sample filter. Returns True if paragraph is admissible."""
    p_lower = " " + para.lower() + " "
    # reject this sample's subject/gold (must not appear at all)
    for w in (sample["subject"].lower(), sample["gold"].lower()):
        if (" " + w + " ") in p_lower or (" " + w + ",") in p_lower or (" " + w + ".") in p_lower:
            return False
    # reject any other gold from same relation (prevent "the capital of France is Paris"
    # appearing in an irrelevant paragraph for the (Spain, Madrid) sample)
    for w in rel_leak_set:
        if w == sample["subject"].lower() or w == sample["gold"].lower():
            continue
        if (" " + w + " ") in p_lower or (" " + w + ",") in p_lower or (" " + w + ".") in p_lower:
            return False
    return True


def deterministic_pick(sample_id, candidates):
    """Stable hash of sample_id → index into candidate list."""
    h = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16)
    return candidates[h % len(candidates)]


def main():
    # ---- Step 1: load or build the wikipedia pool ----
    if WIKI_POOL_PATH.exists():
        print(f"reusing existing pool at {WIKI_POOL_PATH}", flush=True)
        pool = json.load(WIKI_POOL_PATH.open())
    else:
        paragraphs = load_wiki_paragraphs()
        print(f"collected {len(paragraphs)} paragraphs after length+quality filter", flush=True)
        # take a stratified slice: sort by length, take every nth so we span the range
        paragraphs.sort(key=lambda p: len(p.split()))
        if len(paragraphs) > POOL_TARGET_SIZE:
            step = len(paragraphs) / POOL_TARGET_SIZE
            paragraphs = [paragraphs[int(i*step)] for i in range(POOL_TARGET_SIZE)]
        pool = [{"id": f"wiki_{i:05d}", "text": p, "n_tokens_ws": len(p.split())}
                for i, p in enumerate(paragraphs)]
        json.dump(pool, WIKI_POOL_PATH.open("w"), ensure_ascii=False, indent=2)
        print(f"wrote pool of {len(pool)} paragraphs to {WIKI_POOL_PATH}", flush=True)

    # ---- Step 2: load source samples ----
    src = json.load(SRC_DATASET.open())
    samples = src["samples"]
    print(f"loaded {len(samples)} samples from {SRC_DATASET}", flush=True)

    rel_leaks = collect_relation_leaks(samples)
    print(f"forbidden-word sets per relation: {[(r, len(s)) for r, s in rel_leaks.items()]}",
          flush=True)

    # ---- Step 3: per-sample pick ----
    pick_lens = []
    n_pool_after_filter = []
    rejection_examples = []

    for samp in samples:
        rel_leak = rel_leaks[samp["relation"]]
        candidates = [p for p in pool if filter_paragraph_for_sample(p["text"], samp, rel_leak)]
        n_pool_after_filter.append(len(candidates))
        if not candidates:
            # fallback: drop the relation-leak filter, only require no subject/gold
            candidates = [p for p in pool
                          if filter_paragraph_for_sample(p["text"], samp, set())]
            rejection_examples.append(samp["sample_id"])
        chosen = deterministic_pick(samp["sample_id"], candidates)
        samp["variants"]["long_irrelevant"] = {
            "context": chosen["text"],
            "distractor": "__auto__",
            "wiki_chunk_id": chosen["id"],
            "n_tokens_ws": chosen["n_tokens_ws"],
        }
        pick_lens.append(chosen["n_tokens_ws"])

    # ---- Step 4: save ----
    src["classes"] = list(src.get("classes", []))
    if "long_irrelevant" not in src["classes"]:
        src["classes"].append("long_irrelevant")
    src["long_irrelevant_metadata"] = {
        "source_dataset": "wikitext-103-raw-v1 train split (HuggingFace)",
        "pool_size": len(pool),
        "target_token_range_ws": [TARGET_MIN_TOKS, TARGET_MAX_TOKS],
        "selection_rule": "deterministic sha256(sample_id) mod filtered_pool",
        "filter": "no subject/gold appearance + no other-sample-gold-from-same-relation appearance",
        "median_pick_tokens_ws": int(sorted(pick_lens)[len(pick_lens)//2]),
        "mean_pool_after_filter": sum(n_pool_after_filter) / len(n_pool_after_filter),
        "samples_using_fallback_filter": rejection_examples,
    }
    OUT_DATASET.write_text(json.dumps(src, ensure_ascii=False, indent=2))
    print(f"\nWROTE: {OUT_DATASET}", flush=True)
    print(f"  pool: {len(pool)} paragraphs", flush=True)
    print(f"  per-sample post-filter pool: mean={sum(n_pool_after_filter)/len(n_pool_after_filter):.0f}  min={min(n_pool_after_filter)}",
          flush=True)
    print(f"  pick length (ws-tokens): median={sorted(pick_lens)[len(pick_lens)//2]}  "
          f"min={min(pick_lens)}  max={max(pick_lens)}", flush=True)
    print(f"  samples needing fallback: {len(rejection_examples)}", flush=True)

    print("\nEXAMPLE (sample_id=cou_001):")
    sample = next(s for s in samples if s["sample_id"] == "cou_001")
    print(f"  subject: {sample['subject']}  gold: {sample['gold']}")
    print(f"  long_irr context (first 300 chars):\n    {sample['variants']['long_irrelevant']['context'][:300]!r}...")


if __name__ == "__main__":
    main()
