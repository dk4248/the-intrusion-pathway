"""Stage 2 — build (query fact, context fact) pairs.

Every context is a templated LRE fact. The class is the RELATIONSHIP between the
context fact and the query fact, so surface form/length/register are matched:

  golden          same relation, same subject, true object        -> contains the answer
  counterfactual  same relation, same subject, WRONG object       -> competing answer
  related         same relation, different subject, true object   -> one-shot demo
  near_irr        DIFFERENT relation, SAME subject                -> about the entity, answers nothing
  mid_irr         different relation, different subject, TYPE OVERLAP with the query
  far_irr         different relation, different subject, TYPES DISJOINT
  random          content-free token

Leakage control is structural (type-disjointness from LRE's domain/range labels) plus a
literal entity-overlap check. No substring heuristics on free text, no embedding threshold.
"""
import json, random, sys, argparse, re
from pathlib import Path
from collections import defaultdict, Counter
sys.path.insert(0, str(Path(__file__).parent))
from lre_pool import load_relations, query_prompt, context_sentence, type_lexicon
REPO = Path(__file__).resolve().parents[2]  # repository root

ROOT = Path(__file__).resolve().parent.parent
RES, DATA = ROOT / "results", ROOT / "data"

ap = argparse.ArgumentParser()
ap.add_argument("--min-rank1", type=int, default=3, help="models that must rank gold #1")
ap.add_argument("--cap", type=int, default=40, help="max query facts per relation")
ap.add_argument("--k", type=int, default=3, help="context instances per irrelevant class")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--wiki-pool",
                default=str(REPO / "lre_dataset/data/wiki_pool.json"))
ap.add_argument("--long-target", type=int, default=120,
                help="whitespace tokens for the length-matched far_irr_long control")
ap.add_argument("--classes", default="golden,near_irr,mid_irr,far_irr",
                help="context classes to build (nctx is implicit)")
ap.add_argument("--out", default="pairs.json")
ap.add_argument("--allow-multitoken-gold", action="store_true",
                help="keep facts whose gold is not single-token in all 4 tokenizers "
                     "(rank/DLA still use the FIRST gold token)")
a = ap.parse_args()
rng = random.Random(a.seed)
CLASSES = [c.strip() for c in a.classes.split(",") if c.strip()]

R = load_relations()
screens = {p.stem.replace("screen_", ""): json.load(p.open())
           for p in sorted(RES.glob("screen_*.json"))}
models = sorted(screens)
print(f"screens: {models}")

# ---- 1. parametric filter -------------------------------------------------
per_fact = defaultdict(dict)           # (rel,subj) -> {model: rec}
for mk, s in screens.items():
    for r in s["records"]:
        per_fact[(r["relation"], r["subject"])][mk] = r

# first gold token must be unambiguous inside its own relation, otherwise
# "Mexico City" (-> " Mexico") is indistinguishable from the country Mexico.
first_id_counts = defaultdict(Counter)
for mk, sc in screens.items():
    for r in sc["records"]:
        first_id_counts[(mk, r["relation"])][r["gold_first_id"]] += 1

kept, stats = [], defaultdict(lambda: Counter())
for (rel, subj), by_m in per_fact.items():
    if len(by_m) != len(models):
        continue
    if not R[rel]["query_ok"]:
        stats[rel]["total"] += 1
        continue
    n_rank1 = sum(1 for m in models if by_m[m]["rank"] == 1)
    n_single = sum(1 for m in models if by_m[m]["single_token"])
    stats[rel]["total"] += 1
    stats[rel][f"rank1_{n_rank1}of4"] += 1
    if n_single == len(models):
        stats[rel]["single_tok_all"] += 1
    if n_rank1 >= a.min_rank1:
        stats[rel]["rank_ok"] += 1
    unique_first = all(
        first_id_counts[(m, rel)][by_m[m]["gold_first_id"]] == 1 for m in models)
    if unique_first:
        stats[rel]["unique_first_tok"] += 1
    gold_ok = (n_single == len(models)) or (a.allow_multitoken_gold and unique_first)
    if n_rank1 >= a.min_rank1 and gold_ok:
        kept.append(dict(relation=rel, subject=subj, gold=by_m[models[0]]["gold"],
                         n_rank1=n_rank1, single_token=(n_single == len(models)),
                         n_gold_tokens=max(by_m[m]["n_gold_tokens"] for m in models),
                         mean_margin=sum(by_m[m]["margin"] for m in models) / len(models)))
        stats[rel]["kept"] += 1

print(f"\n{'relation':22s} {'q?':>3s} {'total':>6s} {'1tok/4':>7s} {'4of4':>6s} {'3of4':>6s} "
      f"{'rank_ok':>8s} {'KEPT':>6s}")
for rel in R:
    c = stats[rel]
    print(f"{rel:22s} {('y' if R[rel]['query_ok'] else 'ctx'):>3s} {c['total']:6d} "
          f"{c['single_tok_all']:7d} {c['rank1_4of4']:6d} {c['rank1_3of4']:6d} "
          f"{c['rank_ok']:8d} {c['kept']:6d}")
print("  (rank_ok = passes the >=%d-of-4 rank-1 filter; KEPT also applies the gold-token rule)"
      % a.min_rank1)

# ---- 2. cap per relation (best margin first, but keep gold diversity) ------
by_rel = defaultdict(list)
for f in kept:
    by_rel[f["relation"]].append(f)
query_facts = []
for rel, fs in by_rel.items():
    fs.sort(key=lambda f: (-f["n_rank1"], -f["mean_margin"]))
    seen_gold, take = Counter(), []
    for f in fs:                       # cap repeats of the same gold token at 3
        if seen_gold[f["gold"]] >= 3:
            continue
        seen_gold[f["gold"]] += 1
        take.append(f)
        if len(take) >= a.cap:
            break
    query_facts.extend(take)
print(f"\nquery facts after cap={a.cap} (max 3 per gold token): {len(query_facts)}")
print("  ", dict(Counter(f['relation'] for f in query_facts)))
print(f"   distinct gold tokens: {len({f['gold'] for f in query_facts})}")

# ---- 3. context eligibility ----------------------------------------------
def types(rel):
    return {R[rel]["domain"], R[rel]["range"]}

def entity_clash(q_subj, q_gold, c_subj, c_obj):
    a_ = {q_subj.lower(), q_gold.lower()}
    b_ = {c_subj.lower(), c_obj.lower()}
    for x in a_:
        for y in b_:
            if x in y or y in x:
                return True
    return False

def sample_ctx(q, kind, k):
    qrel = q["relation"]
    out = []
    cands = []
    for crel, rel in R.items():
        if crel == qrel:
            continue
        overlap = bool(types(crel) & types(qrel))
        if kind == "mid_irr" and not overlap:
            continue
        if kind == "far_irr" and overlap:
            continue
        cands.append(crel)
    rng.shuffle(cands)
    tries = 0
    while len(out) < k and tries < 400:
        tries += 1
        crel = cands[tries % len(cands)] if cands else None
        if crel is None:
            break
        csubj, cobj = rng.choice(R[crel]["facts"])
        if entity_clash(q["subject"], q["gold"], csubj, cobj):
            continue
        if any(o["context_relation"] == crel and o["context_subject"] == csubj for o in out):
            continue
        out.append(dict(context=context_sentence(R[crel], csubj, cobj),
                        context_relation=crel, context_subject=csubj, distractor=cobj))
    return out


# ---- 3b. long contexts: Wikipedia paragraphs + a length-matched LRE control ----
WIKI = json.load(open(a.wiki_pool)) if Path(a.wiki_pool).exists() else []
LEX = type_lexicon()

def _rx(terms, min_len=4):
    # min_len guards the broad relation/type lists against junk matches; the sample's OWN
    # subject and gold must be matched at ANY length (colours like 'red' are 3 chars).
    terms = sorted({t for t in terms if t and len(t) >= min_len}, key=len, reverse=True)
    if not terms:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)

# per query relation: everything that would make a paragraph evidential
REL_FORBID = {}
for rn, rel in R.items():
    terms = [s for s, _ in rel["facts"]] + [o for _, o in rel["facts"]]
    REL_FORBID[rn] = _rx(terms)
# per range type: does the paragraph contain an entity of the ANSWER's type?
TYPE_RX = {rn: _rx(LEX.get(R[rn]["range"], [])) for rn in R}

def wiki_contexts(q, k):
    """Paragraphs with no evidential leakage for this query, flagged for whether they
    contain an entity of the query's answer type."""
    out = []
    subj, gold = q["subject"], q["gold"]
    own = _rx([subj, gold], min_len=2)
    forbid, trx = REL_FORBID[q["relation"]], TYPE_RX[q["relation"]]
    order = list(range(len(WIKI)))
    rng.shuffle(order)
    for i in order:
        if len(out) >= k:
            break
        t = WIKI[i]["text"]
        if own and own.search(t):
            continue
        if forbid and forbid.search(t):
            continue
        hit = trx.search(t) if trx else None
        out.append(dict(context=t, context_relation="wikitext103",
                        context_subject=None, distractor=None,
                        wiki_id=WIKI[i]["id"], n_tokens_ws=WIKI[i]["n_tokens_ws"],
                        has_type_match=bool(hit),
                        type_match=hit.group(0) if hit else None))
    return out

def far_long_contexts(q, k, target):
    """Concatenated type-disjoint LRE facts, matched in length to the wiki paragraphs.
    Same content KIND as far_irr, same LENGTH as long_irr -> separates the two."""
    out = []
    for _ in range(k):
        parts, dstrs, n = [], [], 0
        guard = 0
        while n < target and guard < 60:
            guard += 1
            c = sample_ctx(q, "far_irr", 1)
            if not c:
                break
            if c[0]["context"] in parts:
                continue
            parts.append(c[0]["context"])
            dstrs.append(c[0]["distractor"])
            n = len(" ".join(parts).split())
        if parts:
            out.append(dict(context=" ".join(parts), context_relation="far_concat",
                            context_subject=None, distractor=dstrs[-1],
                            n_facts=len(parts), n_tokens_ws=n, distractors=dstrs))
    return out

# same subject in a different relation of the same family
subj_index = defaultdict(list)          # (family, subject) -> [(rel, obj)]
for rname, rel in R.items():
    for s, o in rel["facts"]:
        subj_index[(rel["family"], s)].append((rname, o))

samples, n_near = [], 0
for q in query_facts:
    qrel = R[q["relation"]]
    v = {}
    v["golden"] = [dict(context=context_sentence(qrel, q["subject"], q["gold"]),
                        context_relation=q["relation"], context_subject=q["subject"],
                        distractor=q["gold"])]
    wrong = [o for o in qrel["objects"] if o != q["gold"]]
    cf = rng.choice(wrong) if wrong else None
    v["counterfactual"] = [dict(context=context_sentence(qrel, q["subject"], cf),
                                context_relation=q["relation"], context_subject=q["subject"],
                                distractor=cf)] if cf else []
    rel_c = []
    pool = [(s, o) for s, o in qrel["facts"] if s != q["subject"] and o != q["gold"]]
    rng.shuffle(pool)
    for s, o in pool[:a.k]:
        rel_c.append(dict(context=context_sentence(qrel, s, o),
                          context_relation=q["relation"], context_subject=s, distractor=o))
    v["related"] = rel_c
    # near_irr SHARES the subject on purpose, so only the context's object is
    # checked for leakage (it must not be, or contain, the gold).
    def obj_leaks(o2):
        x, y = q["gold"].lower(), str(o2).lower()
        return x in y or y in x
    near = [(r2, o2) for r2, o2 in subj_index[(qrel["family"], q["subject"])]
            if r2 != q["relation"] and not obj_leaks(o2)]
    v["near_irr"] = [dict(context=context_sentence(R[r2], q["subject"], o2),
                          context_relation=r2, context_subject=q["subject"], distractor=o2)
                     for r2, o2 in near[:a.k]]
    n_near += bool(v["near_irr"])
    v["mid_irr"] = sample_ctx(q, "mid_irr", a.k)
    v["far_irr"] = sample_ctx(q, "far_irr", a.k)
    v["random"] = [dict(context="Dog.", context_relation=None, context_subject=None, distractor="Dog")]
    if "long_irr" in CLASSES:
        v["long_irr"] = wiki_contexts(q, a.k)
    if "far_irr_long" in CLASSES:
        v["far_irr_long"] = far_long_contexts(q, a.k, a.long_target)
    v = {k: val for k, val in v.items() if k in CLASSES}
    samples.append(dict(sample_id=f"{q['relation'][:4]}_{len(samples):04d}", **q,
                        query=query_prompt(qrel, q["subject"]), variants=v))

print(f"\nsamples: {len(samples)}   with near_irr available: {n_near}")
for cls in CLASSES:
    tot = sum(len(s["variants"].get(cls, [])) for s in samples)
    have = sum(1 for s in samples if s["variants"].get(cls))
    extra = ""
    if cls == "long_irr":
        flagged = sum(1 for s in samples for v in s["variants"].get(cls, []) if v["has_type_match"])
        extra = f"   [{flagged}/{tot} contain an answer-TYPE entity]"
    if cls == "far_irr_long":
        ws = [v["n_tokens_ws"] for s in samples for v in s["variants"].get(cls, [])]
        nf = [v["n_facts"] for s in samples for v in s["variants"].get(cls, [])]
        extra = (f"   [median {sorted(ws)[len(ws)//2]} ws-tokens from "
                 f"{sorted(nf)[len(nf)//2]} facts]" if ws else "")
    print(f"   {cls:16s} contexts: {tot:5d}  (samples with >=1: {have}){extra}")

DATA.mkdir(exist_ok=True)
json.dump(dict(n=len(samples), min_rank1=a.min_rank1, cap=a.cap, k=a.k, seed=a.seed,
               classes=CLASSES, models=models, samples=samples), (DATA / a.out).open("w"), indent=1)
print(f"\nwrote {DATA / a.out}")
