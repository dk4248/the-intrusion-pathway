"""M.1 — do the mid-block relational readers gate on the answer-SLOT frame, or on answer TYPE?

Readers are identified on a DISCOVERY half by two criteria — high attention from END
to the answer position under golden context, and low maximum context attention under an
irrelevant prefix — then the cue factorial is run on HELD-OUT examples.

Five variants, token-length matched where possible, all for the same query:

  v1 slot+gold      "The capital city of Germany is Berlin."     frame present, correct filler
  v2 slot+sametype  "The capital city of Germany is Madrid."     frame present, wrong city
  v3 type_nosLOT    "Madrid is a well known place."              a city, but NO relational frame
  v4 slot+wrongtype "The capital city of Germany is hammer."     frame present, wrong-TYPE filler
  v5 neither        a far-irrelevant sentence                    neither cue

Structural gating predicts v4 (frame, wrong type) opens the gate; semantic-type gating predicts
v3 (right type, no frame) does. Readout is the FULL source-region attention mass onto the filler
span -- not just the single highest-attended token -- plus the gold logit.
"""
import argparse, json, sys, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load
from lre_pool import load_relations

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=90); ap.add_argument("--topk", type=int, default=10)
a = ap.parse_args()
m = load(a.model, a.device); nL, nH = m.cfg.n_layers, m.cfg.n_heads
W_U = m.W_U.float(); R = load_relations()
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"][: a.limit]
rng = random.Random(0)
MID = range(nL // 4, 3 * nL // 4)          # the mid block the claim is about

def fid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False); return i[0] if i else None

def span_positions(prompt, sub):
    """all token positions covered by `sub` inside prompt (the full filler span)"""
    i = prompt.rfind(sub)
    if i < 0: return []
    a_ = m.to_tokens(prompt[:i]).shape[1] - 1
    b_ = m.to_tokens(prompt[:i + len(sub)]).shape[1] - 1
    return list(range(max(a_, 1), b_ + 1))

WRONG_TYPE = ["hammer", "broom", "yellow", "positive", "bicycle", "violin"]
# The filler must sit at the END of the context in every variant: in v1/v2/v4 the relational
# frame puts it last, so a no-slot template with the filler first would confound "no frame"
# with "further from END", and distance alone suppresses attention.
NOSLOT_TPL = ["A well known example is {}.", "People often mention {}.",
              "The one that comes to mind is {}."]

def variants(s):
    rel = R[s["relation"]]
    subj, gold = s["subject"], s["gold"]
    others = [o for _, o in rel["facts"] if o.lower() != gold.lower()]
    if not others: return None
    other = rng.choice(others)
    far = s["variants"].get("far_irr", [])
    if not far: return None
    wt = rng.choice(WRONG_TYPE)
    def ctx(o): return f"{rel['ctx_tpl'].format(subj)} {o}."
    return {
        "v1_slot_gold":      (ctx(gold), gold),
        "v2_slot_sametype":  (ctx(other), other),
        "v3_type_noslot":    (rng.choice(NOSLOT_TPL).format(other), other),
        "v4_slot_wrongtype": (ctx(wt), wt),
        "v5_neither":        (far[0]["context"], far[0]["distractor"]),
    }

half = len(S) // 2
disc, held = S[:half], S[half:]

# ---- identify readers on the discovery half ----
att_gold = torch.zeros(nL, nH, device=W_U.device)
att_irr = torch.zeros(nL, nH, device=W_U.device)
n = 0
with torch.no_grad():
    for s in disc:
        g = s["variants"].get("golden", []); f = s["variants"].get("far_irr", [])
        if not g or not f: continue
        pg = f"{g[0]['context']} {s['query']}"
        pos = span_positions(pg, s["gold"])
        if not pos: continue
        tg = m.to_tokens(pg); _, cg = m.run_with_cache(tg, names_filter=lambda x: "hook_pattern" in x)
        E = tg.shape[1] - 1
        for L in range(nL):
            att_gold[L] += cg["pattern", L][0, :, E, torch.tensor(pos, device=W_U.device)].float().sum(-1)
        pf = f"{f[0]['context']} {s['query']}"
        tf = m.to_tokens(pf); nctx = m.to_tokens(f[0]["context"]).shape[1] - 1
        _, cf = m.run_with_cache(tf, names_filter=lambda x: "hook_pattern" in x)
        Ef = tf.shape[1] - 1
        for L in range(nL):
            att_irr[L] += cf["pattern", L][0, :, Ef, 1:nctx + 1].float().max(-1).values
        n += 1
        del cg, cf
score = ((att_gold - att_irr) / max(n, 1)).cpu()
mask = torch.full((nL, nH), float("-inf"))
for L in MID: mask[L] = 0
sc = (score + mask).flatten()
READERS = [(int(i // nH), int(i % nH)) for i in torch.topk(sc, a.topk).indices.tolist()]
CTRL = [(int(rng.choice(list(MID))), int(rng.randrange(nH))) for _ in range(a.topk)]
print(f"[{a.model}] readers (mid block L{MID.start}-{MID.stop-1}, discovery n={n}): "
      f"{['L%dH%d' % h for h in READERS[:6]]}", flush=True)

# ---- the factorial, on held-out ----
acc = defaultdict(lambda: defaultdict(list))
with torch.no_grad():
    for s in held:
        V = variants(s)
        if not V: continue
        gid = fid(s["gold"])
        if gid is None: continue
        for name, (ctx, filler) in V.items():
            prompt = f"{ctx} {s['query']}"
            pos = span_positions(prompt, filler)
            if not pos: continue
            toks = m.to_tokens(prompt); E = toks.shape[1] - 1
            lg, c = m.run_with_cache(toks, names_filter=lambda x: "hook_pattern" in x)
            row = lg[0, E].float()
            pt = torch.tensor(pos, device=W_U.device)
            for tag, hs in (("reader", READERS), ("control", CTRL)):
                mass = 0.0
                for (L, H) in hs:
                    mass += float(c["pattern", L][0, H, E, pt].float().sum())
                acc[tag][name].append(mass / len(hs))
            acc["model"][name].append(float(row[gid]))
            del c

def mci(xs, n=1500):
    mu = sum(xs) / len(xs)
    d = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return mu, d[int(.025 * n)], d[int(.975 * n)]

ORDER = ["v1_slot_gold", "v2_slot_sametype", "v3_type_noslot", "v4_slot_wrongtype", "v5_neither"]
print(f"\n=== {a.model}: M.1 slot frame vs answer type "
      f"(held-out n={len(acc['model'][ORDER[0]])}) ===")
print(f"  {'variant':20s} {'reader attn to filler span':>28s} {'control heads':>16s} {'gold logit':>12s}")
for v in ORDER:
    if not acc["reader"][v]: continue
    r = mci(acc["reader"][v]); c = mci(acc["control"][v]); g = mci(acc["model"][v])
    print(f"  {v:20s} {r[0]:16.4f} [{r[1]:.3f},{r[2]:.3f}] {c[0]:16.4f} {g[0]:12.3f}")
print("\n  structural gating -> v4 (frame, wrong type) high; semantic-type gating -> v3 high")
json.dump(dict(readers=READERS, controls=CTRL,
               attn={k: {v: mci(x)[0] for v, x in d.items()} for k, d in acc.items()}),
          (ROOT / "results" / f"M1_{a.model}.json").open("w"), indent=1)
