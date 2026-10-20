"""E4.3 — whole-head removal vs finer edits, across conditions.

The question: is there ANY granularity at which the parroting population's
contribution under irrelevant context can be removed without paying for it under useful context?

Granularities, coarse to fine:
  head_zero     zero the head everywhere                    (what "ablate the head" usually means)
  head_resample replace the head's output with a donor's    (non-zero baseline)
  end_only      remove the head's output at END only
  prefix_src    remove only the PREFIX-sourced part of the head's END output, original
                attention weights kept (no renormalisation)
  edge_mask     set pre-softmax END->prefix scores to -inf, so softmax RENORMALISES over the
                remaining positions -- a different operation from prefix_src, not a synonym
  cloud_sub     subtract the validated cloud component (E3.2) from the head's END output

Conditions: nctx · golden · far_irr · golden+far_irr (the mixed case, built by concatenating
the two prefixes) -- because a defense is only interesting if it survives useful evidence.
"""
import argparse, json, sys, time, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load
from lre_pool import load_relations
from assoc_vocab import category_cloud, wiki_attr_cloud

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--limit", type=int, default=50); ap.add_argument("--heads", type=int, default=8)
a = ap.parse_args()
m = load(a.model, a.device); nL, nH = m.cfg.n_layers, m.cfg.n_heads
W_U = m.W_U.float(); R = load_relations()
S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"][: a.limit]
HEADS = [tuple(x) for x in json.load((ROOT / "results" / f"E2_3_{a.model}.json").open())["top_heads"][: a.heads]]
BY_L = defaultdict(list)
for L, H in HEADS:
    BY_L[L].append(H)
STOP = set(m.tokenizer.encode(" the of a an and is in to for on at .", add_special_tokens=False))
rng = random.Random(0)

def fid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False); return i[0] if i else None

def build(s):
    out = {"nctx": (s["query"], 0)}
    for c in ("golden", "far_irr"):
        v = s["variants"].get(c, [])
        if v:
            out[c] = (f"{v[0]['context']} {s['query']}", len(v[0]["context"]))
    g, f = s["variants"].get("golden", []), s["variants"].get("far_irr", [])
    if g and f:
        ctx = f"{g[0]['context']} {f[0]['context']}"
        out["golden+far"] = (f"{ctx} {s['query']}", len(ctx))
    return out

def metrics(logits, gid):
    row = logits[0, -1].float(); lse = torch.logsumexp(row, -1)
    mk = row.clone(); mk[gid] = float("-inf")
    return dict(gold=float(row[gid]), logprob=float(row[gid] - lse),
                margin=float(row[gid] - mk.max()), hit=int(int(row.argmax()) == gid))

res = defaultdict(lambda: defaultdict(list))
t0 = time.time()
with torch.no_grad():
    for si, s in enumerate(S):
        gid = fid(s["gold"])
        if gid is None: continue
        donor = S[(si + 7) % len(S)]
        conds = build(s)
        for cond, (prompt, ctx_chars) in conds.items():
            toks = m.to_tokens(prompt); END = toks.shape[1] - 1
            n_ctx = (m.to_tokens(prompt[:ctx_chars]).shape[1] - 1) if ctx_chars else 0
            pref = [p for p in range(1, n_ctx + 1)] if n_ctx else []
            base_logits, cache = m.run_with_cache(toks)
            base = metrics(base_logits, gid)
            res[(cond, "none")]["gold"].append(base["gold"])
            res[(cond, "none")]["margin"].append(base["margin"])
            res[(cond, "none")]["hit"].append(base["hit"])

            dz = json.loads("{}")
            # precompute per-head prefix-sourced contribution and cloud component at END
            pre_delta, cloud_delta = defaultdict(dict), defaultdict(dict)
            for L, Hs in BY_L.items():
                vv = cache["v", L][0].float(); n_kv = vv.shape[1]
                if n_kv != nH: vv = vv.repeat_interleave(nH // n_kv, dim=1)
                W_O = m.blocks[L].attn.W_O.float()
                for H in Hs:
                    if pref:
                        al = cache["pattern", L][0, H, END, torch.tensor(pref, device=W_U.device)].float()
                        w = vv[torch.tensor(pref, device=W_U.device), H]
                        pre_delta[L][H] = (al.unsqueeze(1) * w).sum(0)     # in d_head space
                    else:
                        pre_delta[L][H] = torch.zeros(vv.shape[-1], device=W_U.device)
            dcache = None
            for mode in ("head_zero", "head_resample", "end_only", "prefix_src", "edge_mask"):
                hooks = []
                if mode == "head_resample" and dcache is None:
                    dt = m.to_tokens(donor["query"])
                    _, dcache = m.run_with_cache(dt, names_filter=lambda n: "hook_z" in n)
                for L, Hs in BY_L.items():
                    if mode == "edge_mask":
                        def mk_edge(Hs=Hs, pref=pref, END=END):
                            def hk(x, hook):
                                if pref:
                                    for H in Hs:
                                        x[:, H, END, torch.tensor(pref, device=x.device)] = -1e9
                                return x
                            return hk
                        hooks.append((f"blocks.{L}.attn.hook_attn_scores", mk_edge()))
                    else:
                        def mk_z(L=L, Hs=Hs, mode=mode, END=END):
                            def hk(x, hook):
                                for H in Hs:
                                    if mode == "head_zero":
                                        x[:, :, H, :] = 0
                                    elif mode == "end_only":
                                        x[:, END, H, :] = 0
                                    elif mode == "prefix_src":
                                        x[:, END, H, :] = x[:, END, H, :] - pre_delta[L][H].to(x.dtype)
                                    elif mode == "head_resample":
                                        dz_ = dcache["z", L][0, -1, H, :]
                                        x[:, :, H, :] = dz_.to(x.dtype)
                                return x
                            return hk
                        hooks.append((f"blocks.{L}.attn.hook_z", mk_z()))
                with m.hooks(fwd_hooks=hooks):
                    lg = m(toks, return_type="logits")
                mt = metrics(lg, gid)
                for k in ("gold", "margin", "hit"):
                    res[(cond, mode)][k].append(mt[k])
            del cache
        if (si + 1) % 20 == 0:
            print(f"  [{a.model}] {si+1}/{len(S)} {time.time()-t0:.0f}s", flush=True)

def mu(xs): return sum(xs) / len(xs) if xs else float("nan")
print(f"\n=== {a.model}: E4.3 intervention granularity x condition "
      f"(d gold logit and d accuracy vs unmodified) ===")
MODES = ["head_zero", "head_resample", "end_only", "prefix_src", "edge_mask"]
print(f"  {'mode':14s} " + " ".join(f"{c:>21s}" for c in ["nctx", "golden", "far_irr", "golden+far"]))
for mode in MODES:
    cells = []
    for cond in ["nctx", "golden", "far_irr", "golden+far"]:
        b = res[(cond, "none")]; e = res[(cond, mode)]
        if not e["gold"]:
            cells.append(f"{'-':>21s}"); continue
        dg = mu(e["gold"]) - mu(b["gold"]); da = 100 * (mu(e["hit"]) - mu(b["hit"]))
        cells.append(f"{dg:+9.3f} / {da:+6.1f}pt")
    print(f"  {mode:14s} " + " ".join(cells))
print("\n  a separable defense would need far_irr gain > 0 with golden / golden+far cost ~ 0")
json.dump({f"{c}|{mo}": {k: mu(v) for k, v in d.items()} for (c, mo), d in res.items()},
          (ROOT / "results" / f"E4_3_{a.model}.json").open("w"), indent=1)
