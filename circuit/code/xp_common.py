"""Shared setup for the two-sided / four-write / route experiments (75, 76, 77).

Everything that must be identical across the three experiments lives here, so a result in one can be
compared with the others without re-deriving definitions:

  * held-out facts (data/c5_split.json) -- the frozen populations were selected on discovery facts
  * populations (61_sep_scores.py, raw DLA + prefix mass, k = 20):
        Istar   the paper's intrusion set: top-20 s_int excluding the top-40 s_ret
        Ronly   R_20 minus I_20
        CTRL    one head per Istar head, same layer, closest discovery prefix-attention mass,
                outside I_20 u R_20 u Istar
        COPIER  the E2.3 copier heads (top-20), used as senders in 77
  * prompts: bare, far_irr, long_irr, golden, golden+irrelevant, and token-exact matched fillers
  * distractor vocabularies, fixed from the input alone and kept separate:
        D_ans    the far context's own answer (variant['distractor']), both casings
        D_lit    every content word that literally occurs in the irrelevant prefix
        D_assoc  association words of the prefix's content (category co-members and Wikipedia-summary
                 words of the distractor and of the context subject), minus literal tokens, gold, the
                 query's own words, its category words and the query relation's other answers
    long_irr has no predefined distractor, so only D_lit is defined for it.
  * readouts of a final distribution: gold logit, gold log-prob, P(D_ans), P(D_lit), P(D_assoc),
    margin vs the strongest non-gold token, margin vs the strongest literal token, accuracy
  * DLA convention of the frozen screen: (z_h W_O^h) . (w_lnf * W_U[:, t])  ("raw")
"""
import json, os, re, random, sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load
from assoc_vocab import (STOPWORDS_FOR_ATTRS as STOP, category_cloud, wiki_attr_cloud,
                         other_valid_answers, query_category_words)
from lre_pool import load_all_relations

ROOT = Path(__file__).resolve().parent.parent
REL = load_all_relations()
FILLER_POOL = json.load((ROOT / "data" / "e53_factorial.json").open())["samples"][0]["filler_pool"]
FACT = {x["sample_id"]: x for x in json.load((ROOT / "data" / "e53_factorial.json").open())["samples"]}


def load_fp32(model, device):
    """fp32 weights and activations. bf16 changes logits by up to ~0.19 between batch sizes and adds
    rounding noise to every patched row, comparable to the smallest effects measured here; fp32 keeps
    both below 1e-4 (checked: identical rows in a batch agree exactly, batch vs single ~3e-5)."""
    from transformer_lens import HookedTransformer
    from common import MODELS
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    m = HookedTransformer.from_pretrained_no_processing(MODELS[model], device=device, dtype=torch.float32)
    m.eval()
    m.cfg.default_prepend_bos = True        # BOS at position 0 for every model (Qwen2 defaults to none); see common.load
    m.tokenizer.padding_side = "right"
    if m.tokenizer.pad_token is None:
        m.tokenizer.pad_token = m.tokenizer.eos_token
    return m


class XP:
    def __init__(self, model, device):
        self.mk, self.dev = model, device
        m = self.m = load_fp32(model, device)
        self.nL, self.nH = m.cfg.n_layers, m.cfg.n_heads
        self.nKV = getattr(m.cfg, "n_key_value_heads", None) or self.nH
        self.REP = self.nH // self.nKV
        lw = getattr(m.ln_final, "w", None)
        self.WUf = m.W_U.float() * (lw.float()[:, None] if lw is not None else 1.0)
        self.WO = [m.blocks[L].attn.W_O.float() for L in range(self.nL)]      # [H, dh, dm]
        S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"]
        held = set(json.load((ROOT / "data" / "c5_split.json").open())["heldout"])
        self.HELD = [s for s in S if s["sample_id"] in held]
        sw = {r["k"]: r for r in json.load((ROOT / "results" / f"SEP_scores_{model}.json").open())
              ["variants"]["raw|prefix"]["sweep"]}
        I20 = {tuple(x) for x in sw[20]["I"]}; R20 = {tuple(x) for x in sw[20]["R"]}
        self.Istar = sorted(tuple(x) for x in sw[20]["Istar"])
        self.Ronly = sorted(R20 - I20)
        man = json.load((ROOT / "results" / f"E5_1_manifest_{model}.json").open())
        abar = {(int(L), int(h)): float(v) for L, h, v in man["a_bar_all"]}
        taken = I20 | R20 | set(self.Istar)
        ctrl = []
        for L, h in self.Istar:
            cands = [(abs(abar[(L, g)] - abar[(L, h)]), g) for g in range(self.nH)
                     if (L, g) not in taken and (L, g) not in ctrl]
            ctrl.append((L, min(cands)[1]))
        self.CTRL = sorted(ctrl)
        self.abar = abar
        self.COPIER = [tuple(x) for x in json.load((ROOT / "results" / f"E2_3_{model}.json").open())["top_heads"]]
        self.POPS = {"Istar": self.Istar, "Ronly": self.Ronly, "CTRL": self.CTRL}

    # ---------------------------------------------------------------- tokens
    def fid(self, w):
        i = self.m.tokenizer.encode(" " + str(w).strip(), add_special_tokens=False)
        return i[0] if i else None

    def word_id(self, w):
        """Output token for a vocabulary word: ' w' if it is one token; otherwise its first token only
        when that piece is itself a word-like string (>= 4 letters), so sub-word fragments such as
        ' Po' or ' Th' never enter a distractor set."""
        ids = self.m.tokenizer.encode(" " + str(w).strip(), add_special_tokens=False)
        if not ids: return None
        if len(ids) == 1: return ids[0]
        first = self.m.tokenizer.decode([ids[0]]).strip()
        return ids[0] if first.isalpha() and len(first) >= 4 else None

    def forms(self, w):
        w = str(w).strip()
        return {i for i in (self.word_id(w), self.word_id(w[:1].upper() + w[1:]),
                            self.word_id(w[:1].lower() + w[1:])) if i is not None}

    def ids_of_words(self, words, exclude_words=()):
        ex = {e.lower() for e in exclude_words if e}
        out = set()
        for w in words:
            if not w or len(w) < 3 or w.lower() in STOP or w.lower() in ex: continue
            i = self.word_id(w)
            if i is not None: out.add(i)
        return out

    def ntok(self, text):
        return len(self.m.tokenizer.encode(text, add_special_tokens=False))

    # ---------------------------------------------------------------- prompts
    def matched_filler(self, prefix, query, seed):
        """Filler with EXACTLY the token count of `prefix` inside the full prompt, so every prefix and
        query position aligns with the irrelevant prompt. Neutral pool, seeded order, cut at a token."""
        target = self.m.to_tokens(f"{prefix} {query}").shape[1]
        n = self.ntok(prefix)
        rg = random.Random(seed)
        for attempt in range(40):
            pool = FILLER_POOL[:]; rg.shuffle(pool)
            ids = self.m.tokenizer.encode(" ".join(pool * 4), add_special_tokens=False)
            for cut in (n, n - 1, n + 1, n - 2, n + 2):
                txt = self.m.tokenizer.decode(ids[:cut]).strip()
                if self.m.to_tokens(f"{txt} {query}").shape[1] == target:
                    return txt
        return None

    def example(self, s):
        """All prompts, spans and vocabularies for one held-out fact (None if unusable)."""
        m = self.m
        gid = self.fid(s["gold"])
        if gid is None: return None
        far = (s["variants"].get("far_irr") or [None])[0]
        lng = (s["variants"].get("long_irr") or [None])[0]
        gol = (s["variants"].get("golden") or [None])[0]
        if far is None or lng is None or gol is None: return None
        q = s["query"]
        ex = {"sid": s["sample_id"], "relation": s["relation"], "gid": gid, "gold": s["gold"], "query": q}
        P = {"bare": q, "far": f"{far['context']} {q}", "long": f"{lng['context']} {q}",
             "golden": f"{gol['context']} {q}",
             "ev_irr": f"{far['context']} {gol['context']} {q}"}
        ctx = {"far": far["context"], "long": lng["context"], "golden": gol["context"],
               "ev_irr": f"{far['context']} {gol['context']}"}
        for c, tag in (("far", "fill_far"), ("long", "fill_long")):
            f = self.matched_filler(ctx[c], q, seed=hash((s["sample_id"], c)) % 10**6)
            if f is None: return None
            P[tag] = f"{f} {q}"; ctx[tag] = f
        ex["prompt"] = P; ex["ctx"] = ctx
        ex["toks"] = {k: m.to_tokens(v) for k, v in P.items()}
        ex["T"] = {k: t.shape[1] for k, t in ex["toks"].items()}
        # prefix = context positions 1..p_last (BOS excluded), as in the frozen screen
        ex["p_last"] = {k: (0 if k == "bare" else min(self.ntok(ctx[k]), ex["T"][k] - 2)) for k in P}
        ex["irr_span_evirr"] = self.ntok(far["context"])           # oracle: irrelevant part of ev_irr
        assert ex["T"]["fill_far"] == ex["T"]["far"] and ex["T"]["fill_long"] == ex["T"]["long"]
        # query regions, per prompt: subject tokens / other query tokens / END
        ex["regions"] = {k: self._regions(P[k], q, s["subject"], ex["T"][k], ex["p_last"][k]) for k in P}
        # vocabularies
        qwords = re.findall(r"[A-Za-z]+", q)
        base_ex = [s["gold"], s["subject"]] + qwords
        lit_far = self.ids_of_words(re.findall(r"[A-Za-z]+", far["context"]), base_ex)
        d = far.get("distractor")
        D_ans = (self.forms(d) - {gid}) if d not in (None, "None") else set()
        lit_far |= D_ans
        lit_long = self.ids_of_words(re.findall(r"[A-Za-z]+", lng["context"]), base_ex)
        assoc_words = []
        cr = REL.get(far.get("context_relation"))
        if d not in (None, "None"):
            if cr and cr.get("range"): assoc_words += category_cloud(d, cr["range"], exclude=[s["gold"]])
            assoc_words += wiki_attr_cloud(d, exclude=[s["gold"]])[:60]
        cs = far.get("context_subject")
        if cs not in (None, "None"):
            assoc_words += wiki_attr_cloud(cs, exclude=[s["gold"]])[:60]
            assoc_words += wiki_attr_cloud(str(cs).rstrip("s"), exclude=[s["gold"]])[:60]
        forbid = set(base_ex) | set(query_category_words(s["relation"])) | \
            set(other_valid_answers(s["relation"], s["gold"], max_n=500)) | \
            set(re.findall(r"[A-Za-z]+", far["context"]))
        D_assoc = self.ids_of_words(assoc_words, forbid) - lit_far - {gid}
        ex["D"] = {"ans": sorted(D_ans), "lit": sorted(lit_far - {gid}), "assoc": sorted(D_assoc),
                   "lit_long": sorted(lit_long - {gid})}
        return ex

    def _regions(self, prompt, query, subject, T, p_last):
        """Masks over positions: bos / prefix / subj / rel / end. Char offsets -> token positions."""
        tok = self.m.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
        offs = tok["offset_mapping"]
        shift = T - len(offs)                                    # 1 when a BOS was prepended
        qs = len(prompt) - len(query)
        cs = prompt.rfind(subject); ce = cs + len(subject)
        R = {k: torch.zeros(T, dtype=torch.bool) for k in ("bos", "prefix", "subj", "rel", "end")}
        R["bos"][:shift] = True
        for i, (a, b) in enumerate(offs):
            p = i + shift
            if p == T - 1: R["end"][p] = True
            elif p <= p_last and p_last > 0: R["prefix"][p] = True
            elif cs >= qs and a < ce and b > cs: R["subj"][p] = True
            else: R["rel"][p] = True
        tot = sum(r.long() for r in R.values())
        assert int(tot.min()) == 1 and int(tot.max()) == 1, "regions must partition positions"
        return R

    # ---------------------------------------------------------------- readouts
    def readout(self, logits_end, ex, lit_key="lit"):
        """Final-distribution readouts at END. logits_end: [vocab] tensor (any dtype)."""
        x = logits_end.float(); lp = torch.log_softmax(x, -1); p = lp.exp()
        g = ex["gid"]
        o = x.clone(); o[g] = -1e9
        out = {"gold": float(x[g]), "logp": float(lp[g]), "margin": float(x[g] - o.max()),
               "acc": float(int(x.argmax()) == g)}
        for k in ("ans", "lit", "assoc", "lit_long"):
            ids = ex["D"][k]
            if ids:
                t = torch.tensor(ids, device=x.device)
                out[f"P_{k}"] = float(p[t].sum())
                out[f"m_vs_{k}"] = float(x[g] - x[t].max())
        return out

    def dla(self, o, ids):
        """o [..., d_model] head writes -> mean raw DLA over token ids (fold_ln convention)."""
        if not ids: return None
        return (o @ self.WUf[:, list(ids)]).mean(-1)

    def values(self, cache, L):
        v = cache["v", L][0].float()                             # [T, nKV, dh]
        return v.repeat_interleave(self.REP, dim=1) if v.shape[1] != self.nH else v


def byL(heads):
    d = defaultdict(list)
    for L, h in heads: d[L].append(h)
    return d


# ---------------------------------------------------------------- batched interventions
@torch.no_grad()
def run_variants(xp, toks, variants, grab=None, chunk=48):
    """Run one prompt under several interventions in a single batch (noise-matched to its own base).

    variant = dict with any of
        z    {L: (heads LongTensor, values [len(heads), d_head])}   patch hook_z at END
        q    {L: (heads LongTensor, values [len(heads), d_head])}   patch hook_q at END (pre-rotary)
        mask {L: (heads LongTensor, positions LongTensor)}         pre-softmax -1e9, END row only
    grab = {L: heads LongTensor}: capture z[END] and pattern[END] for those heads in every element.
    Returns logits at END [B, vocab] (float32) and, if grab, {L: (z [B,h,dh], pat [B,h,T])}.
    """
    m = xp.m
    chunk = int(os.environ.get("XP_CHUNK", chunk))      # smaller batches bound activation memory
    outs, grabs = [], defaultdict(list)
    for c0 in range(0, len(variants), chunk):
        part = variants[c0:c0 + chunk]
        B, T = len(part), toks.shape[1]
        tb = toks.expand(B, T).contiguous()
        zp, qp, mp = defaultdict(list), defaultdict(list), defaultdict(list)
        for b, v in enumerate(part):
            for L, (hs, vals) in v.get("z", {}).items(): zp[L].append((b, hs, vals))
            for L, (hs, vals) in v.get("q", {}).items(): qp[L].append((b, hs, vals))
            for L, (hs, pos) in v.get("mask", {}).items(): mp[L].append((b, hs, pos))
        hooks = []
        for L, items in zp.items():
            def fz(x, hook, items=items):
                for b, hs, vals in items: x[b, -1, hs, :] = vals.to(x.dtype)
                return x
            hooks.append((f"blocks.{L}.attn.hook_z", fz))
        for L, items in qp.items():
            def fq(x, hook, items=items):
                for b, hs, vals in items: x[b, -1, hs, :] = vals.to(x.dtype)
                return x
            hooks.append((f"blocks.{L}.attn.hook_q", fq))
        for L, items in mp.items():
            def fm(x, hook, items=items):
                for b, hs, pos in items:
                    sub = x[b, :, -1, :]                     # view [H, Tk]
                    sub[hs[:, None], pos[None, :]] = -1e9
                return x
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", fm))
        store = {}
        if grab:
            for L, hs in grab.items():
                def gz(x, hook, L=L, hs=hs):
                    store[("z", L)] = x[:, -1, hs, :].float().clone(); return x
                def gp(x, hook, L=L, hs=hs):
                    store[("p", L)] = x[:, hs, -1, :].float().clone(); return x
                hooks += [(f"blocks.{L}.attn.hook_z", gz), (f"blocks.{L}.attn.hook_pattern", gp)]
        lg = m.run_with_hooks(tb, fwd_hooks=hooks)
        outs.append(lg[:, -1].float())
        if grab:
            for L in grab: grabs[L].append((store[("z", L)], store[("p", L)]))
        del lg
    logits = torch.cat(outs)
    if grab:
        return logits, {L: (torch.cat([a for a, _ in v]), torch.cat([b for _, b in v])) for L, v in grabs.items()}
    return logits


def lt(xs, dev):
    return torch.tensor(sorted(xs), device=dev, dtype=torch.long)
