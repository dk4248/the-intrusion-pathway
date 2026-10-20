"""Entrainment heads, by a faithful port of Niu et al. (2025), github.com/frankniujc/entrainment.

Everything below is taken from `head_search/head_analysis.py`, `head_search/data/lre_dataset.py`
and `head_search/circuit_lms/transformer_blocks.py`; deviations are listed explicitly. Relation files
are read from external/lre_dataset (the LRE release, byte-identical to the copy in their repo).

DATA (lre_dataset.get_headsearch_data) -- SAME-relation contexts whose object is a competing answer:
    for query (s, o) and context sample (s', o') of the SAME relation with o != o':
        context = prompt_template.format(s') + ' ' + o' + '.'     e.g. "The capital city of France is Paris."
        query   = prompt_template_zs.format(s)                    e.g. "The capital of Germany is the city of"
        gold = o (Berlin), dstr = o' (Paris)
    combos capped at 100 per split; x every (template, zs-template) pair; 90/10 sample split into
    train/test, then 90/10 of train into train/dev.
MASK (transformer_blocks.apply_mask / gumbel_sigmoid / HeadTransformerBlock.init_masks):
    one logit per head, init N(0, 0.01); applied to z at EVERY position (masked_z = z * mask)
    training: gumbel-sigmoid, hard 0/1 forward with straight-through gradient
    evaluation: deterministic, head ON iff logit > 0
OBJECTIVE (head_analysis.search):
    loss = CE([gold_logit, dstr_logit], target=gold) - lambda * mean(sigmoid(mask_logits)),  lambda=1
    AdamW(lr=1.0) (default weight decay 0.01), batch 16, up to 500 epochs
EVALUATION (head_analysis.evaluate): acc = P(gold) > P(dstr); avg logit delta = gold - dstr;
    KL of masked output vs the un-masked with-context and no-context runs.
ENTRAINMENT SET E = heads switched OFF at the selected epoch (native budget, not fixed at k).

Deviations, all deliberate:
  1. search() reads logits[:, -1] after right padding, which lands on a pad token for any shorter
     prompt in the batch. evaluate() in the same file reads lengths-1. The last REAL position is used
     everywhere here -- that is the evident intent.
  2. The published epoch (406) has no documented selection rule. Selection here is on DEV:
     highest dev acc, then highest mask density (fewest heads off), then latest epoch. The whole
     trajectory is saved so a different rule can be applied afterwards.
  3. Their token-id helper has no pythia branch; the first token of ' '+word is used for all models,
     which matches their llama-3.1 and gpt2 branches.
  4. Dev/test evaluated every `--eval_every` epochs instead of every epoch (compute).
"""
import argparse, json, sys, random, math, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch, torch.nn.functional as F
from common import load, true_lengths

ROOT = Path(__file__).resolve().parent.parent
LRE_ROOT = ROOT.parent / "external" / "lre_dataset"      # LRE release (Hernandez et al., 2024)
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda:0")
ap.add_argument("--relation", default="country_capital_city")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=500)
ap.add_argument("--eval_every", type=int, default=5)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1.0)
ap.add_argument("--pad_bug", action="store_true",
                help="replicate head_analysis.search(): read logits[:, -1] after right padding in the "
                     "TRAINING loss (evaluation still reads the true last position, as theirs does)")
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
m = load(a.model, a.device); nL, nH = m.cfg.n_layers, m.cfg.n_heads
for p in m.parameters(): p.requires_grad_(False)
if m.tokenizer.pad_token is None: m.tokenizer.pad_token = m.tokenizer.eos_token

# ---------------- data (port of get_headsearch_data) ----------------
rel = json.load(open(glob.glob(str(LRE_ROOT / "*" / f"{a.relation}.json"))[0]))
rng = random.Random(a.seed); nprng = np.random.RandomState(a.seed)
def tid(w):
    i = m.tokenizer.encode(" " + str(w), add_special_tokens=False); return i[0] if i else None
def get_ds(ss):
    combs = [(s["subject"], s["object"], c["subject"], c["object"])
             for s in ss for c in rel["samples"] if s["object"] != c["object"]]
    if len(combs) > 100: combs = rng.sample(combs, 100)
    out = []
    for zs in rel["prompt_templates_zs"]:
        for t in rel["prompt_templates"]:
            for subj, obj, csubj, cobj in combs:
                g, d = tid(obj), tid(cobj)
                if g is None or d is None or g == d: continue
                q = zs.format(subj)
                out.append(dict(wctx=f"{t.format(csubj)} {cobj}. {q}", nctx=q, gold=g, dstr=d))
    return out
perm = nprng.permutation(len(rel["samples"])); nte = max(1, math.ceil(0.1 * len(perm)))
test_s = [rel["samples"][i] for i in perm[:nte]]; train_s = [rel["samples"][i] for i in perm[nte:]]
td = get_ds(train_s); rng.shuffle(td); ndev = max(1, math.ceil(0.1 * len(td)))
DEV, TRAIN, TEST = td[:ndev], td[ndev:], get_ds(test_s)
print(f"  {a.relation}: train {len(TRAIN)} | dev {len(DEV)} | test {len(TEST)} prompts")

# ---------------- mask (port of apply_mask / gumbel_sigmoid) ----------------
theta = torch.nn.Parameter(torch.randn(nL, nH, device=a.device) * 0.01)
def gumbel_sigmoid(logits, eps=1e-10):
    u = logits.new_empty([2] + list(logits.shape)).uniform_(0, 1)
    noise = -((u[1] + eps).log() / (u[0] + eps).log() + eps).log()
    res = torch.sigmoid(logits + noise)
    return ((res > 0.5).type_as(res) - res).detach() + res
def hooks(mask):
    return [(f"blocks.{L}.attn.hook_z",
             (lambda L=L: lambda z, hook: z * mask[L].view(1, 1, -1, 1).to(z.dtype))())
            for L in range(nL)]
def batches(rows, shuffle):
    idx = list(range(len(rows)))
    if shuffle: rng.shuffle(idx)
    for i in range(0, len(idx), a.bs): yield [rows[j] for j in idx[i:i + a.bs]]
def last_logits(texts, mask=None, bug=False):
    toks = m.to_tokens(texts, padding_side="right")
    lens = true_lengths(m, toks)
    lg = m.run_with_hooks(toks, fwd_hooks=hooks(mask)) if mask is not None else m(toks)
    pos = torch.full_like(lens, toks.shape[1] - 1) if bug else lens - 1
    return lg[torch.arange(len(texts)), pos].float()

with torch.no_grad():
    ref = {}
    for nm, rows in (("dev", DEV), ("test", TEST)):
        w, n = [], []
        for b in batches(rows, False):
            w.append(last_logits([r["wctx"] for r in b]).cpu())
            n.append(last_logits([r["nctx"] for r in b]).cpu())
        ref[nm] = (torch.cat(w), torch.cat(n))
def kl(p_logits, q_logits):
    return float(F.kl_div(p_logits.log_softmax(-1), q_logits.log_softmax(-1),
                          log_target=True, reduction="none").sum(-1).mean())
@torch.no_grad()
def evaluate(rows, nm):
    det = (theta > 0).float()
    outs, acc, dl = [], [], []
    for b in batches(rows, False):
        lg = last_logits([r["wctx"] for r in b], det)
        g = torch.tensor([r["gold"] for r in b], device=lg.device)
        d = torch.tensor([r["dstr"] for r in b], device=lg.device)
        ar = torch.arange(len(b)); pr = lg.softmax(-1)
        acc += (pr[ar, g] > pr[ar, d]).tolist(); dl += (lg[ar, g] - lg[ar, d]).tolist()
        outs.append(lg.cpu())
    o = torch.cat(outs)
    return dict(acc=sum(acc) / len(acc), logit_delta=sum(dl) / len(dl),
                kl_wctx=kl(o, ref[nm][0]), kl_nctx=kl(o, ref[nm][1]),
                density=int(det.sum()), off=[[int(L), int(h)] for L, h in (det == 0).nonzero().tolist()])

@torch.no_grad()
def evaluate_unmasked(rows):
    acc, dl = [], []
    for b in batches(rows, False):
        lg = last_logits([r["wctx"] for r in b])
        g = torch.tensor([r["gold"] for r in b], device=lg.device)
        d = torch.tensor([r["dstr"] for r in b], device=lg.device)
        ar = torch.arange(len(b)); pr = lg.softmax(-1)
        acc += (pr[ar, g] > pr[ar, d]).tolist(); dl += (lg[ar, g] - lg[ar, d]).tolist()
    return dict(acc=sum(acc) / len(acc), logit_delta=sum(dl) / len(dl))
# NOTE: the baseline must not use evaluate(), which applies the deterministic mask of the
# random init (theta ~ N(0,0.01) -> ~half the heads OFF) and would report a half-ablated model.
base = evaluate_unmasked(DEV)
print(f"  unmasked dev: acc {base['acc']:.3f}  logit delta {base['logit_delta']:+.2f}"
      f"   (training loss reads {'logits[:, -1] -- pad bug replicated' if a.pad_bug else 'the true last position'})")
opt = torch.optim.AdamW([theta], lr=a.lr)
traj = []
for ep in range(a.epochs):
    for b in batches(TRAIN, True):
        mask = gumbel_sigmoid(theta)
        lg = last_logits([r["wctx"] for r in b], mask, bug=a.pad_bug)
        ar = torch.arange(len(b))
        g = torch.tensor([r["gold"] for r in b], device=lg.device)
        d = torch.tensor([r["dstr"] for r in b], device=lg.device)
        eff = F.cross_entropy(torch.stack([lg[ar, g], lg[ar, d]], -1),
                              torch.zeros(len(b), dtype=torch.long, device=lg.device))
        loss = eff - a.lam * torch.sigmoid(theta).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if ep % a.eval_every == 0 or ep == a.epochs - 1:
        dv, te = evaluate(DEV, "dev"), evaluate(TEST, "test")
        traj.append(dict(epoch=ep, dev=dv, test=te))
        print(f"  ep {ep:3d}  dev acc {dv['acc']:.3f} Δ {dv['logit_delta']:+6.2f} "
              f"KLn {dv['kl_nctx']:.3f} | test acc {te['acc']:.3f} | heads off {nL*nH-dv['density']:4d}",
              flush=True)

best = max(traj, key=lambda r: (r["dev"]["acc"], r["dev"]["density"], r["epoch"]))
E = [tuple(x) for x in best["dev"]["off"]]
print(f"\n=== {a.model} / {a.relation} / seed {a.seed}: selected epoch {best['epoch']} ===")
print(f"  dev acc {best['dev']['acc']:.3f}  test acc {best['test']['acc']:.3f}  "
      f"logit delta {best['test']['logit_delta']:+.2f}  |E| = {len(E)} heads off")
print(f"  E = {sorted(E)}")
out = ROOT / "results" / f"NIU_E_{a.model}_{a.relation}_s{a.seed}{'_padbug' if a.pad_bug else ''}.json"
json.dump(dict(model=a.model, relation=a.relation, seed=a.seed, selected=best, E=sorted(E),
               unmasked_dev=base, trajectory=traj), out.open("w"))
print(f"  -> {out}")
