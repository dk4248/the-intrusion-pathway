"""G0.2 — audit the copying diagnostic BEFORE using it to select heads.

Two formulas are in play and they are not the same object.

  WRITTEN (projection form):  rho = <proj_{u_ts}(o_h), u_t*> / <o_h, u_t*>
     With u_t* = u_ts (single-source usage) this is ALGEBRAICALLY 1 for every
     possible o_h, because proj_u(o) has the same inner product with u that o does.

  IMPLEMENTED (per-source form):
     rho_impl = sum_p <hat_p, uhat_p> <uhat_p, u_d>  /  sum_p <hat_p, u_d>
     where hat_p = alpha_p (v_p W_O) is the per-source contribution and uhat_p the unit
     unembedding of the token AT p. This one is a real quantity.

  DIFFUSE (attention-weighted form):  rho_attn = sum_p alpha_p <u_tp, u*> / <o_h, u*>,  u* = sum alpha_p u_tp
     The numerator does not depend on the head at all: it equals ||u*||^2. So rho_attn is
     ||u*||^2 / <o_h, u*>, i.e. an estimate of 1/gain, and it is blind to anything the head
     writes orthogonal to u*.

Ground truth here is the copy fraction ||copy component|| / ||o_h||, which we control by
construction. Token directions are REAL unembedding rows, so they are non-orthogonal as in the
model. No thresholds are taken from this file; it exists to say which readings are meaningful.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from common import load

ROOT = Path(__file__).resolve().parent.parent
m = load("gpt2-xl", "cuda:0")
W_U = m.W_U.float()
torch.manual_seed(0)
d_model = W_U.shape[0]
DEV = W_U.device

# real token directions
ids = [m.tokenizer.encode(" " + w, add_special_tokens=False)[0]
       for w in ["brown", "hammer", "Paris", "green", "river", "music", "table", "Spain"]]
U = torch.stack([W_U[:, i] for i in ids], 0)        # [k, d_model]
Uh = U / U.norm(dim=1, keepdim=True)

def metrics(hat, alpha, u_d, u_star):
    """hat: [k, d_model] per-source contributions (already attention-weighted)."""
    o = hat.sum(0)
    num_impl = sum(float(hat[p] @ Uh[p]) * float(Uh[p] @ u_d) for p in range(len(hat)))
    den_impl = float(o @ u_d)
    rho_impl = num_impl / den_impl if abs(den_impl) > 1e-9 else float("nan")
    num_attn = float(u_star @ u_star)
    den_attn = float(o @ u_star)
    rho_attn = num_attn / den_attn if abs(den_attn) > 1e-9 else float("nan")
    proj = (o @ u_d) / (u_d @ u_d) * u_d
    rho_written = float((proj @ u_d) / (o @ u_d))
    return rho_written, rho_impl, rho_attn

rows = []
def case(name, hat, alpha, true_frac, d_idx=0):
    u_d = Uh[d_idx]
    u_star = (alpha.unsqueeze(1) * Uh).sum(0)
    rw, ri, ra = metrics(hat, alpha, u_d, u_star)
    rows.append(dict(case=name, true_copy_fraction=true_frac,
                     rho_written=rw, rho_impl=ri, rho_attn=ra))

k = len(ids)
# 1. pure single-source copy, all attention on source 0
a = torch.zeros(k, device=DEV); a[0] = 1.0
case("pure single-source copy", (a.unsqueeze(1) * Uh).clone(), a, 1.00)
# 2. same, gain 3
case("single-source copy, gain 3", 3 * (a.unsqueeze(1) * Uh), a, 1.00)
# 3. SUPPRESSION of the same token
case("single-source SUPPRESSION (-1)", -(a.unsqueeze(1) * Uh), a, 1.00)
# 4. weighted multi-source copy
a2 = torch.tensor([.35, .25, .15, .10, .07, .04, .03, .01], device=DEV)
case("multi-source copy (unequal attn)", (a2.unsqueeze(1) * Uh), a2, 1.00)
# 5. multi-source copy, gain 3
case("multi-source copy, gain 3", 3 * (a2.unsqueeze(1) * Uh), a2, 1.00)
# 6. non-copy write (random directions)
R = torch.randn(k, d_model, device=DEV); R = R / R.norm(dim=1, keepdim=True)
case("non-copy write (random dirs)", (a2.unsqueeze(1) * R), a2, 0.00)
# 7. copy + orthogonal noise at 3x the copy norm
cp = (a2.unsqueeze(1) * Uh)
nz = torch.randn(k, d_model, device=DEV); nz = nz / nz.norm(dim=1, keepdim=True) * cp.norm(dim=1, keepdim=True) * 3
case("copy + 3x orthogonal noise", cp + nz, a2, float(cp.norm() / (cp + nz).norm()))
# 8. copies the WRONG token (always source 3's direction)
case("copies the WRONG token", (a2.unsqueeze(1) * Uh[3].expand(k, d_model)), a2, 0.00)

print(f"\n{'construction':34s} {'true copy':>10s} {'rho_written':>12s} {'rho_impl':>10s} {'rho_attn':>10s}")
for r in rows:
    print(f"{r['case']:34s} {r['true_copy_fraction']:10.2f} {r['rho_written']:12.3f} "
          f"{r['rho_impl']:10.3f} {r['rho_attn']:10.3f}")
json.dump(rows, (ROOT / "results" / "G0_2_copy_audit.json").open("w"), indent=1)
print(f"\n-> results/G0_2_copy_audit.json")
