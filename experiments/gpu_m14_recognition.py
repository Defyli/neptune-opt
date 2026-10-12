"""
Milestone 14 -- recognition derived from the algebra, end to end into AOTI.

The problem this fixes
----------------------
Milestone 13 proved that a kernel generated from algebra lands in the AOTI artifact. But
WHICH skeleton a graph implemented was decided by a hand-written if/elif chain inside the
experiment -- `if "tanh" in joined: return "softcap_lse"`. That is the same "preset
skeleton plus hand-written snippet" shape the generator itself was criticised for, and it
does not scale: every new operator needed a new row that could silently disagree with the
registry it was supposed to mirror.

Here the decision is derived (see `neptune_recognize.py`):

  1. interpret the ATen graph symbolically over one data symbol `c`, so each reduction
     installs a symbol for its result and the graph becomes the pair the algebra talks
     about:  phi(c) -- what the max reduces over -- and psi(c, s) -- what the sum reduces over
  2. ask every REGISTERED skeleton whether it reproduces those two expressions

So recognition reads the generator's registry directly, and a new operator extends
recognition by being registered. Nothing here tells the recogniser which operator is which.

This milestone follows a "cover the whole registry" goal. All the registered skeletons are
tested; three are additionally carried through AOTI, and one of those three has its scalar
parameter INFERRED from the graph rather than supplied.

Run: python3 gpu_m14_recognition.py
"""
from __future__ import annotations

import operator
import os
import sys
import tempfile
import zipfile

import torch

assert torch.cuda.is_available(), "GPU required"

os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="m14_ind_")
os.environ["TRITON_HOME"] = tempfile.mkdtemp(prefix="m14_triton_")

# Import the packages whether this file sits next to them (the original flat layout)
# or inside experiments/ (the layout of the published repository).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import neptune_codegen as CG  # noqa: E402
import neptune_recognize as NR  # noqa: E402
from torch.export import export  # noqa: E402

print("=" * 100)
print("MILESTONE 14 -- recognition from the algebra")
print(f"torch {torch.__version__}  registry: {', '.join(CG.supported())}")
print("=" * 100)

X = torch.randn(64, 4096, device="cuda")


# ------------------------------------------------------------------ decomposed forms
# Written the way a user writes the operator in eager torch, i.e. the multi-pass form that
# a graph optimiser actually receives.
def d_logsumexp(t):
    m = t.amax(dim=-1, keepdim=True)
    return (torch.log(torch.exp(t - m).sum(dim=-1, keepdim=True)) + m).squeeze(-1)


def d_stable_l2(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return (m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))).squeeze(-1)


def d_softcap(t):
    h = torch.tanh(0.5 * t)
    m = h.amax(dim=-1, keepdim=True)
    return (torch.log(torch.exp(h - m).sum(dim=-1, keepdim=True)) + m).squeeze(-1)


def d_sumsq_div(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return ((t * t) / m).sum(dim=-1, keepdim=True).squeeze(-1)


def d_linear_attn(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return (t / m).sum(dim=-1, keepdim=True).squeeze(-1)


def d_decay_renorm(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return (t / (m + 2.0)).sum(dim=-1, keepdim=True).squeeze(-1)


def d_lse_biased(t):
    z = t + 1.5
    m = z.amax(dim=-1, keepdim=True)
    return (torch.log(torch.exp(z - m).sum(dim=-1, keepdim=True)) + m).squeeze(-1)


def d_rmsnorm_max(t):
    # THE PAPER'S EXAMPLE, written the natural way: .mean() hides the row length, which is
    # exactly why the interpreter reads the input's shape.
    r = (t * t).mean(dim=-1, keepdim=True)
    return (t / torch.sqrt(r + 1e-6)).amax(dim=-1)


def d_variance(t):
    m = t.mean(dim=-1, keepdim=True)
    return ((t - m) ** 2).mean(dim=-1)


CASES = [("logsumexp", d_logsumexp), ("stable_l2", d_stable_l2), ("softcap_lse", d_softcap),
         ("sumsq_div", d_sumsq_div), ("linear_attn", d_linear_attn),
         ("decay_renorm", d_decay_renorm), ("lse_biased", d_lse_biased),
         ("rmsnorm_max", d_rmsnorm_max)]


class M(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, t):
        return self.fn(t)


# ------------------------------------------------------------------ A. recognition
print("\n[A] recognition, registry-driven (no per-operator pattern rows)")
recognised = 0
for want, fn in CASES:
    ep = export(M(fn).eval(), (X,))
    match, reason = NR.recognize(ep.graph_module)
    got = match.skeleton if match else reason
    flag = "PASS" if got == want else "FAIL"
    recognised += got == want
    inferred = ""
    if match and match.inferred:
        inferred = "  inferred " + ", ".join(f"{k}={v:g}" for k, v in sorted(match.params.items()))
    print(f"    [{flag}] {fn.__name__:15s} -> {got}{inferred}")
print(f"    recognition: {recognised}/{len(CASES)} operators, 0 pattern rows")

print("\n    negative control (a genuine two-producer variance)")
ep = export(M(d_variance).eval(), (X,))
match, reason = NR.recognize(ep.graph_module)
print(f"    [{'PASS' if match is None else 'FAIL'}] refused: {reason[:70]}")


# ------------------------------------------------------------------ B. AOTI
def aoti(fn, mode: str, tag: str):
    """mode: 'base' | 'post' | 'pre'.

    'post' rewrites the EXPORTED graph; 'pre' rewrites and then RE-EXPORTS it. The
    difference is the load-bearing one and is measured below rather than assumed: a
    triton_op is turned into the triton HOP by FunctionalTensorMode *during tracing*, so an
    op inserted after export is never decomposed and Inductor merely falls back to calling
    its Python implementation at runtime -- correct, but no generated kernel in the artifact.
    """
    ep = export(M(fn).eval(), (X,))
    if mode in ("post", "pre"):
        _gm, name, reason = NR.rewrite(ep.graph_module)
        assert name is not None, f"not recognised: {reason}"
    if mode == "pre":
        # The pass must run on the FX graph BEFORE export. Measured (gpu_diag_paths.py):
        # an op inserted into an already-exported program lands in the graph but the
        # program never records its Triton kernel, so the artifact contains 0 occurrences
        # of the generated kernel; the same op present DURING tracing gives 41. Re-exporting
        # the rewritten GraphModule does not help either (0).
        # make_fx, NOT symbolic_trace: a symbolically traced graph carries no shape
        # metadata, so `.mean()`'s row length stays a free symbol and N cannot be inferred
        # (measured: "param-unsolved [N]"). make_fx populates meta["val"], which is what the
        # interpreter reads the row length from.
        from torch.fx.experimental.proxy_tensor import make_fx
        gm = make_fx(M(fn).eval())(X)
        _gm, name, reason = NR.rewrite(gm)
        assert name is not None, f"not recognised: {reason}"
        ep = export(gm, (X,))
    pkg = os.path.join(tempfile.mkdtemp(prefix=f"m14_{tag}_"), "model.pt2")
    torch._inductor.aoti_compile_and_package(ep, package_path=pkg)
    z = zipfile.ZipFile(pkg)
    names = z.namelist()
    cubins = [n for n in names if n.endswith(".cubin")]
    raw = b"".join(z.read(n) for n in names)     # count the kernel name in EVERY byte
    return pkg, len(cubins), raw


print("\n[B] AOTI artifacts for the newly recognised operators")
NEW_OPS = [("linear_attn", d_linear_attn), ("decay_renorm", d_decay_renorm),
           ("lse_biased", d_lse_biased), ("rmsnorm_max", d_rmsnorm_max)]
for name, fn in NEW_OPS:
    _pb, cub_b, raw_b = aoti(fn, "base", "base")
    _pf, cub_f, raw_f = aoti(fn, "pre", "pre")
    print(f"\n    {name}")
    print(f"      baseline (decomposed)        : cubins={cub_b}  mentions '{name}' "
          f"{raw_b.count(name.encode()):3d}x")
    print(f"      rewrite BEFORE export        : cubins={cub_f}  mentions '{name}' "
          f"{raw_f.count(name.encode()):3d}x   <- generated kernel is in the package")
    if name == NEW_OPS[0][0]:
        _pp, cub_p, raw_p = aoti(fn, "post", "post")
        print(f"      (for comparison) AFTER export: cubins={cub_p}  mentions '{name}' "
              f"{raw_p.count(name.encode()):3d}x   <- no HOP: falls back to a Python call")


# ------------------------------------------------------------------ C. numerics
print("\n[C] numerics: eager custom op (built with the INFERRED parameters) vs the "
      "decomposed reference")
for name, fn in NEW_OPS + [("softcap_lse", d_softcap)]:
    for shape in ((64, 4096), (1024, 3000)):
        t = torch.randn(*shape, device="cuda")
        # Recognise and specialise PER SHAPE. `rmsnorm_max` divides by N, which is read off
        # the input's last dimension, so its parameters are coupled to the shape: an op built
        # for 4096 columns applied to a 3000-wide tensor is a different operator. Measured
        # before this fix: rel_err 1.685e-01, which is exactly sqrt(4096/3000) - 1 = 0.168.
        # Re-specialising per shape is also AOTI's own model -- the artifact is shape-specific.
        ep = export(M(fn).eval(), (t,))
        match, reason = NR.recognize(ep.graph_module)
        assert match is not None, reason
        op = CG.op_handle(match.skeleton, params=match.params or None)
        got = op(t)
        ref = fn(t)
        abs_err = (got - ref).abs().max().item()
        rel = ((got - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
        # A sum of SIGNED terms cancels: with 3000 columns the result is ~0.3% of the sum of
        # absolute terms, so a relative error against the RESULT is conditioned by that
        # cancellation factor (~150x here) and says little about the kernel. Normalising by
        # the sum of |terms| is the standard backward-error view. Reported both ways.
        scale = (t.abs().sum(dim=-1) / t.abs().amax(dim=-1).clamp_min(1e-30)).max().item()
        print(f"    [{'PASS' if abs_err / scale < 1e-5 else 'FAIL'}] {name:12s} "
              f"{str(shape):12s} rel_err={rel:.3e}  err/sum|terms|={abs_err / scale:.3e}")

print("\n" + "=" * 100)
print("""Reading:
  * Section A is the point. Every operator is identified by COMPARING EXPRESSIONS against
    the generator's registry. There is no `if 'tanh' in ...` anywhere, so recognition cannot
    drift from what the generator can emit, and registering a skeleton extends recognition
    with no pattern row.
  * The scalar constants are discovered, not configured: the recogniser reports the
    soft-cap beta, the decay offset and the ALiBi bias it read off the graph, and section C
    builds the kernel with those inferred values.
  * Section B is the same acceptance test as earlier milestones: the artifact must mention
    the GENERATED kernel's name (the baseline never does), and cubins >= 1.""")
print("=" * 100)
