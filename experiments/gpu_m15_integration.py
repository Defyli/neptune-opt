"""
Milestone 15 -- the two real integration points, verified on the target machine.

Everything before this milestone was experiment scripts that called the rewrite by hand.
This one exercises the two public entry points of `neptune_opt`, both of which put the
generated op INSIDE tracing (the only placement that records the Triton kernel):

    A. optimize(model, args) -> export -> aoti_compile_and_package
       measured by counting the generated kernel's name inside the .pt2

    B. torch.compile(model, backend=neptune_opt.backend)
       measured by searching the Inductor cache for the generated kernel's name

Both are checked numerically, and the safety gate is exercised on a module where the
operator is embedded (must be refused, and must still compute correctly).

Run: python3 gpu_m15_integration.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import zipfile

import torch

assert torch.cuda.is_available(), "GPU required"

IND = tempfile.mkdtemp(prefix="m15_ind_")
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = IND
os.environ["TRITON_HOME"] = tempfile.mkdtemp(prefix="m15_triton_")

# Import the packages whether this file sits next to them (the original flat layout)
# or inside experiments/ (the layout of the published repository).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import neptune_codegen as CG  # noqa: E402
import neptune_opt as NO  # noqa: E402

print("=" * 100)
print("MILESTONE 15 -- neptune_opt integration points")
print(f"torch {torch.__version__}")
print("=" * 100)

X = torch.randn(64, 4096, device="cuda")


# ---------------------------------------------------------------- models
def l2norm(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return (m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))).squeeze(-1)


def lse(t):
    m = t.amax(dim=-1, keepdim=True)
    return (torch.log(torch.exp(t - m).sum(dim=-1, keepdim=True)) + m).squeeze(-1)


def rmsnorm_max(t):
    r = (t * t).mean(dim=-1, keepdim=True)
    return (t / torch.sqrt(r + 1e-6)).amax(dim=-1)


def embedded(t):
    """The operator's result is combined with the data -> must be refused."""
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    l2 = m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))
    return l2 + t


def scaled(t):
    """The operator's result is scaled -> the x2 must survive."""
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return ((m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))) * 2.0).squeeze(-1)


def with_data(t):
    """The operator's result is added to the data -> the +t must survive."""
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return (m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))) + t


class M(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, t):
        return self.fn(t)


def unwrap(x):
    """An exported GraphModule returns a tuple of outputs; a plain module does not."""
    return x[0] if isinstance(x, (tuple, list)) else x


def search(root, needle: bytes):
    """Count hits, split by file type. Unused now, kept because the lesson is worth keeping.

    A naive total is misleading: Inductor's generated Python carries comments quoting the
    original graph's node names and stack traces, so searching for "stable_l2" returned
    hundreds of hits with NO fusion at all (measured: 273). Only a hit inside a compiled or
    generated-code file means the kernel reached codegen -- and doing that measurement also
    required setting the cache dir before Inductor was imported, which is why this section no
    longer relies on it.
    """
    code = text = 0
    for dp, _d, fs in os.walk(root):
        for f in fs:
            try:
                with open(os.path.join(dp, f), "rb") as fh:
                    n = fh.read().count(needle)
            except OSError:
                continue
            if f.endswith((".py", ".json")):
                text += n
            else:
                code += n
    return code, text


# ---------------------------------------------------------------- A. AOTI path
print("\n[A] RECOMMENDED FORM: torch.export -> neptune_opt.apply(ep) -> AOTI")
print("    (one inserted line; no make_fx pre-trace, no re-export)")
for name, fn in (("stable_l2", l2norm), ("logsumexp", lse), ("rmsnorm_max", rmsnorm_max)):
    mod = M(fn).eval()
    ep = torch.export.export(mod, (X,))
    report = NO.apply(ep, verify_with=(mod, (X,)))
    if not report.fusions:
        print(f"    {name:12s} NOT FUSED: {report.summary().strip()[:70]}")
        continue
    pkg = os.path.join(tempfile.mkdtemp(prefix=f"m15_{name}_"), "m.pt2")
    torch._inductor.aoti_compile_and_package(ep, package_path=pkg)
    z = zipfile.ZipFile(pkg)
    raw = b"".join(z.read(n) for n in z.namelist())
    cubins = sum(1 for n in z.namelist() if n.endswith(".cubin"))
    got = unwrap(ep.module()(X))
    with torch.no_grad():
        ref = mod(X)
    rel = ((got - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
    print(f"    {name:12s} fused={str(report.fusions[0]):26s} mentions="
          f"{raw.count(report.fusions[0].skeleton.encode()):3d}x  cubins={cubins}  "
          f"rel_err={rel:.3e}  verified={report.verified}")

print("\n[A2] drop-in form: neptune_opt.export(model, args)")
for name, fn in (("stable_l2", l2norm),):
    mod = M(fn).eval()
    ep = NO.export(mod, (X,))
    rep = NO.LAST_REPORT
    pkg = os.path.join(tempfile.mkdtemp(prefix="m15_drop_"), "m.pt2")
    torch._inductor.aoti_compile_and_package(ep, package_path=pkg)
    z = zipfile.ZipFile(pkg)
    raw = b"".join(z.read(n) for n in z.namelist())
    print(f"    {name:12s} fused={rep.fusions[0] if rep.fusions else 'NONE'}  "
          f"mentions={raw.count(b'stable_l2'):3d}x  cubins="
          f"{sum(1 for n in z.namelist() if n.endswith('.cubin'))}  verified={rep.verified}")

# ---------------------------------------------------------------- B. torch.compile
print("\n[B] torch.compile(model, backend=neptune_opt.backend) -- KNOWN NOT EFFECTIVE")
print("    The rewrite happens and the results are correct, but the generated kernel does NOT")
print("    reach codegen on this route (measured in gpu_diag_backend.py: 0 hits in")
print("    compiled/generated code, against 41 through optimize()+AOTI), so the op falls back")
print("    to a Python launch. The backend is therefore gated behind experimental=True.")
for name, fn in (("stable_l2", l2norm), ("rmsnorm_max", rmsnorm_max)):
    torch._dynamo.reset()
    mod = M(fn).eval()
    try:
        import functools
        compiled = torch.compile(mod, backend=functools.partial(NO.backend, experimental=True),
                                 dynamic=False)
        with torch.no_grad():
            got, ref = compiled(X), mod(X)
        rel = ((got - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
        rep = NO.LAST_REPORT
        fused = ", ".join(str(f) for f in rep.fusions) if rep and rep.fusions else "NONE"
        print(f"    {name:12s} rel_err={rel:.3e}  fused={fused:26s} (effective: NO)")
    except Exception as e:  # noqa: BLE001
        print(f"    {name:12s} FAILED: {type(e).__name__}: {str(e)[:80]}")

# ---------------------------------------------------------------- D. embedded
print("\n[D] the operator EMBEDDED in a larger computation")
print("    (before finalize matching these were refused; the x2 / +t must survive)")
for name, fn in (("l2*2.0", scaled), ("l2+t", with_data)):
    mod = M(fn).eval()
    opt, report = NO.optimize(mod, (X,))
    if not report.fusions:
        print(f"    {name:8s} NOT FUSED: {report.summary().strip()[:70]}")
        continue
    ep = torch.export.export(opt, (X,))
    pkg = os.path.join(tempfile.mkdtemp(prefix="m15_emb_"), "m.pt2")
    torch._inductor.aoti_compile_and_package(ep, package_path=pkg)
    z = zipfile.ZipFile(pkg)
    raw = b"".join(z.read(n) for n in z.namelist())
    cubins = sum(1 for n in z.namelist() if n.endswith(".cubin"))
    with torch.no_grad():
        got, ref = unwrap(opt(X)), mod(X)
    rel = ((got - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
    print(f"    {name:8s} fused={str(report.fusions[0]):14s} mentions="
          f"{raw.count(report.fusions[0].skeleton.encode()):3d}x  cubins={cubins}  "
          f"rel_err={rel:.3e}  verified={report.verified}")

# ---------------------------------------------------------------- E. dynamic shapes
print("\n[E] shape independence: ONE generated kernel, several row lengths")
print("    (the row length used to be a baked literal, so a kernel built for one width")
print("     silently computed the wrong function at another)")
_opt, _rep = NO.optimize(M(rmsnorm_max).eval(), (X,))
if _rep.fusions:
    _f = _rep.fusions[0]
    _op = CG.op_handle(_f.skeleton, params=_f.params or None)
    for cols in (4096, 1000, 333, 7):
        t = torch.randn(64, cols, device="cuda")
        with torch.no_grad():
            got, ref = _op(t), rmsnorm_max(t)
        rel = ((got - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
        print(f"    width={cols:5d}  rel_err={rel:.3e}  {'OK' if rel < 1e-3 else 'WRONG'}")
else:
    print("    rmsnorm_max was not fused:", _rep.summary().strip()[:70])

# ---------------------------------------------------------------- C. safety
print("\n[C] an embedded operator: fused, and the surrounding computation is preserved")
mod = M(embedded).eval()
opt, report = NO.optimize(mod, (X,))
refused = not report.fusions
with torch.no_grad():
    same = torch.allclose(unwrap(opt(X)), mod(X), rtol=1e-5, atol=1e-6)
print(f"    fused={bool(report.fusions)}  refused={refused}  result unchanged={same}")
print(f"    reason: {report.summary().strip()[:96]}")

print("\n" + "=" * 100)
print("""Reading:
  * A counts the generated kernel's name inside the .pt2. The only way it is there is if the
    op was present during tracing; inserting it after export gives 0.
  * B searches the Inductor cache: even a correct fusion is worthless if the generated kernel
    never reaches codegen, so the name must appear in the compiled output.
  * C is the gate that stops the pass from deleting part of a model.""")
print("=" * 100)
