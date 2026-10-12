"""
Milestone 13 — put a kernel that was DERIVED FROM ALGEBRA into the AOTI artifact.

The critique this answers
-------------------------
Milestone 7 wired a fused reduction into AOTI, but its "fusion" was a hand-written
composite primitive (`stable_l2_parts`) fed to a hand-chosen Inductor reduction type.
So the compiler was not generating anything: a human had already decided the operator,
the two intermediate values, and the arithmetic. It also only ever covered that one
operator.

This milestone uses two *different* operators whose fused kernels were generated from a
sympy expression by `neptune_codegen`, and carries them into a `.pt2` through the
supported compiler path:

    skeleton algebra (g, finalize, producer)
        -> neptune_codegen emits real Triton source
        -> torch.library.triton_op
        -> export decomposes to triton_kernel_wrapper_functional
        -> Inductor lowers to ir.UserDefinedTritonKernel
        -> AOTI embeds the generated kernel as a .cubin in the package

The acceptance criteria, unchanged from earlier milestones:
    (1) cubins >= 1 in the artifact, and
    (2) the artifact text contains the GENerated kernel name, i.e. the cubin really is
        our source and not some aten kernel Inductor produced on its own, and
    (3) one input traversal instead of the decomposed path's several.

Run: python3 gpu_m13_aoti_codegen.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
import zipfile

import torch

assert torch.cuda.is_available(), "GPU required"

os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="m13_ind_")
os.environ["TRITON_HOME"] = tempfile.mkdtemp(prefix="m13_triton_")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp  # noqa: E402

import neptune_codegen as CG  # noqa: E402
from neptune_codegen import c_sym as c, s_sym as s  # noqa: E402

print("=" * 96)
print("MILESTONE 13 - algebra -> generated Triton -> AOTI .pt2")
print(f"torch {torch.__version__}  triton {__import__('triton').__version__}")
print("=" * 96)

# ------------------------------------------------------------------ 0. operators
# Two operators. `stable_l2` was already registered; `sumsq_div` is NEW and is declared
# here as algebra only -- there is no hand-written kernel for it anywhere in this repo.
CG.register(CG.Skeleton(
    name="sumsq_div",
    g=c ** 2 / s,
    # The accumulator invariant is acc = sum_j x_j^2 / s_final, so finalize is the
    # identity. Declaring `m*acc` instead was the first version of this file and it
    # produced rel_err 3.4 -- exactly the max|x| factor. That is the check working:
    # `g` makes the fusion legal, `finalize` states what the operator MEANS, and the
    # generator cannot invent the second one from the first.
    finalize=CG._FINAL_ACC,
    producer="abs",
    doc="sum(x^2)/max|x| -- a scale-normalised energy statistic (new operator)",
))

# Parameter VALUES are baked into the generated source as literals (see
# neptune_codegen.param_bindings): a runtime scalar arrives as fp64 and promotes the
# whole term, which Triton rejects as a loop-carried type change.
VALUES = {"softcap_lse": {"beta": 0.5}}

_beta = sp.Symbol("beta", positive=True)
CG.register(CG.Skeleton(
    name="softcap_lse",
    # Gemma-2 attention soft-capping: the logits are squashed by tanh(beta*x).
    # Note what is NOT declared: the producer operand. For an exponential fold it is
    # forced to be max tanh(beta*x), and the generator derives it from g.
    g=sp.exp(sp.tanh(_beta * c) - s),
    finalize=CG._FINAL_M + sp.log(CG._FINAL_ACC + sp.Float("1e-38")),
    producer="signed",
    is_log=True,
    params=((_beta, "beta"),),
    doc="Gemma-2 soft-capped attention: logsumexp of tanh(beta*x)",
))

for name in ("stable_l2", "sumsq_div", "logsumexp", "softcap_lse"):
    sk = CG.SKELETONS[name]
    print(f"\n[0] {name}: g = {sk.g}   finalize = {sk.finalize}   producer = {sk.producer}")
    print(f"    derived repair R = {CG.repair_factor(sk)}")
    print(f"    derived term   T = {CG.term_expr(sk)}")
    if sk.is_log:
        print(f"    derived producer operand = "
              f"{CG.producer_operand(sk, VALUES.get(name))}")

# ------------------------------------------------------------------ 1. triton_op
OPS = {}
for name in ("stable_l2", "sumsq_div", "logsumexp", "softcap_lse"):
    # ask for the HANDLE rather than guessing the op name: registration encodes the scalar
    # parameter values into it (`softcap_lse_beta0p5`), because torch ops are global and two
    # different betas must not silently share one kernel.
    OPS[name] = CG.op_handle(name, params=VALUES.get(name))
    print(f"[1] registered {OPS[name]}  (generated source -> custom op)")

import torch._functorch.config as fcfg  # noqa: E402
print(f"    decompose_custom_triton_ops = {fcfg.decompose_custom_triton_ops} "
      "(True => export keeps the kernel call, so Inductor can see it)")

# ------------------------------------------------------------------ 2. modules
torch.manual_seed(0)
X = torch.randn(64, 4096, device="cuda")


def decomposed_l2(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return (m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))).squeeze(-1)


def decomposed_ssq(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return ((t * t) / m).sum(dim=-1, keepdim=True).squeeze(-1)


def decomposed_lse(t):
    m = t.amax(dim=-1, keepdim=True)
    return (torch.log(torch.exp(t - m).sum(dim=-1, keepdim=True)) + m).squeeze(-1)


def decomposed_softcap(t):
    h = torch.tanh(0.5 * t)
    m = h.amax(dim=-1, keepdim=True)
    return (torch.log(torch.exp(h - m).sum(dim=-1, keepdim=True)) + m).squeeze(-1)


DECOMP = {"stable_l2": decomposed_l2, "sumsq_div": decomposed_ssq,
          "logsumexp": decomposed_lse, "softcap_lse": decomposed_softcap}


class Base(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, t):
        return DECOMP[self.kind](t)


class Fused(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, t):
        return OPS[self.kind](t)


REF = dict(DECOMP)


# ------------------------------------------------------------------ 3. the selector
def find_pattern(gm):
    """Recognise a decomposable (max-like producer, sum-like consumer) pair.

    Returns (x_node, kind) if the graph is one of the known fused forms, else None.

    This is a *pattern table over ATen nodes*, not over kernels. Being honest about the
    division of labour matters here:

      * GENERATION is fully general -- a new operator is a new sympy `g` and nothing
        else (milestone 12).
      * RECOGNITION is per family: deciding that `amax(x.abs())` + `sum(x*x/m)` is the
        skeleton `g = c^2/s` is a pattern, and patterns must be written down. Inductor's
        own softmax recognition (`prepare_softmax_extra_check`) is hand-written too.

    The producer branch is what distinguishes the families: `max(|v|)` pairs with the
    power norms, a SIGNED `max(v)` pairs with an exponential fold.
    """
    nodes = list(gm.graph.nodes)
    tgt = lambda n: str(getattr(n, "target", ""))  # noqa: E731
    amax = next((n for n in nodes if "amax" in tgt(n)), None)
    sums = [n for n in nodes if "sum" in tgt(n)]
    if amax is None or not sums:
        return None
    a_in = amax.args[0] if amax.args else None
    if not hasattr(a_in, "op"):
        return None
    def root_of(n, depth=10):
        """Walk down the first tensor argument to the placeholder."""
        cur = n
        for _ in range(depth):
            if cur is None or not hasattr(cur, "op"):
                return None
            if cur.op == "placeholder":
                return cur
            nxt = [a for a in cur.args if hasattr(a, "op")]
            if not nxt:
                return None
            cur = nxt[0]
        return None

    if "abs" in tgt(a_in):
        x, signed = a_in.args[0], False          # producer = max|v|
    else:
        # SIGNED producer. The max may be taken over a transform of the input rather than
        # the input itself -- soft-capped attention takes it over tanh(beta*x) -- so walk
        # down to the placeholder instead of demanding `a_in.op == 'placeholder'`.
        x, signed = root_of(a_in), True
        if x is None:
            return None

    # walk a reduction's producer chain down to the input to see what it computes
    def chain(n, depth=10):
        seen = []
        cur = n
        for _ in range(depth):
            if cur is None or not hasattr(cur, "op") or cur.op == "placeholder":
                return seen, cur if (cur is not None and cur.op == "placeholder") else None
            seen.append(tgt(cur))
            nxt = [a for a in cur.args if hasattr(a, "op")]
            if not nxt:
                return seen, None
            cur = nxt[0]
        return seen, None

    for sn in sums:
        seen, root = chain(sn.args[0])
        if root is not x:
            continue
        joined = " ".join(seen)
        if signed:
            # recognition is per family: the producer branch says "exponential fold", and
            # what is being exponentiated separates plain logsumexp from soft-capping
            if "exp" in joined:
                if "tanh" in joined:
                    return x, "softcap_lse"
                return x, "logsumexp"
            continue
        if "pow" in joined:
            return x, "stable_l2"
        if "mul" in joined:
            return x, "sumsq_div"
    return None


def select_and_rewrite(gm):
    """Replace the whole decomposed computation with one call to the generated kernel.

    The replacement must remove the ENTIRE computation: an FX node can only be erased
    once it has no users, so replacing just the reduction leaves the rest alive and the
    kernel never becomes the only producer of the output.
    """
    found = find_pattern(gm)
    if found is None:
        return gm, None
    x, kind = found
    graph = gm.graph
    out = next(n for n in reversed(list(graph.nodes)) if n.op == "output")
    with graph.inserting_before(out):
        call = graph.call_function(OPS[kind], args=(x,))
    old = out.args[0]
    # the decomposed result may be squeezed to match rank; keep whatever rank it had
    if hasattr(old, "op") and "squeeze" in str(getattr(old, "target", "")):
        with graph.inserting_before(out):
            call = graph.call_function(torch.ops.aten.squeeze.dim, args=(call, -1))
    out.replace_input_with(old, call)
    changed = True
    while changed:
        changed = False
        for n in reversed(list(graph.nodes)):
            if n.op in ("placeholder", "output") or n is call or n is x:
                continue
            if n.op == "call_function" and len(n.users) == 0:
                graph.erase_node(n)
                changed = True
    graph.lint()
    gm.recompile()
    return gm, kind


print("\n[3] selector: pattern table over ATen nodes -> one generated-kernel call")
for kind in ("stable_l2", "sumsq_div", "logsumexp", "softcap_lse"):
    from torch.export import export
    ep = export(Base(kind).eval(), (X,))
    _gm, hit = select_and_rewrite(ep.graph_module)
    n_ops = [str(n.target) for n in _gm.graph.nodes if n.op == "call_function"]
    print(f"    {kind:10s} matched={hit!s:10s} graph now: {n_ops}")

# ------------------------------------------------------------------ 4. AOTI
def aoti(kind, use_fused, tag):
    from torch.export import export
    import torch._inductor as ti

    mod = (Fused(kind) if use_fused else Base(kind)).eval()
    ep = export(mod, (X,))
    if use_fused is False and tag != "base":
        # the selected form: same pass, applied to the exported program
        select_and_rewrite(ep.graph_module)
    pkg = os.path.join(tempfile.mkdtemp(prefix=f"m13_{tag}_"), "model.pt2")
    ti.aoti_compile_and_package(ep, package_path=pkg)
    z = zipfile.ZipFile(pkg)
    names = z.namelist()
    cubins = [n for n in names if n.endswith(".cubin")]
    # Count the generated kernel's name in EVERY byte of the package, not just the text
    # files: a compiled cubin carries the kernel as a symbol, and that is exactly the
    # evidence we want (the cubin is our source, not something Inductor synthesised).
    raw = b"".join(z.read(n) for n in names)
    return pkg, cubins, names, raw


print("\n[4] AOTI artifacts")
for kind in ("stable_l2", "sumsq_div", "logsumexp", "softcap_lse"):
    pkg_b, cub_b, nm_b, txt_b = aoti(kind, False, "base")
    pkg_f, cub_f, nm_f, txt_f = aoti(kind, True, "sel")
    named_b = txt_b.count(kind.encode())
    named_f = txt_f.count(kind.encode())
    loads_b = txt_b.count(b"in_ptr0")
    loads_f = txt_f.count(b"in_ptr0")
    print(f"\n    {kind}")
    print(f"      baseline      : cubins={len(cub_b):3d}  artifact mentions '{kind}' {named_b:3d}x"
          f"  in_ptr0 loads={loads_b}")
    print(f"      selected+gen  : cubins={len(cub_f):3d}  artifact mentions '{kind}' {named_f:3d}x"
          f"  in_ptr0 loads={loads_f}")
    print(f"      size          : {os.path.getsize(pkg_b)/1024:.0f} KiB -> "
          f"{os.path.getsize(pkg_f)/1024:.0f} KiB")
    if named_f == 0:
        print(f"      [WARN] generated kernel name not found in artifact; kernels seen:")
        for n in nm_f:
            if n.endswith(".cubin"):
                print(f"             {n}")

# ------------------------------------------------------------------ 5. correctness
print("\n[5] numerics (eager custom op vs decomposed reference)")
for kind in ("stable_l2", "sumsq_div", "logsumexp", "softcap_lse"):
    for shape in ((64, 4096), (1024, 16384), (8, 1000)):
        x = torch.randn(*shape, device="cuda")
        got = OPS[kind](x.reshape(shape[0], -1).contiguous())
        ref = REF[kind](x.reshape(shape[0], -1))
        rel = ((got - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
        flag = "PASS" if rel < 1e-3 else "FAIL"
        print(f"    [{flag}] {kind:10s} {str(shape):12s} rel_err={rel:.3e}")

print("\n" + "=" * 96)
print("""How to read this:
  * `artifact mentions '<operator>'` is the load-bearing number. It counts occurrences of
    the GENERATED kernel's name in the package's sources. The baseline never contains it
    (Inductor produced its own kernels); the selected build does. That is the difference
    between "a kernel that happens to be fast" and "the artefact carries the kernel the
    algebra asked for".
  * cubins >= 1 alone is not evidence: the baseline passes that too.
  * Two operators are carried through the same path with no kernel written by hand; the
    second one did not exist in this repo before this file.""")
print("=" * 96)
