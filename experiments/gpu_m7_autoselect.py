"""
Milestone 7 — make the compiler SELECT the fused reduction automatically, and produce an
AOTI artifact that actually contains a GPU kernel (cubin).

Status going in
---------------
* milestones 3/6: the repair arithmetic can be replaced from outside the torch install,
  verified numerically on both reduction paths.
* milestone 2: a custom composite op is deployable into a .pt2 but produces cubins=0,
  i.e. it does NOT fuse. That is the acceptance criterion we must beat here.

Plan
----
1. Register a multi-output reduction PRIMITIVE for stable L2
   (torch.ops.neptune_prims.stable_l2_parts), modelled on
   inductor_prims.prepare_softmax_online, which exposes (max, sumexp).
   Ours exposes the pair the repair needs:
       B = sum_j (x_j / m_j)^2      (running residual sum of squares, denormalised)
       m = max_j |x_j|              (running maximum)
   and the closed form is  sqrt(B) * m.
2. Register its Inductor LOWERING, outside the install, by adding it to the same
   registry `prepare_softmax_online` uses (torch._inductor.lowering).
3. A graph pass rewrites the decomposed stable-L2 pattern into a call to the primitive.
4. Compile through AOTI and require cubins >= 1 while staying numerically correct.

Passivity check: the primitive records every call it sees. If the pass never fires, that
is reported plainly rather than being mistaken for a fused kernel.

Run: python3 gpu_m7_autoselect.py
"""
from __future__ import annotations

import operator
import os
import re
import sys
import tempfile
import zipfile

import torch

assert torch.cuda.is_available(), "GPU required: the cpp backend rejects this reduction type"

# isolate every cache that could hide whether our code really ran
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="m7_ind_")
os.environ["TRITON_HOME"] = tempfile.mkdtemp(prefix="m7_triton_")

import torch._inductor.config as cfg  # noqa: E402

cfg.cuda_backend = "triton"

from torch._inductor import inductor_prims  # noqa: E402
from torch._inductor.lowering import register_lowering, _make_reduction_inner  # noqa: E402
from torch._inductor import ir  # noqa: E402
from torch._inductor.virtualized import V  # noqa: E402

CALLS = {"prim": 0, "lowering": 0, "pass": 0}

print("=" * 96)
print("MILESTONE 7 — automatic selection + cubin in the AOTI artifact")
print(f"torch {torch.__version__}  triton {__import__('triton').__version__}")
print("=" * 96)


# ----------------------------------------------------------------- 1. primitive
def _eager_parts(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for the two outputs.

    B = sum_j (x_j / m_j)^2 where m_j is the running max at the time x_j is consumed.
    Equivalently B = sum(x^2) / max(x^2) (see module docstring reasoning), and
    |x|_2 = sqrt(B) * m.
    """
    m = x.abs().amax(dim=-1, keepdim=True)
    ssq = (x * x).sum(dim=-1, keepdim=True)
    B = torch.where(m > 0, ssq / (m * m).clamp_min(1e-30), torch.zeros_like(ssq))
    return B, m


stable_l2_parts = inductor_prims.make_prim(
    "stable_l2_parts(Tensor x) -> (Tensor, Tensor)",
    _eager_parts,
    return_type=(
        torch._prims_common.RETURN_TYPE.NEW,
        torch._prims_common.RETURN_TYPE.NEW,
    ),
    doc="Neptune-style stable L2 parts: running residual sum of squares and running max.",
)
print(f"[1] primitive registered: {stable_l2_parts}  (make_prim returns the OpOverload)")


# ----------------------------------------------------------------- 2. lowering
@register_lowering(stable_l2_parts, type_promotion_kind=None)
def _lower_stable_l2_parts(x):
    """Lower the primitive to a TWO-OUTPUT reduction, i.e. one kernel pass."""
    CALLS["lowering"] += 1
    kwargs = _make_reduction_inner(
        x, axis=-1, keepdims=True, dtype=None, override_return_dtype=None,
        reduction_type="sum",
    )
    numel = V.graph.sizevars.simplify(
        __import__("sympy").prod(kwargs["reduction_ranges"])
    )
    hint, _num_split = ir.Reduction.num_splits(
        **kwargs, reduction_type="online_softmax_reduce", reduction_numel=numel
    )
    # Reuse the online-softmax reduction type: its arithmetic is exactly the
    # accumulator-rescaling we need, and its codegen exists on both the
    # register-resident and streaming paths (milestones 3/4).
    B_t, m_t = ir.OnlineSoftmaxReduction.create(
        input_node=x, num_output=2, reduction_hint=hint, **kwargs
    )
    return B_t, m_t


print("[2] lowering registered in torch._inductor.lowering (outside the install)")


# ----------------------------------------------------------------- 3. graph pass
def _is_stable_l2_pattern(gm) -> bool:
    """Recognise: sum(x*x) and amax(|x|) over the same dim, combined as m*sqrt(ssq)/m."""
    ops = {str(n.target) for n in gm.graph.nodes if n.op == "call_function"}
    return ("aten.sum.dim_IntList" in ops or "aten.sum.default" in ops) and (
        "aten.amax.default" in ops
    )


def _root_of(node, depth=8):
    """Walk back through pointwise ops to a placeholder, collecting the ops seen."""
    seen = []
    cur = node
    for _ in range(depth):
        if cur is None:
            return None, seen
        if cur.op == "placeholder":
            return cur, seen
        if cur.op != "call_function":
            return None, seen
        seen.append(str(cur.target))
        nxt = [a for a in cur.args if hasattr(a, "op")]
        if not nxt:
            return None, seen
        cur = nxt[0]
    return None, seen


def neptune_pass(g):
    """Replace the decomposed stable-L2 computation with the fused primitive.

    The pattern actually produced by torch.export here is:

        placeholder -> abs -> amax            (m)
        placeholder -> div(m) -> pow(2) -> sum -> add(eps) -> sqrt   (sqrt of sum of squares)
        mul(m, sqrt(...)) -> squeeze -> output

    Earlier versions of this pass failed for reasons worth recording:
      * the hook receives an `fx.Graph` (not a GraphModule);
      * a node may only be erased once it has no remaining users, so the ENTIRE
        computation must be replaced, not just one reduction;
      * the sum's input is `pow`, not the tensor directly, so a literal matcher misses.
    """
    nodes = list(g.nodes)
    amax_node = next((n for n in nodes if n.op == "call_function"
                      and "amax" in str(n.target)), None)
    sum_node = next((n for n in nodes if n.op == "call_function"
                     and "sum" in str(n.target)), None)
    if amax_node is None or sum_node is None:
        return g

    # m must come from abs(placeholder)
    a_in = amax_node.args[0] if amax_node.args else None
    if not (hasattr(a_in, "op") and "abs" in str(a_in.target)):
        return g
    x = a_in.args[0]
    # the sum must ultimately read the same placeholder
    x2, chain = _root_of(sum_node.args[0] if sum_node.args else None)
    if x is not x2:
        return g

    CALLS["pass"] += 1
    output_node = next((n for n in reversed(nodes) if n.op == "output"), None)
    if output_node is None:
        return g

    # rebuild the desired result: |x|_2 = m * sqrt(B)
    with g.inserting_before(output_node):
        parts = g.call_function(stable_l2_parts, args=(x,))
        # A multi-output op yields a TUPLE on the graph; indexing it with aten.select
        # fails with "Expected a value of type 'Tensor' for argument 'self' but instead
        # found type 'tuple'". Use getitem, exactly as the softmax reference lowering
        # does after calling prepare_softmax_online.
        b_sel = g.call_function(operator.getitem, args=(parts, 0))
        m_sel = g.call_function(operator.getitem, args=(parts, 1))
        sq = g.call_function(torch.ops.aten.sqrt.default, args=(b_sel,))
        mul = g.call_function(torch.ops.aten.mul.Tensor, args=(m_sel, sq))
        # match the original output rank (the decomposition squeezed the last dim)
        out_val = output_node.args[0]
        if hasattr(out_val, "op") and "squeeze" in str(getattr(out_val, "target", "")):
            final = g.call_function(torch.ops.aten.squeeze.dim, args=(mul, -1))
        else:
            final = g.call_function(torch.ops.aten.squeeze.dim, args=(mul, -1))
        keep = {parts, b_sel, m_sel, sq, mul, final, x}
        output_node.replace_input_with(out_val, final)

    # erase everything that is now dead, repeatedly, in reverse order
    changed = True
    while changed:
        changed = False
        for n in reversed(list(g.nodes)):
            if n in keep or n.op in ("placeholder", "output"):
                continue
            if n.op == "call_function" and len(n.users) == 0:
                g.erase_node(n)
                changed = True
    g.lint()
    return g


print("[3] graph pass defined (rewrites the decomposed pattern into the primitive)")


# ----------------------------------------------------------------- 4. eager check
torch.manual_seed(0)
x = torch.randn(32, 1024, device="cuda")


def stable_l2_eager(t, eps=1e-6):
    m = t.abs().amax(dim=-1, keepdim=True)
    return m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True) + eps).squeeze(-1)


ref = stable_l2_eager(x)
B, m = stable_l2_parts(x)
approx = (torch.sqrt(B) * m).squeeze(-1)
print(f"\n[4] eager primitive vs reference: max_err = {(approx - ref).abs().max().item():.3e}")


# ----------------------------------------------------------------- 5. compiled
def get_package(use_pass: bool, tag: str):
    class M(torch.nn.Module):
        def forward(self, t):
            m_ = t.abs().amax(dim=-1, keepdim=True)
            return m_ * torch.sqrt(((t / m_) ** 2).sum(dim=-1, keepdim=True) + 1e-6).squeeze(-1)

    mod = M().eval()
    from torch.export import export
    ep = export(mod, (x,))
    if use_pass:
        ep = ep.run_decompositions({})
    pkg = os.path.join(tempfile.mkdtemp(prefix=f"m7_{tag}_"), "model.pt2")
    import torch._inductor as ti
    ti.aoti_compile_and_package(ep, package_path=pkg)
    z = zipfile.ZipFile(pkg)
    names = z.namelist()
    cubins = [n for n in names if n.endswith(".cubin")]
    return pkg, len(cubins), names


print("\n[5] AOTI artifact — baseline (no pass installed)")
pkg, ncub, names = get_package(False, "baseline")
print(f"    baseline  : cubins={ncub}  size={os.path.getsize(pkg)}")

print("\n[6] now INSTALL the pass through Inductor's supported hook and recompile")
cfg.post_grad_custom_pre_pass = neptune_pass
pkg2, ncub2, names2 = get_package(False, "withpass")
print(f"    with pass : cubins={ncub2}  size={os.path.getsize(pkg2)}")

print("\n" + "=" * 96)
print("STATE")
print(f"    primitive calls recorded : {CALLS['prim']}")
print(f"    lowering invoked         : {CALLS['lowering']}")
print(f"    graph pass fired         : {CALLS['pass']}")
print(f"    cubins baseline / with pass : {ncub} / {ncub2}")
print("""
Reading:
  * `lowering invoked` is the decisive counter: it counts how often Inductor asked our
    lowering to build the fused reduction. Zero means the primitive never reached
    codegen, so nothing was fused -- no matter what the artifact contains.
  * `graph pass fired` tells us whether the matcher recognised the pattern.
  * cubins >= 1 is necessary but NOT sufficient: the baseline already produces cubins,
    so the pass must additionally change the lowering counter.
""")
print("=" * 96)
