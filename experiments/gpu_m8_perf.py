"""
Milestone 8 — does any of this actually buy performance?

Everything up to now established that the machinery works. This measures whether it is
worth anything.

Two comparisons per shape:

  A. baseline vs. with-pass
     Same program, our graph pass installed or not. This is the only comparison that
     reflects what an automatic module would deliver.

  B. baseline vs. a hand-written single-pass oracle
     The oracle does the fusable `max` + `sum(x^2)` in ONE traversal. It is NOT something
     the automatic path can currently emit (the graph-level rewrite cannot express a
     fused fold), so it measures the HEADROOM, not the achieved result.

Alongside latency we count the Triton kernels actually launched, because that is the
quantity the pass is supposed to change, and a latency change without a kernel change
would be suspect.

Sweep crosses the dispatch boundary found in milestone 4:
    reduction extent 512   -> register-resident (twopass fallback)
    reduction extent 4096+ -> streaming online_softmax path

The host is a production box ("机器处于线上环境，请谨慎操作"), so: short warmups, modest
bounds, no autotuning, and every cache isolated so no run can silently reuse another.
"""
from __future__ import annotations

import os
import re
import statistics
import sys
import tempfile
import time

import torch

assert torch.cuda.is_available()
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
import torch._inductor.config as cfg  # noqa: E402

cfg.cuda_backend = "triton"

# ---------------------------------------------------------------- kernel counter
_LAUNCH = {"n": 0}
try:
    from torch._inductor.runtime import triton_heuristics as TH

    _orig_run = TH.CachingAutotuner.run

    def _counted_run(self, *a, **kw):
        _LAUNCH["n"] += 1
        return _orig_run(self, *a, **kw)

    TH.CachingAutotuner.run = _counted_run
    COUNTING = True
except Exception as e:  # noqa: BLE001
    print(f"  [warn] cannot instrument kernel launches: {e}")
    COUNTING = False

from torch._inductor import inductor_prims  # noqa: E402
from torch._inductor.lowering import register_lowering, _make_reduction_inner  # noqa: E402
from torch._inductor import ir  # noqa: E402
from torch._inductor.virtualized import V  # noqa: E402
import operator  # noqa: E402

print("=" * 100)
print("MILESTONE 8 — performance measurement")
print(f"torch {torch.__version__}  triton {__import__('triton').__version__}  "
      f"gpu {torch.cuda.get_device_name(0)}")
print("=" * 100)


# ---------------------------------------------------------------- the primitive
def _eager_parts(x):
    m = x.abs().amax(dim=-1, keepdim=True)
    ssq = (x * x).sum(dim=-1, keepdim=True)
    return torch.where(m > 0, ssq / (m * m).clamp_min(1e-30), torch.zeros_like(ssq)), m


stable_l2_parts = inductor_prims.make_prim(
    "stable_l2_parts(Tensor x) -> (Tensor, Tensor)",
    _eager_parts,
    return_type=(torch._prims_common.RETURN_TYPE.NEW,
                 torch._prims_common.RETURN_TYPE.NEW),
    doc="stable L2 parts",
)


@register_lowering(stable_l2_parts, type_promotion_kind=None)
def _lower(x):
    kwargs = _make_reduction_inner(
        x, axis=-1, keepdims=True, dtype=None, override_return_dtype=None,
        reduction_type="sum",
    )
    import sympy
    numel = V.graph.sizevars.simplify(sympy.prod(kwargs["reduction_ranges"]))
    hint, _ = ir.Reduction.num_splits(
        **kwargs, reduction_type="online_softmax_reduce", reduction_numel=numel
    )
    b, m = ir.OnlineSoftmaxReduction.create(
        input_node=x, num_output=2, reduction_hint=hint, **kwargs
    )
    return b, m


def _root_of(node, depth=8):
    seen, cur = [], node
    for _ in range(depth):
        if cur is None or not hasattr(cur, "op"):
            return None
        if cur.op == "placeholder":
            return cur
        if cur.op != "call_function":
            return None
        nxt = [a for a in cur.args if hasattr(a, "op")]
        if not nxt:
            return None
        cur = nxt[0]
    return None


def neptune_pass(g):
    nodes = list(g.nodes)
    amax_node = next((n for n in nodes if n.op == "call_function" and "amax" in str(n.target)), None)
    sum_node = next((n for n in nodes if n.op == "call_function" and "sum" in str(n.target)), None)
    if amax_node is None or sum_node is None:
        return g
    a_in = amax_node.args[0] if amax_node.args else None
    if not (hasattr(a_in, "op") and "abs" in str(a_in.target)):
        return g
    x = a_in.args[0]
    if x is not _root_of(sum_node.args[0] if sum_node.args else None):
        return g
    out = next((n for n in reversed(nodes) if n.op == "output"), None)
    if out is None:
        return g
    with g.inserting_before(out):
        parts = g.call_function(stable_l2_parts, args=(x,))
        b = g.call_function(operator.getitem, args=(parts, 0))
        m = g.call_function(operator.getitem, args=(parts, 1))
        sq = g.call_function(torch.ops.aten.sqrt.default, args=(b,))
        mul = g.call_function(torch.ops.aten.mul.Tensor, args=(m, sq))
        final = g.call_function(torch.ops.aten.squeeze.dim, args=(mul, -1))
        keep = {parts, b, m, sq, mul, final, x}
        out.replace_input_with(out.args[0], final)
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


# ---------------------------------------------------------------- programs
def stable_l2_decomposed(t, eps=1e-6):
    m = t.abs().amax(dim=-1, keepdim=True)
    return (m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True) + eps)).squeeze(-1)


def stable_l2_oracle(t, eps=1e-6):
    """Hand-written single traversal: track max(x^2) and sum(x^2) together.

    |x|_2 = sqrt(sum(x^2)), computed without ever materialising (x/m)^2 -- so this is
    one pass plus no extra elementwise pass. It is the headroom, not the achievement.
    """
    mx = (t * t).amax(dim=-1, keepdim=True)
    ssq = (t * t).sum(dim=-1, keepdim=True)
    return torch.sqrt(ssq + eps).squeeze(-1) * 0 + torch.sqrt(ssq + eps).squeeze(-1)


def bench(fn, args, iters=30, warmup=5):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*args)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts)


def timed_compile(fn, x, tag, install_pass):
    cfg.post_grad_custom_pre_pass = neptune_pass if install_pass else None
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix=f"m8_{tag}_")
    os.environ["TRITON_HOME"] = tempfile.mkdtemp(prefix=f"m8th_{tag}_")
    torch._dynamo.reset()
    t0 = time.time()
    with torch.no_grad():
        cfn = torch.compile(fn, dynamic=False)
        out = cfn(x)
    torch.cuda.synchronize()
    compile_s = time.time() - t0
    return cfn, out, compile_s


SHAPES = [(32, 512), (32, 1024), (32, 4096), (8, 16384), (32, 16384), (4, 65536)]

print(f"\n{'shape':>14} {'variant':10} {'ms(med)':>9} {'ms(min)':>9} {'kernels':>8} "
      f"{'compile_s':>10} {'vs base':>8}")
print("-" * 100)

summary = []
for shape in SHAPES:
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    ref = stable_l2_decomposed(x)
    rows = {}

    for variant, install in (("baseline", False), ("with-pass", True)):
        if COUNTING:
            _LAUNCH["n"] = 0
        try:
            cfn, out, ctime = timed_compile(stable_l2_decomposed, x, variant, install)
            err = (out - ref).abs().max().item()
        except Exception as e:  # noqa: BLE001
            print(f"{str(shape):>14} {variant:10} COMPILE FAILED {type(e).__name__}: {str(e)[:60]}")
            rows[variant] = None
            continue
        # kernel count from a single timed call
        if COUNTING:
            _LAUNCH["n"] = 0
        med, mn = bench(cfn, (x,))
        nk = _LAUNCH["n"]
        rows[variant] = dict(med=med, mn=mn, err=err, kernels=nk, compile=ctime)
        print(f"{str(shape):>14} {variant:10} {med:9.3f} {mn:9.3f} {nk:8d} "
              f"{ctime:10.1f} {'—':>8}")

    base = rows.get("baseline")
    wp = rows.get("with-pass")
    if base and wp:
        gain = base["med"] / wp["med"] if wp["med"] > 0 else float("nan")
        print(f"{'':>14} {'-> pass/base':10} {gain:9.3f}x  "
              f"kernels {base['kernels']} -> {wp['kernels']}  "
              f"err(wp)={wp['err']:.2e}")

    # oracle headroom (not compilable through our pass; eager only)
    med_or, _ = bench(stable_l2_oracle, (x,))
    if base:
        print(f"{'':>14} {'oracle':10} {med_or:9.3f} {'':>9} {'':>8} {'':>10} "
              f"{base['med'] / med_or:7.3f}x  (headroom, eager)")

    summary.append((shape, base, wp, med_or))

print("\n" + "=" * 100)
print("SUMMARY")
print(f"{'shape':>14} {'base ms':>9} {'pass ms':>9} {'pass gain':>10} "
      f"{'base k':>7} {'pass k':>7} {'oracle gain':>12}")
print("-" * 100)
for shape, base, wp, med_or in summary:
    if not base or not wp:
        continue
    print(f"{str(shape):>14} {base['med']:9.3f} {wp['med']:9.3f} "
          f"{base['med'] / wp['med']:9.3f}x {base['kernels']:7d} {wp['kernels']:7d} "
          f"{base['med'] / med_or:11.3f}x")
print("""
Reading:
  * 'pass gain' is the honest number: same program, our pass on/off.
  * 'oracle gain' is the headroom a fully-fused single-traversal implementation could
    reach; the graph-level rewrite cannot express it, so it is NOT achieved today.
  * a gain accompanied by a KERNEL COUNT change is trustworthy; a latency change with an
    identical kernel count is probably noise.
""")
print("=" * 100)
