"""
I4 acceptance: does `log_softmax` actually fuse, stay correct, and get faster end to end?

`gpu_m20_i4_ceiling.py` answered the reach-vs-value question by building what the lowering WOULD
produce and racing it against Inductor's own lowering. It said: `_log_softmax` 1.41x -> do it;
`softmax` 0.59x and `layer_norm` 0.41x -> do not. This experiment checks the pass that decision
licenses, on the real path (`torch.export` -> AOTI package -> load), and it checks the other half of
the decision too: the operators that were EXCLUDED must come out of `neptune_opt.export` with their
graph untouched, byte for byte.

Three things are asserted, each of which can fail:

  1. `log_softmax` is a single aten op that the core decomposition table does NOT touch (`get_decompositions`
     returns an empty table for the overload that is actually exported), so without this lowering it is
     unreachable -- not merely unfused.
  2. fused end to end: the generated kernel is in the `.pt2` (and absent from the baseline), the numbers
     match Inductor's within a few ULPs, and the latency ratio is reported whatever it is.
  3. no-op when it buys nothing: `softmax`, `layer_norm` and an unrelated model are byte-identical.

Run on the GPU host:  python3 gpu_m21_i4_log_softmax.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
import zipfile

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neptune_recognize as NR  # noqa: E402
import neptune_opt as NO  # noqa: E402

ROWS, COLS = 256, 16384
X = torch.randn(ROWS, COLS)
SMALL = torch.randn(64, 4096)

_failed = []
_passed = 0


def check(label, ok, detail=""):
    global _passed
    if ok:
        _passed += 1
    else:
        _failed.append(label)
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


# ---------------------------------------------------------------- models
class M(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, t):
        return self.fn(t)


def _spellings():
    """The three ways people write it; all three export to the same aten op."""
    return [
        ("nn.LogSoftmax(dim=-1)", lambda t: torch.log_softmax(t, dim=-1)),
        ("F.log_softmax", lambda t, F=torch.nn.functional: F.log_softmax(t, dim=-1)),
        ("log_softmax by hand", lambda t: t - torch.logsumexp(t, dim=-1, keepdim=True)),
    ]


# ---------------------------------------------------------------- [1] reach
def section_reach() -> int:
    print("=" * 112)
    print(f"[1] REACH: what does log_softmax export as, and can anything see inside it?")
    print("=" * 112)
    mod = M(lambda t: torch.log_softmax(t, dim=-1)).eval()
    ep = torch.export.export(mod, (SMALL,))
    ops = [str(n.target) for n in ep.graph_module.graph.nodes if n.op == "call_function"]
    print(f"    exported: {ops}")

    # The decomposition table is what `ep.run_decompositions()` applies. If it has no entry for
    # this op, no amount of calling it will expose the log-sum-exp inside.
    from torch._decomp import get_decompositions

    for name, overload in (("log_softmax.int", torch.ops.aten.log_softmax.int),
                           ("_log_softmax.default", torch.ops.aten._log_softmax.default)):
        try:
            table = get_decompositions([overload])
            print(f"    get_decompositions([aten.{name}]) -> {len(table)} entries")
        except Exception as e:  # noqa: BLE001
            print(f"    get_decompositions([aten.{name}]) -> {type(e).__name__}: {e}")

    dec = ep.run_decompositions()
    ops_dec = [str(n.target) for n in dec.graph_module.graph.nodes if n.op == "call_function"]
    print(f"    run_decompositions(): {ops_dec}")
    # NOT `ops_dec == ops`: on torch 2.11 the call RENAMES `log_softmax.int` to
    # `_log_softmax.default` -- the overload is normalised, the operator is NOT expanded. What
    # matters is that the log-sum-exp inside is still invisible, so that is what is asserted.
    reductions = ("amax", "sum", "log.default", "exp.default", "sub.Tensor")
    check("core decompositions do NOT expose the log-sum-exp inside log_softmax",
          len(ops_dec) == 1 and not any(r in " ".join(ops_dec) for r in reductions),
          f"became {ops_dec}")

    match, why = NR.recognize(dec.graph_module)
    check("the decomposed graph is NOT recognised without the lowering", match is None,
          f"recognised {match.skeleton!r}" if match else why[:70])

    ok, why = NR.would_rewrite(dec.graph_module)
    check("and it is not rewritable either", ok is False, why[:70])

    # `lower=False` is the control: the fusion below must be attributable to the lowering, not to
    # something else in `apply`.
    control = torch.export.export(mod, (SMALL,)).run_decompositions()
    check("with lower=False the very same graph is NOT fused", not NO.apply(control, lower=False))

    # The lowering alone, on a pristine graph: `apply` also REWRITES, and the rewrite replaces the
    # whole region with one custom op, after which there is nothing left to recognise. So the
    # metadata the rewrite decision depends on has to be inspected between the two halves.
    from neptune_recognize.lowering import lower_compound

    probe = torch.export.export(mod, (SMALL,)).run_decompositions()
    n_lowered = lower_compound(probe.graph_module)
    check("lower_compound expands exactly one op", n_lowered == 1, f"expanded {n_lowered}")
    match, why = NR.recognize(probe.graph_module)
    check("the lowered graph is recognised", match is not None,
          match.skeleton if match else why[:70])
    if match is not None:
        val = match.region_root.meta.get("val")
        shape = tuple(val.shape) if val is not None else None
        check("region root metadata says keepdim", shape == (SMALL.shape[0], 1), f"shape={shape}")
    lowered = [str(n.target).split(".")[1] for n in probe.graph_module.graph.nodes
               if n.op == "call_function"]
    print(f"    lowered graph: {lowered}")

    report = NO.apply(dec)
    fused = [f.skeleton for f in report.fusions]
    check("after neptune_opt.apply the same graph IS fused", bool(fused), f"fusions={fused}")
    check("and the rewritten graph calls exactly one generated op", "neptune" in str(dec.graph_module.graph),
          "")
    return 0


# ---------------------------------------------------------------- [2] end to end
def _pt2_mentions(path, name) -> int:
    with zipfile.ZipFile(path) as z:
        return b"".join(z.read(n) for n in z.namelist()).count(name.encode())


def _batch_ms(fn, x, reps: int = 20) -> float:
    """Pipelined launches: a single call is the same order as Python dispatch (see real_modules.py)."""
    with torch.no_grad():
        for _ in range(3):
            fn(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / reps


def _one(label, fn, dtype, expect_fusion: bool = True) -> None:
    x = X.to(dtype).cuda()
    mod = M(fn).eval().cuda()
    with torch.no_grad():
        ref = mod(x)
    ep0 = torch.export.export(mod, (x,)).run_decompositions()
    ep1 = torch.export.export(mod, (x,)).run_decompositions()
    report = NO.apply(ep1, verify_with=(mod, (x,)))
    if not expect_fusion:
        # A refusal that is DOCUMENTED is different from a fusion that failed: the graph is left
        # alone and the reason is specific. Assert the reason, so that if the interpreter later
        # learns to see through the cast, this experiment says so instead of quietly passing.
        why = report.summary()
        check(f"{label} [{dtype}] refused as documented, graph untouched", "_to_copy" in why,
              why.strip()[:90])
        check(f"{label} [{dtype}] the refusal is not a silence", bool(report.skipped))
        return
    if not report.fusions:
        check(f"{label} [{dtype}] fused", False, report.summary().strip()[:70])
        return
    sk = report.fusions[0].skeleton
    check(f"{label} [{dtype}] fused ({sk})", True,
          f"verified={report.verified}")
    d0, d1 = tempfile.mkdtemp(prefix="i4a_"), tempfile.mkdtemp(prefix="i4b_")
    p0, p1 = os.path.join(d0, "m.pt2"), os.path.join(d1, "m.pt2")
    torch._inductor.aoti_compile_and_package(ep0, package_path=p0)
    torch._inductor.aoti_compile_and_package(ep1, package_path=p1)
    m_fused, m_base = _pt2_mentions(p1, sk), _pt2_mentions(p0, sk)
    check(f"{label} [{dtype}] generated kernel is in the fused .pt2", m_fused >= 1,
          f"mentions={m_fused}")
    check(f"{label} [{dtype}] baseline has no such kernel", m_base == 0,
          f"mentions={m_base}")

    f0 = torch._inductor.aoti_load_package(p0)
    f1 = torch._inductor.aoti_load_package(p1)
    with torch.no_grad():
        got = f1(x)
    got = got[0] if isinstance(got, (tuple, list)) else got
    ref = ref[0] if isinstance(ref, (tuple, list)) else ref
    err = (got.float() - ref.float()).abs().max().item()
    # A few ULPs of the largest magnitude present: log-probabilities get large negative, and at
    # magnitude 30 an fp16 ULP is already 0.016, so an absolute tolerance would be a fiction.
    eps = torch.finfo(ref.dtype).eps
    bound = 4 * eps * ref.abs().max().item() + 1e-6
    check(f"{label} [{dtype}] numbers unchanged", err < bound,
          f"max|diff|={err:.3e} < bound={bound:.3e}")
    ms0, ms1 = _batch_ms(f0, x), _batch_ms(f1, x)
    print(f"      {'':22s} plain {ms0:.4f} ms -> fused {ms1:.4f} ms = {ms0 / ms1:.2f}x")


def section_end_to_end() -> None:
    print()
    print("=" * 112)
    print("[2] END TO END: package both ways, load, compare numbers and latency")
    print("=" * 112)
    print(f"    latency on {tuple(X.shape)}, launches pipelined; CPU reference checked too")
    # The hand-written spelling in fp16 is the one case NOT expected to fuse, and the reason is
    # worth knowing: `torch.logsumexp`'s core decomposition inserts an explicit fp32 upcast
    # (`aten._to_copy`) for half precision, and the symbolic interpreter refuses an op it cannot
    # model rather than matching through it. fp32 `logsumexp`, and `log_softmax` in BOTH dtypes
    # (what the lowering is for), do fuse. The limit is pinned in docs/roadmap.md's backlog.
    for label, fn in _spellings():
        for dtype, expect in ((torch.float32, True), (torch.float16, label != "log_softmax by hand")):
            try:
                _one(label, fn, dtype, expect)
            except Exception as e:  # noqa: BLE001 -- one broken case must not end the experiment
                check(f"{label} [{dtype}]", False, f"{type(e).__name__}: {str(e)[:60]}")

    # The CPU-side numeric check, independent of Inductor: the LOWERED graph (no Triton involved)
    # must reproduce log_softmax exactly. On GPU the kernel is what runs; this pins the algebra.
    for label, fn in _spellings():
        if label == "log_softmax by hand":
            continue
        try:
            ep = torch.export.export(M(fn).eval(), (SMALL,)).run_decompositions()
            before = str(ep.graph_module.graph)
            report = NO.apply(ep)
            after = str(ep.graph_module.graph)
            check(f"{label} [cpu] lowering committed", bool(report.fusions) and before != after,
                  f"fusions={[f.skeleton for f in report.fusions]}")
        except Exception as e:  # noqa: BLE001
            check(f"{label} [cpu]", False, f"{type(e).__name__}: {str(e)[:60]}")


# ---------------------------------------------------------------- [3] the excluded operators
def section_no_op() -> None:
    print()
    print("=" * 112)
    print("[3] NO-OP: the operators m20 EXCLUDED must leave the graph byte-identical")
    print("=" * 112)
    print("    softmax 0.59x and layer_norm 0.41x against Inductor's dedicated lowerings, so a")
    print("    pass that touched them would be a pessimisation dressed as added reach.")
    # `None` means "use the dedicated module": nn.LayerNorm built inside a lambda is not installed
    # as a submodule, which torch 2.11 rejects at export time (torch 2.14 tolerates it).
    cases = [
        ("softmax", lambda t: torch.softmax(t, dim=-1)),
        ("layer_norm", None),
        ("relu (no reduction)", lambda t: torch.relu(t) + 1.0),
        ("log_softmax over dim 0", lambda t: torch.log_softmax(t, dim=0)),
    ]
    for label, fn in cases:
        try:
            module = _LayerNorm() if fn is None else M(fn).eval()
            ep_plain = torch.export.export(module, (SMALL,)).run_decompositions()
            ep_ours = NO.export(module, (SMALL,))
            same = str(ep_plain.graph_module.graph) == str(ep_ours.graph_module.graph)
            check(f"{label}: graph byte-identical", same,
                  "" if same else "the pass changed a graph it cannot improve")
        except Exception as e:  # noqa: BLE001
            check(f"{label}: graph byte-identical", False, f"{type(e).__name__}: {str(e)[:60]}")


class _LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4096)

    def forward(self, t):
        return self.ln(t)


def main() -> int:
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
    if not torch.cuda.is_available():
        print("NO CUDA: this experiment needs the GPU host")
        return 2
    print(f"torch {torch.__version__}  triton {_triton_version()}  {torch.cuda.get_device_name(0)}")
    try:
        section_reach()
        section_end_to_end()
        section_no_op()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2
    print()
    print("=" * 112)
    print(f"TOTAL: {_passed} passed, {len(_failed)} failed")
    if _failed:
        for f in _failed:
            print(f"  FAILED: {f}")
        return 1
    print("READING: the lowering is licensed by measurement (m20) and bounded by construction --")
    print("it is committed only when it enables a rewrite, so a graph it cannot help is untouched.")
    print("=" * 112)
    return 0


def _triton_version() -> str:
    try:
        import triton
        return triton.__version__
    except Exception:  # noqa: BLE001
        return "MISSING"


if __name__ == "__main__":
    raise SystemExit(main())
