"""
Does this work on the code people actually write?

The coverage study says 26 of 28 families, but that study is a list *this project* wrote: it
measures the algebra, not the world. This experiment asks the uncomfortable question with real
`torch.nn` modules and real `torch.*` ops, and it reports the answer whether or not it flatters
the project.

Two things it establishes, both of which the family study cannot:

  1. Real modules very often export as ONE high-level aten op (`aten.layer_norm.default`,
     `aten.softmax.int`, `aten.logsumexp.default`), not as a producer/consumer pair. The frame
     therefore needs the CORE-ATEN DECOMPOSITIONS of the graph, which `ExportedProgram` can apply
     itself (`ep.run_decompositions()`). Without that step almost nothing real is recognised.
  2. Even decomposed, an operator needs TWO reductions (a producer and a consumer) to be in
     frame. A single reduction with elementwise work around it -- RMSNorm, LayerNorm, softmax's
     normalisation -- is outside the frame no matter how the algebra is extended. That is a
     structural limit, and it is the single most useful thing to know before investing more.

Run on the GPU host:  python3 real_modules.py
On CPU it still runs the recognition half and skips the timing.
"""
from __future__ import annotations

import copy
import os
import statistics
import sys
import tempfile
import time
import zipfile
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neptune_codegen as CG  # noqa: E402
import neptune_recognize as NR  # noqa: E402
import neptune_opt as NO  # noqa: E402
from neptune_recognize.lowering import lower_compound  # noqa: E402

X = torch.randn(64, 4096)
# Timing input: large enough that the kernels, not the launch, dominate the measurement.
PERF_X = torch.randn(256, 16384)


# ---------------------------------------------------------------- what real code looks like
def _rmsnorm_max(t):
    """RMSNorm's normaliser with a row max instead of a scale -- IN frame (two reductions)."""
    r = (t * t).mean(dim=-1, keepdim=True)
    return (t / torch.sqrt(r + 1e-6)).amax(dim=-1)


def _stable_l2(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return (m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))).squeeze(-1)


def _lse_minus_max(t):
    """What cross-entropy is built from: a log-sum-exp that stays in the graph."""
    return torch.logsumexp(t, dim=-1)


def _log_softmax_by_hand(t):
    return t - torch.logsumexp(t, dim=-1, keepdim=True)


def _attn_scores(t):
    """A realistic head: scale, mask-free max-subtracted log-sum-exp (what softmax costs)."""
    return torch.logsumexp(t / 8.0, dim=-1)


CASES = [
    # (label, factory, in-frame?)
    ("nn.LayerNorm", lambda t: nn.LayerNorm(t.shape[-1])(t), False),
    ("nn.RMSNorm", lambda t: nn.RMSNorm(t.shape[-1])(t), False),
    ("F.layer_norm", lambda t: F.layer_norm(t, (t.shape[-1],)), False),
    ("nn.Softmax", lambda t: nn.Softmax(dim=-1)(t), False),
    ("nn.LogSoftmax", lambda t: nn.LogSoftmax(dim=-1)(t), True),   # in frame via I4's lowering
    ("F.cross_entropy", None, False),          # filled in below (needs targets)
    ("torch.logsumexp", lambda t: torch.logsumexp(t, dim=-1), True),
    ("log_softmax by hand", _log_softmax_by_hand, True),
    ("attention lse (t/8)", _attn_scores, True),
    ("RMSNorm + row max", _rmsnorm_max, True),
    ("stable L2 norm", _stable_l2, True),
]


class M(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, t):
        return self.fn(t)


def _verdict(obj):
    """(skeleton | None, reason) for an ExportedProgram OR a bare GraphModule."""
    gm = getattr(obj, "graph_module", obj)
    match, reason = NR.recognize(gm)
    return (match.skeleton if match else None), reason


def test_recognition() -> list:
    """Which real modules are recognised, and what does the REFUSAL say?"""
    print("=" * 112)
    print("RECOGNITION ON REAL MODULES   (64 x 4096, torch %s)" % torch.__version__)
    print("=" * 112)
    print(f"    {'module':24s} {'as exported':28s} {'after core decompositions':34s}")
    print("    " + "-" * 106)
    fusable = []
    for label, fn, in_frame in CASES:
        if fn is None:
            continue
        try:
            ep = torch.export.export(M(fn).eval(), (X,))
        except Exception as e:  # noqa: BLE001
            print(f"    {label:24s} EXPORT FAILED: {type(e).__name__}: {str(e)[:48]}")
            continue
        raw_sk, raw_why = _verdict(ep)
        lowered = ""
        try:
            dep = ep.run_decompositions()
            dep_sk, dep_why = _verdict(dep)
        except Exception as e:  # noqa: BLE001
            dep, dep_sk, dep_why = None, None, f"run_decompositions failed: {type(e).__name__}"
        if dep_sk is None and dep is not None:
            # The core decompositions leave some operators WHOLE (`_log_softmax`: no table entry
            # for the overload export produces). The lowering pass expands those, so the honest
            # question is what the graph looks like AFTER it -- not what the decompositions alone
            # can see. This is the column I4 changed.
            probe = copy.deepcopy(dep.graph_module)
            try:
                if lower_compound(probe):
                    got = _verdict(probe)[0]
                    if got:
                        dep_sk, lowered = got, " (lowered)"
            except Exception:  # noqa: BLE001 -- a failed trial must not end the table
                pass
        raw = raw_sk or _short(raw_why)
        dec = (dep_sk + lowered) if dep_sk else _short(dep_why)
        print(f"    {label:24s} {raw:28s} {dec:34s}")
        if dep_sk:
            fusable.append((label, dep_sk, fn))
    print()
    print("    The second column is the point: real modules export as ONE high-level aten op, so")
    print("    the frame needs the CORE-ATEN DECOMPOSITIONS to see a producer/consumer pair.")
    return fusable


def _short(reason: str) -> str:
    reason = str(reason)
    for key, tag in (("unfamiliar op", "unfamiliar op"),
                     ("only 0 reduction", "no reduction"),
                     ("only 1 reduction", "ONE reduction -- out of frame"),
                     ("none depends on another", "reductions are independent")):
        if key in reason:
            if "unfamiliar op" in key:
                # name the op, it is the actionable part
                return "needs decomposition: " + reason.split("'")[1].split(".")[-2]
            return tag
    return reason[:34]


def test_end_to_end(fusable) -> None:
    """For the ones that ARE in frame: does the fused version run and is it faster?"""
    if not torch.cuda.is_available():
        print("\n(no CUDA: skipping the timing half)")
        return
    print("\n" + "=" * 112)
    print("END TO END: package both ways, load, compare numbers and latency")
    print("=" * 112)
    print(f"    {'module':24s} {'fused as':10s} {'mentions':>8s} {'base':>5s} {'rel_err':>10s} "
          f"{'plain ms':>9s} {'fused ms':>9s} {'x':>6s}")
    print("    'mentions'/'base' = occurrences of the generated kernel's name in each .pt2")
    print(f"    latency on {tuple(PERF_X.shape)}, launches pipelined (see _batch_ms)")
    for label, sk, fn in fusable:
        try:
            _one_case(label, sk, fn)
        except Exception as e:  # noqa: BLE001 -- one broken case must not end the experiment
            print(f"    {label:24s} FAILED: {type(e).__name__}: {str(e)[:60]}")


def _one_case(label, sk, fn) -> None:
    # ONE shape for everything. An AOTI package is specialised to the shapes it was
    # compiled for, so compiling at 64x4096 and then timing at 256x16384 was an
    # out-of-bounds access (measured: "CUDA error: an illegal memory access", which then
    # poisoned the context for every following case). The larger shape is used throughout,
    # because that is the one where the kernels rather than the launch dominate.
    x = PERF_X.cuda()
    mod = M(fn).eval().cuda()
    with torch.no_grad():
        ref = mod(x)
    # plain
    ep0 = torch.export.export(mod, (x,)).run_decompositions()
    # fused
    ep1 = torch.export.export(mod, (x,)).run_decompositions()
    report = NO.apply(ep1, verify_with=(mod, (x,)))
    if not report.fusions:
        print(f"    {label:24s} NOT FUSED: {report.summary().strip()[:48]}")
        return
    p0 = os.path.join(tempfile.mkdtemp(prefix="rm0_"), "m.pt2")
    p1 = os.path.join(tempfile.mkdtemp(prefix="rm1_"), "m.pt2")
    torch._inductor.aoti_compile_and_package(ep0, package_path=p0)
    torch._inductor.aoti_compile_and_package(ep1, package_path=p1)
    sk_name = report.fusions[0].skeleton
    with zipfile.ZipFile(p1) as z:
        raw1 = b"".join(z.read(n) for n in z.namelist())
    with zipfile.ZipFile(p0) as z:
        raw0 = b"".join(z.read(n) for n in z.namelist())
    mentions = raw1.count(sk_name.encode())
    # The comparison is only meaningful if the BASELINE lacks the generated kernel. Printing
    # it, rather than assuming it, is cheap and it is the difference between a measurement and
    # a story.
    base_mentions = raw0.count(sk_name.encode())
    f0 = torch._inductor.aoti_load_package(p0)
    f1 = torch._inductor.aoti_load_package(p1)
    with torch.no_grad():
        got = f1(x)
    got = got[0] if isinstance(got, (tuple, list)) else got
    rel = ((got - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
    # TWO timings, because the first one cannot answer the question. A single call takes
    # ~0.023 ms, which is the same order as Python + one launch, so per-call overhead hides
    # whatever the kernels are doing; replaying a captured CUDA graph removes that overhead
    # and leaves the GPU work, which is the thing being optimised.
    # A second, LARGER input for timing: 64x4096 is too small for the kernels to dominate.
    ms0, ms1 = _batch_ms(f0, x), _batch_ms(f1, x)
    sp = ms0 / ms1 if ms1 > 0 else float("nan")
    print(f"    {label:24s} {sk_name:10s} {mentions:9d} {base_mentions:8d} {rel:10.2e} "
          f"{ms0:9.4f} {ms1:9.4f} {sp:7.2f}x")
    print(f"      [{'PASS' if rel < 1e-3 else 'FAIL'}] numbers unchanged; "
          f"[{'PASS' if mentions >= 1 else 'FAIL'}] generated kernel in the fused .pt2; "
          f"[{'PASS' if base_mentions == 0 else 'WARN'}] absent from the baseline")


def _batch_ms(fn, x, reps: int = 20) -> float:
    """Per-call time with the launches PIPELINED, which is what makes the comparison meaningful.

    A single call at 64x4096 takes ~0.023 ms, the same order as Python dispatch plus one launch,
    so timing single calls measures the harness rather than the kernels -- and it produced
    differences of a few percent that could not be attributed to anything. Issuing `reps` calls
    back to back amortises that overhead and lets the GPU work dominate.

    (Capturing a CUDA graph instead was tried and abandoned: an AOTI-loaded package allocates its
    output on every call, and capture fails with "operation failed due to a previous error", which
    then poisons the context for the rest of the run.)
    """
    with torch.no_grad():
        for _ in range(3):
            fn(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / reps


def _median(fn, iters: int = 30) -> float:
    with torch.no_grad():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main() -> int:
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
    try:
        fusable = test_recognition()
        test_end_to_end(fusable)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2
    print()
    print("=" * 112)
    print("READING: 'after core decompositions' is not a detail -- it is the difference between")
    print("one op and a producer/consumer pair. 'ONE reduction -- out of frame' is a STRUCTURAL")
    print("verdict, not a missing pattern: this frame fuses two dependent reductions, so an")
    print("operator whose consumer is elementwise (RMSNorm, LayerNorm, softmax's normalisation)")
    print("cannot be served by adding algebra.")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
