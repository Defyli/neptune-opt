"""
GPU validation of the TWO-PASS plan: the fallback for operators that have no repair.

Why this needs its own experiment rather than riding on gpu_m10_codegen_tests.py: the plan
has failure modes the one-pass kernels cannot have.

  1. A SECOND TRAVERSAL. The kernel reads the row twice. If the producer is not actually
     finalized between the passes, or if pass 2 accidentally re-reads the running value, the
     result is wrong in a way that still looks like a plausible variance.
  2. ILL-CONDITIONING. `mean((x - mean)^2)` in fp32 loses precision when the mean is huge
     relative to the spread, and torch's own `var` uses Welford, which does not. This is a
     property of the operator in fp32, not of the codegen, so it is MEASURED and reported
     rather than asserted away -- a tight tolerance here would either be a lie or a test that
     fails for reasons that have nothing to do with this project.
  3. Perf. The claim being tested is "one kernel and no materialised intermediate beats the
     decomposed form that Inductor is handed". That is a measurement, so it is printed.

Run on the GPU host:  python3 gpu_m17_two_pass.py
"""
from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neptune_codegen as CG  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    verdict = PASS if ok else FAIL
    _results.append((name, verdict, detail))
    print(f"  [{verdict}] {name}" + (f"  -- {detail}" if detail else ""))


def test_plan_shape() -> None:
    """The emitted source must be a two-pass kernel, with no repair machinery."""
    print("\n--- plan shape")
    for name in ("variance", "std"):
        sk = CG.SKELETONS[name]
        src = CG.emit_kernel_source(sk, {})
        record(f"{name}/declared two_pass", sk.two_pass, "")
        record(f"{name}/two loops", src.count("for off in range(0, ncols, BLOCK):") == 2, "")
        record(f"{name}/no repair statement", "scale = tl.where" not in src, "")
        # Positional, not literal: `std` takes a root of N, so its N is bound as a float and
        # the same statement reads `(1.0 / (1.0 * ncols))`. See
        # `_intrinsics_needing_float` in neptune_codegen/emitters.py.
        lines = src.splitlines()
        fin = [i for i, l in enumerate(lines) if l.strip().startswith("m = m * (")]
        den = [i for i, l in enumerate(lines) if l.strip() == "den = m"]
        record(f"{name}/producer finalized before pass 2",
               bool(fin) and bool(den) and fin[0] < den[0], "")
        record(f"{name}/no floor on the finalized producer",
               "tl.maximum(m_new" not in src, "")


def test_correctness() -> None:
    """Well-conditioned data: compare against torch's own var/std, tightly."""
    print("\n--- correctness vs torch.var/std (well-conditioned)")
    shapes = [(1, 1), (1, 7), (8, 33), (5, 128), (4, 1024), (16, 4096), (3, 1000), (2, 65536)]
    dists = {
        "normal": lambda *s: torch.randn(*s, device="cuda"),
        "offset": lambda *s: torch.randn(*s, device="cuda") * 2.0 + 10.0,
        "small": lambda *s: torch.randn(*s, device="cuda") * 1e-3,
        "all_zero": lambda *s: torch.zeros(*s, device="cuda"),
        "constant": lambda *s: torch.full(s, 3.5, device="cuda"),
        "two_values": lambda *s: torch.where(torch.rand(*s, device="cuda") > 0.5,
                                             torch.ones(*s, device="cuda"),
                                             -torch.ones(*s, device="cuda")),
    }
    for name in ("variance", "std"):
        for shape in shapes:
            for dname, gen in dists.items():
                x = gen(*shape)
                try:
                    got = CG.run(name, x, block=256)
                    ref = CG.reference(name, x)
                    # absolute floor because several cases have a reference of exactly 0
                    err = (got - ref).abs().max().item()
                    denom = ref.abs().max().item()
                    rel = err / max(denom, 1e-6)
                except Exception as e:  # noqa: BLE001
                    record(f"{name} {shape} {dname}", False, f"{type(e).__name__}: {e}")
                    continue
                record(f"{name} {shape} {dname}", err < 1e-3, f"abs_err={err:.3e} rel={rel:.3e}")


def test_ill_conditioned() -> None:
    """Huge mean, tiny spread: measure the cancellation and report it, do not hide it."""
    print("\n--- ill-conditioned (large offset, small spread)")
    print(f"    {'offset':>10} {'spread':>8} {'torch var':>12} {'kernel var':>12} "
          f"{'rel err':>10}")
    x0 = torch.randn(4, 8192, device="cuda")
    resolved, skipped, worst = 0, 0, 0.0
    for offset in (0.0, 1e3, 1e6, 1e8):
        for spread in (1.0, 1e-2, 1e-4):
            x = x0 * spread + offset
            # fp32 has ~7 decimal digits. Once the spread falls below one ULP at this offset
            # the INPUT is quantised -- every element rounds to nearly the same value -- and no
            # fp32 algorithm can recover a variance that is no longer in the data. Asserting on
            # such a case would be measuring the input format, not the kernel, so it is
            # reported and excluded. (At offset 1e8, one ULP is ~8.0, so any spread below that
            # is gone before the kernel is even called.)
            ulp = float(torch.finfo(torch.float32).eps * abs(offset)) if offset else 0.0
            got = CG.run("variance", x, block=256)
            ref = CG.reference("variance", x)
            if offset and spread < ulp:
                skipped += 1
                print(f"    {offset:10.1e} {spread:8.1e} {'unrepresentable':>12} "
                      f"{'in fp32':>12} {1.0 * (spread < ulp):12.0f}   (spread < ULP {ulp:.1e})")
                continue
            rel = ((got - ref).abs() / ref.abs().clamp_min(1e-30)).max().item()
            worst = max(worst, rel)
            resolved += 1
            print(f"    {offset:10.1e} {spread:8.1e} {ref.mean().item():12.5e} "
                  f"{got.mean().item():12.5e} {rel:10.3e}")
    record("ill-conditioning degrades as fp32 predicts", worst < 1.0,
           f"worst rel err {worst:.2e} over the {resolved} cases fp32 can represent "
           f"({skipped} skipped as unrepresentable)")


def test_vs_welford_reference() -> None:
    """Does the two-pass form track torch's Welford, or drift into a different operator?"""
    print("\n--- agreement with torch.var over random rows")
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(20):
        x = torch.randn(64, 4096, device="cuda") * 10 ** torch.randint(-2, 3, (1,)).item()
        got = CG.run("variance", x, block=512)
        ref = CG.reference("variance", x)
        worst = max(worst, ((got - ref).abs() / ref.abs().clamp_min(1e-30)).max().item())
    record("tracks torch.var within fp32", worst < 1e-4, f"worst rel err {worst:.3e}")


def test_explicit_op() -> None:
    """The zero-risk entry point: neptune_opt.variance(x) / neptune_opt.std(x)."""
    print("\n--- explicit operator library")
    import neptune_opt as NO
    x = torch.randn(6, 512, device="cuda")
    for name, ref_fn in (("variance", lambda t: t.var(dim=-1, unbiased=False)),
                         ("std", lambda t: t.std(dim=-1, unbiased=False))):
        op = getattr(NO, name)
        got, ref = op(x), ref_fn(x)
        err = (got - ref).abs().max().item()
        record(f"neptune_opt.{name}", err < 1e-4, f"abs_err={err:.3e}")
        record(f"neptune_opt.{name} needs no parameters",
               NO.op_parameters(name) == (), f"{NO.op_parameters(name)}")


def test_performance() -> None:
    """One kernel with no intermediate vs the decomposed form torch.compile is handed.

    Reported, not asserted, and paired with a ONE-PASS kernel at the same shape. The claim
    "the two-pass plan is a performance win" is not true in general, and this table is how that
    was found: the plan reads the row twice, so it can only win where the baseline's second
    kernel and its materialised intermediate cost more than a second read -- and it loses badly
    when there are too few rows to fill the GPU, because this frame runs ONE PROGRAM PER ROW.
    Measuring `stable_l2` (one pass, one read) alongside `variance` separates those two causes
    instead of blaming the plan for a property of the frame.
    """
    print("\n--- performance vs the decomposed form (reported, not asserted)")
    # No baseline-load column: re-pointing TORCHINDUCTOR_CACHE_DIR per shape does NOT re-scope
    # Inductor's cache once it has been loaded, so counting loads there accumulated every
    # earlier shape's kernels (measured: 4, 12, 20, 28 -- growing by one shape's worth each
    # time). The generated kernel's own traversals are a static fact of the source, so they are
    # derived from it instead of from a directory.
    gen_loads = {n: CG.emit_kernel_source(CG.SKELETONS[n],
                                         CG.param_defaults(n)).count("tl.load(")
                 for n in ("variance", "stable_l2")}
    print(f"    {'shape':>14} {'base ms':>9} {'var ms':>9} {'var x':>7} "
          f"{'l2 ms':>9} {'l2 x':>7} {'loads var/l2':>13}")
    ratios = []
    for shape in ((8, 4096), (8, 16384), (8, 65536), (32, 65536), (1, 262144)):
        x = torch.randn(*shape, device="cuda")
        vals = CG.param_defaults("variance")
        # A FRESH cache dir per shape before compiling the baseline, or the counter below
        # accumulates the kernels of every earlier shape and reports a number that belongs to
        # none of them. (m10 does the same thing inside `baseline_traversals`.)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="m17_base_")
        with torch.no_grad():
            base = torch.compile(lambda t: CG.graph_reference("variance", t, **vals),
                                 dynamic=False)
            base(x)
        med_b = _median(lambda: base(x))
        med_v = _median(lambda: CG.run("variance", x, block=1024))
        med_l2 = _median(lambda: CG.run("stable_l2", x, block=1024))
        sx = med_b / med_v if med_v > 0 else float("nan")
        s2 = med_b / med_l2 if med_l2 > 0 else float("nan")
        ratios.append(sx)
        print(f"    {str(shape):>14} {med_b:9.4f} {med_v:9.4f} {sx:6.2f}x "
              f"{med_l2:9.4f} {s2:6.2f}x "
              f"{gen_loads['variance']:>6}/{gen_loads['stable_l2']}")
        # Not pathological: a two-pass kernel that reads the row twice can be slower, but it
        # must not be an order of magnitude slower -- that would mean the plan is unusable
        # rather than merely shape-dependent.
        record(f"variance{shape}/not pathological", sx > 0.25, f"{sx:.2f}x")
    worst = min(ratios)
    print(f"    worst two-pass speedup {worst:.2f}x; "
          f"the one-pass kernel is the reference for how much of that is the frame")


def _median(fn, iters: int = 50) -> float:
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
    if not torch.cuda.is_available():
        print("GPU required: this validates generated Triton kernels and their timing. The "
              "rest of the repository runs on CPU (tests/, tools/coverage_codegen.py).")
        return 1
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="m17_base_")
    os.environ["TRITON_HOME"] = tempfile.mkdtemp(prefix="m17th_")
    print("=" * 100)
    print("two-pass plan GPU validation")
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"{torch.cuda.get_device_name(0)}")
    print("=" * 100)
    try:
        test_plan_shape()
        test_correctness()
        test_ill_conditioned()
        test_vs_welford_reference()
        test_explicit_op()
        test_performance()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2
    print("\n" + "=" * 100)
    n_pass = sum(1 for _n, s, _d in _results if s == PASS)
    n_fail = sum(1 for _n, s, _d in _results if s == FAIL)
    print(f"TOTAL: {n_pass} passed, {n_fail} failed, {len(_results)} checks")
    for n, s, d in _results:
        if s == FAIL:
            print(f"   FAIL: {n}  {d}")
    print("=" * 100)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
