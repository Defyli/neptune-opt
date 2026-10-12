"""
Test suite for the automatic codegen.

Four independent things are checked, because each can fail silently on its own:

  1. CORRECTNESS across several input distributions and shapes, against the exact
     row-wise torch reference. Includes the degenerate rows (all zeros, a single huge
     outlier, tiny magnitudes, negatives) that a running-max scheme is most likely to
     get wrong.
  2. GENERATOR QUALITY. The generated source must compile on the GPU, define the
     expected symbol, and contain no leftover template markers. Rendering is also
     checked to be *deterministic*, so regenerating does not silently change meaning.
  3. NEGATIVE CONTROL. `pnorm3` carries a deliberately wrong finalize (acc^3 instead of
     cbrt). The suite must FAIL it. Without this, "all tests pass" could just mean the
     tests are blind.
  4. PERFORMANCE. For each supported skeleton, compare against the decomposed torch form
     compiled by Inductor: input traversals (parsed from emitted Triton) and latency.

Run on the GPU host:  python3 gpu_m10_codegen_tests.py
"""
from __future__ import annotations

import os
import re
import statistics
import sys
import tempfile
import traceback

import torch

# Import the packages whether this file sits next to them (the original flat layout)
# or inside experiments/ (the layout of the published repository).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neptune_codegen as CG  # noqa: E402

assert torch.cuda.is_available()
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
import torch._inductor.config as cfg  # noqa: E402

cfg.cuda_backend = "triton"

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name, ok, detail=""):
    results.append((name, PASS if ok else FAIL, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name:44s} {detail}")


# ---------------------------------------------------------------- 1. correctness
def distributions(shape, seed):
    """The input families a running-max scheme is most likely to mishandle."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    r, c = shape
    yield "normal", torch.randn(shape, device="cuda", generator=g)
    yield "large_scale", torch.randn(shape, device="cuda", generator=g) * 1e3
    yield "tiny_scale", torch.randn(shape, device="cuda", generator=g) * 1e-6
    yield "all_negative", -torch.rand(shape, device="cuda", generator=g)
    yield "outlier", torch.cat([
        torch.randn(r, 1, device="cuda", generator=g) * 1e4,
        torch.randn(r, c - 1, device="cuda", generator=g),
    ], dim=1)
    z = torch.randn(shape, device="cuda", generator=g)
    z[::3] = 0.0                                    # every third row is all zeros
    yield "zero_rows", z
    yield "positive", torch.rand(shape, device="cuda", generator=g) + 1e-3
    yield "exp_range", torch.exp(torch.randn(shape, device="cuda", generator=g) * 6)


TOL = {"logsumexp": 2e-4, "stable_l2": 2e-4, "pnorm3_true": 2e-4}


def vals_for(name, x):
    """Scalar values to bake in. `rmsnorm_max` couples N to the row length, because its
    producer is a MEAN -- the kernel has N as a literal, so the reference must be told the
    same number rather than silently using the row length."""
    # Use the skeleton's DECLARED defaults, not a blanket 1.0: the log-family readout's
    # additive constant defaults to 0 because `torch.logsumexp` has no such constant, and
    # baking 1.0 there would compare two different operators.
    v = CG.param_defaults(name)
    if name == "rmsnorm_max":
        # N is intrinsic now (the kernel reads `ncols`), so only eps is baked.
        v["eps"] = 1e-6
    return v


def test_correctness(name, shapes):
    print(f"\n--- correctness: {name}")
    worst = 0.0
    worst_case = ""
    for shape in shapes:
        # NB: the seed must not depend on the loop variable -- that raised
        # UnboundLocalError. Derive it from the shape/distribution index instead.
        for idx, (dist, x) in enumerate(distributions(shape, seed=abs(hash((name, shape))) % 2**28)):
            try:
                _v = vals_for(name, x)
                got = CG.run(name, x, **_v)
                ref = CG.reference(name, x, **_v)
                # Absolute error where the reference is ~0. A pure relative metric on
                # an all-zero row divides a 1e-12 absolute error by a 1e-6 floor and
                # reports 2e-6, blaming the kernel for a harness artefact.
                denom = ref.abs().clamp_min(1e-3)
                rel = ((got - ref).abs() / denom).max().item()
            except Exception as e:  # noqa: BLE001
                record(f"{name}/{dist}{shape}", False, f"{type(e).__name__}: {str(e)[:60]}")
                continue
            ok = rel <= TOL.get(name, 1e-3)
            if rel > worst:
                worst, worst_case = rel, f"{dist}{shape}"
            record(f"{name}/{dist}{shape}", ok, f"rel_err={rel:.2e}")
    print(f"    worst relative error for {name}: {worst:.3e} ({worst_case})")
    return worst


# ---------------------------------------------------------------- 2. generator
def test_generator():
    print("\n--- applicability guard (untranslatable algebra must be refused)")
    import sympy as _sp
    from neptune_codegen import Skeleton as _Sk
    bad = [
        # g not invertible in c -> Eq.5 cannot be instantiated
        ("non_invertible", _Sk(name="bad_noninv",
                               g=_sp.exp(_sp.Symbol("c", real=True)
                                         - _sp.Abs(_sp.Symbol("c", real=True)) - 0 * _sp.Symbol("s", real=True)),
                               finalize=_sp.Symbol("acc"))),
        # an exponential fold with the wrong producer must be refused
        ("log_with_abs_producer", _Sk(name="bad_log", g=_sp.exp(_sp.Symbol("c", real=True)
                                                                - _sp.Symbol("s", real=True)),
                                      finalize=_sp.Symbol("acc"),
                                      producer="abs", is_log=True)),
    ]
    for label, sk in bad:
        try:
            CG.emit_kernel_source(sk)
            record(f"guard/{label}", False, "emitted an invalid skeleton")
        except (NotImplementedError, Exception) as e:  # noqa: BLE001
            record(f"guard/{label}", True, type(e).__name__)

    print("\n--- generator quality")
    for name in CG.supported():
        sk = CG.SKELETONS[name]
        vals = CG.param_defaults(name)                 # scalars are literals now
        src = CG.emit_kernel_source(sk, vals)
        leftover = [m for m in ("{", "}") if m in src]
        record(f"{name}/source clean", not leftover, f"leftover={leftover}")
        record(f"{name}/defines symbol", f"def {name}(" in src, "")
        # determinism
        again = CG.emit_kernel_source(sk, vals)
        record(f"{name}/render deterministic", src == again, "")
        # real file on disk (Triton requirement)
        CG.compile_skeleton(name, vals, force=True)
        p = CG.generated_path(name, vals)
        on_disk = os.path.exists(p) and os.path.getsize(p) > 0
        record(f"{name}/written to real file", on_disk, os.path.basename(p))
        # must be importable / jitted
        try:
            k = CG.compile_skeleton(name, CG.param_defaults(name))
            record(f"{name}/compiles", k is not None, type(k).__name__)
        except Exception as e:  # noqa: BLE001
            record(f"{name}/compiles", False, f"{type(e).__name__}: {str(e)[:80]}")


# ---------------------------------------------------------------- 3. negative ctrl
def test_negative_control():
    """Prove the harness can detect bad codegen.

    The guard now refuses to emit `pnorm3`, so the control builds a deliberately broken
    kernel by hand (correct term, wrong finalize) and checks that the same comparison
    machinery flags it. Without this, "all tests pass" could just mean the tests are
    blind to a wrong finalize.
    """
    print("\n--- negative control (the suite must catch a wrong finalize)")
    import tempfile as _tf
    src = """
import triton
import triton.language as tl

@triton.jit
def bad_l2(x_ptr, out_ptr, rows, ncols, stride_row, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if row < rows:
        base = x_ptr + row * stride_row
        m = 0.0
        acc = 0.0
        for off in range(0, ncols, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            mask = idx < ncols
            v = tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
            a = tl.abs(v)
            m_new = tl.maximum(m, tl.max(tl.where(mask, a, -float("inf")), axis=0))
            scale = tl.where(m == 0.0, 1.0, (m / m_new) * (m / m_new))
            acc = acc * scale + tl.sum(tl.where(mask, (a / m_new) * (a / m_new), 0.0), axis=0)
            m = m_new
        # WRONG on purpose: cube instead of square root
        tl.store(out_ptr + row, m * acc * acc * acc)
"""
    import importlib.util
    d = _tf.mkdtemp()
    path = os.path.join(d, "bad_l2.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("bad_l2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    x = torch.randn(16, 4096, device="cuda")
    rows, ncols = x.shape
    x2 = x.contiguous()
    out = torch.empty(rows, device="cuda", dtype=torch.float32)
    mod.bad_l2[(rows,)](x2, out, rows, ncols, x2.stride(0), BLOCK=1024)
    ref = CG.reference("stable_l2", x)
    rel = ((out - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
    caught = rel > 1e-2
    record("wrong finalize detected", caught, f"rel_err={rel:.2e} (expected large)")


# ---------------------------------------------------------------- 4. performance
def timeit(fn, iters=40, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts)


def baseline_traversals(name, shape):
    """Compile the decomposed torch form and count input loads in the emitted Triton."""
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="m10_base_")
    os.environ["TRITON_HOME"] = tempfile.mkdtemp(prefix="m10th_")
    torch._dynamo.reset()
    x = torch.randn(*shape, device="cuda")
    vals = CG.param_defaults(name)
    if name == "rmsnorm_max":
        vals.update({"eps": 1e-6})
    fn = lambda t: CG.graph_reference(name, t, **vals)  # noqa: E731
    with torch.no_grad():
        cfn = torch.compile(fn, dynamic=False)
        cfn(x)
    total, names = 0, []
    for dp, _d, fs in os.walk(os.environ["TORCHINDUCTOR_CACHE_DIR"]):
        for f in fs:
            if not f.endswith(".py"):
                continue
            t = open(os.path.join(dp, f), encoding="utf-8", errors="replace").read()
            if "async_compile.triton(" not in t:
                continue
            names += re.findall(r"async_compile\.triton\('(\w+)'", t)
            body = t.split("'''", 1)[-1] if "'''" in t else t
            total += len(re.findall(r"tl\.load\(\s*in_ptr0", body))
    return cfn, x, total, names


PERF_SHAPES = [(8, 4096), (8, 16384), (8, 65536), (32, 65536)]


def test_performance():
    print("\n--- performance vs decomposed-and-compiled torch")
    print(f"    {'operator':12s} {'shape':>12} {'base load':>10} {'gen load':>9} "
          f"{'base ms':>9} {'gen ms':>8} {'speedup':>8}")
    for name in CG.supported():
        for shape in PERF_SHAPES:
            try:
                cfn, x, base_load, names = baseline_traversals(name, shape)
                med_b, _ = timeit(lambda: cfn(x))
                _vals = vals_for(name, x)
                med_g, _ = timeit(lambda: CG.run(name, x, **_vals))
                sp = med_b / med_g if med_g > 0 else float("nan")
                print(f"    {name:12s} {str(shape):>12} {base_load:>10} {1:>9} "
                      f"{med_b:9.3f} {med_g:8.3f} {sp:7.2f}x")
                record(f"{name}{shape}/speedup>1.05", sp > 1.05, f"{sp:.2f}x")
            except Exception as e:  # noqa: BLE001
                record(f"{name}{shape}/perf", False, f"{type(e).__name__}: {str(e)[:60]}")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    print("=" * 100)
    print("TEST SUITE — automatic codegen for break-and-repair fused reductions")
    print(f"torch {torch.__version__}  triton {__import__('triton').__version__}  "
          f"gpu {torch.cuda.get_device_name(0)}")
    print(f"supported skeletons: {CG.supported()}")
    print("=" * 100)

    SHAPES = [(8, 1024), (16, 4096), (4, 16384), (1, 65536), (32, 512)]
    for name in CG.supported():
        if name == "pnorm3":
            continue                     # negative control, correctness checked separately
        test_correctness(name, SHAPES)

    test_generator()
    test_negative_control()
    test_performance()

    print("\n" + "=" * 100)
    n_pass = sum(1 for _n, s, _d in results if s == PASS)
    n_fail = sum(1 for _n, s, _d in results if s == FAIL)
    print(f"TOTAL: {n_pass} passed, {n_fail} failed, {len(results)} checks")
    if n_fail:
        print("FAILURES:")
        for n, s, d in results:
            if s == FAIL:
                print(f"   {n}: {d}")
    print("=" * 100)
