"""
GPU validation of the AFFINE repair: `h(t, s, s') = a(s,s')*t + b(s,s')`.

Why this needs a GPU run of its own: the affine path is the only part of the generator whose
emitted source contains a statement the multiplicative kernels never had --

    bias = tl.where(m == -inf, 0.0, <bias>)
    acc  = tl.maximum(acc * scale + bias, tl.max(...))

-- and "the algebra is right" does not imply "Triton compiles it and the numbers come out".
Every registered skeleton has b = 0, so not one of the existing 395 checks in
gpu_m10_codegen_tests.py executes that line. This experiment is the missing coverage.

The families used are synthetic on purpose. The additive-shift operator from the coverage
study, g = c - s, is DEGENERATE: its value is identically zero, so a kernel that is wrong
everywhere except zero can still pass. `g = k*c + s` with a max consumer is not degenerate --
its reference is exactly (k+1)*max(c) -- so a wrong slope, a wrong bias, a bias applied on the
first block, or a floored producer all show up as a first-order error.

Run on the GPU host:  python3 gpu_m16_affine.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch

# Import the packages whether this file sits next to them (the original flat layout) or inside
# experiments/ (the layout of the published repository).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neptune_codegen as CG  # noqa: E402
from neptune_codegen.symbols import (_FINAL_ACC, _PARAM_N, c_sym, s2_sym,
                                      s_sym)  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    verdict = PASS if ok else FAIL
    _results.append((name, verdict, detail))
    print(f"  [{verdict}] {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------- families
def _register_families() -> None:
    """Register the affine families. Not in `library.py` on purpose.

    They are validation vehicles for the repair path, not operators anyone asked for, and
    `library.py` is the shipped registry. Registering here keeps the shipped coverage study
    (tools/coverage_codegen.py) and the GPU test independent of each other.
    """
    # g = k*c + s. Derivative matching gives a = 1, b = s' - s. Non-degenerate reference:
    #   out = max_j g(max c, c_j) = max_j (k*c_j + max c) = k*max c + max c = (k+1)*max c
    CG.register(CG.Skeleton(
        name="affine_plus",
        g=_PARAM_N * c_sym + s_sym,
        finalize=_FINAL_ACC,
        producer="signed",
        producer_reducer="max",
        consumer_reducer="max",
        params=((_PARAM_N, "k"),),
        defaults=((_PARAM_N, 2.0),),
        doc="affine-repair validation: max_j (k*x_j + max x)",
    ))
    # g = c - s, the coverage study's additive-shift family. DEGENERATE (value 0), included
    # because it is the family that motivated the affine branch and it is cheap to run.
    CG.register(CG.Skeleton(
        name="affine_shift",
        g=c_sym - s_sym,
        finalize=_FINAL_ACC,
        producer="signed",
        producer_reducer="max",
        consumer_reducer="max",
        doc="affine-repair validation (degenerate): max_j (x_j - max x) == 0",
    ))


def _references(name: str, x: torch.Tensor, k: float | None = None) -> torch.Tensor:
    """Closed-form reference for the registered affine families.

    For `g = k*c + s` with a max producer, `out = max_j g(max c, c_j)` is

        max_j (k*c_j) + max c

    and NOT `(1+k)*max c`. The two agree only while k > 0, because `max_j(k*c_j)` is
    `k*max c` for positive k and `k*min c` for negative k. Writing the short form cost a
    full GPU run: 88 checks passed and the 23 failures were ALL this reference being wrong
    for k = -0.5, with the kernel exactly right. That is the same class of mistake the
    design notes warn about -- a reference written from a reading of the operator rather
    than from its definition -- so the reference is spelled out here in full.
    """
    mx = x.amax(dim=-1)
    if name == "affine_plus":
        return (k * x).amax(dim=-1) + mx
    if name == "affine_shift":
        return torch.zeros_like(mx)
    raise KeyError(name)


def test_derivation() -> None:
    """The derived repair must be the one the derivation claims, before any kernel runs."""
    print("\n--- derivation")
    a1, b1 = CG.repair_pair(CG.SKELETONS["affine_plus"])
    record("affine_plus/slope is 1", a1 == 1, f"a = {a1}")
    record("affine_plus/bias is s'-s",
           CG._simplify(b1 - (s2_sym - s_sym)) == 0, f"b = {b1}")
    a2, b2 = CG.repair_pair(CG.SKELETONS["affine_shift"])
    record("affine_shift/slope is 1", a2 == 1, f"a = {a2}")
    record("affine_shift/bias is s-s'",
           CG._simplify(b2 - (s_sym - s2_sym)) == 0, f"b = {b2}")
    # and the emitted source must actually contain the bias statement
    src = CG.emit_kernel_source(CG.SKELETONS["affine_plus"], {"k": 2.0})
    record("affine_plus/bias statement emitted", "bias = tl.where" in src
           and "+ bias" in src, "")
    # The floor must NOT be applied: an additive fold never divides by the producer.
    record("affine_plus/producer is unfloored",
           CG.denominator_expr(CG.SKELETONS["affine_plus"]) == "m_new", "")
    # A max consumer must still be inside the template's guard for the first block.
    record("affine_plus/registered + supported",
           "affine_plus" in CG.supported(), "")


def test_correctness() -> None:
    """Run the generated kernels and compare against the closed-form reference."""
    print("\n--- correctness")
    shapes = [(1, 1), (1, 7), (8, 33), (5, 128), (4, 1024), (16, 4096), (3, 1000)]
    dists = {
        "normal": lambda *s: torch.randn(*s, device="cuda"),
        # all negative: the case the multiplicative-only generator got wrong for this family,
        # and where a floored producer is most visible
        "all_negative": lambda *s: -torch.rand(*s, device="cuda") - 0.5,
        "all_zero": lambda *s: torch.zeros(*s, device="cuda"),
        "outlier": lambda *s: torch.randn(*s, device="cuda") * 1e-6,
        "huge": lambda *s: torch.randn(*s, device="cuda") * 1e4,
    }
    # k < 0 is included on purpose: it is the case where max_j(k*c_j) selects the MINIMUM
    # of c, so a reference that assumes k > 0 is silently wrong (see _references).
    for name, k in (("affine_plus", 2.0), ("affine_plus", -0.5), ("affine_plus", -3.0),
                    ("affine_shift", None)):
        params = {"k": k} if k is not None else {}
        for shape in shapes:
            for dname, gen in dists.items():
                x = gen(*shape)
                if shape == (3, 1000) and dname == "outlier":
                    continue
                try:
                    got = CG.run(name, x, block=256, **params)
                    ref = _references(name, x, k)
                    # scale-aware: these operators are homogeneous, so a relative test is the
                    # meaningful one; clamp only to avoid dividing by an exact zero.
                    denom = ref.abs().clamp_min(1e-3 * max(1.0, x.abs().max().item()))
                    err = ((got - ref).abs() / denom).max().item()
                except Exception as e:  # noqa: BLE001
                    record(f"{name} k={k} {shape} {dname}", False,
                           f"{type(e).__name__}: {str(e)[:90]}")
                    continue
                # slack 2e-5: fp32 with a running max of |x| ~ 1e4 loses a few ulp on the
                # repair, and the reference itself is exact by construction.
                record(f"{name} k={k} {shape} {dname}", err < 2e-5, f"rel_err={err:.3e}")


def test_first_block_identity() -> None:
    """A single-block row must never see the bias: the guard has to zero it.

    With one block the accumulator is still the initial -inf when the repair runs, so the
    guard `tl.where(m == -inf, ...)` is what makes the repair the identity. If it were
    missing, `-inf * scale + bias` would still be -inf for a finite bias (so this would pass
    by accident) -- which is why the real check is that a one-element row, where the operator
    is `k*x + x`, comes out exactly right.
    """
    print("\n--- first-block identity")
    x = torch.tensor([[3.0], [-2.5], [0.0]], device="cuda")
    got = CG.run("affine_plus", x, block=256, k=2.0)
    ref = 3.0 * x.amax(dim=-1)
    err = (got - ref).abs().max().item()
    record("one-element rows avoid the bias", err < 1e-6, f"err={err:.3e} got={got.tolist()}")


def test_negative_control() -> None:
    """Change the reference and the test must fail: a guard against a blind suite."""
    print("\n--- negative control")
    x = torch.randn(8, 64, device="cuda")
    got = CG.run("affine_plus", x, block=256, k=2.0)
    good = (got - _references("affine_plus", x, 2.0)).abs().max().item()
    # deliberately wrong reference: pretend the slope is still 1 (the multiplicative answer)
    bad = (got - _references("affine_plus", x, 0.0)).abs().max().item()
    record("wrong reference is detected", good < 1e-5 < bad, f"good={good:.3e} bad={bad:.3e}")


def main() -> int:
    # Ask before describing: `get_device_name` raises on a build without CUDA, so a CPU-only
    # machine used to get a traceback from the banner instead of a one-line explanation.
    if not torch.cuda.is_available():
        print("GPU required: this validates generated Triton kernels, which need a CUDA "
              "device. Everything else in the repository runs on CPU "
              "(tests/, tools/coverage_codegen.py, examples/).")
        return 1
    print("=" * 100)
    print("affine-repair GPU validation")
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"{torch.cuda.get_device_name(0)}")
    print("=" * 100)
    try:
        _register_families()
        test_derivation()
        test_correctness()
        test_first_block_identity()
        test_negative_control()
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
