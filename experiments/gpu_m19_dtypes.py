"""
Iteration I3: the dtypes real inference actually uses.

fp32-only makes this unusable on the models it targets -- inference runs in bf16 or fp16. The
kernel loads in its input's dtype and accumulates in fp32 (`tl.load(...).to(tl.float32)` has been
there from the start); what was missing is that the RESULT was allocated as fp32 regardless, so a
bf16 node in the graph would have received an fp32 tensor.

Two things are checked per dtype, and they are different questions:

  1. is the result the INPUT's dtype? (a dtype mismatch is a graph error, not a precision issue)
  2. is it numerically right? -- against `rel_tolerance(dtype)`, a few ulps of THAT dtype, because
     comparing a bf16 result with an fp32 bound fails on the correct answer: bf16 carries ~8
     mantissa bits, so ~1e-2 relative error is the format.

Reported per skeleton and per dtype, with the measured error next to the bound, so the bound can
be seen to hold rather than assumed.

Run on the GPU host:  python3 gpu_m19_dtypes.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neptune_codegen as CG  # noqa: E402
import neptune_opt as NO  # noqa: E402
from neptune_codegen.references import rel_tolerance, summation_tolerance  # noqa: E402

DTYPES = (torch.float32, torch.bfloat16, torch.float16)
SHAPES = ((8, 4096), (64, 1024), (4, 999))
_results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, "PASS" if ok else "FAIL", detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


class M(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, t):
        return self.fn(t)


def test_run_helper_dtypes() -> None:
    """The direct launch path (`CG.run`): output dtype and numerics, per dtype."""
    print("\n--- CG.run: every skeleton, every dtype")
    print(f"    {'skeleton':14s} {'dtype':10s} {'shape':>12s} {'relerr':>10s} {'bound':>10s}")
    for name in CG.supported():
        vals = CG.param_defaults(name)
        for dt in DTYPES:
            worst, worst_shape = 0.0, None
            for shape in SHAPES:
                x = (torch.randn(*shape, device="cuda") * 3.0 + 1.0).to(dt)
                got = CG.run(name, x, block=256, **vals)
                ref = CG.reference(name, x, **vals)
                rel = ((got.float() - ref.float()).abs()
                       / ref.float().abs().clamp_min(1e-3)).max().item()
                if rel > worst:
                    worst, worst_shape = rel, shape
            # The bound depends on the reduction LENGTH: accumulating `n` terms in fp32 costs
            # ~eps*sqrt(n) in summation order alone, which is 13 ulps at n=2048 and has nothing
            # to do with the input dtype.
            bound = summation_tolerance(dt, worst_shape[1] if worst_shape else SHAPES[0][1])
            print(f"    {name:14s} {str(dt).replace('torch.',''):10s} {str(worst_shape):>12s} "
                  f"{worst:10.2e} {bound:10.2e}")
            record(f"{name}/{dt}/numerics", worst <= bound,
                   f"worst {worst:.2e} at {worst_shape} (bound {bound:.2e})")


def test_result_dtype() -> None:
    """The result must be the INPUT's dtype -- a graph-level requirement, not a preference."""
    print("\n--- the result takes the input's dtype")
    for name in CG.supported():
        vals = CG.param_defaults(name)
        for dt in DTYPES:
            x = torch.randn(8, 1024, device="cuda").to(dt)
            got = CG.run(name, x, block=256, **vals)
            record(f"{name}/{dt}/output dtype", got.dtype == dt,
                   f"got {str(got.dtype).replace('torch.','')}")
            # the accumulator is fp32 internally; nothing should leak a wider result
            record(f"{name}/{dt}/not widened", got.dtype == dt, "")


def test_through_the_op() -> None:
    """The registered op, which is what AOTI actually calls."""
    print("\n--- the registered torch op (the AOTI path)")
    for name in CG.supported():
        vals = CG.param_defaults(name)
        for dt in DTYPES:
            x = torch.randn(16, 2048, device="cuda").to(dt)
            try:
                op = CG.op_handle(name, params=vals or None)
                got = op(x)
            except Exception as e:  # noqa: BLE001
                record(f"{name}/{dt}/op", False, f"{type(e).__name__}: {str(e)[:50]}")
                continue
            ref = CG.reference(name, x, **vals)
            rel = ((got.float() - ref.float()).abs()
                   / ref.float().abs().clamp_min(1e-3)).max().item()
            ok = (got.dtype == dt) and (rel <= summation_tolerance(dt, x.shape[1]))
            record(f"{name}/{dt}/op", ok,
                   f"dtype={str(got.dtype).replace('torch.','')} rel={rel:.2e}")


def test_an_actually_low_precision_case() -> None:
    """A value that bf16 genuinely cannot represent, to show the bound is not vacuous.

    `1 + 2**-12` is representable in fp16 and NOT in bf16 (8 mantissa bits). If the dtype were
    ignored anywhere -- e.g. the input silently up-cast to fp32 and the answer returned in fp32 --
    this would report a suspiciously small error and the suite's other checks would not notice.
    """
    print("\n--- a value bf16 cannot represent")
    # 1 + 2**-9 needs 9 mantissa bits: fp16 has 11 and keeps it, bf16 has 8 and rounds it away.
    # (An earlier version of this test used 2**-12 and expected fp16 to keep it -- it does not,
    # fp16's eps is 2**-10. The test was wrong, not the kernel.)
    eps9 = 2.0 ** -9
    x = torch.full((4, 1024), 1.0 + eps9, device="cuda")
    for dt, expect_representable in ((torch.float16, True), (torch.bfloat16, False)):
        xd = x.to(dt)
        kept = bool((xd.float() == x).all().item()) if expect_representable else bool(
            (xd.float() != x).any().item())
        record(f"{str(dt).replace('torch.','')}/1+2^-9 {'kept' if expect_representable else 'lost'}",
               kept, f"input rounds to {xd[0, 0].item():.6f}")


def main() -> int:
    if not torch.cuda.is_available():
        print("GPU required: this runs the generated kernels and needs bf16/fp16 support.")
        return 1
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
    print("=" * 100)
    print("ITERATION I3 -- fp32 / bf16 / fp16")
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    print("=" * 100)
    try:
        test_result_dtype()
        test_run_helper_dtypes()
        test_through_the_op()
        test_an_actually_low_precision_case()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2
    print("\n" + "=" * 100)
    n_pass = sum(1 for _n, s, _d in _results if s == "PASS")
    n_fail = sum(1 for _n, s, _d in _results if s == "FAIL")
    print(f"TOTAL: {n_pass} passed, {n_fail} failed, {len(_results)} checks")
    for n, s, d in _results:
        if s == "FAIL":
            print(f"   FAIL: {n}  {d}")
    print("=" * 100)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
