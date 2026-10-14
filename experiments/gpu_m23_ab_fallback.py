"""
I8 verification: does the A/B fallback actually FIRE when the fusion is slower?

An A/B that always picks the fused package is not a fallback, and an A/B that always picks the
baseline is not a fusion. Both directions have to be shown, on the machine, with real AOTI packages:

  §1 MECHANISM  a case that fuses: both packages are built, both are timed, and the one that ships is
                the faster of the two -- in whichever direction the machine decides. (On the 5090 this
                is the 1.54x case; on the 4090 D the same shape measured 0.97x, i.e. the fusion did
                NOT win there, and the fallback is exactly what should handle that.)
  §2 LOSES      `var(unbiased=False)` at (1, 262144) -- the measured 0.30x case (exp31) -- must ship
                the UNMODIFIED baseline, and what comes back must still compute the right answer.
                NB the DEFAULT `x.var(-1)` is not this case: it is refused by design, because its
                default is the /(N-1) form while the kernel computes /N.
  §3 NO-OP  a model with nothing to fuse must compile ONE package and time nothing

Run on a CUDA host:  python experiments/gpu_m23_ab_fallback.py
"""
from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neptune_opt as NO  # noqa: E402

_failed, _passed = [], 0


def check(label, ok, detail=""):
    global _passed
    if ok:
        _passed += 1
    else:
        _failed.append(label)
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")


class M(nn.Module):
    def __init__(self, fn, build=None):
        super().__init__()
        self.fn = fn
        if build is not None:
            self.sub = build()

    def forward(self, t):
        return self.fn(t)


def _run(label, module, x, expect_fusion, dtypes=None):
    try:
        choice = NO.compile_best(module, (x,), verbose=True)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        check(f"{label}: compile_best did not raise", False)
        return None
    print(f"      -> {choice}")
    check(f"{label}: the fused package was built", bool(choice.report.fusions),
          f"fusions={[f.skeleton for f in choice.report.fusions]}")
    if expect_fusion is None:
        # Mechanism: the shipped package must be the faster one, whichever that is.
        faster_is_fused = choice.fused_ms < choice.baseline_ms
        check(f"{label}: shipped the faster of the two packages",
              choice.used_fusion is faster_is_fused,
              f"used_fusion={choice.used_fusion}, {choice.speedup:.2f}x, {choice.reason}")
    else:
        check(f"{label}: chose the {'FUSED' if expect_fusion else 'BASELINE'} package",
              choice.used_fusion is expect_fusion,
              f"used_fusion={choice.used_fusion}, {choice.speedup:.2f}x, {choice.reason}")
    check(f"{label}: both packages exist", os.path.exists(choice.baseline_package)
          and (choice.fused_package is None or os.path.exists(choice.fused_package)))
    # whatever was shipped must be loadable and numerically right
    fn = torch._inductor.aoti_load_package(choice.package_path)
    with torch.no_grad():
        got = fn(x)
        ref = module(x)
    got = got[0] if isinstance(got, (tuple, list)) else got
    ref = ref[0] if isinstance(ref, (tuple, list)) else ref
    err = (got.float() - ref.float()).abs().max().item()
    check(f"{label}: the shipped package matches eager", err < 1e-2, f"max|diff|={err:.2e}")
    if dtypes is not None:
        check(f"{label}: dtype preserved", got.dtype == ref.dtype, f"{got.dtype} vs {ref.dtype}")
    return choice


def main() -> int:
    if not torch.cuda.is_available():
        print("NO CUDA: compile_best needs a device to time on")
        return 2
    print("=" * 110)
    print(f"I8: THE A/B FALLBACK   torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    print("=" * 110)

    print("\n[1] MECHANISM: torch.logsumexp, 256x16384 -- an A/B whose direction is the machine's call")
    x = torch.randn(256, 16384, device="cuda")
    _run("logsumexp", M(lambda t: torch.logsumexp(t, dim=-1)).eval().cuda(), x, None)

    print("\n[2] LOSES: var(unbiased=False) at (1, 262144) -- the measured 0.30x case")
    x2 = torch.randn(1, 262144, device="cuda")
    _run("var(1, 262144)", M(lambda t: torch.var(t, dim=-1, unbiased=False)).eval().cuda(), x2, False)

    print("\n[3] NO-OP: a model with nothing to fuse pays for ONE compilation")
    import tempfile

    d = tempfile.mkdtemp(prefix="i8_noop_")
    x3 = torch.randn(64, 4096, device="cuda")
    choice = NO.compile_best(M(lambda t: torch.relu(t) + 1.0).eval().cuda(), (x3,), package_dir=d,
                             verbose=True)
    packages = [f for f in os.listdir(d) if f.endswith(".pt2")]
    check("no-op: exactly one package was compiled", len(packages) == 1, f"{packages}")
    check("no-op: the baseline was shipped", choice.used_fusion is False and "nothing to fuse" in choice.reason)
    check("no-op: nothing was timed", choice.baseline_ms != choice.baseline_ms)   # nan

    print()
    print("=" * 110)
    print(f"TOTAL: {_passed} passed, {len(_failed)} failed")
    if _failed:
        for f in _failed:
            print(f"  FAILED: {f}")
        return 1
    print("READING: the fallback fires in the direction that matters -- a fusion measured 3x slower")
    print("is dropped and the unmodified AOTI package is what ships.")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
