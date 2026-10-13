"""
Iteration I1: the SYMBOLIC row length, for every in-frame skeleton.

`gpu_m15_integration.py` §E shows that one generated kernel serves several widths, but it reaches
that kernel through a STATIC export. The symbolic path -- `torch.export` with a `Dim` on the row
length -- is a different code path, and it is the one a user with variable-length rows takes:

    * the row length is a free symbol rather than a number, so the recogniser must match against
      `$ncols` instead of solving for a value (this is what `intrinsic_params` is for, and the
      earlier `int(shape[-1])` crash lived here);
    * the kernel must not be specialised to the width it was compiled with, or the `.pt2` would
      quietly compute the wrong thing at another width.

Verified for every supported skeleton here, rather than for `rmsnorm_max` alone as before.

Run on the GPU host:  python3 gpu_m18_dynamic_shapes.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
import zipfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neptune_codegen as CG  # noqa: E402
import neptune_opt as NO  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []

# Widths deliberately include values that are NOT the example width and are not multiples of the
# block size, because a shape bug shows up at exactly those.
WIDTHS = (1024, 999, 333, 7)
EXAMPLE = (64, 1024)


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, PASS if ok else FAIL, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


class M(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, t):
        return self.fn(t)


def _module_for(name: str):
    """The DECOMPOSED torch form of a registered operator, from the project's own references.

    Using `graph_reference` rather than hand-written modules is deliberate: it is the same
    definition the kernels are checked against, so this cannot drift from what the registry
    claims to support.
    """
    vals = CG.param_defaults(name)

    def fn(t):
        return CG.graph_reference(name, t, **vals)

    return fn, vals


def _dim():
    return {1: torch.export.Dim("w", min=2)}


def test_one_by_one() -> None:
    print("\n--- symbolic row length, per skeleton")
    print(f"    {'skeleton':14s} {'fused':14s} {'mentions':>8s} "
          + " ".join(f"{'w' + str(w):>10s}" for w in WIDTHS))
    for name in CG.supported():
        fn, vals = _module_for(name)
        try:
            mod = M(fn).eval().cuda()
            ex = torch.randn(*EXAMPLE, device="cuda")
            # Symbolic export: dim 1 is a Dim, not a constant.
            ep = NO.export(mod, (ex,), dynamic_shapes=({1: torch.export.Dim("w", min=2)},))
            report = NO.LAST_REPORT
            if not report.fusions:
                print(f"    {name:14s} NOT FUSED: {report.summary().strip()[:56]}")
                record(f"{name}/symbolic fusion", False, "no fusion")
                continue
            sk = report.fusions[0].skeleton
            pkg = os.path.join(tempfile.mkdtemp(prefix=f"m18_{name}_"), "m.pt2")
            torch._inductor.aoti_compile_and_package(ep, package_path=pkg)
            with zipfile.ZipFile(pkg) as z:
                raw = b"".join(z.read(n) for n in z.namelist())
                cubins = sum(1 for n in z.namelist() if n.endswith(".cubin"))
            mentions = raw.count(sk.encode())
            fn_pkg = torch._inductor.aoti_load_package(pkg)
            errs, worst = [], 0.0
            for w in WIDTHS:
                t = torch.randn(64, w, device="cuda") * 3.0 + 1.0
                with torch.no_grad():
                    got = fn_pkg(t)
                got = got[0] if isinstance(got, (tuple, list)) else got
                ref = CG.reference(name, t, **vals)
                rel = ((got - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
                errs.append(rel)
                worst = max(worst, rel)
            print(f"    {name:14s} {sk:14s} {mentions:8d} "
                  + " ".join(f"{e:10.2e}" for e in errs))
            record(f"{name}/symbolic fusion", True, sk)
            record(f"{name}/kernel in the .pt2", mentions >= 1 and cubins >= 1,
                   f"{mentions}x, cubins={cubins}")
            # ONE kernel, compiled once, correct at every width. This is the property that a
            # baked-in row length would break -- silently, since it would still return numbers.
            record(f"{name}/same kernel for every width", worst < 1e-3,
                   f"worst rel {worst:.2e} over {WIDTHS}")
        except Exception as e:  # noqa: BLE001
            print(f"    {name:14s} ERROR {type(e).__name__}: {str(e)[:60]}")
            record(f"{name}/symbolic path", False, f"{type(e).__name__}: {str(e)[:50]}")


def test_no_baked_shape() -> None:
    """A static counterpart to the width sweep, and an honest one about its own weakness.

    The real test of shape independence is the sweep above: ONE compiled kernel, several widths.
    All this can do statically is confirm that a skeleton which declares a shape-coupled
    parameter actually emits the kernel's own `ncols` rather than something derived from a value
    -- which is the mechanism, not the property.
    """
    print("\n--- shape-coupled parameters are emitted as the kernel's `ncols`")
    for name in CG.supported():
        sk = CG.SKELETONS[name]
        src = CG.emit_kernel_source(sk, CG.param_defaults(name))
        if not sk.intrinsic_params:
            continue
        record(f"{name}/emits ncols", "ncols" in src,
               ", ".join(f"{sym}->{n}" for sym, n in sk.intrinsic_params))


def main() -> int:
    if not torch.cuda.is_available():
        print("GPU required: this packages and runs the generated kernels.")
        return 1
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
    print("=" * 100)
    print("ITERATION I1 -- symbolic row length for every in-frame skeleton")
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    print("=" * 100)
    try:
        test_one_by_one()
        test_no_baked_shape()
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
