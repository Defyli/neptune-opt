"""
`compile_best`: the performance fallback.

Every other gate in this package protects CORRECTNESS. This one protects SPEED, because a fusion
that is numerically right can still be slower -- measured: the two-pass plan for `variance`/`std` is
**0.30x** at `(1, 262144)`, and a single very long row is 0.67x against Inductor's own lowering
(docs/evidence/exp31, exp37). Without this, "fused" in the report was all a user got.

The mechanism is the only honest one: build BOTH packages, time them on the machine that will run
them, and ship the faster. There is no way to know in advance -- the cost depends on the shape, the
GPU and the launch configuration -- so it is measured.

The price is real and is paid deliberately: two AOTI compilations instead of one, example inputs, a
CUDA device, and static shapes (an AOTI package is specialised to the shapes it was compiled for).
It is a separate entry point rather than something `apply()` does, because `apply()` runs before any
compilation exists and therefore has nothing to time. When a model has nothing to fuse -- the common
case -- the second compilation is skipped entirely.

    choice = neptune_opt.compile_best(model, (x,))
    print(choice)                      # chosen: fused (1.54x), or: baseline (fused was 0.30x)
    ship(choice.package_path)
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
import time

# How much faster the fused package must be before it is preferred. A tie keeps the baseline: the
# safe direction to be wrong in is "did not change my model", and two packages that measure the same
# do not justify the risk of a different kernel.
MARGIN = 0.03


@dataclasses.dataclass
class Choice:
    """What the A/B measured, and which package to ship."""

    package_path: str                  # the package to ship
    used_fusion: bool
    baseline_ms: float
    fused_ms: float
    speedup: float                     # baseline / fused; > 1 means the fusion was faster
    reason: str
    report: object                     # the `Report` from `apply`
    baseline_package: str
    fused_package: str | None = None

    def __str__(self) -> str:
        if self.baseline_ms != self.baseline_ms:              # nan: nothing was timed
            return f"neptune_opt: {self.reason}"
        which = "fused" if self.used_fusion else "baseline"
        return (f"neptune_opt: shipped the {which} package -- {self.reason} "
                f"(baseline {self.baseline_ms:.4f} ms, fused {self.fused_ms:.4f} ms, "
                f"{self.speedup:.2f}x)")


def choose_faster(baseline_ms: float, fused_ms: float, margin: float = MARGIN) -> tuple:
    """Which package to ship: `(use_fusion, speedup, reason)`. Pure, so it can be tested.

    The fused package is preferred only if it is faster by more than `margin`. Anything else --
    slower, equal, or unmeasurable -- keeps the baseline, which is the whole point of the fallback.
    """
    if not (baseline_ms > 0) or not (fused_ms > 0):
        return False, float("nan"), "one of the two packages could not be timed: keeping the baseline"
    speedup = baseline_ms / fused_ms
    if speedup >= 1.0 + margin:
        return True, speedup, f"the fused package is {speedup:.2f}x faster"
    if speedup <= 1.0 - margin:
        return False, speedup, f"the fused package is {1 / speedup:.2f}x SLOWER, so it was dropped"
    return False, speedup, (f"the two are within the {margin:.0%} margin ({speedup:.2f}x), so the "
                            f"unmodified baseline was kept")


# ---------------------------------------------------------------- the pieces, injectable for tests
def _export_pair(module, args, export_kwargs):
    """(baseline ep, fused ep, Report). Exported twice: `apply` mutates in place, so one object
    cannot be both sides of the A/B."""
    import copy

    import torch

    import neptune_opt

    ep_base = torch.export.export(module, tuple(args), **export_kwargs).run_decompositions()
    try:
        ep_fused = copy.deepcopy(ep_base)
    except Exception:  # noqa: BLE001 -- deepcopy of an ExportedProgram is not guaranteed: re-export
        ep_fused = torch.export.export(module, tuple(args), **export_kwargs).run_decompositions()
    report = neptune_opt.apply(ep_fused, verify_with=(module, tuple(args)))
    return ep_base, ep_fused, report


def _package(ep, directory: str, name: str) -> str:
    import torch

    path = os.path.join(directory, f"{name}.pt2")
    torch._inductor.aoti_compile_and_package(ep, package_path=path)
    return path


def _load(path: str):
    import torch

    return torch._inductor.aoti_load_package(path)


def _outputs(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _same_numbers(fa, fb, args) -> tuple:
    """(ok, detail). Never trust a faster package whose numbers moved."""
    import torch

    try:
        with torch.no_grad():
            a, b = _outputs(fa(*args)), _outputs(fb(*args))
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    if len(a) != len(b):
        return False, f"output arity {len(b)} != {len(a)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            return False, f"output {i} is not a tensor"
        if x.shape != y.shape:
            return False, f"output {i} shape {tuple(y.shape)} != {tuple(x.shape)}"
        if not torch.allclose(x.float(), y.float(), rtol=1e-3, atol=1e-4):
            return False, f"output {i} max|diff|={(x.float() - y.float()).abs().max().item():.3e}"
    return True, ""


def _ab_time(fa, fb, args, reps: int = 20, rounds: int = 3, warmup: int = 3) -> tuple:
    """`(baseline_ms, fused_ms)`, ROUNDS INTERLEAVED and the minimum kept per side.

    Interleaved because a clock change or another tenant on the box must not land on one side only;
    minimum because a stall only ever adds time -- the same reasoning as `autotune.search`, and the
    same bug (a noisy single round) was found there on a shared machine.
    """
    import torch

    with torch.no_grad():
        for _ in range(warmup):
            fa(*args)
            fb(*args)
        torch.cuda.synchronize()
        a, b = [], []
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(reps):
                fa(*args)
            torch.cuda.synchronize()
            a.append((time.perf_counter() - t0) * 1e3 / reps)
            t0 = time.perf_counter()
            for _ in range(reps):
                fb(*args)
            torch.cuda.synchronize()
            b.append((time.perf_counter() - t0) * 1e3 / reps)
    return min(a), min(b)


# ---------------------------------------------------------------- the entry point
def compile_best(module, args, *, package_dir: str | None = None, margin: float = MARGIN,
                 reps: int = 20, rounds: int = 3, verify: bool = True,
                 verbose: bool = True, **export_kwargs) -> Choice:
    """Compile the unmodified AND the fused package, time both, and return the faster one.

    `args` are the example inputs: an AOTI package is specialised to the shapes it was compiled for,
    so these are also the shapes that get timed. `package_dir` defaults to a fresh temporary
    directory holding both packages, so the loser stays inspectable.

    Never raises on a slow fusion: the baseline is returned with the measurement in `reason`.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "compile_best needs a CUDA device: it times two AOTI packages to decide between them. "
            "On a CPU-only machine use neptune_opt.export(..., decompose=True) instead."
        )
    args = tuple(args)
    directory = package_dir or tempfile.mkdtemp(prefix="neptune_ab_")
    os.makedirs(directory, exist_ok=True)

    ep_base, ep_fused, report = _export_pair(module, args, export_kwargs)
    if not report.fusions:
        # Nothing was rewritten, so there is nothing to A/B: paying for a second compilation to
        # discover that would double the cost of every model this pass cannot help.
        path = _package(ep_base, directory, "baseline")
        choice = Choice(path, False, float("nan"), float("nan"), 1.0,
                        f"nothing to fuse ({report.summary().strip() or 'no fusion'}), so no A/B was "
                        f"run and only one package was compiled",
                        report, path, None)
        if verbose:
            print(choice)
        return choice

    base_path = _package(ep_base, directory, "baseline")
    try:
        fused_path = _package(ep_fused, directory, "fused")
    except Exception as e:  # noqa: BLE001 -- a fusion that does not compile must not break the build
        choice = Choice(base_path, False, float("nan"), float("nan"), 1.0,
                        f"the fused package failed to compile ({type(e).__name__}: {str(e)[:80]}), "
                        f"so the baseline was shipped", report, base_path, None)
        if verbose:
            print(choice)
        return choice

    if verify:
        ok, detail = _same_numbers(_load(fused_path), _load(base_path), args)
        if not ok:
            choice = Choice(base_path, False, float("nan"), float("nan"), 1.0,
                            f"the fused package does not reproduce the baseline's numbers "
                            f"({detail}), so it was dropped", report, base_path, fused_path)
            if verbose:
                print(choice)
            return choice

    base_ms, fused_ms = _ab_time(_load(base_path), _load(fused_path), args, reps=reps, rounds=rounds)
    use_fusion, speedup, why = choose_faster(base_ms, fused_ms, margin)
    choice = Choice(fused_path if use_fusion else base_path, use_fusion, base_ms, fused_ms, speedup,
                    why, report, base_path, fused_path)
    if verbose:
        print(choice)
        print(report.summary())
    return choice
