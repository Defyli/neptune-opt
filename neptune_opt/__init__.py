"""
neptune_opt -- Neptune-style break-and-repair fusion as a usable module.

What it does
------------
Finds a reduction pair in a model (a producer reduction whose result is consumed inside a
second reduction), asks the algebraic registry in `neptune_codegen` which operator that is,
generates a single-pass Triton kernel for it from algebra, and rewires the graph to call
that kernel.

Why there are exactly two entry points
-------------------------------------
The placement of the rewrite is not a stylistic choice, it is forced by how the compiler
records user-defined Triton kernels. Measured on the target machine
(`gpu_diag_paths.py`, torch 2.11.0+cu128):

    op created during tracing                 -> 41 occurrences of the kernel in the .pt2
    op inserted into an exported program      ->  0
    op inserted then the graph re-exported    -> 41

A `torch.library.triton_op` becomes the Triton HOP only while `FunctionalTensorMode` is
active, i.e. during AOTAutograd/export tracing. Every `*_custom_pass` hook that Inductor
exposes (`post_grad_custom_pre_pass`, `joint_custom_pre_pass`, `pre_grad_custom_pass`, ...)
runs AFTER that, so none of them can be used no matter how convenient they look.

That leaves two real integration points, both of which are ordinary public APIs:

    optimize(module, example_inputs)   rewrite before `torch.export`, for the AOTI path
    backend(gm, example_inputs)        a `torch.compile` backend; `compile_fx` calls
                                       `aot_autograd` internally, so rewriting the Dynamo
                                       graph first lands the op inside tracing

Safety model
------------
Three independent gates, because a fusion pass that is occasionally wrong is worse than no
pass at all:

  1. REGION GATE  the recognised operator must be the graph, not a fragment of it. If the
                  result is combined with other data (a residual add, a matmul) the pass
                  refuses rather than delete that computation. See
                  `neptune_recognize.region_root`.
  2. VERIFY GATE  the rewritten module is executed and compared against the original on the
                  example inputs; on any mismatch the ORIGINAL module is returned.
  3. FAILURE GATE nothing raises: any unexpected exception at any stage returns the original
                  module with the reason recorded in the report.

Known limits (not bugs, but real)
---------------------------------
  * The operator must BE the computation handed to the pass. Fusing a region embedded in a
    larger model needs the `finalize` to be matched too; not implemented.
  * Static shapes only: a skeleton whose parameter comes from the row length (RMSNorm's N)
    cannot be resolved when the shape is symbolic, and is refused.
  * Not supported by the algebra at all: `NEEDS_STATE` families (variance / Welford /
    LayerNorm), a `mul` consumer (needs a carried count), and a runtime-variable exponent.
"""
from __future__ import annotations

import dataclasses
import functools
import sys
import warnings
from typing import Any

import torch

import neptune_codegen as CG

# The sibling packages live next to this one in a clone. Make them importable when running
# straight from the repository WITHOUT shadowing an installed copy.
import importlib.util as _ilu
import os as _os
import sys as _sys

if _ilu.find_spec("neptune_codegen") is None:
    _PARENT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _PARENT not in _sys.path:
        _sys.path.insert(0, _PARENT)

from neptune_recognize import rewrite as _rewrite, recognize as _recognize

# The A/B fallback lives in its own module (it needs compilation and timing, nothing else here
# does); it is imported after the definitions above so it can call back into `apply`.
from neptune_opt.ab import Choice, choose_faster, compile_best  # noqa: E402

__all__ = ["apply", "export", "optimize", "backend", "preview", "Report", "Fusion",
           "LAST_REPORT", "op_parameters", "compile_best", "Choice"]

# The most recent report produced by any entry point. `backend` has nowhere to return one
# (torch.compile owns its return value), and a fusion pass that cannot be observed is a
# fusion pass nobody can debug.
LAST_REPORT: Report = None


@dataclasses.dataclass
class Fusion:
    skeleton: str
    params: dict

    def __str__(self) -> str:
        if not self.params:
            return self.skeleton
        return f"{self.skeleton}({', '.join(f'{k}={v:g}' for k, v in sorted(self.params.items()))})"


@dataclasses.dataclass
class Report:
    """What the pass did, and what it declined to do. Never empty: silence is not a result."""

    fusions: list = dataclasses.field(default_factory=list)
    skipped: list = dataclasses.field(default_factory=list)   # (reason, detail)
    verified: bool = False

    def __bool__(self) -> bool:
        return bool(self.fusions)

    def summary(self) -> str:
        lines = []
        for f in self.fusions:
            lines.append(f"  fused   : {f}")
        for reason, detail in self.skipped:
            lines.append(f"  skipped : {reason}: {detail}")
        if self.fusions and self.verified:
            lines.append("  verified: fused output matches the original on the example inputs")
        return "\n".join(lines) or "  (nothing to do)"


def _outputs(x: Any) -> tuple:
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _verify(module: torch.nn.Module, gm: torch.fx.GraphModule, args: tuple) -> tuple[bool, str]:
    """Execute both and compare. This gate is what makes the pass safe to leave switched on."""
    try:
        with torch.no_grad():
            a = _outputs(module(*args))
            b = _outputs(gm(*args))
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
            err = (x.float() - y.float()).abs().max().item()
            return False, f"output {i} max|diff|={err:.3e}"
    return True, ""


def _trace(module: torch.nn.Module, args: tuple) -> torch.fx.GraphModule:
    """Trace to an ATen FX graph.

    `make_fx`, not `symbolic_trace`: a symbolically traced graph carries no shape metadata,
    and the recogniser reads the row length from the input's shape to resolve parameters
    such as RMSNorm's N (measured: symbolic_trace gives "param-unsolved [N]").
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            return make_fx(module)(*args)
    finally:
        module.train(was_training)


def _rewrite_graph(gm: torch.fx.GraphModule, values: dict | None = None) -> tuple:
    """Rewrite one traced graph, returning (gm, Report). Never raises."""
    report = Report()
    try:
        match, reason = _recognize(gm, values)
    except Exception as e:  # noqa: BLE001
        report.skipped.append(("recognise-failed", f"{type(e).__name__}: {e}"))
        return gm, report
    if match is None:
        report.skipped.append(("no-fusion", reason))
        return gm, report
    try:
        gm2, name, reason = _rewrite(gm, values)
    except ModuleNotFoundError as e:
        # The common case on a CPU-only machine, and worth its own message: recognition
        # SUCCEEDED, only the kernel could not be generated. Reporting this as a generic
        # failure would hide the interesting half of the result.
        if "triton" in str(e):
            report.skipped.append((
                "triton-missing",
                f"recognised {match.skeleton!r}, but generating its Triton kernel needs Triton "
                "(pip install triton)",
            ))
            return gm, report
        report.skipped.append(("rewrite-failed", f"{type(e).__name__}: {e}"))
        return gm, report
    except Exception as e:  # noqa: BLE001
        report.skipped.append(("rewrite-failed", f"{type(e).__name__}: {e}"))
        return gm, report
    if name is None:
        report.skipped.append(("refused", reason))
        return gm, report
    report.fusions.append(Fusion(name, dict(match.params)))
    return gm2, report


def apply(exported_program, values: dict | None = None,
          verify_with: tuple | None = None, lower: bool = True) -> Report:
    """Rewrite a recognised operator inside an `ExportedProgram`, **in place**.

    This is the recommended entry point. Insert it into an existing AOTI flow:

        ep = torch.export.export(model, args)
        report = neptune_opt.apply(ep)
        torch._inductor.aoti_compile_and_package(ep, package_path="m.pt2")

    Why it works at this point: `aoti_compile_and_package` re-runs `aot_autograd` on the
    exported program, and that functionalisation step is what turns the generated op into the
    Triton HOP. Rewriting the exported graph therefore reaches codegen without a re-export --
    measured at 41 kernel occurrences in the `.pt2`, 4 runs out of 4 (the Inductor
    `post_grad_custom_pre_pass` hook, by contrast, comes out with `cubins=0`).

    `verify_with=(module, example_inputs)` adds the numeric self-check, which is otherwise
    unavailable here because `apply` has no access to the original module.

    `lower=True` (the default) first tries to expand compound ops the core decomposition table
    leaves alone -- currently `aten._log_softmax`, whose log-sum-exp otherwise has no node for the
    region gate to replace. It is committed only when it enables a fusion, so a graph it cannot
    help is left byte-identical. See `neptune_recognize/lowering.py` and the measurement in
    `experiments/gpu_m20_i4_ceiling.py` for why softmax and layer_norm are deliberately excluded.

    **Real modules need the graph decomposed first.** `nn.LayerNorm`, `nn.Softmax` and
    `torch.logsumexp` export as ONE high-level aten op rather than as a producer/consumer pair,
    and this frame needs the pair. `neptune_opt.export(..., decompose=True)` does that for you;
    on an already-exported program, call `ep = ep.run_decompositions()` before `apply`. When a
    refusal looks like this case, the report says so rather than leaving the user to guess.
    """
    global LAST_REPORT
    if lower:
        # Expand the compound ops the core decomposition table leaves alone (`_log_softmax`), so
        # the log-sum-exp inside them becomes a node the region gate can replace. It is only
        # KEPT when it enables a fusion: `lower_if_it_helps` trials on a copy, because the
        # decomposed form of an operator that does not fuse is more kernels than the operator.
        from neptune_recognize.lowering import lower_if_it_helps

        report_lowered = lower_if_it_helps(exported_program.graph_module, values)
        if report_lowered:
            report = _rewrite_graph(exported_program.graph_module, values)[1]
            LAST_REPORT = report
            if report.fusions and verify_with is not None:
                module, args = verify_with
                ok, detail = _verify(module, exported_program.graph_module, tuple(args))
                report.verified = ok
            return report
    report = _rewrite_graph(exported_program.graph_module, values)[1]
    if not report.fusions and _recognize(exported_program.graph_module, values)[0] is None:
        # The given graph was not recognised AT ALL, so try a decomposed copy purely to
        # diagnose: if it matches there, the graph was simply not decomposed yet, and saying so
        # is far more useful than "no match".
        #
        # Both halves of that condition matter. Firing on "no fusions" alone was wrong: a graph
        # that WAS recognised but could not be generated (no Triton installed, say) has no
        # fusions either, and the report then claimed it "needs decomposition" while the very
        # next line said it had already recognised the operator. Measured on the dynamic-shape
        # cases, where every report carried both lines at once.
        try:
            dec = exported_program.run_decompositions()
            m, _why = _recognize(dec.graph_module, values)
        except Exception:  # noqa: BLE001 -- a diagnosis must never break the call
            m = None
        if m is not None:
            report.skipped.insert(0, (
                "needs-decomposition",
                f"would recognise {m.skeleton!r} after the core-aten decompositions: call "
                f"ep = ep.run_decompositions() first, or use "
                f"neptune_opt.export(..., decompose=True)",
            ))
    LAST_REPORT = report
    if report.fusions and verify_with is not None:
        module, args = verify_with
        ok, detail = _verify(module, exported_program.graph_module, tuple(args))
        report.verified = ok
        if not ok:
            # The graph is already mutated; the caller is told the rewrite is untrustworthy
            # rather than being handed a silent wrong answer.
            report.fusions, report.skipped = [], [("verify-failed", detail)]
    return report


def preview(exported_program, values: dict | None = None):
    """What WOULD be fused? Recognition only -- no kernel generation, so no Triton needed.

    Useful on a CPU machine and in CI, where installing Triton is not an option.
    """
    match, reason = _recognize(exported_program.graph_module, values)
    return ([match] if match is not None else []), reason


def export(module: torch.nn.Module, example_inputs: tuple, decompose: bool = True,
           **export_kwargs):
    """`torch.export.export` followed by `apply`, returning the ExportedProgram.

    A drop-in replacement for `torch.export.export` that also fuses what it recognises. All
    other keyword arguments are forwarded to `torch.export.export` (`dynamic_shapes`, `strict`,
    ...), so the caller keeps whatever export semantics they already relied on.

    `decompose=True` (the default) runs the CORE-ATEN DECOMPOSITIONS first, and for real models
    that is not optional: `nn.LayerNorm`, `nn.Softmax` and `torch.logsumexp` each export as one
    high-level op, which this frame cannot see. The decomposed form is also what AOTAutograd
    would hand Inductor anyway, so compiling it is the normal path rather than a detour. Pass
    `decompose=False` to fuse only what is already spelled out in the graph.
    """
    ep = torch.export.export(module, tuple(example_inputs), **export_kwargs)
    if decompose:
        ep = ep.run_decompositions()
    report = apply(ep, verify_with=(module, tuple(example_inputs)))
    if report.fusions and not report.verified:
        # optimise() would have returned the original module here; export cannot un-export,
        # so the failure is surfaced instead of hidden.
        warnings.warn(f"neptune_opt: fusion failed verification and was left in the graph: "
                      f"{report.summary()}", RuntimeWarning, stacklevel=2)
    return ep


def op_parameters(name: str) -> tuple:
    """The scalar parameters a fused operator REQUIRES, by keyword name.

    Only parameters with no declared default are listed. Two other kinds are deliberately
    absent: shape-coupled parameters (RMSNorm's row length) come from the kernel's own runtime
    arguments -- which is what makes it shape-independent -- and a parameter with a declared
    default (the log readout's additive constant, which is 0) is filled in automatically.
    """
    sk = CG.SKELETONS[name]
    declared = {sym for sym, _value in sk.defaults}
    return tuple(arg for sym, arg in sk.params if sym not in declared)


def _make_op(name: str):
    def op(x, **params):
        missing = [p for p in op_parameters(name) if p not in params]
        if missing:
            raise TypeError(
                f"neptune_opt.{name}() needs {', '.join(missing)}; "
                f"the fused kernel bakes scalar parameters into its source as literals"
            )
        values = {**CG.param_defaults(name), **params}
        return CG.op_handle(name, params=values or None)(x)

    op.__name__ = name
    op.__qualname__ = name
    op.__doc__ = (f"Fused {name} in a single pass, using a kernel generated from algebra.\n\n"
                  f"Scalar parameters: {op_parameters(name) or '(none)'}")
    return op


# The explicit operator library. This is the zero-risk path: no recognition, no graph
# rewriting -- the caller names the operator and the kernel is generated for it. Useful when
# recognition declines, and useful as a sanity check on what recognition would have chosen.
for _op_name in CG.supported():
    globals()[_op_name] = _make_op(_op_name)
__all__ += [n for n in CG.supported()]


def optimize(module: torch.nn.Module, example_inputs: tuple, *, verbose: bool = False):
    """Legacy form: returns a rewritten module that the caller then exports.

    Superseded by `export()` / `apply()`. It is kept because it verifies numerically before
    returning, which is a useful property, but it costs an extra export: the caller re-exports
    the module this returns. `export()` reaches the same graph in one step and inherits the
    caller's own export arguments, so prefer it.

    On any failure the ORIGINAL module is returned, so the result is always usable.
    """
    warnings.warn(
        "neptune_opt.optimize() is superseded by neptune_opt.export() / neptune_opt.apply(): "
        "they rewrite the ExportedProgram directly and avoid an extra export. optimize() now "
        "costs one export here plus the caller's re-export.",
        DeprecationWarning, stacklevel=2,
    )
    report = Report()
    try:
        ep = torch.export.export(module, tuple(example_inputs))
    except Exception as e:  # noqa: BLE001
        return module, Report(skipped=[("export-failed", f"{type(e).__name__}: {e}")])
    report = apply(ep, verify_with=(module, tuple(example_inputs)))
    if not report.fusions:
        return module, report
    if verbose:
        print(report.summary(), file=sys.stderr)
    return ep.graph_module, report


def backend(gm: torch.fx.GraphModule, example_inputs: list, experimental: bool = False):
    """A `torch.compile` backend that rewrites the Dynamo graph before AOTAutograd.

        torch.compile(model, backend=functools.partial(neptune_opt.backend, experimental=True))

    NOT YET EFFECTIVE -- measure before using. The rewrite itself works (the graph gets the
    generated op and results are numerically correct), but on the target machine the
    generated kernel does NOT reach codegen on this path: searching the Inductor cache for
    the kernel name finds 0 occurrences in compiled/generated code, against 41 for the same
    operator through `optimize()` + AOTI. An op that is not lowered falls back to launching
    its Python implementation, which is slower than not fusing at all, so this is gated
    behind an explicit flag until the placement is sorted out.

    Rewriting before `compile_fx` is still the right idea: `compile_fx` calls `aot_autograd`
    internally, so the op IS present during functionalisation. Why the decomposition into
    `triton_kernel_wrapper_functional` does not happen on that route is the open question.
    """
    if not experimental:
        raise RuntimeError(
            "neptune_opt.backend is not verified effective yet (the generated kernel does not "
            "reach codegen on this path; see the docstring). Pass experimental=True to try it "
            "anyway, or use neptune_opt.optimize() with torch.export + AOTI, which is verified."
        )
    import torch._inductor

    gm2, report = _rewrite_graph(gm)
    if report.fusions and not report.verified:
        # compile() has no separate verification pass available here; the numeric check the
        # caller can make is comparing against eager. Report the fusion regardless so it is
        # never silently claimed.
        report.verified = False
    global LAST_REPORT
    LAST_REPORT = report
    if report:
        print(f"neptune_opt: {report.summary()}", file=sys.stderr)
    return torch._inductor.compile(gm2, list(example_inputs))
