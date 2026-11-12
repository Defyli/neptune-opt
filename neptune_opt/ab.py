"""
`compile_best`: a conservative, measured performance fallback.

Build the unmodified baseline separately from the fusion candidate, verify the actual packages,
and ship the candidate only after a repeatable speed win. Measurements cover only `args` and any
explicit `verification_inputs`; they are not a claim about all dynamic shapes. Dynamic samples
must be supported by the same exported package (use the appropriate export kwargs).
"""
from __future__ import annotations

import copy
import dataclasses
import math
import os
import statistics
import tempfile
import time
from collections.abc import Iterable, Sequence

MARGIN = 0.03


@dataclasses.dataclass
class Choice:
    """What the A/B measured, and which package to ship."""

    package_path: str
    used_fusion: bool
    baseline_ms: float
    fused_ms: float
    speedup: float
    reason: str
    report: object
    baseline_package: str
    fused_package: str | None = None
    sample_metrics: list[dict] | None = None
    selected_plan: str | None = None
    candidate_metrics: list[dict] | None = None

    def __str__(self) -> str:
        if not math.isfinite(self.baseline_ms) or not math.isfinite(self.fused_ms):
            return f"neptune_opt: {self.reason}"
        which = "fused" if self.used_fusion else "baseline"
        return (f"neptune_opt: shipped the {which} package -- {self.reason} "
                f"(baseline {self.baseline_ms:.4f} ms, fused {self.fused_ms:.4f} ms, "
                f"{self.speedup:.2f}x)")


def _validate_nonnegative(name, value):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def choose_faster(baseline_ms: float, fused_ms: float, margin: float = MARGIN) -> tuple:
    """Prefer fusion only for a finite, positive measurement with a win strictly above margin."""
    _validate_nonnegative("margin", margin)
    if not all(math.isfinite(t) and t > 0 for t in (baseline_ms, fused_ms)):
        return False, float("nan"), "one of the two packages could not be timed: keeping the baseline"
    speedup = baseline_ms / fused_ms
    if not math.isfinite(speedup) or speedup <= 0:
        return False, float("nan"), "unreliable timing ratio: keeping the baseline"
    if speedup > 1.0 + margin:
        return True, speedup, f"the fused package is {speedup:.2f}x faster"
    if speedup <= 1.0 - margin:
        return False, speedup, f"the fused package is {1 / speedup:.2f}x SLOWER, so it was dropped"
    return False, speedup, (f"the two are within the {margin:.0%} margin ({speedup:.2f}x), so the "
                            f"unmodified baseline was kept")


# ---------------------------------------------------------------- the pieces, injectable for tests
def _export_pair(module, args, export_kwargs, plan=None, *, baseline=None):
    """Clone the original baseline per plan; failed copies re-export, never globally decompose."""
    import torch

    import neptune_opt

    # Baseline export errors deliberately propagate. Never decompose this side unconditionally.
    ep_base = baseline if baseline is not None else torch.export.export(
        module, tuple(args), **export_kwargs)
    try:
        try:
            ep_fused = copy.deepcopy(ep_base)
            # Refuse incomplete copies rather than risk modifying the original graph or weights.
            if ep_fused is ep_base or ep_fused.graph_module is ep_base.graph_module:
                raise RuntimeError("candidate copy shares the baseline graph")
            for name, value in ep_base.state_dict.items():
                cloned = ep_fused.state_dict[name]
                if value is cloned or (isinstance(value, torch.Tensor) and value.numel()
                                       and value.data_ptr() == cloned.data_ptr()):
                    raise RuntimeError("candidate copy shares baseline weights")
        except Exception:  # noqa: BLE001 -- ExportedProgram deepcopy is version dependent
            # A fresh export of the original module would still share its parameter storage.
            # Copy the module too; if that fails, refuse the candidate rather than share weights.
            ep_fused = torch.export.export(copy.deepcopy(module), tuple(args), **export_kwargs)
        # apply owns targeted lowering; global decomposition would bias the comparison.
        kwargs = {} if plan is None else {"plan": plan}
        report = neptune_opt.apply(ep_fused, verify_with=(module, tuple(args)), **kwargs)
    except Exception as e:  # noqa: BLE001 -- the optional candidate must not break the baseline
        ep_fused = None
        report = neptune_opt.Report(skipped=[("candidate-failed", f"{type(e).__name__}: {e}")])
    return ep_base, ep_fused, report


def _package(ep, directory: str, name: str) -> str:
    import torch

    path = os.path.join(directory, f"{name}.pt2")
    torch._inductor.aoti_compile_and_package(ep, package_path=path)
    return path


def _load(path: str):
    import torch

    return torch._inductor.aoti_load_package(path)


def _same_numbers(fa, fb, args, *, rtol: float = 1e-3, atol: float = 1e-4) -> tuple:
    """Compare tensor pytrees without casting integers, doubles or complex outputs to float32."""
    import torch
    from torch.utils._pytree import tree_flatten

    _validate_nonnegative("rtol", rtol)
    _validate_nonnegative("atol", atol)
    try:
        with torch.no_grad():
            a, spec_a = tree_flatten(fa(*args))
            b, spec_b = tree_flatten(fb(*args))
            if spec_a != spec_b:
                return False, "output pytree structure differs"
            for i, (x, y) in enumerate(zip(a, b)):
                if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                    return False, f"output {i} is not a tensor"
                if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device:
                    return False, f"output {i} shape/dtype/device differs"
                if x.is_floating_point() or x.is_complex():
                    if not (torch.isfinite(x).all().item() and torch.isfinite(y).all().item()):
                        return False, f"output {i} contains nonfinite values"
                    torch.testing.assert_close(x, y, rtol=rtol, atol=atol, equal_nan=False)
                elif not torch.equal(x, y):
                    return False, f"output {i} integer/bool values differ"
    except Exception as e:  # noqa: BLE001 -- execution and comparison are both verification failures
        return False, f"{type(e).__name__}: {e}"
    return True, ""


def _ab_time(fa, fb, args, reps: int = 20, rounds: int = 5, warmup: int = 3) -> tuple:
    """Warm both sides, alternate AB/BA rounds, and retain the median of every measured round."""
    import torch
    from torch.utils._pytree import tree_flatten

    for name, value, minimum in (("reps", reps, 1), ("rounds", rounds, 1), ("warmup", warmup, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    devices = {x.device for x in tree_flatten(args)[0]
               if isinstance(x, torch.Tensor) and x.device.type == "cuda"}

    def synchronize():
        for device in devices:
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup):
            fa(*args)
            fb(*args)
        synchronize()
        samples = ([], [])
        functions = (fa, fb)
        for round_index in range(rounds):
            for side in ((0, 1) if round_index % 2 == 0 else (1, 0)):
                synchronize()
                t0 = time.perf_counter()
                for _ in range(reps):
                    functions[side](*args)
                synchronize()
                samples[side].append((time.perf_counter() - t0) * 1e3 / reps)
    return statistics.median(samples[0]), statistics.median(samples[1])


def _input_key(args):
    """Shape/stride/dtype/device/structure key, without inspecting tensor data."""
    import torch
    from torch.utils._pytree import tree_flatten

    leaves, spec = tree_flatten(args)
    return (repr(spec), tuple(("tensor", tuple(x.shape), tuple(x.stride()), str(x.dtype), str(x.device))
                             if isinstance(x, torch.Tensor) else (type(x).__qualname__, repr(x))
                             for x in leaves))


# ---------------------------------------------------------------- the entry point
def compile_best(module, args, *, package_dir: str | None = None, margin: float = MARGIN,
                 reps: int = 20, rounds: int = 5, verify: bool = True,
                 verbose: bool = True, verification_inputs: Sequence[tuple] | None = None,
                 rtol: float = 1e-3, atol: float = 1e-4,
                 regression_margin: float = MARGIN, plans: Iterable[str] = ("serial",),
                 **export_kwargs) -> Choice:
    """Compile and select on `args` plus every explicitly supplied verification sample.

    Aggregation is the equally weighted mean latency across samples. Each input key must also
    avoid a latency regression greater than `regression_margin`. Default coverage is ONLY `args`,
    not a full-dynamic guarantee. Supply export `dynamic_shapes` if the package needs them.
    With `verify=True`, both apply's verified report and actual package checks are required.
    `verify=False` is an explicit opt-out of correctness checks, not a verified result.
    `plans` opts into serial/split candidates; the original baseline is compiled once. The fastest
    eligible mean candidate latency wins (ties retain plan order), using its paired baseline timings.
    Baseline export/compile errors raise; optional candidate and measurement failures fall back.
    """
    import torch

    for name, value in (("margin", margin), ("regression_margin", regression_margin),
                        ("rtol", rtol), ("atol", atol)):
        _validate_nonnegative(name, value)
    for name, value in (("reps", reps), ("rounds", rounds)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(plans, (str, bytes)):
        raise ValueError("plans must be a nonempty finite iterable of serial/split names")
    try:
        iterator = iter(plans)
        # Only two distinct plans exist: bound consumption even for an infinite iterable.
        normalized = []
        for _ in range(3):
            try:
                plan = next(iterator)
            except StopIteration:
                break
            if not isinstance(plan, str) or plan not in ("serial", "split") or plan in normalized:
                raise ValueError("plans must contain distinct serial/split names")
            normalized.append(plan)
        else:
            raise ValueError("plans must contain at most two distinct plans")
        plans = tuple(normalized)
    except TypeError as e:
        raise ValueError("plans must be a nonempty finite iterable of serial/split names") from e
    if not plans:
        raise ValueError("plans must not be empty")
    args = tuple(args)
    cases = [args]
    if verification_inputs is not None:
        if not isinstance(verification_inputs, Sequence):
            raise TypeError("verification_inputs must be a sequence of argument tuples")
        if any(not isinstance(case, tuple) for case in verification_inputs):
            raise TypeError("each verification_inputs sample must be an argument tuple")
        cases.extend(verification_inputs)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "compile_best needs a CUDA device: it times AOTI packages to decide between them. "
            "On a CPU-only machine use neptune_opt.export(..., decompose=True) instead."
        )
    directory = package_dir or tempfile.mkdtemp(prefix="neptune_ab_")
    os.makedirs(directory, exist_ok=True)
    default_plan = plans == ("serial",)
    # Preserve old three-argument injectable exports in the default path.
    if default_plan:
        ep_base, ep_fused, report = _export_pair(module, args, export_kwargs)
    else:
        ep_base, ep_fused, report = _export_pair(module, args, export_kwargs, plan=plans[0])
    # Deliberately outside candidate error handling: a broken baseline is a build error.
    base_path = _package(ep_base, directory, "baseline")
    fused_path = None
    metrics = [{"case_index": i, "input_key": _input_key(case), "verified": None,
                "baseline_ms": None, "fused_ms": None, "reason": "not evaluated"}
               for i, case in enumerate(cases)]
    scope = f"coverage: {len(cases)} supplied case(s) only, not all dynamic inputs"
    candidate_metrics = []
    active_plan = None
    evaluating = False

    def finish(reason, base_ms=float("nan"), fused_ms=float("nan"), speedup=1.0,
               use_fusion=False):
        choice = Choice(fused_path if use_fusion else base_path, use_fusion, base_ms, fused_ms,
                        speedup, f"{reason}; {scope}", report, base_path, fused_path, metrics,
                        active_plan if use_fusion else None, candidate_metrics)
        if verbose and not evaluating:
            print(choice)
        return choice

    def same_numbers(fa, fb, case):
        # Existing three-argument monkeypatches remain valid with the backward-compatible defaults.
        if rtol == 1e-3 and atol == 1e-4:
            return _same_numbers(fa, fb, case)
        return _same_numbers(fa, fb, case, rtol=rtol, atol=atol)

    try:
        baseline = _load(base_path)
        if verify:
            failures = []
            for case, metric in zip(cases, metrics):
                ok, detail = same_numbers(module, baseline, case)
                metric["baseline_verified"] = ok
                if not ok:
                    metric["reason"] = detail
                    failures.append(f"case {metric['case_index']}: {detail}")
            if failures:
                return finish(f"baseline package numbers verification failed ({'; '.join(failures)}), "
                              "keeping the baseline without a fusion claim")
    except Exception as e:  # noqa: BLE001
        return finish(f"baseline loading/verification failed ({type(e).__name__}: {e}), keeping baseline")

    def evaluate(candidate):
        nonlocal fused_path
        if report is None or not report.fusions or candidate is None:
            summary = report.summary().strip() if report is not None else "no candidate report"
            return finish(f"nothing to fuse ({summary or 'no fusion'}), so no A/B was "
                          "run for this candidate")
        if verify and getattr(report, "verified", False) is not True:
            return finish("apply report is unverified: the fusion was dropped before compilation")
        try:
            name = "fused" if default_plan else f"fused_{active_plan}"
            fused_path = _package(candidate, directory, name)
        except Exception as e:  # noqa: BLE001
            return finish(f"the fused package failed to compile ({type(e).__name__}: {e}), "
                          "so the baseline was kept")
        try:
            fused = _load(fused_path)
            # Check every accuracy sample before timing this candidate at all.
            if verify:
                failures = []
                for case, metric in zip(cases, metrics):
                    ok, detail = same_numbers(baseline, fused, case)
                    metric["verified"] = ok
                    metric["reason"] = detail or "verified"
                    if not ok:
                        failures.append(f"case {metric['case_index']}: {detail}")
                if failures:
                    return finish("the fused package does not reproduce the baseline's numbers "
                                  f"({'; '.join(failures)}), so it was dropped")
            for case, metric in zip(cases, metrics):
                base_ms, fused_ms = _ab_time(baseline, fused, case, reps=reps, rounds=rounds)
                _, _, reason = choose_faster(base_ms, fused_ms, margin)
                metric.update(baseline_ms=base_ms, fused_ms=fused_ms, reason=reason)
        except Exception as e:  # noqa: BLE001
            return finish(f"candidate loading/verification/timing failed ({type(e).__name__}: {e}), "
                          "so the baseline was kept")
        if any(not all(math.isfinite(m[k]) and m[k] > 0 for k in ("baseline_ms", "fused_ms"))
               for m in metrics):
            return finish("unreliable nonfinite or nonpositive case timing: keeping the baseline")
        base_ms = statistics.mean(m["baseline_ms"] for m in metrics)
        fused_ms = statistics.mean(m["fused_ms"] for m in metrics)
        use_fusion, speedup, why = choose_faster(base_ms, fused_ms, margin)
        groups = {}
        for metric in metrics:
            groups.setdefault(metric["input_key"], []).append(metric)
        for key_metrics in groups.values():
            key_base = statistics.mean(m["baseline_ms"] for m in key_metrics)
            key_fused = statistics.mean(m["fused_ms"] for m in key_metrics)
            if key_fused / key_base > 1.0 + regression_margin:
                use_fusion = False
                why = (f"critical per-key regression (case {key_metrics[0]['case_index']}): "
                       "the fusion is SLOWER on this input key, so it was dropped")
                break
        return finish(why, base_ms, fused_ms, speedup, use_fusion)

    baseline_metrics = metrics
    first_candidate, first_report = ep_fused, report
    results = []
    evaluating = True
    for index, active_plan in enumerate(plans):
        metrics = [dict(metric) for metric in baseline_metrics]
        fused_path = None
        if index == 0:
            ep_fused, report = first_candidate, first_report
        else:
            try:
                # Reuse the ORIGINAL export, never a rewritten candidate or a replacement baseline.
                _, ep_fused, report = _export_pair(module, args, export_kwargs,
                                                  plan=active_plan, baseline=ep_base)
            except Exception as e:  # noqa: BLE001 -- subsequent exports are optional candidates
                import neptune_opt

                ep_fused = None
                report = neptune_opt.Report(skipped=[("candidate-failed", f"{type(e).__name__}: {e}")])
        result = evaluate(ep_fused)
        candidate_metrics.append({"plan": active_plan, "package_path": result.fused_package,
                                  "eligible": result.used_fusion, "selected": False,
                                  "baseline_ms": result.baseline_ms, "fused_ms": result.fused_ms,
                                  "speedup": result.speedup, "reason": result.reason,
                                  "report": report, "sample_metrics": metrics})
        results.append(result)
    eligible = [result for result in results if result.used_fusion]
    if eligible:
        choice = min(eligible, key=lambda result: result.fused_ms)
        for result, metric in zip(results, candidate_metrics):
            metric["selected"] = result is choice
        if not default_plan:
            choice.reason = f"selected {choice.selected_plan}: {choice.reason}"
    else:
        # Preserve the default's detailed timings and reason; multi-plan failure explains every plan.
        choice = results[0]
        if not default_plan:
            choice.reason = "no eligible plan; keeping baseline: " + "; ".join(
                f"{m['plan']}: {m['reason']}" for m in candidate_metrics)
    if verbose:
        print(choice)
    return choice
