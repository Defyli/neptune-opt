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

import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
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


def _run(label, module, x, expect_fusion, dtypes=None, args=None):
    try:
        options = {} if args is None else {"reps": args.calls, "rounds": args.repeats}
        if args is not None:
            directory = Path(args.output_dir) / label.replace(" ", "_").replace(",", "_")
            directory.mkdir(parents=True, exist_ok=True)
            options["package_dir"] = str(directory)
        choice = NO.compile_best(module, (x,), verbose=True, **options)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        check(f"{label}: compile_best did not raise", False)
        return None
    print(f"      -> {choice}")
    check(f"{label}: the fused package was built", bool(choice.report.fusions),
          f"fusions={[f.skeleton for f in choice.report.fusions]}")
    if expect_fusion is None:
        # A small win is not enough: use the same margin as compile_best.
        from neptune_opt.ab import choose_faster

        faster_is_fused = choose_faster(choice.baseline_ms, choice.fused_ms)[0]
        check(f"{label}: shipped according to the margin rule",
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


def main(args=None) -> int:
    device = torch.device(args.device if args is not None else "cuda:0")
    if device.type != "cuda" or not torch.cuda.is_available():
        print("NO CUDA: compile_best needs a device to time on")
        return 2
    print("=" * 110)
    print(f"I8: THE A/B FALLBACK   torch {torch.__version__}  {torch.cuda.get_device_name(device)}")
    print("=" * 110)

    print("\n[1] MECHANISM: torch.logsumexp, 256x16384 -- an A/B whose direction is the machine's call")
    if args is None or _selected("logsumexp", args.only):
        x = torch.randn(256, 16384, device=device)
        _run("logsumexp", M(lambda t: torch.logsumexp(t, dim=-1)).eval().to(device), x, None, args=args)

    print("\n[2] LOSES: var(unbiased=False) at (1, 262144) -- the measured 0.30x case")
    if args is None or _selected("var", args.only):
        x2 = torch.randn(1, 262144, device=device)
        _run("var(1, 262144)", M(lambda t: torch.var(t, dim=-1, unbiased=False)).eval().to(device),
             x2, False, args=args)

    print("\n[3] NO-OP: a model with nothing to fuse pays for ONE compilation")
    if args is None or _selected("noop", args.only):
        with tempfile.TemporaryDirectory(prefix="i8_noop_") as d:
            x3 = torch.randn(64, 4096, device=device)
            choice = NO.compile_best(M(lambda t: torch.relu(t) + 1.0).eval().to(device),
                                     (x3,), package_dir=d, verbose=True)
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


class ProjectionAndLSE(nn.Module):
    """Parameterized, multi-input pipeline: projection + residual + LSE + tail."""

    def __init__(self, hidden):
        super().__init__()
        self.projection = nn.Linear(hidden, hidden)

    def forward(self, x, residual):
        return torch.sigmoid(torch.logsumexp(self.projection(x) + residual, dim=-1) * 0.125)


class SyntheticRMSMax(nn.Module):
    """Synthetic RMS normalization + row amax, NOT a claim about the real model."""

    def forward(self, x):
        x = x.float()  # Preserve explicit FP32 arithmetic after BF16 input conversion.
        normalized = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)
        return normalized.amax(-1)  # Original mathematical output is explicitly FP32.


# name, workload, rows, width, dtype, dynamic axis (or None)
_NATIVE_CASES = [
    ("lse_small32x512", "lse", 32, 512, torch.float32, None),
    ("lse_model_width1125x512bf16", "lse", 1125, 512, torch.bfloat16, None),
    ("lse_large256x16384fp32", "lse", 256, 16384, torch.float32, None),
    ("lse_large256x16384bf16", "lse", 256, 16384, torch.bfloat16, None),
    ("lse_singlelong1x262144fp32", "lse", 1, 262144, torch.float32, None),
    ("rms_model_width1125x512bf16", "rms", 1125, 512, torch.bfloat16, None),
    ("rms_large256x16384fp32", "rms", 256, 16384, torch.float32, None),
    ("projection_lse128x1024fp32", "projection", 128, 1024, torch.float32, None),
    ("projection_lse64x512fp32", "projection", 64, 512, torch.float32, None),
    ("lse_dynamic_rows256x4096fp32", "lse", 256, 4096, torch.float32, 0),
    ("lse_dynamic_width32x4096fp32", "lse", 32, 4096, torch.float32, 1),
    ("lse_rank1_4096fp32", "rank1", 1, 4096, torch.float32, None),
]


def _selected(name, only):
    from fnmatch import fnmatchcase

    return not only or any(fnmatchcase(name, pattern) for pattern in only.split(","))


def _identity():
    root = Path(_HERE).parent
    files = [Path(__file__).resolve()]
    for directory in ("neptune_opt", "neptune_recognize", "neptune_codegen", "repairlib"):
        files.extend(sorted((root / directory).rglob("*.py")))
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in files}
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "files_sha256": hashes}


def _oplist(ep):
    return [str(node.target) for node in ep.graph.nodes if node.op == "call_function"]


def _tensors(value):
    values = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    if not values or not all(isinstance(x, torch.Tensor) for x in values):
        raise AssertionError("expected Tensor or flat tuple/list of Tensors")
    return values


def _bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.device == b.device and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def _compare(got, reference):
    got, reference = _tensors(got), _tensors(reference)
    if len(got) != len(reference):
        raise AssertionError("output arity differs")
    metrics = []
    for x, y in zip(got, reference):
        if (x.shape, x.dtype, x.device) != (y.shape, y.dtype, y.device):
            raise AssertionError("output shape/dtype/device differs")
        if not (torch.isfinite(x).all().item() and torch.isfinite(y).all().item()):
            raise AssertionError("nonfinite output")
        diff = (x.float() - y.float()).abs()
        strict = torch.allclose(x, y, rtol=1e-3, atol=1e-4)
        ulp_used = False
        if not strict and y.dtype in (torch.bfloat16, torch.float16):
            # At most one local dtype ULP; never silently grant atol=0.1.
            up = torch.nextafter(y, torch.full_like(y, float("inf")))
            down = torch.nextafter(y, torch.full_like(y, -float("inf")))
            bound = torch.maximum((up.float() - y.float()).abs(), (y.float() - down.float()).abs())
            ulp_used = bool((diff <= bound).all().item())
        if not strict and not ulp_used:
            raise AssertionError(f"numerical mismatch: max_abs={diff.max().item():.6g}")
        metrics.append({"shape": list(y.shape), "dtype": str(y.dtype), "device": str(y.device),
                        "max_abs": diff.max().item(), "strict_rtol_1e-3_atol_1e-4": strict,
                        "dtype_ulp_bound_used": ulp_used, "dtype_ulp_limit": 1 if ulp_used else None})
    return metrics


class PoisonedCUDA(RuntimeError):
    pass


def _error(error):
    text = f"{type(error).__name__}: {error}"
    if any(token in text.lower() for token in
           ("illegal memory access", "device-side assert", "unspecified launch failure", "context is destroyed")):
        raise PoisonedCUDA(text) from error  # No more CUDA calls in this process.
    return {"status": "error", "error": text}


def _native_case(case, args, device, artifact_root, result):
    from infer_compiler.backends.aoti import _inductor_options
    from infer_compiler.spec.precision import PrecisionPolicy, math_environment

    name, workload, rows, width, dtype, dynamic_axis = case
    policy = PrecisionPolicy("bfloat16" if dtype == torch.bfloat16 else "float32", dtype, dtype, None)
    result.update(name=name, synthetic=True, workload=workload, precision=policy.to_manifest(),
                   variants={variant: {"status": "skipped", "reason": "baseline not yet validated"}
                             for variant in ("baseline", "serial", "split")},
                   dynamic_axis=dynamic_axis, cold_move_validation="not_executed_separate_task")
    # One policy context covers export, functionalization, rewrites, compile, verification and timing.
    with policy.execution_context(device, autocast=False):
        generator = torch.Generator(device="cpu").manual_seed(20261112)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261112)
            module = (ProjectionAndLSE(width) if workload == "projection" else
                      SyntheticRMSMax() if workload == "rms" else
                      M(lambda x: torch.logsumexp(x, dim=-1)))
        module = module.eval().to(device=device, dtype=dtype)
        shapes = [(rows, width)] if workload != "rank1" else [(width,)]
        if dynamic_axis == 0:
            shapes.extend([(17, width), (128, width)])
        elif dynamic_axis == 1:
            shapes.extend([(rows, 512), (rows, 2048)])
        # Two independent values per static shape; dynamic samples share the same exported contract.
        samples = [tuple(torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
                         for _ in range(2 if workload == "projection" else 1))
                   for shape in shapes for _ in range(2)]
        snapshots = [tuple(x.clone() for x in sample) for sample in samples]
        result["inputs"] = [[{"shape": list(x.shape), "dtype": str(x.dtype)} for x in s] for s in samples]
        result["math_environment"] = math_environment()
        dynamic = None if dynamic_axis is None else ({dynamic_axis: torch.export.Dim(
            "rows" if dynamic_axis == 0 else "width", min=1, max=rows if dynamic_axis == 0 else width)},)
        ep = torch.export.export(module, samples[0], dynamic_shapes=dynamic).run_decompositions({})
        result["baseline_core_ops"] = _oplist(ep)
        result["functionalization"] = "run_decompositions({}); no default core decomposition"
        state = [(x, x.clone()) for x in (*module.parameters(), *module.buffers(), *ep.state_dict.values())]
        runners, references = {}, []
        for variant in ("baseline", "serial", "split"):
            metadata = result["variants"][variant] = {"status": "pending"}
            try:
                trial = copy.deepcopy(ep)
                if variant != "baseline":
                    report = NO.apply(trial, plan=variant, lower=True, verify_with=(module, samples[0]),
                                      verification_inputs=samples[1:],
                                      max_ulps=1 if dtype in (torch.bfloat16, torch.float16) else None)
                    metadata["report"] = dataclasses.asdict(report)
                    metadata["verification_policy"] = {
                        "rtol": 1e-3, "atol": 1e-4,
                        "explicit_dtype_max_ulps": 1 if dtype in (torch.bfloat16, torch.float16) else None,
                        "scope": "synthetic low-precision operator experiment, not model tolerance"}
                    # apply catches execution errors; inspect refusals for a poisoned context too.
                    for reason, detail in report.skipped:
                        _error(RuntimeError(f"{reason}: {detail}"))
                    if not report.fusions or not report.verified:
                        metadata["status"] = "noop" if all(reason == "no-fusion" for reason, _ in report.skipped) else "declined"
                        metadata["expected_dynamic_width_refusal"] = variant == "split" and dynamic_axis == 1 and any(
                            "concrete width" in detail for _, detail in report.skipped)
                        metadata["timing"] = "not_executed_invalid_or_noop_candidate"
                        metadata["core_ops"] = _oplist(trial)
                        continue
                ops = _oplist(trial)
                metadata.update(core_ops=ops, neptune_ops=[op for op in ops if "neptune" in op],
                                split_launch={"block": 1024, "num_warps": 4,
                                              "scratch": "four rows*nparts buffers (fp32, fp32, int32, int32)",
                                              "selection": "fixed default, not measurement-selected"}
                                if variant == "split" else None)
                directory = artifact_root / name / variant
                directory.mkdir(parents=True)
                so = directory / "model.so"
                options = _inductor_options(so)
                metadata.update(artifact_path=str(so), compile_options={k: v for k, v in options.items()
                                if k != "aot_inductor.output_path"}, compile_invocations=1,
                                artifact_retained=args.keep_artifacts)
                metadata["stage"] = "compile"
                start = time.perf_counter()
                torch._inductor.aot_compile(trial.module(), samples[0], options=options)
                metadata["compile_wall_s"] = time.perf_counter() - start
                if not so.is_file():
                    raise RuntimeError("compiler did not emit model.so")
                metadata["so_sha256"] = hashlib.sha256(so.read_bytes()).hexdigest()
                metadata["stage"] = "initial_native_load"
                start = time.perf_counter()
                runner = torch._C._aoti.AOTIModelContainerRunnerCuda(str(so), 1, str(device))
                metadata["initial_load_wall_s"] = time.perf_counter() - start
                checks = metadata["checks"] = []
                metadata["stage"] = "native_validation"
                for index, sample in enumerate(samples):
                    observed = tuple(x.clone() for x in _tensors(runner.run(list(sample))))
                    repeated = _tensors(runner.run(list(sample)))
                    torch.cuda.synchronize(device)
                    if len(observed) != len(repeated) or not all(
                            _bits_equal(a, b) for a, b in zip(observed, repeated)):
                        raise AssertionError("native repeated output not bitwise equal")
                    if not all(_bits_equal(a, b) for a, b in zip(sample, snapshots[index])):
                        raise AssertionError("native runner modified inputs")
                    if not all(_bits_equal(a, b) for a, b in state):
                        raise AssertionError("execution modified model/export state")
                    reference = module(*sample) if variant == "baseline" else references[index]
                    checks.append(_compare(observed, reference))
                    if variant == "baseline":
                        references.append(observed)
                metadata.update(status="validated", checks=checks, repeat_bitwise=True, input_readonly=True)
                runners[variant] = runner
            except Exception as error:
                metadata.update(_error(error))
                if variant == "baseline":
                    result["status"] = "error"
                    return
        names = list(runners)
        result["round_orders"] = [names[i % len(names):] + names[:i % len(names)] for i in range(args.repeats)]
        for name_ in names:
            result["variants"][name_].update(timings=[], stage="timing")
        for sample_index, sample in enumerate(samples):
            for runner in runners.values():
                for _ in range(100):
                    runner.run(list(sample))
            torch.cuda.synchronize(device)
            rounds = {name_: [] for name_ in names}
            for order in result["round_orders"]:
                for name_ in order:
                    torch.cuda.synchronize(device)
                    start = time.perf_counter()
                    for _ in range(args.calls):
                        runners[name_].run(list(sample))
                    torch.cuda.synchronize(device)
                    rounds[name_].append((time.perf_counter() - start) * 1000 / args.calls)
            for name_ in names:
                if not all(_bits_equal(a, b) for a, b in zip(sample, snapshots[sample_index])):
                    raise AssertionError("timing modified inputs")
                _compare(runners[name_].run(list(sample)), references[sample_index])
                result["variants"][name_]["timings"].append({"sample": sample_index,
                    "wall_ms_per_call_rounds": rounds[name_], "median_ms": statistics.median(rounds[name_])})
        if not all(_bits_equal(a, b) for a, b in state):
            raise AssertionError("timing modified model/export state")
        baseline_ms = statistics.mean(x["median_ms"] for x in result["variants"]["baseline"]["timings"])
        from neptune_opt.ab import MARGIN, choose_faster
        for name_ in names:
            metadata = result["variants"][name_]
            medians = [x["median_ms"] for x in metadata["timings"]]
            ms = statistics.mean(medians)
            wins, speedup, reason = choose_faster(baseline_ms, ms)
            regressions = [v / b["median_ms"] > 1 + MARGIN for v, b in
                           zip(medians, result["variants"]["baseline"]["timings"])]
            metadata.update(stage="complete", mean_sample_median_ms=ms, speedup_vs_baseline=speedup,
                            eligible_by_margin=name_ != "baseline" and wins and not any(regressions),
                            margin_reason=reason, per_sample_regression=regressions)
        result["status"] = "error" if any(v["status"] == "error" or any(
            reason in ("verify-failed", "rewrite-failed", "candidate-failed")
            and not v.get("expected_dynamic_width_refusal", False)
            for reason, _ in v.get("report", {}).get("skipped", []))
            for v in result["variants"].values()) else "ok"


def native_main(args):
    import triton

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        print("NO CUDA: native experiment requires a CUDA device")
        return 2
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = [case for case in _NATIVE_CASES if _selected(case[0], args.only)]
    if not selected:
        raise ValueError("--only matched no native cases")
    owner = None if args.keep_artifacts else tempfile.TemporaryDirectory(prefix="m23_native_")
    root = Path(tempfile.mkdtemp(prefix="artifacts_", dir=output)) if owner is None else Path(owner.name)
    document = {"mode": "native", "identity": _identity(), "torch": torch.__version__,
                "triton": triton.__version__, "cuda": torch.version.cuda, "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "runner": "torch._C._aoti.AOTIModelContainerRunnerCuda", "seed": 20261112,
                "calls": args.calls, "repeats": args.repeats, "warmup_calls_per_sample_variant": 100,
                "metric": "synchronized wall ms/call including native Python binding and host overhead",
                "coverage": "synthetic supplied inputs only; not full dynamic or real-model precision",
                "profiling": "not_executed; no claims about kernel absence", "cases": []}
    def save():
        (output / "native_results.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    try:
        for case in selected:
            result = {"name": case[0]}
            document["cases"].append(result)
            print(f"[native] {case[0]}: baseline / strengthened serial / explicit split", flush=True)
            try:
                _native_case(case, args, device, root, result)
            except Exception as error:
                result.update(_error(error))
            save()
            print(f"  {result['status']}: " + ", ".join(
                f"{key}={value['status']}" for key, value in result.get("variants", {}).items()), flush=True)
    except PoisonedCUDA as error:
        document.update(aborted_poisoned_cuda=True, error=str(error))
        if document["cases"]:
            document["cases"][-1].update(status="error", error=str(error))
        save()  # CPU-only reporting: do not synchronize or attempt another case.
        return 1
    finally:
        if owner is not None:
            owner.cleanup()
    return int(any(case["status"] == "error" for case in document["cases"]))


def _arguments():
    parser = argparse.ArgumentParser(description="Original compile_best fallback or native three-way AOTI experiment")
    parser.add_argument("--native", action="store_true", help="compile/load real model.so variants instead of PT2 packages")
    parser.add_argument("--output-dir", default=str(Path(_HERE) / "results" / "gpu_m23_ab_fallback"),
                        help="directory for native_results.json (created if needed)")
    parser.add_argument("--repeats", type=int, default=7, help="interleaved timing rounds (default: 7)")
    parser.add_argument("--calls", type=int, default=500, help="calls per timed round (default: 500)")
    parser.add_argument("--only", help="comma-separated names/globs; default cases: logsumexp,var,noop; native cases: " +
                        ", ".join(case[0] for case in _NATIVE_CASES))
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--keep-artifacts", action="store_true", help="retain native .so and compiler files for separate cold-move validation")
    args = parser.parse_args()
    if args.calls < 1 or args.repeats < 1:
        parser.error("--calls and --repeats must be positive")
    names = [case[0] for case in _NATIVE_CASES] if args.native else ["logsumexp", "var", "noop"]
    if not any(_selected(name, args.only) for name in names):
        parser.error("--only matched no cases")
    return args


if __name__ == "__main__":
    arguments = _arguments()
    if arguments.native:
        raise SystemExit(native_main(arguments))
    raise SystemExit(main(arguments))
