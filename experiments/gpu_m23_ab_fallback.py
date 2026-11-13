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


class FullRMS(nn.Module):
    """Full vector RMS; round the fp32 product BEFORE multiplying gamma."""

    def __init__(self, width, gamma_input=False):
        super().__init__()
        if not gamma_input:
            self.weight = nn.Parameter(torch.linspace(0.5, 1.5, width))

    def forward(self, x, gamma=None):
        z = x.float()
        normalized = z * torch.rsqrt((z * z).mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * (self.weight if gamma is None else gamma)


class SegmentPool(nn.Module):
    """Runtime lengths, including empty segments; no precomputed host offsets."""

    def __init__(self, reduce):
        super().__init__()
        self.reduce = reduce

    def forward(self, data, lengths):
        return torch.segment_reduce(data, self.reduce, lengths=lengths, initial=0)


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


def _structured_cases():
    cases = []
    for rows, width in ((1, 512), (32, 512), (1125, 512), (8, 4096), (1, 65536), (32, 333)):
        for dtype, suffix in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            for gamma_input in (False, True):
                gamma = "input" if gamma_input else "parameter"
                cases.append((f"recommend_rms_{rows}x{width}_{suffix}_gamma_{gamma}",
                              "full_rms", rows, width, dtype, None,
                              {"capabilities": ("rms_norm",), "gamma_input": gamma_input}))
    for rows in (512, 4096):
        for segments in (8, 64):
            for reduce in ("sum", "mean"):
                for dtype, suffix in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
                    for distribution in ("uniform", "empty_long_tail"):
                        cases.append((f"recommend_segment_{reduce}_{rows}x64_s{segments}_{suffix}_{distribution}",
                                      "segment", rows, 64, dtype, None,
                                      {"capabilities": ("segment_pool",), "segments": segments,
                                       "reduce": reduce, "distribution": distribution}))
    return cases


def _segment_lengths(rows, segments, distribution):
    if distribution == "uniform":
        values = [rows // segments] * segments
        values[-1] += rows % segments
    else:
        values = [0 if i % 3 == 0 else 1 + i % 7 for i in range(segments - 1)]
        values.append(rows - sum(values))
    if len(values) != segments or min(values) < 0 or sum(values) != rows:
        raise AssertionError("invalid synthetic lengths")
    return values


def _native_cases(args):
    return _NATIVE_CASES + (_structured_cases() if getattr(args, "structured", False) else [])


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
            # A true representable-value distance, including finite maxima. The
            # scope is explicit in the report; no infinite-neighbour spacing slack.
            ulp_used = NO._compare_trees(y, x, rtol=0, atol=0, max_ulps=1)[0]
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

    name, workload, rows, width, dtype, dynamic_axis = case[:6]
    configuration = case[6] if len(case) > 6 else {}
    capabilities = configuration.get("capabilities", ())
    policy = PrecisionPolicy("bfloat16" if dtype == torch.bfloat16 else "float32", dtype, dtype, None)
    result.update(name=name, synthetic=True, workload=workload, precision=policy.to_manifest(),
                   module_kind=workload, capabilities=list(capabilities),
                   correctness_contract={"scope": "supplied argument tuples only; no output aliasing proof",
                                         "rtol": 1e-3, "atol": 1e-4,
                                         "explicit_dtype_max_ulps": 1 if dtype in (torch.bfloat16, torch.float16) else None,
                                         "repeat": "bitwise", "inputs_and_state": "readonly"},
                   variants={variant: {"status": "skipped", "reason": "baseline not yet validated"}
                             for variant in ("baseline", "serial", "split")},
                   dynamic_axis=dynamic_axis, cold_move_validation="not_executed_separate_task")
    # One policy context covers export, functionalization, rewrites, compile, verification and timing.
    with policy.execution_context(device, autocast=False):
        generator = torch.Generator(device="cpu").manual_seed(20261112)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261112)
            module = (ProjectionAndLSE(width) if workload == "projection" else
                      FullRMS(width, configuration["gamma_input"]) if workload == "full_rms" else
                      SegmentPool(configuration["reduce"]) if workload == "segment" else
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
        distributions = []
        if workload == "full_rms" and configuration["gamma_input"]:
            samples = [(sample[0], (0.5 + torch.rand(width, generator=generator)).to(device=device, dtype=dtype))
                       for sample in samples]
        elif workload == "segment":
            # Same exported two-input shapes, different runtime descriptor contents.
            samples = []
            primary = configuration["distribution"]
            alternate = "empty_long_tail" if primary == "uniform" else "uniform"
            for distribution in (primary, alternate):
                lengths = _segment_lengths(rows, configuration["segments"], distribution)
                for _ in range(2):
                    data = torch.randn(rows, width, generator=generator).to(device=device, dtype=dtype)
                    samples.append((data, torch.tensor(lengths, dtype=torch.int64, device=device)))
                    distributions.append({"len_dist": distribution, "lengths": lengths,
                                          "total_rows": sum(lengths), "empty_segments": lengths.count(0),
                                          "max_length": max(lengths)})
            result.update(reduce=configuration["reduce"], initial=0, unsafe=False,
                          descriptor="lengths", primary_len_dist=primary,
                          numerical_stages="ordered per-token storage-dtype RTNE; no reassociation")
        snapshots = [tuple(x.clone() for x in sample) for sample in samples]
        pointers = [tuple(x.data_ptr() for x in sample) for sample in samples]
        result["inputs"] = [[{"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device),
                             "stride": list(x.stride())} for x in s] for s in samples]
        result["sample_descriptors"] = distributions
        result["correctness_contract"]["verification_samples"] = len(samples)
        result["math_environment"] = math_environment()
        dynamic = None if dynamic_axis is None else ({dynamic_axis: torch.export.Dim(
            "rows" if dynamic_axis == 0 else "width", min=1, max=rows if dynamic_axis == 0 else width)},)
        ep = torch.export.export(module, samples[0], dynamic_shapes=dynamic).run_decompositions({})
        result["baseline_core_ops"] = _oplist(ep)
        result["functionalization"] = "run_decompositions({}); no default core decomposition"
        state = [(x, x.clone()) for x in (*module.parameters(), *module.buffers(),
                                         *ep.state_dict.values(), *ep.constants.values())
                 if isinstance(x, torch.Tensor)]
        state_pointers = [x.data_ptr() for x, _ in state]
        runners, references = {}, []
        for variant in ("baseline", "serial", "split"):
            metadata = result["variants"][variant] = {"status": "pending", "stage": "prepare",
                                                       "capabilities": list(capabilities)}
            try:
                trial = copy.deepcopy(ep)
                if variant != "baseline":
                    if capabilities:
                        metadata["stage"] = "preview"
                        matches, reason = NO.preview(ep, capabilities=capabilities)
                        result["preview"] = {"reason": reason, "regions": [
                            {"kind": match.kind, "operation": match.operation, "skeleton": match.skeleton,
                             "params": dict(match.params), "input_names": [x.name for x in match.input_nodes]}
                            for match in matches]}
                        metadata["stage"] = "plan_contract"
                        try:
                            if workload == "full_rms":
                                from neptune_codegen.template import validate_rms_inputs
                                gamma = samples[0][1] if configuration["gamma_input"] else module.weight
                                validate_rms_inputs(samples[0][0], gamma, {"eps": 1e-6},
                                                    plan=variant, metadata=True)
                            else:
                                from neptune_codegen.torch_op import validate_segment_inputs
                                validate_segment_inputs(*samples[0], reduce=configuration["reduce"],
                                                        initial=0, plan=variant, metadata=True)
                        except NotImplementedError as error:
                            metadata.update(status="refused", expected_plan_refusal=True,
                                            reason=str(error), timing="not_executed_unsupported_plan",
                                            compile_invocations=0)
                            print(f"  [{variant}: refused at plan_contract] {error}", flush=True)
                            continue
                        if not any(match.kind == "structured" and match.operation in capabilities
                                   for match in matches):
                            metadata.update(status="no-fusion", reason=result["preview"]["reason"],
                                            timing="not_executed_no_structured_fusion", compile_invocations=0)
                            print(f"  [{variant}: no-fusion] {metadata['reason']}", flush=True)
                            continue
                    metadata["stage"] = "apply"
                    report = NO.apply(trial, plan=variant, lower=True, verify_with=(module, samples[0]),
                                      verification_inputs=samples[1:], capabilities=capabilities,
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
                        no_fusion = all(reason == "no-fusion" for reason, _ in report.skipped)
                        metadata["status"] = ("no-fusion" if no_fusion else "refused") if capabilities else (
                            "noop" if no_fusion else "declined")
                        metadata["reason"] = report.summary()
                        print(f"  [{variant}: {metadata['status']}] {metadata['reason']}", flush=True)
                        metadata["expected_dynamic_width_refusal"] = variant == "split" and dynamic_axis == 1 and any(
                            "concrete width" in detail for _, detail in report.skipped)
                        metadata["timing"] = "not_executed_refused_or_no_fusion_candidate" if capabilities else (
                            "not_executed_invalid_or_noop_candidate")
                        metadata["compile_invocations"] = 0
                        metadata["core_ops"] = _oplist(trial)
                        continue
                if capabilities and variant != "baseline" and not any(
                        fusion.kind == "structured" and fusion.operation in capabilities for fusion in report.fusions):
                    metadata.update(status="no-fusion", reason="only scalar fragments fused; full structured region absent",
                                    timing="not_executed_no_structured_fusion", compile_invocations=0)
                    continue
                ops = _oplist(trial)
                metadata.update(core_ops=ops, neptune_ops=[op for op in ops if "neptune" in op],
                                split_launch={"block": 1024, "num_warps": 4,
                                              "scratch": "rows*nparts totals fp32 and counts int64" if workload == "full_rms" else
                                                         "four rows*nparts buffers (fp32, fp32, int32, int32)",
                                              "selection": "fixed default, not measurement-selected"}
                                if variant == "split" else None)
                directory = artifact_root / name / variant
                directory.mkdir(parents=True)
                so = directory / "model.so"
                options = {**_inductor_options(so), "triton.unique_user_kernel_names": True}
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
                    if distributions:
                        print(f"  [{variant}: validate sample {index}] "
                              f"len_dist={distributions[index]['len_dist']} "
                              f"lengths={distributions[index]['lengths']}", flush=True)
                    observed = tuple(x.clone() for x in _tensors(runner.run(list(sample))))
                    repeated = _tensors(runner.run(list(sample)))
                    torch.cuda.synchronize(device)
                    if len(observed) != len(repeated) or not all(
                            _bits_equal(a, b) for a, b in zip(observed, repeated)):
                        raise AssertionError("native repeated output not bitwise equal")
                    if tuple(x.data_ptr() for x in sample) != pointers[index]:
                        raise AssertionError("native runner changed input pointers")
                    if [x.data_ptr() for x, _ in state] != state_pointers:
                        raise AssertionError("execution changed model/export state pointers")
                    if not all(_bits_equal(a, b) for a, b in zip(sample, snapshots[index])):
                        raise AssertionError("native runner modified inputs")
                    if not all(_bits_equal(a, b) for a, b in state):
                        raise AssertionError("execution modified model/export state")
                    reference = module(*sample) if variant == "baseline" else references[index]
                    checks.append(_compare(observed, reference))
                    if variant == "baseline":
                        references.append(observed)
                metadata.update(status="validated", checks=checks, repeat_bitwise=True, input_readonly=True,
                                input_pointers_preserved=True, state_readonly=True, state_pointers_preserved=True)
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
                if tuple(x.data_ptr() for x in sample) != pointers[sample_index]:
                    raise AssertionError("timing changed input pointers")
                if not all(_bits_equal(a, b) for a, b in zip(sample, snapshots[sample_index])):
                    raise AssertionError("timing modified inputs")
                _compare(runners[name_].run(list(sample)), references[sample_index])
                result["variants"][name_]["timings"].append({"sample": sample_index,
                    "len_dist": distributions[sample_index]["len_dist"] if distributions else None,
                    "wall_ms_per_call_rounds": rounds[name_], "median_ms": statistics.median(rounds[name_])})
        if [x.data_ptr() for x, _ in state] != state_pointers or not all(_bits_equal(a, b) for a, b in state):
            raise AssertionError("timing modified model/export state")
        if not all(tuple(x.data_ptr() for x in sample) == pointer and all(
                _bits_equal(x, saved) for x, saved in zip(sample, snapshot))
                for sample, pointer, snapshot in zip(samples, pointers, snapshots)):
            raise AssertionError("execution modified supplied inputs")
        baseline_ms = statistics.mean(x["median_ms"] for x in result["variants"]["baseline"]["timings"])
        from neptune_opt.ab import MARGIN, choose_faster
        for name_ in names:
            metadata = result["variants"][name_]
            medians = [x["median_ms"] for x in metadata["timings"]]
            ms = statistics.mean(medians)
            wins, speedup, reason = choose_faster(baseline_ms, ms)
            regressions = [v / b["median_ms"] > 1 + MARGIN for v, b in
                           zip(medians, result["variants"]["baseline"]["timings"])]
            if distributions:
                metadata["mean_median_ms_by_len_dist"] = {
                    distribution: statistics.mean(t["median_ms"] for t in metadata["timings"]
                                                  if t["len_dist"] == distribution)
                    for distribution in ("uniform", "empty_long_tail")}
            metadata.update(stage="complete", mean_sample_median_ms=ms, speedup_vs_baseline=speedup,
                            eligible_by_margin=name_ != "baseline" and wins and not any(regressions),
                            margin_reason=reason, per_sample_regression=regressions)
        result["status"] = "error" if any(v["status"] == "error" or any(
            reason in ("verify-failed", "rewrite-failed", "candidate-failed")
            and not v.get("expected_dynamic_width_refusal", False)
            for reason, _ in v.get("report", {}).get("skipped", []))
            for v in result["variants"].values()) else "ok"
        if capabilities:
            eligible = [name_ for name_ in names if result["variants"][name_]["eligible_by_margin"]]
            selected = (min(eligible, key=lambda name_: result["variants"][name_]["mean_sample_median_ms"])
                        if eligible else "baseline")
            result["ab_selection"] = {"variant": selected,
                                      "artifact_path": result["variants"][selected]["artifact_path"],
                                      "rule": "margin win and no supplied-sample regression; otherwise baseline",
                                      "scope": "synthetic supplied samples only",
                                      "artifact_retained": args.keep_artifacts}


def native_main(args):
    import triton

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        print("NO CUDA: native experiment requires a CUDA device")
        return 2
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = [case for case in _native_cases(args) if _selected(case[0], args.only)]
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
                "structured": getattr(args, "structured", False),
                "metric": "synchronized wall ms/call including native Python binding, host overhead, "
                          "allocations and compiled descriptor cumsum/validation",
                "deployment": "compiled model.so via native AOTI runner; no Python operator callback",
                "interpretation": "synthetic operator/pipeline speedups only, never full-model speedup",
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


def genericmodel_native_main(args):
    """Evaluate one declared node, never the whole DAG or a replacement engine."""
    from infer_compiler.backends.aoti import _inductor_options, capture_aoti_node
    from infer_compiler.backends.extensions import aoti_extensions
    from infer_compiler.model.module_loading import prepare_aoti_module
    from infer_compiler.model.package import inspect_parameter_package
    from infer_compiler.model.sample_checks import compare_outputs
    from infer_compiler.model.samples import load_export_samples
    from infer_compiler.pipeline.build import _model_context
    from infer_compiler.spec.precision import math_environment, resolve_precision_policy
    from infer_compiler.spec.shapes import make_shape_probes
    from neptune_opt.ab import MARGIN, choose_faster

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        print("NO CUDA: model native experiment requires a CUDA device")
        return 2
    policy = resolve_precision_policy(args.model_precision)
    capabilities = args.capabilities if args.capabilities is not None else ("segment_pool",)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    document = {"mode": "model_native", "identity": _identity(), "node": args.node,
                "model_package": str(Path(args.model_package).expanduser().resolve()),
                "module_kind": "real_model_node", "synthetic": False,
                "full_dag_performance": False, "full_dag_precision_validated": False,
                "precision": policy.to_manifest(), "capabilities": list(capabilities), "plan": args.plan,
                "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "runner": "torch._C._aoti.AOTIModelContainerRunnerCuda", "num_models": 1,
                "deployment": "native model.so; no Python Neptune operator callback",
                "calls": args.calls, "repeats": args.repeats, "warmup_calls_per_case_variant": 20,
                "metric": "synchronized wall ms/call including native binding, host overhead, allocations and descriptor validation",
                "aggregation": "equally weighted supplied cases; NOT production request weights",
                "correctness_scope": "all captured probes and real samples; input values/pointers and original EP/module state; "
                                     "native internal constants and output aliasing not proved",
                "status": "pending", "stage": "inspect_package", "cases": [], "variants": {},
                "all_real_samples_native_validated": False, "cold_move_validation": "not_executed"}

    def save():
        (output / "native_results.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")

    owner = None if args.keep_artifacts else tempfile.TemporaryDirectory(prefix="m23_model_native_")
    root = Path(tempfile.mkdtemp(prefix="artifacts_", dir=output)) if owner is None else Path(owner.name)
    try:
        package = inspect_parameter_package(Path(args.model_package).expanduser().resolve())
        document["package"] = package.to_dict()
        with package.code_archive.open("rb") as stream:
            document["model_code_archive_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        with tempfile.TemporaryDirectory(prefix="m23_model_source_") as source_tmp:
            # Keep source imports alive through capture, compilation, native checks AND timing.
            with _model_context(package, Path(source_tmp)) as (loaded, graph, _request):
                nodes = [node for node in graph.nodes if node.spec.name == args.node]
                if len(nodes) != 1:
                    raise ValueError(f"unknown/ambiguous node {args.node!r}; declared: {[n.spec.name for n in graph.nodes]}")
                node = nodes[0]
                document["engine"] = node.spec.engine_type
                if node.spec.engine_type != "aoti":
                    raise NotImplementedError("model native mode requires a declared aoti node; engine is never overridden")
                if getattr(node.spec, "aoti_extensions", None):
                    raise NotImplementedError("node-level aoti_extensions declaration is unsupported; not silently omitted")
                tolerances = dict(node.spec.tolerances or {})
                rtol, atol = float(tolerances.get("rtol", 0)), float(tolerances.get("atol", 0))
                document.update(tolerances=tolerances, max_ulps=None, stage="load_samples")
                samples = load_export_samples(package.checkpoint)
                node_samples = samples.for_node(node)
                document.update(sample_identity=dict(samples.identity), export_record_count=len(samples.records),
                                mapped_sample_count=len(node_samples), sample_binding=samples.binding_for_node(node))
                document["stage"] = "prepare_target_node"
                prepared = prepare_aoti_module(node, loaded, package.state_dict)
                document["parameters"] = dict(prepared.state_report)
                document["stage"] = "capture"
                cap = capture_aoti_node(
                    prepared.module, node.spec.input_specs, node.spec.output_specs,
                    precision=policy, device=device, node_name=node.spec.name,
                    example_inputs=node.spec.example_inputs,
                    logical_input_names=tuple(binding.logical_name for binding in node.input_bindings),
                    shape_probes=make_shape_probes(graph.tensor_specs.values()),
                    validation_samples=node_samples, tolerances=tolerances, engine_type=node.spec.engine_type)
                if len(cap.samples) != len(node_samples):
                    raise AssertionError("capture dropped real samples")
                document.update(capture=cap.to_report(), captured_sample_count=len(cap.samples),
                                captured_probe_count=len(cap.examples),
                                input_names=list(cap.module.input_names), output_names=list(cap.module.output_names))
                compiler_sources = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                    for name, module in sorted(sys.modules.items())
                                    if name.startswith("infer_compiler.") and getattr(module, "__file__", None)
                                    and Path(module.__file__).suffix == ".py"}
                document["infer_compiler_sources_sha256"] = compiler_sources
                cases = [(probe, inputs, eager, None, "probe") for probe, inputs, eager in cap.examples]
                cases.extend((sample.probe, sample.inputs, sample.eager, sample.recorded, "real_sample")
                             for sample in cap.samples)
                if not cases:
                    raise AssertionError("capture supplied no validation cases")
                snapshots = [tuple(x.clone() for x in case[1]) for case in cases]
                pointers = [tuple(x.data_ptr() for x in case[1]) for case in cases]
                for probe, inputs, eager, recorded, source in cases:
                    document["cases"].append({
                        "name": probe.name, "source": source, "dimensions": dict(probe.values),
                        "inputs": [{"name": name, "shape": list(x.shape), "dtype": str(x.dtype),
                                    "device": str(x.device), "stride": list(x.stride()),
                                    "bytes": x.numel() * x.element_size()}
                                   for name, x in zip(cap.module.input_names, inputs, strict=True)],
                        "outputs": [{"name": name, "shape": list(x.shape), "dtype": str(x.dtype)}
                                    for name, x in zip(cap.module.output_names, eager, strict=True)],
                        "recorded_available": recorded is not None, "checks": {}, "timings": {}})
                base = cap.exported
                state = [(x, x.clone(), x.data_ptr()) for x in
                         (*cap.module.parameters(), *cap.module.buffers(), *base.state_dict.values(),
                          *base.constants.values()) if isinstance(x, torch.Tensor)]

                def readonly():
                    if any(x.data_ptr() != pointer or not _bits_equal(x, saved) for x, saved, pointer in state):
                        raise AssertionError("original module/export state modified")
                    for case, saved, pointer in zip(cases, snapshots, pointers, strict=True):
                        if tuple(x.data_ptr() for x in case[1]) != pointer or not all(
                                _bits_equal(x, y) for x, y in zip(case[1], saved, strict=True)):
                            raise AssertionError(f"input modified: {case[0].name}")

                def compare(actual, expected, label):
                    ok, detail = NO._compare_trees(_tensors(actual), _tensors(expected),
                                                  rtol=rtol, atol=atol, max_ulps=None)
                    if not ok:
                        raise AssertionError(f"{label}: {detail}")

                def validate(runner, variant, index, baseline_reference=None):
                    probe, inputs, eager, recorded, _source = cases[index]
                    raw = _tensors(runner.run(list(inputs)))
                    if any(not x.is_contiguous() for x in raw):
                        raise AssertionError(f"{probe.name}: native output ABI is not contiguous")
                    actual = tuple(x.clone() for x in raw)
                    repeated = _tensors(runner.run(list(inputs)))
                    torch.cuda.synchronize(device)
                    if len(actual) != len(repeated) or not all(
                            _bits_equal(x, y) for x, y in zip(actual, repeated, strict=True)):
                        raise AssertionError(f"{probe.name}: native repeated outputs differ")
                    if any(not x.is_contiguous() for x in (*actual, *repeated)):
                        raise AssertionError(f"{probe.name}: native output ABI is not contiguous")
                    compare(actual, eager, f"{probe.name}: native vs original eager")
                    if baseline_reference is not None:
                        compare(actual, baseline_reference, f"{probe.name}: candidate vs baseline")
                    recorded_check = None
                    if recorded is not None:
                        # Preserve recorded source dtype/values; never cast saved FP32 gold to BF16.
                        recorded_check = compare_outputs(actual, recorded, cap.module.output_specs, tolerances,
                                                         source=probe.name, stage="native_recorded")
                    readonly()
                    document["cases"][index]["checks"][variant] = {
                        "status": "passed", "native_vs_original_eager": True,
                        "candidate_vs_baseline": baseline_reference is not None,
                        "repeat_bitwise": True, "output_contiguous": True,
                        "inputs_readonly_and_pointers_preserved": True,
                        "original_state_readonly": True, "recorded_comparison": recorded_check}
                    return actual

                runners, baseline_outputs = {}, []
                for variant in ("baseline", "candidate"):
                    metadata = document["variants"][variant] = {"status": "pending", "stage": "graph_trial",
                                                               "compile_invocations": 0}
                    try:
                        # Copy only graph/signature structure, sharing readonly parameter tensors.
                        trial = NO._candidate_program(base, NO._graph_trial(base.graph_module))
                        if variant == "candidate":
                            metadata["stage"] = "apply"
                            document["stage"] = "candidate_apply"
                            verification = [case[1] for case in cases if case[4] == "real_sample"]
                            verification.extend(case[1] for case in cases if case[4] == "probe")
                            with policy.execution_context(device, autocast=False):
                                matches, preview_reason = NO.preview(trial, capabilities=capabilities)
                                metadata["preview"] = {"reason": preview_reason, "regions": [
                                    {"skeleton": match.skeleton, "kind": match.kind,
                                     "operation": match.operation, "params": dict(match.params)}
                                    for match in matches]}
                                report = NO.apply(trial, plan=args.plan, capabilities=capabilities,
                                                  verify_with=(cap.module, verification[0]),
                                                  verification_inputs=verification[1:], rtol=rtol, atol=atol,
                                                  max_ulps=None)
                            metadata["report"] = dataclasses.asdict(report)
                            for reason, detail in report.skipped:
                                _error(RuntimeError(f"{reason}: {detail}"))
                            with policy.execution_context(device, autocast=False):
                                readonly()
                            if not report.fusions or not report.verified:
                                metadata.update(status="no-fusion" if all(reason == "no-fusion" for reason, _ in report.skipped)
                                                else "refused", reason=report.summary(), timing="not_executed_refused_candidate")
                                print(f"[model:{args.node}] candidate {metadata['status']}: {metadata['reason']}", flush=True)
                                continue
                            if not any(fusion.kind == "structured" and fusion.operation in capabilities
                                       for fusion in report.fusions):
                                metadata.update(status="no-fusion", reason="requested structured fusion absent; scalar fragments are not a model candidate",
                                                timing="not_executed_no_structured_fusion")
                                continue
                            if report.verification_cases != len(cases):
                                raise AssertionError("apply did not verify every captured case")
                        directory = root / variant
                        directory.mkdir()
                        so = directory / "model.so"
                        metadata.update(stage="compile", core_ops=_oplist(trial), artifact_path=str(so),
                                        artifact_retained=args.keep_artifacts, compile_invocations=1)
                        document["stage"] = f"{variant}_compile"
                        with policy.execution_context(device, autocast=False), aoti_extensions(cap.module, device) as (
                                extension_options, extension_report):
                            options = {**_inductor_options(so), **extension_options,
                                       "triton.unique_user_kernel_names": True}
                            metadata.update(extensions=extension_report,
                                            compile_options={k: v for k, v in options.items()
                                                             if k != "aot_inductor.output_path" and k not in extension_options})
                            start = time.perf_counter()
                            torch._inductor.aot_compile(trial.module(), cap.examples[0][1], options=options)
                            metadata["compile_wall_s"] = time.perf_counter() - start
                        if not so.is_file():
                            raise RuntimeError("compiler did not emit model.so")
                        metadata.update(stage="native_load", so_sha256=hashlib.sha256(so.read_bytes()).hexdigest())
                        start = time.perf_counter()
                        runner = torch._C._aoti.AOTIModelContainerRunnerCuda(str(so), 1, str(device))
                        metadata["initial_load_wall_s"] = time.perf_counter() - start
                        with policy.execution_context(device, autocast=False):
                            document["math_environment"] = math_environment()
                            metadata["stage"] = "native_validation"
                            document["stage"] = f"{variant}_native_validation"
                            for index, case in enumerate(cases):
                                observed = validate(runner, variant, index,
                                                    baseline_outputs[index] if variant == "candidate" else None)
                                if variant == "baseline":
                                    baseline_outputs.append(observed)
                                print(f"[model:{args.node}] {variant} {index + 1}/{len(cases)} {case[0].name}: passed", flush=True)
                        metadata.update(status="validated", stage="validated", validated_cases=len(cases),
                                        validated_real_samples=len(cap.samples), native_vs_original_eager=True,
                                        recorded_output_comparisons=sum(case[3] is not None for case in cases),
                                        precision_validated=bool(cap.samples),
                                        precision_status="passed_all_captured_samples" if cap.samples else "not_executed_no_samples")
                        runners[variant] = runner
                        if variant == "baseline":
                            document["ab_selection"] = {"variant": "baseline", "used_fusion": False,
                                                        "artifact_path": str(so), "artifact_retained": args.keep_artifacts,
                                                        "reason": "validated baseline retained pending optional candidate and timing"}
                            document["baseline_preserved"] = True
                        save()
                    except Exception as error:
                        metadata.update(_error(error))
                        if variant == "baseline":
                            raise
                        # Failed optional optimization cannot contaminate the retained baseline.
                        with policy.execution_context(device, autocast=False):
                            for x, saved, _pointer in state:
                                if not _bits_equal(x, saved):
                                    x.copy_(saved)
                            for case, saved in zip(cases, snapshots, strict=True):
                                for x, y in zip(case[1], saved, strict=True):
                                    if not _bits_equal(x, y):
                                        x.copy_(y)
                        readonly()
                        save()
                document["all_real_samples_native_validated"] = bool(cap.samples) and all(
                    metadata.get("validated_real_samples") == len(cap.samples)
                    for metadata in document["variants"].values())
                document["baseline_all_real_samples_native_validated"] = bool(cap.samples) and (
                    document["variants"]["baseline"].get("validated_real_samples") == len(cap.samples))
                # All cases on both eligible runners have passed before any timings start.
                with policy.execution_context(device, autocast=False):
                    document["stage"] = "timing"
                    orders = [("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
                              for i in range(args.repeats)] if "candidate" in runners else [("baseline",)] * args.repeats
                    document["round_orders"] = orders
                    for index, case in enumerate(cases):
                        inputs = case[1]
                        for runner in runners.values():
                            for _ in range(20):
                                runner.run(list(inputs))
                        torch.cuda.synchronize(device)
                        rounds = {variant: [] for variant in runners}
                        for order in orders:
                            for variant in order:
                                torch.cuda.synchronize(device)
                                start = time.perf_counter()
                                for _ in range(args.calls):
                                    runners[variant].run(list(inputs))
                                torch.cuda.synchronize(device)
                                rounds[variant].append((time.perf_counter() - start) * 1000 / args.calls)
                        for variant, values in rounds.items():
                            document["cases"][index]["timings"][variant] = {
                                "wall_ms_per_call_rounds": values, "median_ms": statistics.median(values)}
                            validate(runners[variant], variant, index,
                                     baseline_outputs[index] if variant == "candidate" else None)
                        save()
                    readonly()
                baseline_ms = statistics.mean(case["timings"]["baseline"]["median_ms"] for case in document["cases"])
                selection = {"variant": "baseline", "used_fusion": False, "reason": "candidate unavailable or refused",
                             "baseline_mean_case_median_ms": baseline_ms, "margin": MARGIN,
                             "scope": "whole selected node only; not full DAG speedup"}
                if "candidate" in runners:
                    candidate_ms = statistics.mean(case["timings"]["candidate"]["median_ms"] for case in document["cases"])
                    wins, speedup, reason = choose_faster(baseline_ms, candidate_ms)
                    regressions = []
                    for case in document["cases"]:
                        ratio = case["timings"]["candidate"]["median_ms"] / case["timings"]["baseline"]["median_ms"]
                        case.update(candidate_latency_ratio=ratio, critical_regression=ratio > 1 + MARGIN)
                        if case["critical_regression"]:
                            regressions.append(case["name"])
                    use_fusion = wins and not regressions
                    selection.update(variant="candidate" if use_fusion else "baseline", used_fusion=use_fusion,
                                     candidate_mean_case_median_ms=candidate_ms, speedup_vs_baseline=speedup,
                                     reason="critical per-case regression; retain whole-node baseline" if regressions else reason,
                                     critical_regression_cases=regressions)
                selection.update(artifact_path=document["variants"][selection["variant"]]["artifact_path"],
                                 artifact_retained=args.keep_artifacts)
                document.update(ab_selection=selection, status="ok", stage="complete",
                                optimization_status=document["variants"]["candidate"]["status"],
                                baseline_preserved=selection["variant"] == "baseline")
                save()
    except PoisonedCUDA as error:
        document.update(status="error", aborted_poisoned_cuda=True, error=str(error),
                        all_real_samples_native_validated=False)
        save()  # CPU-only; never inspect or restore tensors after CUDA poisoning.
        return 1
    except Exception as error:
        document.update(status="error", error=f"{type(error).__name__}: {error}",
                        all_real_samples_native_validated=False)
        save()
        print(document["error"], flush=True)
        return 1
    finally:
        if owner is not None:
            owner.cleanup()
    print(f"[model:{args.node}] selected {document['ab_selection']['variant']}; "
          f"real samples={document['captured_sample_count']}, probes={document['captured_probe_count']}", flush=True)
    return 0


def _arguments():
    parser = argparse.ArgumentParser(description="Original compile_best fallback or native three-way AOTI experiment")
    parser.add_argument("--native", action="store_true", help="compile/load real model.so variants instead of PT2 packages")
    parser.add_argument("--structured", action="store_true",
                        help="opt in to full RMS and runtime-lengths segment cases (requires --native)")
    parser.add_argument("--model-package", help="trusted parameter package; evaluate only --node using the real capture pipeline")
    parser.add_argument("--node", help="exact declared aoti node name (requires --model-package and --native)")
    parser.add_argument("--model-precision", choices=("float32", "bfloat16", "float16"),
                        default="bfloat16", help="real node deployment precision (default: bfloat16)")
    parser.add_argument("--capabilities", help="model-mode comma-separated rms_norm/segment_pool capabilities; default: segment_pool")
    parser.add_argument("--plan", choices=("serial", "split"), default="serial",
                        help="one model-mode candidate plan; default: serial")
    parser.add_argument("--output-dir", default=str(Path(_HERE) / "results" / "gpu_m23_ab_fallback"),
                        help="directory for native_results.json (created if needed)")
    parser.add_argument("--repeats", type=int, default=7, help="interleaved timing rounds (default: 7)")
    parser.add_argument("--calls", type=int, default=500, help="calls per timed round (default: 500)")
    parser.add_argument("--only", help="comma-separated names/globs; default cases: logsumexp,var,noop; native cases: " +
                        ", ".join(case[0] for case in _NATIVE_CASES) +
                        "; with --structured: recommend_rms*,recommend_segment*")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--keep-artifacts", action="store_true", help="retain native .so and compiler files for separate cold-move validation")
    args = parser.parse_args()
    if args.calls < 1 or args.repeats < 1:
        parser.error("--calls and --repeats must be positive")
    if args.structured and not args.native:
        parser.error("--structured requires --native")
    if bool(args.model_package) != bool(args.node):
        parser.error("--model-package and --node must be supplied together")
    if args.model_package:
        if not args.native:
            parser.error("--model-package requires --native")
        if args.only or args.structured:
            parser.error("model mode validates every captured case; --only/--structured are synthetic-mode options")
        if args.capabilities is not None:
            names = tuple(name.strip() for name in args.capabilities.split(","))
            if not names or len(set(names)) != len(names) or any(
                    name not in ("rms_norm", "segment_pool") for name in names):
                parser.error("--capabilities must contain distinct rms_norm/segment_pool names")
            args.capabilities = names
        return args
    if args.capabilities is not None or args.plan != "serial" or args.model_precision != "bfloat16":
        parser.error("--capabilities/--plan/--model-precision are model-mode options")
    names = [case[0] for case in _native_cases(args)] if args.native else ["logsumexp", "var", "noop"]
    if not any(_selected(name, args.only) for name in names):
        parser.error("--only matched no cases")
    return args


if __name__ == "__main__":
    arguments = _arguments()
    if arguments.model_package:
        raise SystemExit(genericmodel_native_main(arguments))
    if arguments.native:
        raise SystemExit(native_main(arguments))
    raise SystemExit(main(arguments))
