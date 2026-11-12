"""
Tests for the performance fallback (`neptune_opt/ab.py`).

The A/B itself needs two AOTI compilations and a GPU, so what is tested here is everything that
decides: the choice rule (pure), and the orchestration around it with the four heavy pieces replaced
-- export, package, load, time. That includes the branches that matter most and would otherwise only
be exercised by a lucky measurement: a fused package that is SLOWER, one whose numbers moved, one
that does not compile at all, and a model with nothing to fuse.

Run:  python tests/test_ab.py     (or pytest tests/test_ab.py)
"""
from __future__ import annotations

import contextlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neptune_opt  # noqa: E402
from neptune_opt import ab  # noqa: E402


def _report(fusions=("logsumexp",), verified=True):
    return types.SimpleNamespace(fusions=list(fusions), skipped=[], verified=verified,
                                 summary=lambda: "  fused   : logsumexp")


@contextlib.contextmanager
def _fake_compile(fused_ms=0.02, base_ms=0.03, fusions=("logsumexp",), package_fails=None,
                  numbers_ok=True):
    """Replace export/package/load/time, and pretend the machine has a GPU."""
    import tempfile

    import torch

    directory = tempfile.TemporaryDirectory(prefix="neptune_ab_test_")
    calls = {"package": [], "timed": 0}
    real = (ab._export_pair, ab._package, ab._load, ab._ab_time, ab._same_numbers,
            torch.cuda.is_available, ab.tempfile.mkdtemp)

    def fake_export_pair(module, args, kwargs):
        return object(), object(), _report(fusions)

    def fake_package(ep, directory, name):
        calls["package"].append(name)
        if package_fails == name:
            raise RuntimeError("simulated compile failure")
        return os.path.join(directory, f"{name}.pt2")

    def fake_ab_time(fa, fb, args, reps=20, rounds=3, warmup=3):
        calls["timed"] += 1
        return base_ms, fused_ms

    ab._export_pair, ab._package, ab._load = fake_export_pair, fake_package, lambda path: path
    ab._ab_time = fake_ab_time
    ab._same_numbers = lambda fa, fb, args: (numbers_ok, "" if numbers_ok else "output 0 diff")
    torch.cuda.is_available = lambda: True
    ab.tempfile.mkdtemp = lambda **kwargs: directory.name
    try:
        yield calls
    finally:
        (ab._export_pair, ab._package, ab._load, ab._ab_time, ab._same_numbers,
         torch.cuda.is_available, ab.tempfile.mkdtemp) = real
        directory.cleanup()


def test_choose_faster_only_prefers_a_real_win():
    use, speedup, why = ab.choose_faster(0.0300, 0.0195)
    assert use is True and speedup > 1.5 and "faster" in why, (use, speedup, why)
    use, speedup, why = ab.choose_faster(0.0516, 0.1715)          # the measured 0.30x case
    assert use is False and "SLOWER" in why, (use, speedup, why)
    # a tie keeps the baseline: the safe direction to be wrong in is "did not change my model"
    use, _speedup, why = ab.choose_faster(0.0300, 0.0300)
    assert use is False and "margin" in why, why
    use, _speedup, why = ab.choose_faster(0.0300, 0.0295)          # 1.7% faster: inside the margin
    assert use is False, why
    # unmeasurable -> baseline
    for a, b in ((float("nan"), 0.02), (0.03, float("nan")), (0.0, 0.02)):
        assert ab.choose_faster(a, b)[0] is False


def test_the_slow_fusion_is_dropped_and_the_baseline_shipped():
    """The whole point of the fallback: 0.30x must NOT reach the user."""
    with _fake_compile(fused_ms=0.1715, base_ms=0.0516) as calls:
        choice = ab.compile_best(object(), (None,), verbose=False)
    assert choice.used_fusion is False, choice
    assert choice.package_path == choice.baseline_package, choice.package_path
    assert "SLOWER" in choice.reason and "dropped" in choice.reason, choice.reason
    assert choice.speedup < 1 and calls["timed"] == 1
    assert calls["package"] == ["baseline", "fused"], calls["package"]      # timed both, then chose


def test_a_faster_fusion_is_taken():
    with _fake_compile(fused_ms=0.0195, base_ms=0.0300):
        choice = ab.compile_best(object(), (None,), verbose=False)
    assert choice.used_fusion is True, choice
    assert choice.package_path == choice.fused_package
    assert choice.speedup > 1.5


def test_a_fusion_whose_numbers_moved_is_dropped():
    with _fake_compile(numbers_ok=False) as calls:
        choice = ab.compile_best(object(), (None,), verbose=False)
    assert choice.used_fusion is False and "numbers" in choice.reason, choice.reason
    assert calls["timed"] == 0, "a package with wrong numbers must not even be timed"


def test_a_fusion_that_does_not_compile_is_dropped():
    with _fake_compile(package_fails="fused") as calls:
        choice = ab.compile_best(object(), (None,), verbose=False)
    assert choice.used_fusion is False and "failed to compile" in choice.reason, choice.reason
    assert calls["timed"] == 0


def test_nothing_to_fuse_costs_one_compilation():
    """The common case must not pay for a comparison there is nothing to compare."""
    with _fake_compile(fusions=()) as calls:
        choice = ab.compile_best(object(), (None,), verbose=False)
    assert choice.used_fusion is False and "nothing to fuse" in choice.reason, choice.reason
    assert calls["package"] == ["baseline"], calls["package"]
    assert calls["timed"] == 0


def test_a_cpu_only_machine_gets_a_clear_error():
    import torch

    real = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        try:
            ab.compile_best(object(), (None,), verbose=False)
        except RuntimeError as e:
            assert "CUDA device" in str(e) and "export" in str(e), str(e)
        else:
            raise AssertionError("compile_best silently did something on a CPU-only machine")
    finally:
        torch.cuda.is_available = real


def test_it_is_reachable_from_the_package_root():
    """A fallback nobody can find is not a fallback."""
    assert neptune_opt.compile_best is ab.compile_best
    assert "compile_best" in neptune_opt.__all__ and "Choice" in neptune_opt.__all__


def test_same_numbers_preserves_pytree_and_tensor_metadata():
    import torch

    def compare(a, b, **kwargs):
        return ab._same_numbers(lambda: a, lambda: b, (), **kwargs)[0]

    x = torch.ones(2, dtype=torch.float64)
    assert compare({"a": (x, [x])}, {"a": (x.clone(), [x.clone()])})
    assert not compare(x, (x,))
    assert not compare([x], (x,))
    assert not compare({"a": x}, {"b": x})
    assert not compare(x, x.float())
    assert not compare(x, x.reshape(1, 2))
    assert not compare(x, torch.empty(2, device="meta", dtype=x.dtype))
    assert not compare({"a": x, "b": 1}, {"a": x, "b": 1})
    assert compare(torch.empty(0), torch.empty(0))
    for dtype in (torch.float32, torch.float64, torch.complex128):
        for value in (float("inf"), float("-inf"), float("nan")):
            bad = torch.tensor([value], dtype=dtype)
            assert not compare(bad, bad.clone())


def test_same_numbers_never_floatcasts_integer_double_or_complex():
    import torch

    def compare(a, b, **kwargs):
        return ab._same_numbers(lambda: a, lambda: b, (), **kwargs)[0]

    a = torch.tensor([2**60], dtype=torch.int64)
    assert not compare(a, a + 1)
    assert compare(a, a.clone())
    assert not compare(torch.tensor([True]), torch.tensor([False]))
    a = torch.tensor([1.0], dtype=torch.float64)
    assert not compare(a, a + 1e-10, rtol=0, atol=0)
    assert compare(a, a + 1e-10, rtol=0, atol=1e-9)
    a = torch.tensor([1 + 2j], dtype=torch.complex128)
    assert not compare(a, a + 1j)
    assert compare(a, a.clone())
    assert not compare(torch.tensor([1.0]), torch.tensor([1.1]))
    assert compare(torch.tensor([1.0]), torch.tensor([1.1]), rtol=0.2, atol=0)


def test_same_numbers_execution_failure_is_a_refusal():
    def fail():
        raise RuntimeError("execution broke")

    ok, detail = ab._same_numbers(fail, fail, ())
    assert not ok and "execution broke" in detail


def test_choose_faster_rejects_unreliable_times_and_invalid_margin():
    import pytest

    for value in (float("inf"), float("-inf"), float("nan"), 0, -1):
        assert not ab.choose_faster(value, 1)[0]
        assert not ab.choose_faster(1, value)[0]
    assert not ab.choose_faster(1e308, 1e-308)[0]
    assert not ab.choose_faster(1.03, 1, margin=0.03)[0]
    assert not ab.choose_faster(1, 1, margin=0)[0]
    for margin in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="margin"):
            ab.choose_faster(2, 1, margin=margin)
        with pytest.raises(ValueError, match="margin"):
            ab.compile_best(object(), (), margin=margin, verbose=False)


def test_ab_time_alternates_all_rounds_and_uses_medians_after_warmup():
    from unittest.mock import patch

    calls = []
    # Durations in execution order: A/B, B/A, A/B, B/A, A/B.
    durations = [1, 10, 20, 3, 5, 30, 40, 7, 100, 50]
    clock = []
    current = 0
    for duration in durations:
        clock.extend([current, current + duration / 1000])
        current += 1
    with patch.object(ab.time, "perf_counter", side_effect=clock):
        a, b = ab._ab_time(lambda: calls.append("A"), lambda: calls.append("B"), (), reps=1)
    assert calls == list("ABABAB" + "ABBAABBAAB"), calls
    assert abs(a - 5) < 1e-8 and abs(b - 30) < 1e-8, (a, b)


def test_ab_time_rejects_invalid_iteration_counts():
    import pytest

    for kwargs in ({"reps": 0}, {"rounds": 0}, {"warmup": -1}, {"rounds": 1.5}, {"reps": True}):
        with pytest.raises(ValueError):
            ab._ab_time(lambda: None, lambda: None, (), **kwargs)


def test_export_pair_keeps_baseline_separate_and_passes_verify_with():
    from unittest.mock import patch

    import torch

    baseline = types.SimpleNamespace(run_decompositions=lambda: (_ for _ in ()).throw(
        AssertionError("baseline was decomposed")))
    candidate = types.SimpleNamespace(run_decompositions=lambda: candidate)
    module, args = object(), (object(),)
    kwargs = {"strict": False}
    with patch.object(torch.export, "export", side_effect=[baseline, candidate]) as export:
        with patch.object(neptune_opt, "apply", return_value=_report()) as apply:
            a, b, report = ab._export_pair(module, args, kwargs)
    assert a is baseline and b is candidate and report.verified
    assert export.call_count == 2
    assert export.call_args.kwargs == kwargs
    apply.assert_called_once_with(candidate, verify_with=(module, args))


def test_export_pair_candidate_errors_refuse_but_baseline_errors_raise():
    from unittest.mock import patch

    import pytest
    import torch

    base = object()
    with patch.object(torch.export, "export", side_effect=[base, RuntimeError("candidate export")]):
        a, b, report = ab._export_pair(object(), (), {})
    assert a is base and b is None and not report.fusions and not report.verified
    assert "candidate export" in report.summary()
    with patch.object(torch.export, "export", side_effect=RuntimeError("baseline export")):
        with pytest.raises(RuntimeError, match="baseline export"):
            ab._export_pair(object(), (), {})
    candidate = types.SimpleNamespace(run_decompositions=lambda: candidate)
    with patch.object(torch.export, "export", side_effect=[base, candidate]):
        with patch.object(neptune_opt, "apply", side_effect=RuntimeError("apply broke")):
            a, b, report = ab._export_pair(object(), (), {})
    assert a is base and b is None and "apply broke" in report.summary()


def test_unverified_report_never_compiles_fusion_but_validates_baseline():
    from unittest.mock import patch

    for verified in (False, None):
        with _fake_compile() as calls:
            report = _report(verified=verified)
            if verified is None:
                del report.verified
            with patch.object(ab, "_export_pair", return_value=(object(), object(), report)):
                with patch.object(ab, "_same_numbers", return_value=(True, "")) as check:
                    choice = ab.compile_best(object(), (), verbose=False)
            assert not choice.used_fusion and "unverified" in choice.reason
            assert calls["package"] == ["baseline"] and check.call_count == 1
            assert choice.sample_metrics[0]["baseline_verified"]


def test_candidate_load_verify_time_exceptions_fall_back():
    from unittest.mock import patch

    for piece in ("_load", "_same_numbers", "_ab_time"):
        with _fake_compile() as calls:
            if piece == "_load":
                replacement = lambda path: (_ for _ in ()).throw(RuntimeError("load broke"))
            elif piece == "_same_numbers":
                replacement = lambda *a: (_ for _ in ()).throw(RuntimeError("verify broke"))
            else:
                replacement = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("time broke"))
            with patch.object(ab, piece, replacement):
                choice = ab.compile_best(object(), (), verbose=False)
            assert not choice.used_fusion and "failed" in choice.reason
        assert choice.package_path == choice.baseline_package
    with _fake_compile():
        with patch.object(ab, "_load", side_effect=["baseline", RuntimeError("candidate load")]):
            choice = ab.compile_best(object(), (), verbose=False)
        assert not choice.used_fusion and "candidate load" in choice.reason
    with _fake_compile():
        with patch.object(ab, "_same_numbers", side_effect=[(True, ""), RuntimeError("candidate verify")]):
            choice = ab.compile_best(object(), (), verbose=False)
        assert not choice.used_fusion and "candidate verify" in choice.reason


def test_baseline_compile_error_is_not_masked():
    import pytest

    with _fake_compile(package_fails="baseline"):
        with pytest.raises(RuntimeError, match="simulated compile failure"):
            ab.compile_best(object(), (), verbose=False)


def test_verification_distribution_checks_all_cases_before_any_timing():
    from unittest.mock import patch

    with _fake_compile() as calls:
        # All baselines pass; middle candidate fails but last sample must still be verified.
        checks = [(True, "")] * 3 + [(True, ""), (False, "shape mismatch"), (True, "")]
        with patch.object(ab, "_same_numbers", side_effect=checks) as check:
            choice = ab.compile_best(object(), (1,), verification_inputs=[(2,), (3,)], verbose=False)
        assert check.call_count == 6
        assert calls["timed"] == 0 and not choice.used_fusion
        assert [m["verified"] for m in choice.sample_metrics] == [True, False, True]
        assert "case 1" in choice.reason


def test_distribution_aggregate_gain_cannot_hide_critical_key_regression():
    from unittest.mock import patch

    import torch

    args = (torch.zeros(1),)
    other = (torch.zeros(2),)
    with _fake_compile():
        with patch.object(ab, "_ab_time", side_effect=[(100, 10), (1, 2)]) as timing:
            choice = ab.compile_best(object(), args, verification_inputs=[other], verbose=False)
        assert timing.call_count == 2
        assert choice.speedup > 8 and not choice.used_fusion
        assert "per-key regression" in choice.reason
        assert len(choice.sample_metrics) == 2
    with _fake_compile():
        with patch.object(ab, "_ab_time", side_effect=[(100, 10), (1, 1.02)]):
            choice = ab.compile_best(object(), args, verification_inputs=[other], verbose=False)
        assert choice.used_fusion
    with _fake_compile():
        with patch.object(ab, "_ab_time", side_effect=[(1, 1), (1, 1)]):
            choice = ab.compile_best(object(), args, verification_inputs=[other], verbose=False)
        assert not choice.used_fusion and "margin" in choice.reason


def test_distribution_equal_key_aggregation_and_nonfinite_refusal():
    from unittest.mock import patch

    import torch

    args = (torch.zeros(1),)
    with _fake_compile():
        with patch.object(ab, "_ab_time", side_effect=[(100, 10), (1, 2)]):
            choice = ab.compile_best(object(), args, verification_inputs=[args], verbose=False)
        assert choice.used_fusion  # repeated key represents its sampled distribution
    with _fake_compile():
        with patch.object(ab, "_ab_time", side_effect=[(1, 0.5), (float("inf"), 1)]) as timing:
            choice = ab.compile_best(object(), args, verification_inputs=[args], verbose=False)
        assert timing.call_count == 2 and not choice.used_fusion
        assert "unreliable" in choice.reason


def test_tolerance_kwargs_do_not_leak_into_export_and_default_patches_still_work():
    from unittest.mock import patch

    with _fake_compile():
        with patch.object(ab, "_same_numbers", return_value=(True, "")) as check:
            with patch.object(ab, "_export_pair", return_value=(object(), object(), _report())) as export:
                choice = ab.compile_best(object(), (), rtol=1e-6, atol=1e-8,
                                         verification_inputs=[()], strict=False, verbose=False)
        assert choice.used_fusion and check.call_count == 4
        assert check.call_args.kwargs == {"rtol": 1e-6, "atol": 1e-8}
        assert export.call_args.args[2] == {"strict": False}
    with _fake_compile() as calls:
        choice = ab.compile_best(object(), (), verbose=False)
        assert calls["timed"] == 1 and len(choice.sample_metrics) == 1
        assert "not all dynamic inputs" in choice.reason
    old = ab.Choice("base", False, 1, 2, 0.5, "reason", None, "base")
    assert old.sample_metrics is None and old.fused_package is None


def test_verification_opt_out_is_explicit_and_not_marked_verified():
    from unittest.mock import patch

    with _fake_compile():
        with patch.object(ab, "_export_pair", return_value=(object(), object(), _report(verified=False))):
            with patch.object(ab, "_same_numbers", side_effect=AssertionError("must not verify")):
                choice = ab.compile_best(object(), (), verify=False, verbose=False)
        assert choice.used_fusion and choice.sample_metrics[0]["verified"] is None


def test_verification_options_are_validated_before_compilation():
    import pytest

    for kwargs in ({"regression_margin": -1}, {"rtol": float("inf")}, {"atol": -1},
                   {"rounds": 0}, {"reps": 1.5}):
        with pytest.raises(ValueError):
            ab.compile_best(object(), (), verbose=False, **kwargs)
    for cases in ([[]], [1], iter([()])):
        with pytest.raises(TypeError, match="verification_inputs"):
            ab.compile_best(object(), (), verification_inputs=cases, verbose=False)


def test_real_cpu_export_preserves_parameterized_baseline_graph():
    from unittest.mock import patch

    import torch

    model = torch.nn.Linear(3, 2).eval()
    args = (torch.ones(4, 3),)
    seen = []

    def apply(candidate, *, verify_with):
        seen.append(candidate)
        assert verify_with[0] is model and verify_with[1] == args
        candidate.graph_module.meta["candidate_only"] = True
        return _report(fusions=())

    with patch.object(neptune_opt, "apply", side_effect=apply):
        baseline, candidate, report = ab._export_pair(model, args, {})
    assert baseline is not candidate and seen == [candidate]
    assert baseline.graph_module is not candidate.graph_module
    assert "candidate_only" not in baseline.graph_module.meta
    assert any(node.target == torch.ops.aten.linear.default for node in baseline.graph.nodes)
    assert ab._same_numbers(model, baseline.module(), args)[0]
    assert not report.fusions


def test_baseline_validation_records_all_cases_even_on_failure():
    from unittest.mock import patch

    with _fake_compile() as calls:
        with patch.object(ab, "_same_numbers", side_effect=[(False, "bad baseline"), (True, "")]) as check:
            choice = ab.compile_best(object(), (1,), verification_inputs=[(2,)], verbose=False)
        assert check.call_count == 2 and calls["package"] == ["baseline"]
        assert not choice.used_fusion and calls["timed"] == 0
        assert [m["baseline_verified"] for m in choice.sample_metrics] == [False, True]


def test_plans_validation_is_bounded_and_precedes_any_export():
    import itertools
    from unittest.mock import patch

    import pytest

    for plans in (None, "serial", b"split", (), ("unknown",), ("serial", "serial"),
                  (1,), (float("inf"),), itertools.repeat("serial")):
        with patch.object(ab, "_export_pair", side_effect=AssertionError("must not export")):
            with pytest.raises(ValueError, match="plans"):
                ab.compile_best(object(), (), plans=plans, verbose=False)


def test_multiplan_selects_split_and_compiles_original_baseline_only_once():
    from unittest.mock import patch

    original = object()
    reports = {plan: _report() for plan in ("serial", "split")}
    exports = []

    def export(module, args, kwargs, plan=None, *, baseline=None):
        exports.append((plan, baseline, kwargs))
        return original if baseline is None else object(), plan, reports[plan]

    def timing(fa, fb, args, reps=20, rounds=5):
        assert fa.endswith("baseline.pt2")
        return (10, 12) if fb.endswith("fused_serial.pt2") else (10, 5)

    with _fake_compile() as calls:
        with patch.object(ab, "_export_pair", side_effect=export):
            with patch.object(ab, "_ab_time", side_effect=timing):
                choice = ab.compile_best(object(), (), plans=("serial", "split"),
                                         strict=False, verbose=False)
    assert calls["package"] == ["baseline", "fused_serial", "fused_split"]
    assert exports == [("serial", None, {"strict": False}), ("split", original, {"strict": False})]
    assert choice.used_fusion and choice.selected_plan == "split"
    assert choice.package_path.endswith("fused_split.pt2") and choice.report is reports["split"]
    assert [m["eligible"] for m in choice.candidate_metrics] == [False, True]
    assert [m["selected"] for m in choice.candidate_metrics] == [False, True]
    assert choice.sample_metrics is choice.candidate_metrics[1]["sample_metrics"]
    assert choice.candidate_metrics[0]["sample_metrics"][0]["fused_ms"] == 12


def test_multiplan_best_is_lowest_eligible_latency_not_highest_noisy_ratio():
    from unittest.mock import patch

    for timings, selected in (([(10, 5), (100, 6)], "serial"),
                              ([(10, 5), (10, 5)], "serial"),
                              ([(10, 5), (10, 4)], "split")):
        with _fake_compile():
            with patch.object(ab, "_export_pair", return_value=(object(), object(), _report())):
                with patch.object(ab, "_ab_time", side_effect=timings):
                    choice = ab.compile_best(object(), (), plans=("serial", "split"), verbose=False)
        assert choice.selected_plan == selected
        assert sum(m["selected"] for m in choice.candidate_metrics) == 1
        assert all(m["eligible"] for m in choice.candidate_metrics)


def test_split_only_is_opt_in_and_default_serial_keeps_old_package_and_signature():
    from unittest.mock import patch

    with _fake_compile() as calls:
        with patch.object(ab, "_export_pair", return_value=(object(), object(), _report())) as export:
            choice = ab.compile_best(object(), (), plans=iter(["split"]), verbose=False)
        assert export.call_args.kwargs == {"plan": "split"}
        assert calls["package"] == ["baseline", "fused_split"]
        assert choice.selected_plan == "split"
    with _fake_compile() as calls:
        with patch.object(ab, "_export_pair", return_value=(object(), object(), _report())) as export:
            choice = ab.compile_best(object(), (), verbose=False)
        assert len(export.call_args.args) == 3 and export.call_args.kwargs == {}
        assert calls["package"] == ["baseline", "fused"] and choice.selected_plan == "serial"
    old = ab.Choice("base", False, 1, 2, 0.5, "reason", None, "base", None, None)
    assert old.selected_plan is None and old.candidate_metrics is None


def test_candidate_failure_at_each_stage_continues_to_other_plan():
    from unittest.mock import patch

    for stage in ("refused", "unverified", "export", "compile", "load", "verify", "time", "nonfinite"):
        serial_report = _report()
        split_report = _report()
        original = object()
        timed = []
        checked = []

        def export(module, args, kwargs, plan=None, *, baseline=None):
            if plan == "serial":
                if stage == "refused":
                    serial_report.fusions = []
                    serial_report.summary = lambda: "symbolic shape unsupported"
                if stage == "unverified":
                    serial_report.verified = False
                return original, object(), serial_report
            if stage == "export":
                raise RuntimeError("split export broke")
            return object(), object(), split_report

        def load(path):
            if stage == "load" and path.endswith("fused_serial.pt2"):
                raise RuntimeError("serial load broke")
            return path

        def check(fa, fb, args):
            checked.append(fb)
            if stage == "verify" and fb.endswith("fused_serial.pt2"):
                return False, "serial numbers wrong"
            return True, ""

        def timing(fa, fb, args, **kwargs):
            timed.append(fb)
            if stage == "time" and fb.endswith("fused_serial.pt2"):
                raise RuntimeError("serial time broke")
            if stage == "nonfinite" and fb.endswith("fused_serial.pt2"):
                return float("inf"), 1
            return 10, 5

        with _fake_compile(package_fails="fused_serial" if stage == "compile" else None) as calls:
            with patch.object(ab, "_export_pair", side_effect=export), \
                 patch.object(ab, "_load", side_effect=load), \
                 patch.object(ab, "_same_numbers", side_effect=check), \
                 patch.object(ab, "_ab_time", side_effect=timing):
                choice = ab.compile_best(object(), (), plans=("serial", "split"), verbose=False)
        assert calls["package"].count("baseline") == 1
        expected = "serial" if stage == "export" else "split"
        assert choice.selected_plan == expected, (stage, choice.reason)
        assert choice.report is (serial_report if expected == "serial" else split_report)
        failed = choice.candidate_metrics[1 if stage == "export" else 0]
        assert not failed["eligible"] and failed["reason"]
        if stage in ("refused", "unverified", "compile", "load", "verify"):
            assert not any(path.endswith("fused_serial.pt2") for path in timed)
        if stage == "refused":
            assert "symbolic shape unsupported" in failed["reason"]


def test_all_plans_fail_closed_and_explain_every_rejection():
    from unittest.mock import patch

    report = _report(fusions=(), verified=False)
    report.summary = lambda: "split symbolic shape unsupported"
    with _fake_compile() as calls:
        with patch.object(ab, "_export_pair", side_effect=[(object(), object(), _report()),
                                                          (object(), None, report)]):
            with patch.object(ab, "_ab_time", return_value=(10, 12)):
                choice = ab.compile_best(object(), (), plans=("serial", "split"), verbose=False)
    assert not choice.used_fusion and choice.selected_plan is None
    assert choice.package_path == choice.baseline_package
    assert "serial" in choice.reason and "split symbolic shape unsupported" in choice.reason
    assert not any(m["eligible"] or m["selected"] for m in choice.candidate_metrics)
    assert calls["package"] == ["baseline", "fused_serial"]


def test_stride_is_a_distinct_regression_key_for_each_plan():
    from unittest.mock import patch

    import torch

    contiguous = torch.zeros(2, 2)
    transposed = contiguous.t()
    assert contiguous.shape == transposed.shape
    assert ab._input_key((contiguous,)) != ab._input_key((transposed,))
    with _fake_compile():
        with patch.object(ab, "_export_pair", return_value=(object(), object(), _report())):
            with patch.object(ab, "_ab_time", side_effect=[(100, 10), (1, 2), (100, 20), (1, 0.5)]):
                choice = ab.compile_best(object(), (contiguous,), verification_inputs=[(transposed,)],
                                         plans=("serial", "split"), verbose=False)
    assert choice.selected_plan == "split"
    assert "per-key regression" in choice.candidate_metrics[0]["reason"]
    assert all(m["verified"] for m in choice.sample_metrics)
    assert choice.baseline_ms == 50.5 and choice.fused_ms == 10.25


def test_real_cpu_plan_clones_do_not_mutate_baseline_graph_or_weights():
    from unittest.mock import patch

    import torch

    model = torch.nn.Linear(3, 2).eval()
    args = (torch.ones(4, 3),)
    baseline = torch.export.export(model, args)
    code = baseline.graph_module.code
    weights = {name: value.clone() for name, value in baseline.state_dict.items()}
    candidates = []

    def apply(candidate, *, verify_with, plan):
        assert verify_with[0] is model
        assert candidate.graph_module.code == code
        assert "plan" not in candidate.graph_module.meta
        for name, value in candidate.state_dict.items():
            assert torch.equal(value, weights[name])
            assert value.data_ptr() != baseline.state_dict[name].data_ptr()
        candidate.graph_module.meta["plan"] = plan
        with torch.no_grad():
            next(iter(candidate.state_dict.values())).add_(100)
        candidates.append(candidate)
        return _report()

    with patch.object(torch.export, "export", side_effect=AssertionError("must clone existing baseline")):
        with patch.object(neptune_opt, "apply", side_effect=apply):
            for plan in ("serial", "split"):
                original, candidate, report = ab._export_pair(model, args, {}, plan=plan, baseline=baseline)
                assert original is baseline and report.verified
    assert baseline.graph_module.code == code and "plan" not in baseline.graph_module.meta
    assert candidates[0].graph_module is not candidates[1].graph_module
    for name, value in baseline.state_dict.items():
        assert torch.equal(value, weights[name])
        assert torch.equal(model.state_dict()[name], weights[name])


def test_failed_or_incomplete_copy_reexports_candidate_without_replacing_baseline():
    from unittest.mock import patch

    import torch

    for incomplete in (False, True):
        base = types.SimpleNamespace(graph_module=object(), state_dict={})
        fresh = object()
        copy_kwargs = {"side_effect": [base, object()]} if incomplete else {
            "side_effect": [RuntimeError("copy failed"), object()]}
        with patch.object(ab.copy, "deepcopy", **copy_kwargs):
            with patch.object(torch.export, "export", return_value=fresh) as export:
                with patch.object(neptune_opt, "apply", return_value=_report()) as apply:
                    original, candidate, report = ab._export_pair(object(), (), {"strict": False},
                                                                  plan="split", baseline=base)
        assert original is base and candidate is fresh and report.verified
        assert export.call_count == 1 and export.call_args.kwargs == {"strict": False}
        assert apply.call_args.kwargs["plan"] == "split"


def test_multiplan_baseline_compile_and_load_failure_boundaries():
    from unittest.mock import patch

    import pytest

    with _fake_compile(package_fails="baseline"):
        with patch.object(ab, "_export_pair", return_value=(object(), object(), _report())):
            with pytest.raises(RuntimeError, match="simulated compile failure"):
                ab.compile_best(object(), (), plans=("serial", "split"), verbose=False)
    with _fake_compile() as calls:
        with patch.object(ab, "_export_pair", return_value=(object(), object(), _report())) as export:
            with patch.object(ab, "_load", side_effect=RuntimeError("baseline load failed")):
                choice = ab.compile_best(object(), (), plans=("serial", "split"), verbose=False)
        assert export.call_count == 1 and calls["package"] == ["baseline"]
        assert not choice.used_fusion and choice.selected_plan is None
        assert choice.candidate_metrics == [] and "baseline load failed" in choice.reason


def test_fresh_export_fallback_isolates_module_weights_and_preserves_baseline():
    from unittest.mock import patch

    import torch

    model = torch.nn.Linear(3, 2).eval()
    args = (torch.ones(4, 3),)
    baseline = torch.export.export(model, args)
    original_deepcopy = ab.copy.deepcopy
    weight = model.weight.clone()

    def deepcopy(value):
        if value is baseline:
            raise RuntimeError("ExportedProgram copy unsupported")
        return original_deepcopy(value)

    def apply(candidate, *, verify_with, plan):
        assert verify_with[0] is model and plan == "split"
        candidate_weight = candidate.state_dict["weight"]
        assert candidate_weight.data_ptr() != model.weight.data_ptr()
        with torch.no_grad():
            candidate_weight.add_(100)
        return _report()

    with patch.object(ab.copy, "deepcopy", side_effect=deepcopy):
        with patch.object(neptune_opt, "apply", side_effect=apply):
            original, candidate, report = ab._export_pair(model, args, {}, plan="split", baseline=baseline)
    assert original is baseline and candidate is not None and report.verified
    assert torch.equal(model.weight, weight) and torch.equal(baseline.state_dict["weight"], weight)


def test_multiplan_each_accuracy_distribution_is_verified_before_its_timing():
    from unittest.mock import patch

    checked = []
    timed = []

    def check(fa, fb, args):
        checked.append((fb, args))
        if fb.endswith("fused_serial.pt2") and args == (2,):
            return False, "dynamic sample numbers differ"
        return True, ""

    def timing(fa, fb, args, **kwargs):
        timed.append((fb, args))
        assert sum(path == fb for path, case in checked) == 3
        return 10, 5

    with _fake_compile():
        with patch.object(ab, "_export_pair", return_value=(object(), object(), _report())), \
             patch.object(ab, "_same_numbers", side_effect=check), \
             patch.object(ab, "_ab_time", side_effect=timing):
            choice = ab.compile_best(object(), (1,), verification_inputs=[(2,), (3,)],
                                     plans=("serial", "split"), verbose=False)
    assert choice.selected_plan == "split" and len(checked) == 9 and len(timed) == 3
    assert all(path.endswith("fused_split.pt2") for path, case in timed)
    assert [m["verified"] for m in choice.candidate_metrics[0]["sample_metrics"]] == [True, False, True]
    assert all(m["verified"] for m in choice.sample_metrics)


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            failed.append(name)
            print(f"  [FAIL] {name}: {e}")
        else:
            print(f"  [PASS] {name}")
    print(f"TOTAL: {len(tests) - len(failed)} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
