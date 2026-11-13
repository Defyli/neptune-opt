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

import pytest  # noqa: E402

import neptune_opt  # noqa: E402
from neptune_opt import ab  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_dynamo_cache():
    import torch._dynamo

    # Real strict exports specialize a shared wrapper on each module instance.
    # Do not let another test's entries exhaust old Torch's per-frame cache limit.
    torch._dynamo.reset()
    try:
        yield
    finally:
        torch._dynamo.reset()


def _report(fusions=("logsumexp",), verified=True):
    return types.SimpleNamespace(fusions=list(fusions), skipped=[], verified=verified,
                                 summary=lambda: "  fused   : logsumexp")


@contextlib.contextmanager
def _fake_compile(fused_ms=0.02, base_ms=0.03, fusions=("logsumexp",), package_fails=None,
                  numbers_ok=True):
    """Fake the timing gate, but keep real export's CUDA detection isolated."""
    import tempfile
    from unittest.mock import patch

    import torch

    directory = tempfile.TemporaryDirectory(prefix="neptune_ab_test_")
    calls = {"package": [], "timed": 0}
    real = (ab._export_pair, ab._package, ab._load, ab._ab_time, ab._same_numbers,
            torch.cuda.is_available, ab.tempfile.mkdtemp, torch.export.export)
    real_cuda_available = torch.cuda.is_available
    real_export = torch.export.export
    real_mkdtemp = ab.tempfile.mkdtemp

    def cpu_export(*args, **kwargs):
        # preserve_rng_state must not see the GPU gate used by compile_best.
        # Restore the callable, not a cached result, for every real export.
        # Torch 2.6's first export also lazily imports modules creating temp dirs;
        # those must not use compile_best's shared-directory stub.
        with patch.object(torch.cuda, "is_available", real_cuda_available), \
             patch.object(ab.tempfile, "mkdtemp", real_mkdtemp):
            return real_export(*args, **kwargs)

    def fake_export_pair(module, args, kwargs, **candidate_kwargs):
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
    torch.export.export = cpu_export
    ab.tempfile.mkdtemp = lambda **kwargs: directory.name
    try:
        yield calls
    finally:
        (ab._export_pair, ab._package, ab._load, ab._ab_time, ab._same_numbers,
         torch.cuda.is_available, ab.tempfile.mkdtemp, torch.export.export) = real
        directory.cleanup()


def test_fake_compile_real_export_restores_cuda_detection_even_after_failure():
    from unittest.mock import patch

    import pytest
    import torch

    real_cuda_available = torch.cuda.is_available
    real_export = torch.export.export
    real_mkdtemp = ab.tempfile.mkdtemp
    model = torch.nn.Linear(3, 2).eval()
    args = (torch.ones(4, 3),)
    failure = RuntimeError("export failed before tracing")

    def checked_export(module, *args, **kwargs):
        assert torch.cuda.is_available is real_cuda_available
        assert ab.tempfile.mkdtemp is real_mkdtemp
        if module is None:
            raise failure
        return real_export(module, *args, **kwargs)

    with patch.object(torch.export, "export", side_effect=checked_export) as export:
        with _fake_compile() as calls:
            fake_cuda_available = torch.cuda.is_available
            fake_mkdtemp = ab.tempfile.mkdtemp
            assert fake_cuda_available() is True
            for strict in (False, True):
                baseline = torch.export.export(model, args, strict=strict)
                assert ab._same_numbers(model, baseline.module(), args)[0]
                assert torch.cuda.is_available is fake_cuda_available
                assert ab.tempfile.mkdtemp is fake_mkdtemp
            with pytest.raises(RuntimeError) as caught:
                torch.export.export(None, args)
            assert caught.value is failure
            assert torch.cuda.is_available is fake_cuda_available
            assert ab.tempfile.mkdtemp is fake_mkdtemp
            assert calls == {"package": [], "timed": 0}
        assert torch.export.export is export
        assert export.call_count == 3
        assert [call.kwargs for call in export.call_args_list] == [
            {"strict": False}, {"strict": True}, {}]
    assert torch.cuda.is_available is real_cuda_available
    assert torch.export.export is real_export
    assert ab.tempfile.mkdtemp is real_mkdtemp


def test_package_without_unique_user_kernel_names_checks_feature_gate(monkeypatch, tmp_path):
    from unittest.mock import Mock, patch

    import pytest
    import torch
    from torch._inductor import config

    supported = hasattr(config.triton, "unique_user_kernel_names")
    with monkeypatch.context() as scoped:
        scoped.delitem(config._config, "triton.unique_user_kernel_names", raising=False)
        assert not hasattr(config.triton, "unique_user_kernel_names")
        calls = []
        failure = RuntimeError("simulated legacy package failure")

        def compile_and_package(ep, *, package_path):
            assert not hasattr(config.triton, "unique_user_kernel_names")
            calls.append((ep, package_path))
            if package_path.endswith("fused.pt2"):
                raise failure

        scoped.setattr(torch._inductor, "aoti_compile_and_package",
                       compile_and_package, raising=False)
        config_patch = Mock(wraps=config.patch)
        # ConfigModule rejects setattr for methods; shadow it in the module dict.
        with patch.dict(config.__dict__, {"patch": config_patch}):
            ep = object()
            expected_path = str(tmp_path / "baseline.pt2")
            assert ab._package(ep, tmp_path, "baseline") == expected_path
            with pytest.raises(RuntimeError) as caught:
                ab._package(ep, tmp_path, "fused")
            assert caught.value is failure
            assert calls == [(ep, expected_path), (ep, str(tmp_path / "fused.pt2"))]
            expected_options = ({"triton.autotune_at_compile_time": False}
                                if hasattr(config.triton, "autotune_at_compile_time") else {})
            assert [call.args for call in config_patch.call_args_list] == [
                (expected_options,), (expected_options,)]
            for target in ("neptune.state_rms_norm.default", "neptune.segment_mean.default",
                           "neptune.segment_sum.default"):
                structured = types.SimpleNamespace(graph=types.SimpleNamespace(
                    nodes=[types.SimpleNamespace(target=target)]))
                with pytest.raises(NotImplementedError, match="unique_user_kernel_names"):
                    ab._package(structured, tmp_path, "structured")
            assert len(calls) == 2 and config_patch.call_count == 2
        assert not hasattr(config.triton, "unique_user_kernel_names")
    assert hasattr(config.triton, "unique_user_kernel_names") is supported


@pytest.mark.parametrize("previous", (None, True, False))
def test_package_uses_scoped_unique_user_kernel_names(monkeypatch, tmp_path, previous):
    import torch
    from torch._inductor import config

    keys = ("triton.unique_user_kernel_names", "triton.autotune_at_compile_time")
    supported = tuple(hasattr(config.triton, key.split(".")[1]) for key in keys)
    with monkeypatch.context() as scoped:
        for key in keys:
            if key not in config._config:
                # Copy stock metadata, not a plain bool on _ConfigEntry-based Torch.
                scoped.setitem(config._config, key,
                               ab.copy.deepcopy(config._config["triton.unique_kernel_names"]))
        original = (config.triton.unique_user_kernel_names, config.triton.autotune_at_compile_time)
        with config.patch(dict.fromkeys(keys, previous)):
            for result in (None, object()):
                calls = []

                # Keep both wrappers on the same scoped options. This mock does not
                # execute AOTI: real packaging may tune with actual EP inputs.
                def compile_and_package(ep, *, package_path):
                    assert config.triton.unique_user_kernel_names is True
                    assert config.triton.autotune_at_compile_time is False
                    calls.append((ep, package_path))
                    return result

                scoped.setattr(torch._inductor, "aoti_compile_and_package",
                               compile_and_package, raising=False)
                for name in ("baseline", "fused"):
                    ep = object()
                    expected_path = str(tmp_path / f"{name}.pt2")
                    assert config.triton.unique_user_kernel_names is previous
                    assert config.triton.autotune_at_compile_time is previous
                    assert ab._package(ep, tmp_path, name) == expected_path
                    assert len(calls) == (1 if name == "baseline" else 2)
                    assert calls[-1] == (ep, expected_path)
                    assert config.triton.unique_user_kernel_names is previous
                    assert config.triton.autotune_at_compile_time is previous
        assert (config.triton.unique_user_kernel_names,
                config.triton.autotune_at_compile_time) == original
    assert tuple(hasattr(config.triton, key.split(".")[1]) for key in keys) == supported


@pytest.mark.parametrize("previous", (None, True, False))
def test_package_restores_unique_user_kernel_names_after_exception(monkeypatch, tmp_path, previous):
    import torch
    from torch._inductor import config

    keys = ("triton.unique_user_kernel_names", "triton.autotune_at_compile_time")
    supported = tuple(hasattr(config.triton, key.split(".")[1]) for key in keys)
    with monkeypatch.context() as scoped:
        for key in keys:
            if key not in config._config:
                scoped.setitem(config._config, key,
                               ab.copy.deepcopy(config._config["triton.unique_kernel_names"]))
        original = (config.triton.unique_user_kernel_names, config.triton.autotune_at_compile_time)
        with config.patch(dict.fromkeys(keys, previous)):
            calls = []
            failure = RuntimeError("simulated AOTI package failure")

            def compile_and_package(ep, *, package_path):
                assert config.triton.unique_user_kernel_names is True
                assert config.triton.autotune_at_compile_time is False
                calls.append((ep, package_path))
                raise failure

            scoped.setattr(torch._inductor, "aoti_compile_and_package",
                           compile_and_package, raising=False)
            for name in ("baseline", "fused"):
                ep = object()
                expected_path = str(tmp_path / f"{name}.pt2")
                with pytest.raises(RuntimeError) as caught:
                    ab._package(ep, tmp_path, name)
                assert caught.value is failure
                assert len(calls) == (1 if name == "baseline" else 2)
                assert calls[-1] == (ep, expected_path)
                assert config.triton.unique_user_kernel_names is previous
                assert config.triton.autotune_at_compile_time is previous
        assert (config.triton.unique_user_kernel_names,
                config.triton.autotune_at_compile_time) == original
    assert tuple(hasattr(config.triton, key.split(".")[1]) for key in keys) == supported


@pytest.mark.parametrize("autotune_exists", (True, False))
def test_package_missing_config_keys_restore_after_monkeypatch_context(
        monkeypatch, tmp_path, autotune_exists):
    from unittest.mock import Mock, patch

    import torch
    from torch._inductor import config

    keys = ("triton.unique_user_kernel_names", "triton.autotune_at_compile_time")
    originals = {key: ab.copy.deepcopy(config._config[key]) for key in keys if key in config._config}
    original_values = {key: getattr(config.triton, key.split(".")[1]) for key in originals}
    with monkeypatch.context() as scoped:
        if autotune_exists and keys[1] not in config._config:
            scoped.setitem(config._config, keys[1],
                           ab.copy.deepcopy(config._config["triton.unique_kernel_names"]))
        scoped.delitem(config._config, keys[0], raising=False)
        if not autotune_exists:
            scoped.delitem(config._config, keys[1], raising=False)
        options = {keys[1]: False} if autotune_exists else {}
        outer = {keys[1]: True} if autotune_exists else {}
        config_patch = Mock(wraps=config.patch)
        calls = []
        failure = RuntimeError("legacy scalar compile failure")

        def compile_and_package(ep, *, package_path):
            assert not hasattr(config.triton, "unique_user_kernel_names")
            if autotune_exists:
                assert config.triton.autotune_at_compile_time is False
            else:
                assert not hasattr(config.triton, "autotune_at_compile_time")
            calls.append((ep, package_path))
            if package_path.endswith("fused.pt2"):
                raise failure

        scoped.setattr(torch._inductor, "aoti_compile_and_package",
                       compile_and_package, raising=False)
        with config.patch(outer):
            with patch.dict(config.__dict__, {"patch": config_patch}):
                ep = object()
                assert ab._package(ep, tmp_path, "baseline") == str(tmp_path / "baseline.pt2")
                if autotune_exists:
                    assert config.triton.autotune_at_compile_time is True
                with pytest.raises(RuntimeError) as caught:
                    ab._package(ep, tmp_path, "fused")
                assert caught.value is failure
                if autotune_exists:
                    assert config.triton.autotune_at_compile_time is True
                assert calls == [(ep, str(tmp_path / "baseline.pt2")),
                                 (ep, str(tmp_path / "fused.pt2"))]
                assert [call.args for call in config_patch.call_args_list] == [(options,), (options,)]
                for target in ("neptune.state_rms_norm.default", "neptune.segment_mean.default",
                               "neptune.segment_sum.default"):
                    structured = types.SimpleNamespace(graph=types.SimpleNamespace(
                        nodes=[types.SimpleNamespace(target=target)]))
                    with pytest.raises(NotImplementedError, match="unique_user_kernel_names"):
                        ab._package(structured, tmp_path, "structured")
                assert len(calls) == 2 and config_patch.call_count == 2
        assert not hasattr(config.triton, "unique_user_kernel_names")
    for key in keys:
        assert (key in config._config) is (key in originals)
        if key in originals:
            assert type(config._config[key]) is type(originals[key])
            assert getattr(config.triton, key.split(".")[1]) is original_values[key]


def test_package_without_autotune_key_keeps_unique_user_kernel_names(monkeypatch, tmp_path):
    from unittest.mock import Mock, patch

    import torch
    from torch._inductor import config

    unique = "triton.unique_user_kernel_names"
    autotune = "triton.autotune_at_compile_time"
    supported = hasattr(config.triton, "autotune_at_compile_time")
    original_autotune = config.triton.autotune_at_compile_time if supported else None
    with monkeypatch.context() as scoped:
        if unique not in config._config:
            scoped.setitem(config._config, unique,
                           ab.copy.deepcopy(config._config["triton.unique_kernel_names"]))
        scoped.delitem(config._config, autotune, raising=False)
        config_patch = Mock(wraps=config.patch)
        failure = RuntimeError("legacy package failure")
        calls = []

        def compile_and_package(ep, *, package_path):
            assert config.triton.unique_user_kernel_names is True
            assert not hasattr(config.triton, "autotune_at_compile_time")
            calls.append((ep, package_path))
            if package_path.endswith("fused.pt2"):
                raise failure

        scoped.setattr(torch._inductor, "aoti_compile_and_package",
                       compile_and_package, raising=False)
        with config.patch({unique: False}):
            with patch.dict(config.__dict__, {"patch": config_patch}):
                ep = object()
                assert ab._package(ep, tmp_path, "baseline") == str(tmp_path / "baseline.pt2")
                assert config.triton.unique_user_kernel_names is False
                with pytest.raises(RuntimeError) as caught:
                    ab._package(ep, tmp_path, "fused")
                assert caught.value is failure
                assert config.triton.unique_user_kernel_names is False
                assert calls == [(ep, str(tmp_path / "baseline.pt2")),
                                 (ep, str(tmp_path / "fused.pt2"))]
                assert [call.args for call in config_patch.call_args_list] == [
                    ({unique: True},), ({unique: True},)]
        assert not hasattr(config.triton, "autotune_at_compile_time")
    assert hasattr(config.triton, "autotune_at_compile_time") is supported
    if supported:
        assert config.triton.autotune_at_compile_time is original_autotune


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
        with _fake_compile():
            def replacement(*args, **kwargs):
                label = {"_load": "load", "_same_numbers": "verify", "_ab_time": "time"}[piece]
                raise RuntimeError(f"{label} broke")
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


def test_capabilities_validation_is_bounded_before_baseline_work():
    import itertools
    from unittest.mock import patch

    import pytest

    invalid = (None, 1, "rms_norm", b"segment_pool", ("unknown",), (1,),
               ("rms_norm", "rms_norm"), ("RMS_NORM",), itertools.repeat("segment_pool"))
    for capabilities in invalid:
        with patch.object(ab, "_export_pair") as export, patch.object(ab, "_package") as package:
            with pytest.raises(ValueError, match="capabilities"):
                ab.compile_best(object(), (), capabilities=capabilities, verbose=False)
            export.assert_not_called()
            package.assert_not_called()
    consumed = []

    def unbounded():
        for name in itertools.cycle(("rms_norm", "segment_pool")):
            consumed.append(name)
            assert len(consumed) <= 3
            yield name

    with pytest.raises(ValueError, match="capabilities"):
        ab.compile_best(object(), (), capabilities=unbounded(), verbose=False)
    assert len(consumed) == 3
    for max_ulps in (-1, True, 1.5, "1", float("inf")):
        with patch.object(ab, "_package") as package:
            with pytest.raises(ValueError, match="max_ulps"):
                ab.compile_best(object(), (), max_ulps=max_ulps, verbose=False)
            package.assert_not_called()


def test_real_cpu_capabilities_forward_all_samples_and_tolerances_to_apply():
    from unittest.mock import patch

    import torch

    model = torch.nn.Linear(3, 2).eval()
    args = (torch.ones(4, 3),)
    samples = ((torch.zeros(4, 3),), (torch.full((4, 3), 2.0),))
    real_export_pair = ab._export_pair
    with _fake_compile() as calls:
        # Wrap the isolated proxy, not the original captured before _fake_compile.
        real_export = torch.export.export
        with patch.object(ab, "_export_pair", side_effect=real_export_pair), \
             patch.object(torch.export, "export", wraps=real_export) as export, \
             patch.object(neptune_opt, "apply", return_value=_report(("rms_norm",))) as apply, \
             patch.object(ab, "_same_numbers", return_value=(True, "")) as check:
            choice = ab.compile_best(model, args, capabilities=iter(("rms_norm", "segment_pool")),
                                     verification_inputs=samples, rtol=2e-3, atol=2e-4,
                                     max_ulps=2, strict=False, verbose=False)
    assert choice.used_fusion and calls["package"] == ["baseline", "fused"]
    assert export.call_count == 1 and export.call_args.kwargs == {"strict": False}
    kwargs = apply.call_args.kwargs
    assert kwargs["verify_with"][0] is model and kwargs["verify_with"][1][0] is args[0]
    assert kwargs["verification_inputs"] is samples
    assert kwargs["capabilities"] == ("rms_norm", "segment_pool")
    assert kwargs["rtol"] == 2e-3 and kwargs["atol"] == 2e-4 and kwargs["max_ulps"] == 2
    assert check.call_count == 6
    assert all(call.kwargs == {"rtol": 2e-3, "atol": 2e-4, "max_ulps": 2}
               for call in check.call_args_list)
    assert "capabilities: rms_norm, segment_pool" in choice.reason


def test_export_pair_forwards_only_nondefault_options_and_preserves_three_arg_mock():
    from unittest.mock import patch

    import torch

    model = torch.nn.Linear(2, 1).eval()
    args = (torch.ones(1, 2),)
    for options in ({"rtol": 0.2}, {"atol": 0.1}, {"max_ulps": 0},
                    {"verification_inputs": ()}, {"capabilities": ("rms_norm",)}):
        with patch.object(neptune_opt, "apply", return_value=_report()) as apply:
            ab._export_pair(model, args, {}, **options)
        assert apply.call_args.kwargs == {"verify_with": (model, args), **options}
    with _fake_compile():
        def old_export(module, args, export_kwargs):
            return object(), object(), _report()

        with patch.object(ab, "_export_pair", side_effect=old_export) as export:
            choice = ab.compile_best(object(), (), capabilities=(), max_ulps=None, verbose=False)
        assert choice.used_fusion and export.call_args.kwargs == {}


def test_same_numbers_explicit_ulps_keep_finite_tensor_only_policy():
    import pytest
    import torch

    def compare(a, b, **kwargs):
        return ab._same_numbers(lambda: a, lambda: b, (), rtol=0, atol=0, **kwargs)[0]

    for dtype in (torch.float16, torch.bfloat16):
        a = torch.tensor([1.0, -1.0], dtype=dtype)
        b = torch.nextafter(a, torch.full_like(a, float("inf")))
        assert not compare(a, b)
        assert not compare(a, b, max_ulps=0)
        assert compare(a, b, max_ulps=1)
        far = a + 0.125
        assert not compare(a, far, max_ulps=1)
        for value in (float("nan"), float("inf"), -float("inf")):
            bad = torch.tensor([value], dtype=dtype)
            assert not compare(bad, bad.clone(), max_ulps=100)
        assert not compare((a, None), (a.clone(), None), max_ulps=1)
        assert not compare((a,), [a.clone()], max_ulps=1)
        assert not compare(a, a.float(), max_ulps=1)
        assert compare(torch.empty(0, dtype=dtype), torch.empty(0, dtype=dtype), max_ulps=0)
    for dtype in (torch.float32, torch.float64, torch.complex128):
        a = torch.tensor([1.0], dtype=dtype)
        assert not compare(a, a + 0.001, max_ulps=1000)
    a = torch.tensor([2**60], dtype=torch.int64)
    assert compare(a, a.clone(), max_ulps=1000)
    assert not compare(a, a + 1, max_ulps=1000)
    assert not compare(torch.tensor([True]), torch.tensor([False]), max_ulps=1000)
    for value in (-1, True, 0.5):
        with pytest.raises(ValueError, match="max_ulps"):
            compare(a, a, max_ulps=value)


def test_segment_pool_same_key_ragged_case_cannot_be_hidden_by_mean():
    from unittest.mock import patch

    import torch

    args = (torch.zeros(4, 3), torch.tensor([0, 1, 4]))
    other = (torch.zeros(4, 3), torch.tensor([0, 3, 4]))
    assert ab._input_key(args) == ab._input_key(other)
    for capabilities, rejected in (((), False), (("rms_norm",), False),
                                   (("segment_pool",), True), (("rms_norm", "segment_pool"), True)):
        with _fake_compile():
            with patch.object(ab, "_ab_time", side_effect=[(100, 10), (1, 2)]):
                choice = ab.compile_best(object(), args, verification_inputs=[other],
                                         capabilities=capabilities, verbose=False)
        assert choice.used_fusion is not rejected
        assert choice.sample_metrics[0]["input_key"] == choice.sample_metrics[1]["input_key"]
        assert [m["case_index"] for m in choice.sample_metrics] == [0, 1]
        if rejected:
            assert "critical per-case regression (case 1)" in choice.reason
            assert "per-case regression" in choice.sample_metrics[1]["reason"]
            assert not choice.candidate_metrics[0]["eligible"]
            assert choice.package_path == choice.baseline_package
    for timings, accepted in (([(100, 10), (1, 1.03)], True),
                              ([(100, 10), (1, 1.031)], False),
                              ([(1, 2), (100, 10)], False)):
        with _fake_compile():
            with patch.object(ab, "_ab_time", side_effect=timings):
                choice = ab.compile_best(object(), args, verification_inputs=[other],
                                         capabilities=("segment_pool",), verbose=False)
        assert choice.used_fusion is accepted


def test_structured_multiplan_unsupported_split_is_local_and_state_is_isolated():
    from unittest.mock import patch

    import torch

    class BufferedLinear(torch.nn.Linear):
        def __init__(self):
            super().__init__(3, 2)
            self.register_buffer("offset", torch.ones(2))

        def forward(self, x):
            return super().forward(x) + self.offset

    for domain in ("rms_norm", "segment_pool"):
        model = BufferedLinear().eval()
        args = (torch.ones(4, 3),)
        samples = ((torch.zeros(4, 3),),)
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        assert "offset" in weights
        seen = []
        real_export_pair = ab._export_pair

        def apply(candidate, *, verify_with, verification_inputs, capabilities, plan):
            assert capabilities == (domain,) and verification_inputs is samples
            assert verify_with[0] is model
            for name, value in candidate.state_dict.items():
                assert torch.equal(value, weights[name])
                assert value.data_ptr() != model.state_dict()[name].data_ptr()
                if seen:
                    assert value.data_ptr() != seen[0].state_dict[name].data_ptr()
            seen.append(candidate)
            candidate.graph_module.meta["candidate_only"] = plan
            with torch.no_grad():
                candidate.state_dict["weight"].add_(100)
                candidate.state_dict["offset"].add_(100)
            if plan == "split":
                raise RuntimeError(f"{domain} split domain unsupported")
            return _report((domain,))

        with _fake_compile() as calls:
            with patch.object(ab, "_export_pair", side_effect=real_export_pair) as export, \
                 patch.object(neptune_opt, "apply", side_effect=apply):
                choice = ab.compile_best(model, args, verification_inputs=samples,
                                         capabilities=(domain,), plans=("serial", "split"), verbose=False)
        assert choice.selected_plan == "serial"
        assert calls["package"] == ["baseline", "fused_serial"]
        assert len(seen) == 2 and export.call_count == 2
        baseline = export.call_args.kwargs["baseline"]
        assert "candidate_only" not in baseline.graph_module.meta
        for name, value in weights.items():
            assert torch.equal(model.state_dict()[name], value)
            assert torch.equal(baseline.state_dict[name], value)
        assert f"{domain} split domain unsupported" in choice.candidate_metrics[1]["reason"]
        assert not choice.candidate_metrics[1]["eligible"]
        assert export.call_args.kwargs["verification_inputs"] is samples


def test_segment_pool_per_case_rejection_does_not_poison_next_plan():
    from unittest.mock import patch

    import torch

    args = (torch.zeros(2),)
    with _fake_compile() as calls:
        with patch.object(ab, "_export_pair", return_value=(object(), object(), _report(("segment_pool",)))), \
             patch.object(ab, "_ab_time", side_effect=[(100, 10), (1, 2), (100, 20), (1, 0.5)]):
            choice = ab.compile_best(object(), args, verification_inputs=[args],
                                     capabilities=("segment_pool",), plans=("serial", "split"), verbose=False)
    assert choice.selected_plan == "split" and calls["package"].count("baseline") == 1
    assert "per-case regression (case 1)" in choice.candidate_metrics[0]["reason"]
    assert [m["eligible"] for m in choice.candidate_metrics] == [False, True]


def test_structured_noop_and_missing_report_do_not_force_candidate_compile():
    from unittest.mock import patch

    for report in (None, _report((), verified=False)):
        with _fake_compile() as calls:
            with patch.object(ab, "_export_pair", return_value=(object(), object(), report)):
                choice = ab.compile_best(object(), (), capabilities=("rms_norm", "segment_pool"),
                                         plans=("serial", "split"), verbose=False)
        assert not choice.used_fusion and calls["package"] == ["baseline"]
        assert calls["timed"] == 0 and "nothing to fuse" in choice.reason


def test_real_cpu_structured_exports_capture_optin_without_generating_kernels():
    from unittest.mock import patch

    import torch

    class RMS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3))

        def forward(self, x):
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * self.weight

    class Segment(torch.nn.Module):
        def forward(self, x, lengths):
            return torch.segment_reduce(x, reduce="mean", lengths=lengths)

    models = (("rms_norm", RMS().eval(), (torch.randn(2, 3),),
               ((torch.zeros(2, 3),), (torch.ones(2, 3),))),
              ("segment_pool", Segment().eval(), (torch.randn(4, 3), torch.tensor([1, 3])),
               ((torch.randn(4, 3), torch.tensor([3, 1])),)))
    for domain, model, args, samples in models:
        with patch.object(neptune_opt, "apply", return_value=_report(())) as apply:
            baseline, candidate, report = ab._export_pair(
                model, args, {}, capabilities=(domain,), verification_inputs=samples,
                rtol=0.01, atol=0.001, max_ulps=1)
        assert baseline is not candidate and baseline.graph_module is not candidate.graph_module
        assert apply.call_args.kwargs["capabilities"] == (domain,)
        assert apply.call_args.kwargs["verification_inputs"] is samples
        assert not report.fusions
        for case in (args, *samples):
            assert ab._same_numbers(model, baseline.module(), case)[0]
            assert ab._same_numbers(baseline.module(), candidate.module(), case)[0]


def test_optin_fresh_export_fallback_and_failed_apply_preserve_source_weights():
    from unittest.mock import patch

    import torch

    model = torch.nn.Linear(3, 2).eval()
    args = (torch.ones(4, 3),)
    samples = ((torch.zeros(4, 3),),)
    baseline = torch.export.export(model, args)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    code = baseline.graph_module.code
    real_deepcopy = ab.copy.deepcopy

    def deepcopy(value):
        if value is baseline:
            raise RuntimeError("copy unsupported")
        return real_deepcopy(value)

    def apply(candidate, **kwargs):
        assert kwargs["capabilities"] == ("segment_pool",)
        assert kwargs["verification_inputs"] is samples
        assert kwargs["plan"] == "split" and kwargs["max_ulps"] == 0
        with torch.no_grad():
            for name, value in candidate.state_dict.items():
                assert value.data_ptr() != baseline.state_dict[name].data_ptr()
                assert value.data_ptr() != model.state_dict()[name].data_ptr()
                value.add_(100)
        raise RuntimeError("unsupported split")

    with patch.object(ab.copy, "deepcopy", side_effect=deepcopy), \
         patch.object(neptune_opt, "apply", side_effect=apply):
        original, candidate, report = ab._export_pair(
            model, args, {}, baseline=baseline, plan="split", capabilities=("segment_pool",),
            verification_inputs=samples, max_ulps=0)
    assert original is baseline and candidate is None and "unsupported split" in report.summary()
    assert baseline.graph_module.code == code
    for name, value in weights.items():
        assert torch.equal(model.state_dict()[name], value)
        assert torch.equal(baseline.state_dict[name], value)


@pytest.mark.parametrize("message", (
    "CUDA error: device-side assert triggered",
    "CUDAERRORASSERT",
    "CUDA error: an illegal memory access was encountered",
    "cudaErrorIllegalAddress",
    "CUDA ERROR: UNSPECIFIED LAUNCH FAILURE",
    "cudaErrorLaunchFailure",
    "CUDA context is destroyed",
    "context is destroyed",
))
def test_fatal_cuda_helper_detects_tokens_without_gpu_queries(monkeypatch, message):
    from unittest.mock import Mock

    import torch

    forbidden = Mock(side_effect=AssertionError("fatal helper must not touch CUDA"))
    for name in ("synchronize", "is_available", "is_initialized", "current_device", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    original = RuntimeError(message)
    with pytest.raises(ab.FatalCudaError) as caught:
        ab._raise_if_fatal_cuda(original)
    assert caught.value.__cause__ is original
    assert message in str(caught.value)
    assert "baseline" in str(caught.value).lower() and "unsafe" in str(caught.value).lower()
    forbidden.assert_not_called()


@pytest.mark.parametrize("link", ("__cause__", "__context__"))
def test_fatal_cuda_helper_follows_nested_errors_and_preserves_original_cause(monkeypatch, link):
    from unittest.mock import Mock

    import torch

    synchronize = Mock(side_effect=AssertionError("must not synchronize a broken context"))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    original = RuntimeError("CUDA error: device-side assert triggered")
    inner = RuntimeError("wrapped kernel compile failure")
    outer = RuntimeError("wrapped candidate failure")
    setattr(inner, link, original)
    setattr(outer, link, inner)
    with pytest.raises(ab.FatalCudaError) as caught:
        ab._raise_if_fatal_cuda(outer)
    assert caught.value.__cause__ is outer
    assert getattr(getattr(caught.value.__cause__, link), link) is original
    assert str(original) in str(caught.value)
    synchronize.assert_not_called()


def test_fatal_cuda_helper_ordinary_errors_oom_and_cycles_are_recoverable(monkeypatch):
    from unittest.mock import Mock

    import torch

    forbidden = Mock(side_effect=AssertionError("helper must not query or synchronize CUDA"))
    for name in ("synchronize", "is_available", "is_initialized", "current_device", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for error in (RuntimeError("ordinary compile failure"), AssertionError("CPU assertion failed"),
                  RuntimeError("CUDA error: out of memory"), RuntimeError("generic CUDA failure"),
                  torch.cuda.OutOfMemoryError("CUDA out of memory")):
        assert not ab._raise_if_fatal_cuda(error)
    first, second = RuntimeError("first"), AssertionError("second")
    first.__cause__ = second
    second.__context__ = first
    assert not ab._raise_if_fatal_cuda(first)
    second.__cause__ = RuntimeError("CUDA error: device-side assert triggered")
    with pytest.raises(ab.FatalCudaError) as caught:
        ab._raise_if_fatal_cuda(first)
    assert caught.value.__cause__ is first
    forbidden.assert_not_called()


def test_fatal_cuda_helper_reraises_existing_fatal_error_identity():
    fatal = ab.FatalCudaError("already classified context failure")
    wrapper = RuntimeError("outer wrapper")
    wrapper.__context__ = fatal
    for error in (fatal, wrapper):
        with pytest.raises(ab.FatalCudaError) as caught:
            ab._raise_if_fatal_cuda(error)
        assert caught.value is fatal


@pytest.mark.parametrize("failed_side", ("baseline", "candidate"))
def test_fatal_cuda_same_numbers_execution_propagates_instead_of_false(monkeypatch, failed_side):
    from unittest.mock import Mock

    import torch

    synchronize = Mock(side_effect=AssertionError("must not synchronize"))
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    original = RuntimeError("CUDA error: device-side assert triggered")
    failure = RuntimeError("execution wrapper")
    failure.__cause__ = original
    value = torch.ones(2)
    good = Mock(return_value=value)
    fail = Mock(side_effect=failure)
    fa, fb = (fail, good) if failed_side == "baseline" else (good, fail)
    with pytest.raises(ab.FatalCudaError) as caught:
        ab._same_numbers(fa, fb, ())
    assert caught.value.__cause__ is failure
    assert caught.value.__cause__.__cause__ is original
    assert fail.call_count == 1
    assert good.call_count == (0 if failed_side == "baseline" else 1)
    synchronize.assert_not_called()


@pytest.mark.parametrize("stage", ("copy", "reexport", "module_copy", "apply"))
def test_fatal_cuda_export_pair_catches_never_refuse_or_retry(monkeypatch, stage):
    from unittest.mock import Mock

    import torch

    baseline = types.SimpleNamespace(graph_module=object(), state_dict={})
    candidate = types.SimpleNamespace(graph_module=object(), state_dict={})
    model = object()
    original = RuntimeError("CUDA error: device-side assert triggered")
    failure = RuntimeError("candidate export wrapper")
    # Re-export happens inside a normal copy-error handler; Python replaces
    # implicit __context__ there, so retain the injected CUDA error as __cause__.
    failure.__cause__ = original
    copy_calls = []

    def deepcopy(value):
        copy_calls.append(value)
        if value is baseline:
            if stage == "copy":
                raise failure
            if stage in ("reexport", "module_copy"):
                raise RuntimeError("ordinary unsupported copy")
            return candidate
        assert value is model
        if stage == "module_copy":
            raise failure
        return model

    export = Mock(side_effect=failure)
    apply = Mock(side_effect=failure)
    synchronize = Mock(side_effect=AssertionError("must not synchronize"))
    monkeypatch.setattr(ab.copy, "deepcopy", deepcopy)
    monkeypatch.setattr(torch.export, "export", export)
    monkeypatch.setattr(neptune_opt, "apply", apply)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    with pytest.raises(ab.FatalCudaError) as caught:
        ab._export_pair(model, (), {}, baseline=baseline)
    assert caught.value.__cause__ is failure
    assert caught.value.__cause__.__cause__ is original
    assert copy_calls == ([baseline, model] if stage in ("reexport", "module_copy") else [baseline])
    assert export.call_count == (1 if stage == "reexport" else 0)
    assert apply.call_count == (1 if stage == "apply" else 0)
    synchronize.assert_not_called()


@pytest.mark.parametrize("stage", (
    "export_pair", "fused_package", "baseline_load", "candidate_load",
    "baseline_numbers", "candidate_numbers", "ab_time", "subsequent_export",
))
def test_fatal_cuda_pipeline_stops_without_choice_or_next_plan(monkeypatch, tmp_path, stage):
    from unittest.mock import Mock, patch

    import torch

    original = RuntimeError("CUDA error: device-side assert triggered")
    failure = RuntimeError("pipeline stage wrapper")
    failure.__cause__ = original
    baseline_ep = types.SimpleNamespace(graph_module=object(), state_dict={})
    candidate_ep = object()
    real_export_pair = ab._export_pair
    synchronize = Mock(side_effect=AssertionError("broken CUDA context must not be synchronized"))
    apply = Mock(side_effect=failure)
    exported = []
    loaded = []
    checked = []

    def export(module, args, kwargs, plan=None, *, baseline=None):
        exported.append((plan, baseline))
        if stage == "export_pair":
            # Exercise the real catch around apply, without real tracing or GPU work.
            return real_export_pair(module, args, kwargs, plan=plan, baseline=baseline_ep)
        if stage == "subsequent_export" and plan == "split":
            raise failure
        return baseline_ep, candidate_ep, _report()

    with _fake_compile() as calls:
        fake_package = ab._package
        fake_timing = ab._ab_time

        def package(ep, directory, name):
            path = fake_package(ep, directory, name)
            if stage == "fused_package" and name == "fused_serial":
                raise failure
            return path

        def load(path):
            loaded.append(path)
            if ((stage == "baseline_load" and path.endswith("baseline.pt2")) or
                    (stage == "candidate_load" and path.endswith("fused_serial.pt2"))):
                raise failure
            return path

        def check(fa, fb, args):
            checked.append(fb)
            if ((stage == "baseline_numbers" and fb.endswith("baseline.pt2")) or
                    (stage == "candidate_numbers" and fb.endswith("fused_serial.pt2"))):
                raise failure
            return True, ""

        def timing(*args, **kwargs):
            result = fake_timing(*args, **kwargs)
            if stage == "ab_time":
                raise failure
            return result

        with patch.object(ab, "_export_pair", side_effect=export) as export_spy, \
             patch.object(ab, "_package", side_effect=package) as package_spy, \
             patch.object(ab, "_load", side_effect=load), \
             patch.object(ab, "_same_numbers", side_effect=check), \
             patch.object(ab, "_ab_time", side_effect=timing) as timing_spy, \
             patch.object(neptune_opt, "apply", apply), \
             patch.object(torch.cuda, "synchronize", synchronize):
            with pytest.raises(ab.FatalCudaError) as caught:
                ab.compile_best(object(), (), package_dir=str(tmp_path),
                                plans=("serial", "split"), verbose=False)
        assert caught.value.__cause__ is failure
        assert caught.value.__cause__.__cause__ is original
        assert str(original) in str(caught.value)
        assert "baseline" in str(caught.value).lower() and "unsafe" in str(caught.value).lower()
        expected_exports = [("serial", None)]
        if stage == "subsequent_export":
            expected_exports.append(("split", baseline_ep))
        assert exported == expected_exports
        assert export_spy.call_count == len(expected_exports)
        expected_packages = ([] if stage == "export_pair" else ["baseline"]
                             if stage in ("baseline_load", "baseline_numbers")
                             else ["baseline", "fused_serial"])
        assert calls["package"] == expected_packages
        assert package_spy.call_count == len(expected_packages)
        assert calls["package"].count("baseline") == (0 if stage == "export_pair" else 1)
        expected_timing = 1 if stage in ("ab_time", "subsequent_export") else 0
        assert calls["timed"] == timing_spy.call_count == expected_timing
        assert len(loaded) == (0 if stage == "export_pair" else 1
                               if stage in ("baseline_load", "baseline_numbers", "fused_package") else 2)
        assert len(checked) == (0 if stage in ("export_pair", "baseline_load") else 1
                                if stage in ("baseline_numbers", "fused_package", "candidate_load") else 2)
        assert apply.call_count == (1 if stage == "export_pair" else 0)
        synchronize.assert_not_called()


@pytest.mark.parametrize("stage", ["export", "package"])
def test_fatal_cuda_baseline_boundaries_use_marker_without_fallback(monkeypatch, tmp_path, stage):
    from unittest.mock import Mock

    import torch
    from torch._inductor import config

    original = RuntimeError("CUDA error: device-side assert triggered")
    failure = RuntimeError("baseline build wrapper")
    failure.__cause__ = original
    available = hasattr(config.triton, "autotune_at_compile_time")
    previous = config.triton.autotune_at_compile_time if available else None
    unsafe = Mock(side_effect=AssertionError("baseline failure must not continue"))
    if stage == "export":
        monkeypatch.setattr(torch.export, "export", Mock(side_effect=failure))
        monkeypatch.setattr(ab.copy, "deepcopy", unsafe)
        monkeypatch.setattr(neptune_opt, "apply", unsafe)
    else:
        monkeypatch.setattr(torch._inductor, "aoti_compile_and_package", Mock(side_effect=failure), raising=False)
    with pytest.raises(ab.FatalCudaError) as caught:
        if stage == "export":
            ab._export_pair(object(), (), {})
        else:
            ab._package(object(), str(tmp_path), "baseline")
    assert caught.value.__cause__ is failure
    assert "baseline" in str(caught.value) and "unsafe" in str(caught.value)
    unsafe.assert_not_called()
    if available:
        assert config.triton.autotune_at_compile_time is previous


def test_fatal_cuda_changes_do_not_mask_raw_normal_baseline_export_error(tmp_path):
    from unittest.mock import Mock, patch

    import torch

    real_export_pair = ab._export_pair
    failure = RuntimeError("ordinary baseline export failure")
    export = Mock(side_effect=failure)
    package = Mock(side_effect=AssertionError("must not package a failed baseline"))
    synchronize = Mock(side_effect=AssertionError("must not synchronize"))
    with _fake_compile() as calls:
        with patch.object(ab, "_export_pair", real_export_pair), \
             patch.object(torch.export, "export", export), \
             patch.object(ab, "_package", package), \
             patch.object(torch.cuda, "synchronize", synchronize):
            with pytest.raises(RuntimeError) as caught:
                ab.compile_best(object(), (), package_dir=str(tmp_path), verbose=False)
        assert caught.value is failure and not isinstance(caught.value, ab.FatalCudaError)
        assert export.call_count == 1
        package.assert_not_called()
        synchronize.assert_not_called()
        assert calls == {"package": [], "timed": 0}


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
