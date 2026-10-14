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


def _report(fusions=("logsumexp",)):
    return types.SimpleNamespace(fusions=list(fusions), skipped=[],
                                 summary=lambda: "  fused   : logsumexp")


@contextlib.contextmanager
def _fake_compile(fused_ms=0.02, base_ms=0.03, fusions=("logsumexp",), package_fails=None,
                  numbers_ok=True):
    """Replace export/package/load/time, and pretend the machine has a GPU."""
    import torch

    calls = {"package": [], "timed": 0}
    real = (ab._export_pair, ab._package, ab._load, ab._ab_time, ab._same_numbers,
            torch.cuda.is_available)

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
    try:
        yield calls
    finally:
        (ab._export_pair, ab._package, ab._load, ab._ab_time, ab._same_numbers,
         torch.cuda.is_available) = real


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
