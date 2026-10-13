"""
Tests for the launch-configuration chooser (`neptune_codegen/autotune.py`).

The chooser decides a number that ends up baked into a compiled artifact, so the parts that must
hold are: the answer is always a configuration the search could have produced, an unmeasured shape
falls back to the documented default rather than to a guess, and asking does not drag Triton (or
even torch) into a process.

Run:  python tests/test_autotune.py     (or pytest tests/test_autotune.py)
"""
from __future__ import annotations

import os
import subprocess
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neptune_codegen import autotune  # noqa: E402


class _Symbolic:
    """Stands in for a SymInt that has no concrete value."""

    def __int__(self):
        raise RuntimeError("symbolic")


def test_buckets_are_powers_of_two_and_cover_the_row():
    for n in (1, 2, 255, 256, 257, 1000, 4096, 16384, 100000):
        b = autotune.ncols_bucket(n)
        assert b & (b - 1) == 0, f"{n} -> {b} is not a power of two"
        assert autotune.MIN_BUCKET <= b <= autotune.MAX_BUCKET, f"{n} -> {b} out of range"
        if n <= autotune.MAX_BUCKET:
            assert b >= n, f"bucket {b} does not cover a row of {n}"
    # monotone: a longer row never gets a smaller bucket
    buckets = [autotune.ncols_bucket(n) for n in range(1, 5000)]
    assert buckets == sorted(buckets)


def test_rows_class():
    # One row is its own class: measured, a single row wants a different block from four rows of the
    # same length, because a single program has to find all its parallelism inside the block.
    assert autotune.rows_class(1) == "single"
    assert autotune.rows_class(2) == "few"
    assert autotune.rows_class(autotune.THIN_ROWS) == "few"
    assert autotune.rows_class(autotune.THIN_ROWS + 1) == "many"
    assert autotune.rows_class(4096) == "many"
    # A symbolic row count is "many": a dynamic batch is not the single-row case the class is for.
    assert autotune.rows_class(_Symbolic()) == "many"


def test_a_choice_has_to_clear_the_noise_margin():
    """The policy the table encodes. Most candidate spreads are noise; a real win is not marginal."""
    # default is the best of a spread that is entirely inside the margin -> keep the default
    tight = [autotune.Result(*autotune.DEFAULT, 0.1000), autotune.Result(2048, 8, 0.0980)]
    cfg, gain, why = autotune.choose(tight)
    assert cfg == autotune.DEFAULT, f"noise was recorded as a tuning decision: {cfg} ({why})"
    assert 1.0 < gain <= 1.0 + autotune.NOISE + 1e-9, gain

    # a real win -> taken, and the default is still in the comparison
    wide = [autotune.Result(*autotune.DEFAULT, 0.1252), autotune.Result(8192, 8, 0.0412)]
    cfg, gain, why = autotune.choose(wide)
    assert cfg == (8192, 8), f"a 3x win was not taken: {cfg} ({why})"
    assert gain > 3.0, gain

    # the default missing from the measurements -> keep the default, do not invent a comparison
    cfg, gain, why = autotune.choose([autotune.Result(8192, 8, 0.01)])
    assert cfg == autotune.DEFAULT and "not measurable" in why, (cfg, why)


def test_op_key_carries_the_configuration():
    """The claim "the configuration is part of the op's identity", checked without a GPU."""
    a = autotune.op_key("neptune", "logsumexp", {}, True, 1024, 4)
    b = autotune.op_key("neptune", "logsumexp", {}, True, 8192, 8)
    assert a != b, "two configurations produced one op name: they would share one kernel"
    assert a == "neptune::logsumexp_keep_b1024w4", a
    assert autotune.op_key("neptune", "logsumexp", {"T": 2.0}, False) != a


def test_unmeasured_shape_returns_the_documented_default():
    # A bucket that no measurement could plausibly have covered: the answer must still be a real
    # candidate, and it must be the default rather than something invented.
    cfg = autotune.best_config(11, 3, verbose=False)
    assert cfg in autotune.CANDIDATES, cfg
    assert autotune.DEFAULT in autotune.CANDIDATES


def test_every_table_entry_is_a_candidate():
    for key, cfg in autotune.table().items():
        assert tuple(cfg) in autotune.CANDIDATES, f"{key} -> {cfg} is not a candidate"
        assert isinstance(key, tuple) and len(key) == 2, key
        assert key[0] in ("single", "few", "many"), key
        assert key[1] & (key[1] - 1) == 0, key


def test_a_bogus_table_entry_is_dropped():
    """A hand-edited table must not be able to produce a launch the search never tried."""
    module = types.ModuleType("neptune_codegen.launch_table")
    module.TABLE = {("many", 4096): (4096, 4), ("many", 1024): (12345, 77)}
    real = sys.modules.get("neptune_codegen.launch_table")
    sys.modules["neptune_codegen.launch_table"] = module
    try:
        kept = autotune.table()
    finally:
        if real is None:
            sys.modules.pop("neptune_codegen.launch_table", None)
        else:
            sys.modules["neptune_codegen.launch_table"] = real
    assert ("many", 4096) in kept, "a valid entry was dropped"
    assert ("many", 1024) not in kept, "an invalid entry survived"
    assert autotune.best_config(256, 1024) in autotune.CANDIDATES


def test_lookup_does_not_import_torch():
    """The chooser is consulted while a graph is being rewritten, and it must stay pure Python."""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from neptune_codegen import autotune;"
        "cfg = autotune.best_config(256, 16384);"
        "assert cfg in autotune.CANDIDATES, cfg;"
        "assert 'torch' not in sys.modules, 'best_config imported torch';"
        "assert 'triton' not in sys.modules, 'best_config imported triton';"
        "print('pure')" % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "pure" in out.stdout


def test_run_fills_the_skeletons_declared_defaults():
    """`CG.run(name, x)` must work without the caller reciting the skeleton's scalars.

    It did not: `run("logsumexp", x)` raised `KeyError: no value supplied for scalar parameter`,
    which is how a whole measurement run in `gpu_m22_autotune.py` produced `inf` timings and a
    table built from them. The recorder below calls the REAL emitter, so an unfilled parameter
    raises exactly as it did on the GPU.
    """
    import torch

    import neptune_codegen.template as T
    from neptune_codegen.skeleton import SKELETONS

    seen = {}

    class _Recorder:
        def __getitem__(self, grid):                     # the triton launch syntax
            def launcher(*args, **kwargs):
                seen["launched"] = (grid, args, kwargs)
            return launcher

    def fake_compile(name, params=None, force=False):
        T.emit_kernel_source(SKELETONS[name], params)    # raises if a scalar is missing
        seen["params"] = params
        return _Recorder()

    real = T.compile_skeleton
    T.compile_skeleton = fake_compile
    try:
        T.run("logsumexp", torch.randn(4, 8))
    finally:
        T.compile_skeleton = real
    assert "launched" in seen, "the kernel was never launched"
    assert seen["params"], "no parameters reached the generator"


def test_a_search_where_everything_failed_reports_no_winner():
    """The rule that would have caught the empty search: no measurement, no answer."""
    failed = [autotune.Result(256, 1, float("inf")), autotune.Result(1024, 4, float("inf"))]
    try:
        autotune.best_of(failed)
    except RuntimeError as e:
        assert "no candidate launch configuration" in str(e), str(e)
    else:
        raise AssertionError("a run in which every candidate failed produced a winner")
    good = [autotune.Result(1024, 4, 0.05), autotune.Result(256, 1, 0.01)]
    assert [(r.block, r.num_warps) for r in autotune.best_of(good)] == [(256, 1), (1024, 4)]


def test_op_registration_uses_the_shared_key():
    """The name a graph carries and the name the op registers under must be the same string.

    `triton_op` is stubbed so this runs on CPU; what is being tested is the name construction and
    the path around it, not Triton. A leftover local in this function once made EVERY registration
    raise `NameError`, which no CPU test noticed because they all stub `op_handle` -- so this test
    deliberately does not.
    """
    import torch

    import neptune_codegen.torch_op as TO

    if not hasattr(torch.library, "triton_op"):
        # torch 2.4.1 has no `triton_op`, so op REGISTRATION is unavailable there -- not just this
        # test. Verified by CI: the module-level `from torch.library import triton_op` is what fails,
        # which is exactly why the README's portability limit says the integration is verified on
        # one stack while the CPU suite runs on three.
        print("      (skipped: torch %s has no torch.library.triton_op)" % torch.__version__)
        return

    registered = {}
    real_triton_op = torch.library.triton_op
    real_compile = TO.compile_skeleton

    def fake_triton_op(name, **kwargs):
        def decorate(fn):
            registered["name"] = name
            return fn
        return decorate

    torch.library.triton_op = fake_triton_op
    TO.compile_skeleton = lambda name, params=None: object()
    try:
        TO.as_triton_op("logsumexp", block=8192, num_warps=8, lib="neptune",
                        params={"T": 2.0}, keepdim=True)
    finally:
        torch.library.triton_op = real_triton_op
        TO.compile_skeleton = real_compile
    assert registered["name"] == "neptune::logsumexp_keep_T2_b8192w8", registered
    assert registered["name"] == autotune.op_key("neptune", "logsumexp", {"T": 2.0}, True, 8192, 8)


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
