"""
Tests for the launch-configuration chooser (`neptune_codegen/autotune.py`).

The chooser decides a number that ends up baked into a compiled artifact, so the parts that must
hold are: the answer is always a configuration the search could have produced, an unmeasured shape
falls back to the documented default rather than to a guess, and asking does not drag Triton (or
even torch) into a process.

Run:  python tests/test_autotune.py     (or pytest tests/test_autotune.py)
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import types
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neptune_codegen import autotune  # noqa: E402


@contextlib.contextmanager
def fake_registry(**attrs):
    """Swap `neptune_codegen.launch_table` for a stand-in module, and put everything back.

    `from package import submodule` binds an ATTRIBUTE on the package as well as an entry in
    `sys.modules`, so injecting into `sys.modules` alone leaks into the next test that imports the
    data file -- which is exactly how this helper came to exist.
    """
    import neptune_codegen

    module = types.ModuleType("neptune_codegen.launch_table")
    for key, value in attrs.items():
        setattr(module, key, value)
    saved_module = sys.modules.get("neptune_codegen.launch_table")
    saved_attr = getattr(neptune_codegen, "launch_table", None)
    sys.modules["neptune_codegen.launch_table"] = module
    neptune_codegen.launch_table = module
    try:
        yield module
    finally:
        if saved_module is None:
            sys.modules.pop("neptune_codegen.launch_table", None)
        else:
            sys.modules["neptune_codegen.launch_table"] = saved_module
        if saved_attr is None:
            if getattr(neptune_codegen, "launch_table", None) is module:
                delattr(neptune_codegen, "launch_table")
        else:
            neptune_codegen.launch_table = saved_attr


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


def test_machine_key_is_normalised():
    assert autotune.machine_key("NVIDIA GeForce RTX 5090 D v2") == "rtx-5090-d-v2"
    assert autotune.machine_key("NVIDIA GeForce RTX 4090 D") == "rtx-4090-d"
    assert autotune.machine_key("Tesla V100-PCIE-32GB") == "v100-pcie-32gb"
    # the same card spelled differently must land on the same key, or its measurements split in two
    assert autotune.machine_key("RTX 4090 D") == autotune.machine_key("NVIDIA GeForce RTX 4090 D")
    assert autotune.machine_key("") == ""


def test_a_machine_without_an_entry_does_not_borrow_another_machines():
    """The table does NOT transfer between GPUs, so an unmeasured one must get the DEFAULT.

    Measured: the same search on the same stack chose four deviations on one card and one on
    another. Copying them would encode someone else's hardware as a decision, so this is asserted
    rather than left to the accident of a shared dictionary.
    """
    registry = {"rtx-5090-d-v2": {"measured_on": "x", "noise": 0.05,
                                  "table": {("single", 262144): (8192, 8)}}}
    with fake_registry(TABLES=registry):
        assert autotune.best_config(1, 262144, device_name="NVIDIA GeForce RTX 5090 D v2") == (8192, 8)
        assert autotune.best_config(1, 262144, device_name="Tesla V100-PCIE-32GB") == autotune.DEFAULT
        assert autotune.best_config(1, 262144) == autotune.DEFAULT          # no machine at all
        assert autotune.machines() == ["rtx-5090-d-v2"]


def test_merging_one_machine_keeps_the_others():
    """Adding a GPU is the common case now, so it must not erase the machine measured before it."""
    existing = autotune.tables()
    merged = autotune.merge_table(existing, "some-new-gpu", "a stack", {("single", 4096): (4096, 4)})
    assert set(existing) <= set(merged), "merging dropped a machine"
    assert merged["some-new-gpu"]["table"] == {("single", 4096): (4096, 4)}
    for m in existing:
        assert merged[m]["table"] == existing[m]["table"], f"{m} was changed by another machine's merge"
    # and the merge validates: a configuration nobody measured cannot enter the registry
    bad = autotune.merge_table(existing, "g", "x", {("single", 4096): (12345, 77)})
    assert bad["g"]["table"] == {}
    # the rendered file round-trips through exec, which is what the loader does
    namespace = {}
    exec(autotune.render_tables(merged), namespace)                          # noqa: S102
    assert namespace["TABLES"]["some-new-gpu"]["table"] == {("single", 4096): (4096, 4)}


def test_every_table_entry_is_a_candidate():
    for machine, entry in autotune.tables().items():
        for key, cfg in entry["table"].items():
            assert tuple(cfg) in autotune.CANDIDATES, f"{machine}/{key} -> {cfg} is not a candidate"
            assert isinstance(key, tuple) and len(key) == 2, key
            assert key[0] in ("single", "few", "many"), key
            assert key[1] & (key[1] - 1) == 0, key
        assert entry.get("measured_on"), f"{machine} has no provenance string"


def test_a_bogus_table_entry_is_dropped():
    """A hand-edited table must not be able to produce a launch the search never tried."""
    with fake_registry(TABLE={("many", 4096): (4096, 4), ("many", 1024): (12345, 77)}):
        kept = autotune.table()
        assert ("many", 4096) in kept, "a valid entry was dropped"
        assert ("many", 1024) not in kept, "an invalid entry survived"
        assert autotune.best_config(256, 1024) in autotune.CANDIDATES
        # a file in the pre-registry format has no machine attached, so it applies rather than
        # vanishing -- a FORMAT fallback, which the registry format itself must not have
        assert autotune.best_config(256, 4096) == (4096, 4)


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

    # Only execution is stubbed: validate the real shape/dtype contract, bypassing the
    # CUDA-only device requirement for this CPU launch recorder.
    validate = T._validate_row_last
    with mock.patch.object(T, "compile_skeleton", fake_compile), mock.patch.object(
        T, "_validate_row_last", side_effect=lambda x, **kw: validate(x, metadata=True)
    ):
        T.run("logsumexp", torch.randn(4, 8))
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
    saved_ops = dict(TO._TRITON_OPS)

    def fake_triton_op(name, **kwargs):
        def decorate(fn):
            registered["name"] = name
            return fn
        return decorate

    torch.library.triton_op = fake_triton_op
    TO.compile_skeleton = lambda name, params=None: object()
    try:
        TO.as_triton_op("logsumexp", block=8192, num_warps=8, lib="neptune",
                        params={"tiny": 0.0}, keepdim=True)
    finally:
        torch.library.triton_op = real_triton_op
        TO.compile_skeleton = real_compile
        TO._TRITON_OPS.clear()
        TO._TRITON_OPS.update(saved_ops)
    assert registered["name"] == "neptune::logsumexp_keep_tiny0_b8192w8", registered
    assert registered["name"] == autotune.op_key("neptune", "logsumexp", {"tiny": 0.0}, True, 8192, 8)


def test_exact_scalar_keys_paths_and_compile_cache():
    import neptune_codegen.template as T

    values = [1.0, 1.0000001, 1.0000002, -0.0, 0.0, 1e-8, 1e8]
    keys = [autotune.op_key("neptune", "softcap_lse", {"beta": v}, False) for v in values]
    paths = [T.generated_path("softcap_lse", {"beta": v}) for v in values]
    assert len(set(keys)) == len(values)
    assert len(set(paths)) == len(values)
    assert autotune.op_key("neptune", "logsumexp", {"T": 2.0}, True) == "neptune::logsumexp_keep_T2"
    assert all(key.split("::")[1].isidentifier() for key in keys)
    a = {"alpha": 1.0000001, "beta": 2.0}
    b = dict(reversed(list(a.items())))
    assert autotune.parameter_identity(a) == autotune.parameter_identity(b)
    assert autotune.parameter_tag(a) == autotune.parameter_tag(b)
    assert autotune.parameter_tag({"a": 1, "b": 2}) != autotune.parameter_tag({"a1_b": 2})
    assert autotune.parameter_tag({"a1": 2}) != autotune.parameter_tag({"a": 12})

    # Exercise the real cache and importer without importing or executing Triton.
    counter = []

    def load(mod):
        mod.softcap_lse = object()
        counter.append(mod.softcap_lse)

    spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=load))
    saved = dict(T._CACHE)
    module_names = ["neptune_gen_softcap_lse" + autotune.parameter_tag({"beta": v, "tiny": 0.0})
                    for v in (1.0000001, 1.0000002)]
    modules = {name: sys.modules.get(name) for name in module_names}
    try:
        T._CACHE.clear()
        with mock.patch("builtins.open", mock.mock_open()), mock.patch.object(
            T.importlib.util, "spec_from_file_location", return_value=spec
        ), mock.patch.object(T.importlib.util, "module_from_spec", side_effect=lambda _: types.ModuleType("stub")):
            first = T.compile_skeleton("softcap_lse", {"beta": 1.0000001, "tiny": 0.0})
            second = T.compile_skeleton("softcap_lse", {"beta": 1.0000002, "tiny": 0.0})
            assert first is not second
            assert T.compile_skeleton("softcap_lse", {"beta": 1.0000001, "tiny": 0.0}) is first
            assert T.compile_skeleton("softcap_lse", {"beta": 1.0000001, "tiny": 0.0}, force=True) is not first
            assert len(counter) == 3
    finally:
        T._CACHE.clear()
        T._CACHE.update(saved)
        for name, module in modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_symbolic_config_never_concretises_even_hinted_rows():
    import torch
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    class Hinted:
        def __int__(self):
            raise AssertionError("must not specialise hinted symbolic shapes")

    registry = {"test-gpu": {"table": {("many", 4096): (4096, 4)}}}
    with fake_registry(TABLES=registry):
        assert autotune.best_config(Hinted(), 4096, device_name="test-gpu") == autotune.DEFAULT
        assert autotune.best_config(32, Hinted(), device_name="test-gpu") == autotune.DEFAULT
        env = ShapeEnv()
        symbolic = env.create_unbacked_symint()
        assert isinstance(symbolic, torch.SymInt)
        assert autotune.best_config(symbolic, 4096, device_name="test-gpu") == autotune.DEFAULT
        assert autotune.rows_class(symbolic) == "many"
        from torch._subclasses.fake_tensor import FakeTensorMode
        from neptune_codegen.template import _row_last_layout
        with FakeTensorMode(shape_env=env):
            x = torch.empty(symbolic, 3, 4096)
            flat, rows, width = _row_last_layout(x)
            assert isinstance(rows, torch.SymInt)
            assert flat.shape == (symbolic * 3, 4096)
            assert width == 4096
            assert autotune.best_config(rows, width, device_name="test-gpu") == autotune.DEFAULT


def test_cpu_config_never_queries_cuda_and_device_selection_is_explicit():
    import torch

    with mock.patch.object(torch.cuda, "get_device_name", side_effect=AssertionError("no CUDA query")):
        assert autotune.best_config(1, 4096, device="cpu") == autotune.DEFAULT
        assert autotune.best_config(1, 4096) == autotune.DEFAULT
        assert autotune.device_name_of(torch.device("cpu")) is None
    with mock.patch.object(torch.cuda, "get_device_name", return_value="Test GPU") as query:
        assert autotune.device_name_of(torch.device("cuda:3")) == "Test GPU"
        query.assert_called_with(torch.device("cuda:3"))
        assert autotune.device_name_of(torch.device("cuda")) == "Test GPU"
        query.assert_called_with(torch.device("cuda"))


def test_all_template_loads_use_column_stride_and_source_is_deterministic():
    import ast
    import neptune_codegen.template as T
    from neptune_codegen.skeleton import SKELETONS, param_defaults, supported

    plans = set()
    for name in supported():
        sk = SKELETONS[name]
        params = param_defaults(name)
        src = T.emit_kernel_source(sk, params)
        assert src == T.emit_kernel_source(sk, dict(reversed(list(params.items()))))
        tree = ast.parse(src)
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        assert [arg.arg for arg in fn.args.args][-2:] == ["stride_col", "BLOCK"]
        loads = [node for node in ast.walk(fn) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "load"]
        assert len(loads) == (2 if sk.two_pass else 1)
        assert all("idx * stride_col" in ast.unparse(load.args[0]) for load in loads)
        assert "tl.exp(tl.maximum(tl.minimum(" not in src
        plans.add(sk.two_pass)
    assert plans == {False, True}


def test_true_exp_preserves_underflow_and_positive_overflow():
    import numpy as np
    import sympy as sp
    from neptune_codegen.sym_to_triton import to_triton

    symbol = sp.Symbol("x", real=True)
    source = to_triton(sp.exp(symbol), {symbol: "x"})
    assert source == "tl.exp(x)"
    x = np.array([-100.0, 85.0, 100.0], dtype=np.float32)
    with np.errstate(over="ignore", under="ignore"):
        actual = eval(source, {"tl": types.SimpleNamespace(exp=np.exp), "x": x})
        expected = np.exp(x)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] < np.exp(np.float32(-80))
    assert actual[1] > np.exp(np.float32(80))
    assert np.isposinf(actual[2])
    # Repair must compose even when shifted logits are below -80. Hidden clipping
    # destroys this identity: clipped exp(-90) * exp(-10) != clipped exp(-100).
    c, old, fresh = sp.symbols("c old fresh", real=True)
    term = to_triton(sp.exp(c - old), {c: "c", old: "old"})
    repair = to_triton(sp.exp(old - fresh), {old: "old", fresh: "fresh"})
    target = to_triton(sp.exp(c - fresh), {c: "c", fresh: "fresh"})
    bindings = {"tl": types.SimpleNamespace(exp=np.exp), "c": -90.0, "old": 0.0, "fresh": 10.0}
    np.testing.assert_allclose(eval(term, bindings) * eval(repair, bindings),
                               eval(target, bindings), rtol=1e-14, atol=0)


def test_row_last_runtime_shapes_strides_copies_and_zero_rows():
    import torch
    import neptune_codegen.template as T
    import neptune_codegen.torch_op as TO

    class Recorder:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def launch(x, out, rows, width, stride_row, stride_col, **kwargs):
                assert grid == (rows,)
                assert x.shape == (rows, width)
                assert x.stride() == (stride_row, stride_col)
                self.calls.append(x)
                out.copy_(torch.logsumexp(x, dim=-1).reshape(out.shape))
            return launch

    recorder = Recorder()
    validate = T._validate_row_last
    # All CPU tensors still run real validation for rank/layout/dtype/width; only device
    # execution is substituted, and the recorder checks every argument of the real launch.
    # These bridge APIs are absent in older Torch: install CPU recorder substitutes
    # explicitly rather than skip any layout/stride assertions.
    with mock.patch.object(T, "compile_skeleton", return_value=recorder), mock.patch.object(
        TO, "compile_skeleton", return_value=recorder
    ), mock.patch.object(T, "_validate_row_last", side_effect=lambda x, **kw: validate(x, metadata=True)), mock.patch.object(
        TO, "_validate_row_last", side_effect=lambda x, **kw: validate(x, metadata=True)
    ), mock.patch.object(torch.library, "triton_op", side_effect=lambda *a, **kw: lambda fn: fn,
                         create=True), mock.patch.object(
        torch.library, "wrap_triton", side_effect=lambda kernel: kernel, create=True
    ), mock.patch.dict(TO._TRITON_OPS, {}, clear=True):
        base = torch.randn(4, 6, 8)
        inputs = [torch.randn(8), torch.randn(3, 8), torch.randn(2, 3, 8),
                  torch.randn(2, 3, 4, 8), torch.randn(3, 16)[:, ::2],
                  torch.randn(8, 3).t(), torch.randn(3, 1).expand(3, 8),
                  base[:, :, ::2], base.transpose(0, 1)]
        for keep in (False, True):
            op = TO.as_triton_op("logsumexp", lib="cpu_recorder", keepdim=keep)
            for x in inputs:
                expected = torch.logsumexp(x, dim=-1, keepdim=keep)
                for call in (lambda: T.run("logsumexp", x, keepdim=keep), lambda: op(x)):
                    actual = call()
                    assert actual.shape == expected.shape
                    assert actual.dtype == x.dtype
                    torch.testing.assert_close(actual, expected)
                    flattened = recorder.calls[-1]
                    if x.ndim <= 2 or x is inputs[-2]:
                        assert flattened.data_ptr() == x.data_ptr(), "unnecessary contiguous copy"
                    if x is inputs[-1]:
                        assert flattened.data_ptr() != x.data_ptr(), "non-viewable leading dims must copy"
            count = len(recorder.calls)
            for shape in ((0, 8), (2, 0, 8)):
                x = torch.empty(shape)
                assert T.run("logsumexp", x, keepdim=keep).shape == shape[:-1] + ((1,) if keep else ())
                assert op(x).shape == shape[:-1] + ((1,) if keep else ())
            assert len(recorder.calls) == count, "zero rows launched a kernel"


def test_runtime_rejects_unsupported_inputs_but_meta_infers_shapes():
    import torch
    import neptune_codegen.template as T

    cases = [(torch.empty(2, 8), ValueError, "device"),
             (torch.empty((), device="meta"), ValueError, "rank"),
             (torch.empty(2, 0, device="meta"), RuntimeError, "width"),
             (torch.empty(2, 8, device="meta", dtype=torch.float64), TypeError, "dtype"),
             (torch.empty(2, 8, device="meta", dtype=torch.int32), TypeError, "dtype")]
    for x, error, message in cases:
        try:
            T.run("logsumexp", x)
        except error as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"unsupported input accepted: {x.shape} {x.dtype} {x.device}")
    with mock.patch.object(T, "compile_skeleton", side_effect=AssertionError("metadata compiled Triton")):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for shape in ((8,), (2, 3, 8), (2, 0, 8)):
                for keep in (False, True):
                    out = T.run("logsumexp", torch.empty(shape, device="meta", dtype=dtype), keepdim=keep)
                    assert out.shape == shape[:-1] + ((1,) if keep else ())
                    assert out.dtype == dtype


def test_real_op_fake_and_dynamic_export_without_gpu_launch():
    import torch
    import neptune_codegen.torch_op as TO
    from torch._subclasses.fake_tensor import FakeTensorMode

    if not hasattr(torch.library, "triton_op"):
        return
    with mock.patch.object(TO, "compile_skeleton", return_value=object()):
        op = TO.as_triton_op("logsumexp", lib="neptune_cpu_metadata", keepdim=True)
        handle = TO.op_handle("logsumexp", lib="neptune_cpu_metadata", keepdim=True)
    assert handle is not None
    with FakeTensorMode():
        for shape in ((8,), (2, 3, 8), (0, 3, 8)):
            x = torch.empty(shape, dtype=torch.bfloat16)
            out = op(x)
            assert out.shape == shape[:-1] + (1,)
            assert out.dtype == torch.bfloat16

    class Model(torch.nn.Module):
        def forward(self, x):
            return op(x)

    exported = torch.export.export(Model(), (torch.empty(2, 3, 8),),
                                   dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=16)}})
    node = next(n for n in exported.graph.nodes if n.op == "call_function")
    assert node.meta["val"].shape[0] == next(n for n in exported.graph.nodes if n.op == "placeholder").meta["val"].shape[0]
    assert isinstance(node.meta["val"].shape[0], torch.SymInt)


def test_generated_directory_is_owned_and_released_at_exit():
    # Child process keeps this process's live Triton sources untouched. No helper file.
    result = subprocess.run([sys.executable, "-c",
        "import neptune_codegen.template as t; print(t._GEN_DIR, flush=True)"],
        check=True, capture_output=True, text=True)
    path = result.stdout.strip()
    assert path and not os.path.exists(path), "TemporaryDirectory leaked at interpreter exit"
    result = subprocess.run([sys.executable, "-c",
        "import neptune_codegen.template as t; t._release_generated_dir(); t._release_generated_dir(); "
        "\ntry: t.generated_path('logsumexp')"
        "\nexcept RuntimeError: pass"
        "\nelse: raise AssertionError('released owner accepted new source')"],
        check=True, capture_output=True, text=True)
    assert result.returncode == 0


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
