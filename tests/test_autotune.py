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


# Segment codegen/bridge coverage (kept separate from the existing launch-policy tests).
def test_segment_source_is_serial_storage_order_and_memory_guarded():
    import ast
    import neptune_codegen.torch_op as TO

    for name in ("segment_sum", "segment_mean"):
        for initial in (None, 0.0, -0.0, -2.125, 1e300):
            source = TO.emit_segment_source(name, initial=initial, offsets=True)
            ast.parse(source)
            assert source == TO.emit_segment_source(name, initial=initial, offsets=True)
            assert "for token in range(start, end)" in source
            assert '(acc + v).to(storage_dtype, fp_downcast_rounding="rtne")' in source
            assert "if valid:" in source and "end <= nrows" in source
            assert "token * stride_row + channel * stride_channel" in source
            assert "tl.sum" not in source and "tl.reduce" not in source
            if name == "segment_mean":
                assert "(end - start).to(storage_dtype).to(tl.float32)" in source
                assert "tl.div_rn" in source
    import pytest
    with pytest.raises(NotImplementedError, match="ordered"):
        TO.emit_segment_source("segment_sum", plan="split")
    with pytest.raises(ValueError, match="finite"):
        TO.emit_segment_source("segment_sum", initial=float("nan"))
    with pytest.raises(ValueError, match="unknown segment parameters"):
        TO.compile_segment_reduction("segment_sum", {"typo": 1})


def test_segment_exact_identity_cache_collision_and_owner():
    import neptune_codegen.torch_op as TO
    import neptune_codegen.template as T
    import hashlib
    import importlib.util
    import pytest

    saved = dict(TO._SEGMENT_KERNELS), dict(TO._SEGMENT_NAMES)
    loaded_names = []
    counter = []

    def load(module):
        module.segment_sum = object()
        loaded_names.append(module.__name__)
        counter.append(module.segment_sum)

    spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=load))
    try:
        TO._SEGMENT_KERNELS.clear()
        TO._SEGMENT_NAMES.clear()
        # Force the short name digest to collide: exact tuple still distinguishes every op/file.
        with mock.patch.object(hashlib, "sha256", return_value=types.SimpleNamespace(hexdigest=lambda: "0" * 10)), mock.patch(
            "builtins.open", mock.mock_open()
        ) as files, mock.patch.object(importlib.util, "spec_from_file_location", return_value=spec), mock.patch.object(
            importlib.util, "module_from_spec", side_effect=lambda _: types.ModuleType("segment_stub")
        ):
            options = [{"initial": v} for v in (None, 0.0, -0.0, 1.0000001, 1.0000002)]
            options += [{"offsets": True}, {"unsafe": True}]
            kernels = [TO.compile_segment_reduction("segment_sum", p) for p in options]
            assert len({id(k) for k in kernels}) == len(options)
            assert len({call.args[0] for call in files.call_args_list}) == len(options)
            for p, kernel in zip(options, kernels):
                assert TO.compile_segment_reduction("segment_sum", p) is kernel
            assert len(counter) == len(options)
            with mock.patch.object(T, "_GEN_RELEASED", True), pytest.raises(RuntimeError, match="released"):
                TO.compile_segment_reduction("segment_sum")
    finally:
        TO._SEGMENT_KERNELS.clear()
        TO._SEGMENT_KERNELS.update(saved[0])
        TO._SEGMENT_NAMES.clear()
        TO._SEGMENT_NAMES.update(saved[1])
        for name in loaded_names:
            sys.modules.pop(name, None)


def test_segment_metadata_fake_zero_and_symbolic_dimensions():
    import torch
    import pytest
    import neptune_codegen.torch_op as TO
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    with mock.patch.object(TO, "compile_segment_reduction", side_effect=AssertionError("metadata compiled Triton")):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for shape, count, offsets in (((0,), 0, False), ((3, 0, 2), 2, False),
                                           ((0, 2), 1, True), ((4, 2, 3), 4, True)):
                data = torch.empty(shape, device="meta", dtype=dtype)
                segments = torch.empty(count, device="meta", dtype=torch.int32)
                out = TO.run_segment_pool(data, segments, offsets=offsets)
                assert out.shape == ((count - 1 if offsets else count),) + shape[1:]
                assert out.dtype == dtype
        with pytest.raises(RuntimeError, match="boundary"):
            TO.validate_segment_inputs(torch.empty(3), torch.empty(0, dtype=torch.int64),
                                       metadata=True, offsets=True)
        for data, segments, error in ((torch.empty(()), torch.empty(2, dtype=torch.int64), ValueError),
                                      (torch.empty(3, dtype=torch.float64), torch.empty(2, dtype=torch.int64), TypeError),
                                      (torch.empty(3), torch.empty(2), TypeError),
                                      (torch.empty(3), torch.empty(2, 1, dtype=torch.int64), ValueError),
                                      (torch.empty(3), torch.empty(2, device="meta", dtype=torch.int64), ValueError)):
            with pytest.raises(error):
                TO.validate_segment_inputs(data, segments, metadata=True)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with pytest.raises(RuntimeError, match="overflow"):
                TO.validate_segment_inputs(torch.empty(3, dtype=dtype), torch.tensor([3]),
                                           metadata=True, initial=1e300)
        with pytest.raises(ValueError, match="CUDA"):
            TO.run_segment_pool(torch.empty(3), torch.tensor([3]))
        env = ShapeEnv()
        n, s, d = (env.create_unbacked_symint() for _ in range(3))
        with FakeTensorMode(shape_env=env):
            data = torch.empty(n, d, 2)
            segments = torch.empty(s, dtype=torch.int64)
            out = TO.run_segment_pool(data, segments)
            assert isinstance(out.shape[0], torch.SymInt)
            assert isinstance(out.shape[1], torch.SymInt)
            assert out.shape == (s, d, 2)


def test_segment_prefix_validation_readonly_and_exportable():
    import torch
    import pytest
    import neptune_codegen.torch_op as TO

    x = torch.empty(7, 2)
    lengths = torch.tensor([0, 2, 0, 5], dtype=torch.int32)
    original = lengths.clone()
    spans = TO._segment_spans(x, lengths, offsets=False, unsafe=False)
    assert spans.dtype == torch.int64
    torch.testing.assert_close(spans, torch.tensor([0, 0, 2, 2, 7]))
    torch.testing.assert_close(lengths, original)
    offsets = torch.tensor([1, 1, 4, 6])
    assert TO._segment_spans(x, offsets, offsets=True, unsafe=False) is offsets
    for indices, is_offsets, unsafe in (([-1, 8], False, False), ([3, 3], False, False),
                                        ([8], False, True), ([4, 4], False, True),
                                        ([-1, 4], True, True), ([0, 8], True, False),
                                        ([4, 3], True, True)):
        with pytest.raises(RuntimeError):
            TO._segment_spans(x, torch.tensor(indices), offsets=is_offsets, unsafe=unsafe)
    torch.testing.assert_close(TO._segment_spans(x, torch.tensor([2]), offsets=False, unsafe=True),
                               torch.tensor([0, 2]))

    class Prefix(torch.nn.Module):
        def forward(self, data, segments):
            return TO._segment_spans(data, segments, offsets=False, unsafe=False)

    exported = torch.export.export(Prefix(), (x, lengths), dynamic_shapes={
        "data": {0: torch.export.Dim("segment_data_rows", min=1, max=16)},
        "segments": {0: torch.export.Dim("segment_count", min=1, max=16)}})
    targets = {str(node.target) for node in exported.graph.nodes if node.op == "call_function"}
    assert "aten.cumsum.default" in targets
    assert "aten.cat.default" in targets
    assert "aten._assert_async.msg" in targets


def test_segment_cpu_recorder_strides_fresh_output_and_bridge():
    import torch
    import neptune_codegen.torch_op as TO

    calls = []

    class Recorder:
        def __getitem__(self, grid):
            def launch(flat, spans, out, n, d, s, row_stride, channel_stride, span_stride, **kwargs):
                assert grid == (s, (d + 63) // 64)
                assert flat.shape == (n, d)
                assert flat.stride() == (row_stride, channel_stride)
                assert spans.stride(0) == span_stride
                assert kwargs == {"BLOCK": 64, "num_warps": 4, "enable_fp_fusion": False}
                calls.append(flat)
                out.zero_()
            return launch

    validate = TO.validate_segment_inputs
    with mock.patch.object(TO, "validate_segment_inputs", side_effect=lambda *a, **kw: validate(
        *a, **{**kw, "metadata": True}
    )), mock.patch.object(TO, "compile_segment_reduction", return_value=Recorder()), mock.patch.object(
        torch.library, "triton_op", side_effect=lambda *a, **kw: lambda fn: fn, create=True
    ), mock.patch.object(torch.library, "wrap_triton", side_effect=lambda kernel: kernel, create=True), mock.patch.dict(
        TO._SEGMENT_OPS, {}, clear=True
    ):
        base = torch.randn(7, 4, 6)
        inputs = [torch.randn(7), torch.randn(7, 6)[:, ::2], torch.randn(6, 7).t(),
                  torch.randn(7, 1).expand(7, 70), base[:, :, ::2], base.transpose(1, 2)]
        segments = torch.tensor([0, 99, 2, 99, 5, 99], dtype=torch.int32)[::2]
        for x in inputs:
            saved = x.clone()
            for call in (lambda: TO.run_segment_pool(x, segments),
                         lambda: TO.as_segment_op("segment_sum", lib="segment_recorder")(x, segments)):
                out = call()
                assert out.shape == (3,) + x.shape[1:]
                assert out.dtype == x.dtype
                assert out._base is None
                if x.ndim <= 2:
                    assert calls[-1].untyped_storage() is x.untyped_storage()
                torch.testing.assert_close(x, saved)
        count = len(calls)
        assert TO.run_segment_pool(torch.empty(0, 4), torch.empty(0, dtype=torch.int64)).shape == (0, 4)
        assert TO.run_segment_pool(torch.empty(7, 0), segments).shape == (3, 0)
        assert len(calls) == count


def test_segment_real_custom_op_fake_and_hinted_export():
    import torch
    import neptune_codegen.torch_op as TO
    from torch._subclasses.fake_tensor import FakeTensorMode

    if not hasattr(torch.library, "triton_op"):
        return  # CPU metadata/recorder contracts above still run on Torch 2.4/2.6.
    with mock.patch.object(TO, "compile_segment_reduction", side_effect=AssertionError("Fake compiled Triton")):
        op = TO.as_segment_op("segment_mean", {"initial": -2.0}, lib="segment_metadata_tests")
        handle = TO.segment_op_handle("segment_mean", {"initial": -2.0}, lib="segment_metadata_tests")
        assert isinstance(handle, torch._ops.OpOverload)
        with FakeTensorMode():
            assert op(torch.empty(3, 0), torch.empty(2, dtype=torch.int64)).shape == (2, 0)
            assert op(torch.empty(0, 3), torch.empty(0, dtype=torch.int64)).shape == (0, 3)

        class Model(torch.nn.Module):
            def forward(self, data, segments):
                return op(data, segments)

        exported = torch.export.export(Model(), (torch.empty(7, 2, 3), torch.tensor([2, 5])),
                                       dynamic_shapes={"data": {0: torch.export.Dim("rows", min=1, max=16),
                                                                1: torch.export.Dim("channels", min=1, max=8)},
                                                       "segments": {0: torch.export.Dim("segments", min=1, max=8)}})
        val = next(n.meta["val"] for n in exported.graph.nodes if n.op == "call_function")
        assert isinstance(val.shape[0], torch.SymInt) and isinstance(val.shape[1], torch.SymInt)


def test_segment_import_and_metadata_without_triton():
    result = subprocess.run([sys.executable, "-c",
        "import sys; sys.modules['triton'] = None; import torch; "
        "import neptune_codegen.torch_op as t; "
        "x=torch.empty(3,2,device='meta'); s=torch.empty(2,device='meta',dtype=torch.int64); "
        "assert t.run_segment_pool(x,s).shape == (2,2); "
        "assert 'for token in range' in t.emit_segment_source('segment_sum')"],
        check=True, capture_output=True, text=True)
    assert result.returncode == 0


import pytest as _segment_pytest  # noqa: E402
import torch as _segment_torch  # noqa: E402


@_segment_pytest.mark.skipif(not _segment_torch.cuda.is_available(), reason="segment correctness requires CUDA")
def test_segment_cuda_matches_native_ordered_sum_mean():
    import torch
    import neptune_codegen.torch_op as TO

    if not torch.cuda.is_available():
        return  # Preserve this file's standalone CPU runner as well as pytest.
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        base = torch.full((1027, 2, 6), 0.1, device="cuda", dtype=dtype)
        # Business nonfinites are not filtered/clamped: include cancellation and NaN propagation.
        base[1, 0, 0] = float("inf")
        base[2, 0, 0] = -float("inf")
        base[3, 1, 2] = float("nan")
        base[4, 1, 4] = -float("inf")
        for data in (base[:, :, ::2], base.transpose(1, 2)):
            before = data.clone()
            lengths = torch.tensor([0, 999, 2, 999, 1025, 999, 0, 999], device="cuda", dtype=torch.int32)[::2]
            offsets = torch.tensor([1, 999, 1, 999, 3, 999, 1026, 999], device="cuda", dtype=torch.int64)[::2]
            for initial in (None, 0.0, -2.125):
                for reduce in ("sum", "mean"):
                    for segments, is_offsets in ((lengths, False), (offsets, True)):
                        expected = torch.segment_reduce(data, reduce, initial=initial,
                                                        **{("offsets" if is_offsets else "lengths"): segments})
                        actual = TO.run_segment_pool(data, segments, reduce=reduce, initial=initial, offsets=is_offsets)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                        assert actual.dtype == dtype and actual.shape == expected.shape
                        assert actual._base is None
            torch.testing.assert_close(data, before, rtol=0, atol=0, equal_nan=True)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        data = torch.tensor([[1e-40, -0.0], [1e-40, 0.0]], device="cuda", dtype=dtype)
        lengths = torch.tensor([0, 2], device="cuda")
        for reduce in ("sum", "mean"):
            actual = TO.run_segment_pool(data, lengths, reduce=reduce, initial=-0.0)
            expected = torch.segment_reduce(data, reduce, lengths=lengths, initial=-0.0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            assert torch.equal(torch.signbit(actual), torch.signbit(expected))
    data = torch.zeros(1027, 3, device="cuda", dtype=torch.bfloat16)
    lengths = torch.tensor([1027], device="cuda")
    op = TO.as_segment_op("segment_mean", {"initial": 7.0}, lib="segment_gpu_tests")
    torch.testing.assert_close(op(data, lengths), torch.segment_reduce(data, "mean", lengths=lengths, initial=7.0),
                               rtol=0, atol=0, equal_nan=True)


def test_state_certificate_builtin_schema_and_exact_proof_without_sampling():
    import dataclasses
    import sympy as sp
    import neptune_codegen as CG
    import neptune_codegen.algebra as A

    A.derive_state_certificate.cache_clear()
    with mock.patch.object(A, "_numerically_zero", side_effect=AssertionError("proof sampled")) as sampling:
        for name in ("rms_norm", "segment_sum", "segment_mean"):
            spec = CG.STATE_REDUCTIONS[name]
            assert isinstance(spec, CG.StateReduction)
            assert isinstance(spec.numeric, CG.NumericalContract)
            assert all(isinstance(field, CG.StateField) for field in spec.fields)
            total, count = spec.fields
            assert total.dtype == "float32"
            assert count.dtype == "int64" and count.symbol.is_integer is True
            assert count.symbol.is_nonnegative is True and count.term == 1
            assert total.extent == ("scalar" if name == "rms_norm" else "channel")
            certificate = CG.derive_state_certificate(spec)
            assert certificate.state_fields == ("total", "count")
            assert certificate.proof_kind == "exact-structured-real-algebra"
            assert certificate.ieee_associative is False and certificate.assumptions
            assert certificate.merge == tuple(a + b for a, b in zip(spec.left, spec.right))
            # The proof accepts the supported sympifiable numeric grammar, not just Expr objects.
            converted = dataclasses.replace(spec, fields=(total, dataclasses.replace(
                count, term="1", identity="0")))
            assert CG.derive_state_certificate(converted).merge == certificate.merge
            source = CG.emit_additive_state_helpers(spec)
            assert "a_total + b_total" in source and "a_count + b_count" in source
            assert sp.sympify(total.term).free_symbols <= set(spec.inputs)
        sampling.assert_not_called()
    for name in ("StateField", "NumericalContract", "StateReduction", "derive_state_certificate",
                 "emit_rms_state_source", "run_rms_norm", "as_state_op", "state_op_handle"):
        assert name in CG.__all__ and hasattr(CG, name)


def test_state_certificate_rejects_unknown_schema_unbound_lift_and_invalid_merge():
    import dataclasses
    import sympy as sp
    import pytest
    import neptune_codegen as CG
    import neptune_codegen.algebra as A

    spec = CG.STATE_REDUCTIONS["rms_norm"]
    total, count = spec.fields
    unknown = sp.Symbol("unbound_state_lift", real=True)
    malformed = [
        dataclasses.replace(spec, fields=()),
        dataclasses.replace(spec, fields=(total, total)),
        dataclasses.replace(spec, fields=(dataclasses.replace(total, dtype="float64"), count)),
        dataclasses.replace(spec, fields=(dataclasses.replace(total, extent="unknown"), count)),
        dataclasses.replace(spec, domain="unknown"),
        dataclasses.replace(spec, output_kind="unknown"),
        dataclasses.replace(spec, fields=(dataclasses.replace(total, term=unknown), count)),
        dataclasses.replace(spec, fields=(dataclasses.replace(total, term=total.symbol), count)),
        dataclasses.replace(spec, fields=(dataclasses.replace(total, identity=unknown), count)),
        dataclasses.replace(spec, readout=spec.readout + unknown),
        dataclasses.replace(spec, merge=(spec.left[0] + spec.right[0],)),
        dataclasses.replace(spec, merge=(spec.left[0] + spec.right[0] + unknown,
                                        spec.left[1] + spec.right[1])),
    ]
    for expression in (spec.left[0] - spec.right[0], spec.right[0] - spec.left[0],
                       spec.left[0] + spec.right[0] + 1,
                       spec.left[0] + spec.right[0] + spec.left[0] * spec.right[0] *
                       (spec.left[0] - spec.right[0])):
        malformed.append(dataclasses.replace(spec, merge=(expression, spec.left[1] + spec.right[1])))
    A.derive_state_certificate.cache_clear()
    with mock.patch.object(A, "_numerically_zero", side_effect=AssertionError("proof sampled")):
        for invalid in malformed:
            with pytest.raises(NotImplementedError):
                CG.derive_state_certificate(invalid)
        # A custom additive merge is certifiable but is not authorized by this backend.
        custom = dataclasses.replace(spec, merge=spec.merge_expressions)
        CG.derive_state_certificate(custom)
        with pytest.raises(NotImplementedError, match="componentwise"):
            CG.emit_additive_state_helpers(custom)


def test_state_certificate_rejects_noninteger_count_lift_domain():
    import dataclasses
    import sympy as sp
    import neptune_codegen as CG

    spec = CG.STATE_REDUCTIONS["rms_norm"]
    total, count = spec.fields
    accepted = []
    for fractional in (sp.Rational(1, 2), spec.inputs[0]):
        invalid = dataclasses.replace(spec, fields=(total, dataclasses.replace(count, term=fractional)))
        try:
            CG.derive_state_certificate(invalid)
        except NotImplementedError as error:
            assert any(word in str(error) for word in ("integer", "int64", "domain"))
        else:
            accepted.append(str(fractional))
    assert not accepted, f"int64 state lift lacks an integer-domain proof: {accepted}"


def test_rms_backend_source_ast_masks_strides_rounding_and_certified_square():
    import ast
    import neptune_codegen as CG
    import neptune_codegen.emitters as E

    spec = CG.STATE_REDUCTIONS["rms_norm"]
    helper = CG.emit_additive_state_helpers(spec)
    square = CG.to_triton(spec.fields[0].term, {spec.inputs[0]: "value"})
    for plan in ("serial", "split"):
        with mock.patch.object(E, "emit_additive_state_helpers", wraps=E.emit_additive_state_helpers) as derived:
            source = CG.emit_rms_state_source(spec, {"eps": 1e-6}, plan=plan)
        derived.assert_called_once_with(spec)
        assert helper in source and f"total = tl.sum({square}, 0)" in source
        assert source == CG.emit_rms_state_source(spec, {"eps": 1e-6}, plan=plan)
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        assert set(functions) == ({"rms_norm_state_merge", "rms_norm"} if plan == "serial" else
                                  {"rms_norm_state_merge", "rms_norm_partial", "rms_norm_map"})
        for name, fn in functions.items():
            if name.endswith("state_merge"):
                continue
            loads = [node for node in ast.walk(fn) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "load"]
            for load in loads:
                pointer = ast.unparse(load.args[0])
                keywords = {kw.arg: ast.unparse(kw.value) for kw in load.keywords}
                assert "mask" in keywords and keywords.get("other") == "0"
                if "x_ptr" in pointer:
                    assert "row * stride_row" in pointer and "column * stride_col" in pointer
                    assert keywords["mask"] == "column < ncols"
                elif "gamma_ptr" in pointer:
                    assert "column * stride_gamma" in pointer
                    assert keywords["mask"] == "column < ncols"
                else:
                    assert keywords["mask"] == "part < nparts"
            if name in ("rms_norm", "rms_norm_map"):
                body = ast.unparse(fn)
                assert body.index("tl.div_rn") < body.index("libdevice.rsqrt_rn")
                assert body.index("fp_downcast_rounding='rtne'") < body.index("result = normalized * gamma")
                assert "normalized.to(out_ptr.dtype.element_ty" in body
                stores = [node for node in ast.walk(fn) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Attribute) and node.func.attr == "store"]
                assert len(stores) == 1
                assert "row * ncols + column" in ast.unparse(stores[0].args[0])
                assert any(kw.arg == "mask" and ast.unparse(kw.value) == "column < ncols"
                           for kw in stores[0].keywords)
        assert "count.to(tl.float32)" in source and ".to(tl.int64)" in source
        if plan == "split":
            assert "tl.reduce((total, count), 0, rms_norm_state_merge)" in source


def test_rms_backend_source_and_cache_exact_eps_fail_closed():
    import dataclasses
    import pytest
    import neptune_codegen as CG
    import neptune_codegen.template as T

    spec = CG.STATE_REDUCTIONS["rms_norm"]
    loaded = []

    def load(module):
        module.rms_norm = object()
        loaded.append(module.__name__)

    importer = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=load))
    with mock.patch.dict(T._STATE_CACHE, {}, clear=True), mock.patch("builtins.open", mock.mock_open()) as files, \
            mock.patch.object(T.importlib.util, "spec_from_file_location", return_value=importer), \
            mock.patch.object(T.importlib.util, "module_from_spec", side_effect=lambda _: types.ModuleType("rms_stub")):
        try:
            first = T.compile_state_reduction("rms_norm", {"eps": 1.0000001e-6})
            second = T.compile_state_reduction("rms_norm", {"eps": 1.0000002e-6})
            assert first is not second
            assert T.compile_state_reduction("rms_norm", {"eps": 1.0000001e-6}) is first
            assert len(loaded) == 2 and len({call.args[0] for call in files.call_args_list}) == 2
            for params in ({}, {"eps": 1e-6, "typo": 0}, *({"eps": value} for value in
                           (True, "1e-6", None, 0, -1, float("nan"), float("inf")))):
                with pytest.raises(ValueError):
                    T.compile_state_reduction("rms_norm", params)
            with pytest.raises(NotImplementedError):
                T.compile_state_reduction("unknown", {"eps": 1e-6})
            with pytest.raises(ValueError):
                T.compile_state_reduction("rms_norm", {"eps": 1e-6}, plan="typo")
            with mock.patch.object(T, "_GEN_RELEASED", True), pytest.raises(RuntimeError, match="released"):
                T.compile_state_reduction("rms_norm", {"eps": 1e-6})
            assert len(loaded) == 2, "invalid metadata reached compilation"
        finally:
            for name in loaded:
                sys.modules.pop(name, None)
    for changed in (dataclasses.replace(spec, name="other"),
                    dataclasses.replace(spec, numeric=dataclasses.replace(spec.numeric, stages=())),
                    dataclasses.replace(spec, readout=spec.readout + 1)):
        with pytest.raises(NotImplementedError, match="registered"):
            CG.emit_rms_state_source(changed, {"eps": 1e-6})


def test_rms_backend_metadata_fake_shapes_readonly_without_cuda_or_compile():
    import torch
    import neptune_codegen.template as T
    from torch._subclasses.fake_tensor import FakeTensorMode

    import builtins
    # Torch initializes its own optional Triton discovery when FakeTensorMode is
    # constructed. Initialize that framework first; guard the backend calls below.
    FakeTensorMode()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("triton"), "metadata imported Triton"
        return original_import(name, *args, **kwargs)

    with mock.patch.object(T, "compile_state_reduction", side_effect=AssertionError("metadata compiled")), \
            mock.patch.object(T, "launch_rms_norm", side_effect=AssertionError("metadata launched")), \
            mock.patch.object(torch.cuda, "get_device_name", side_effect=AssertionError("metadata queried CUDA")), \
            mock.patch.object(builtins, "__import__", side_effect=guarded_import):
        for plan in ("serial", "split"):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for shape in ((17,), (2, 3, 17), (0, 17), (2, 0, 17)):
                    for fake in (False, True):
                        with FakeTensorMode() if fake else contextlib.nullcontext():
                            x = torch.empty(shape, dtype=dtype, device="cpu" if fake else "meta")
                            gamma = torch.empty(34, dtype=dtype, device=x.device)[::2]
                            before = (x.shape, x.stride(), x.dtype, x.device, gamma.shape, gamma.stride())
                            out = T.run_rms_norm(x, gamma, plan=plan)
                            assert out.shape == x.shape and out.dtype == dtype and out.device == x.device
                            assert out.is_contiguous() and out._base is None
                            assert out is not x and out is not gamma
                            assert (x.shape, x.stride(), x.dtype, x.device, gamma.shape, gamma.stride()) == before
        # Dense noncanonical input layout is supported; the output is fresh canonical storage.
        x = torch.empty(512, 2, device="meta").t()
        assert x.stride() == (1, 2)
        out = T.run_rms_norm(x, torch.empty(512, device="meta"))
        assert out.stride() == (512, 1) and x.stride() == (1, 2)


def test_rms_backend_metadata_validation_static_width_and_capacity():
    import torch
    import pytest
    import neptune_codegen.template as T

    def meta(shape, dtype=torch.float32):
        return torch.empty(shape, device="meta", dtype=dtype)

    cases = [
        (meta(()), meta((1,)), ValueError, "rank"),
        (meta((2, 0)), meta((0,)), RuntimeError, "width"),
        (meta((2, 8), torch.uint8), meta((8,), torch.uint8), TypeError, "dtype"),
        (meta((2, 8), torch.int32), meta((8,), torch.int32), TypeError, "dtype"),
        (meta((2, 8), torch.float16), meta((8,)), ValueError, "dtype/device"),
        (meta((2, 8)), meta((8,), torch.int32), ValueError, "dtype/device"),
        (meta((2, 8)), meta((1, 8)), ValueError, "1D"),
        (meta((2, 8)), meta((7,)), RuntimeError, "width"),
        (meta((2, 8)), torch.empty(8), ValueError, "dtype/device"),
    ]
    with mock.patch.object(T, "compile_state_reduction", side_effect=AssertionError("invalid input compiled")):
        for x, gamma, error, message in cases:
            with pytest.raises(error, match=message):
                T.run_rms_norm(x, gamma)
        with pytest.raises(ValueError, match="CUDA"):
            T.run_rms_norm(torch.empty(2, 8), torch.empty(8))
        for width, plan in ((32768, "serial"), (32769, "split"), (4194304, "split")):
            assert T.validate_rms_inputs(meta((0, width)), meta((width,)), {"eps": 1e-6},
                                         plan=plan, metadata=True) == (0, width)
        for width, plan, message in ((32769, "serial", "32768"), (4194305, "split", "capacity")):
            with pytest.raises(NotImplementedError, match=message):
                T.validate_rms_inputs(meta((0, width)), meta((width,)), {"eps": 1e-6},
                                      plan=plan, metadata=True)
        for eps in (True, "1e-6", None, 0, -1, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="eps"):
                T.run_rms_norm(meta((2, 8)), meta((8,)), eps=eps)


def test_rms_backend_symbolic_batch_never_concretizes_and_dynamic_width_rejects():
    import torch
    import pytest
    import neptune_codegen.template as T
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    env = ShapeEnv()
    batch, width = env.create_unbacked_symint(), env.create_unbacked_symint()
    with FakeTensorMode(shape_env=env):
        x, gamma = torch.empty(batch, 3, 8), torch.empty(8)
        with mock.patch.object(torch.SymInt, "__int__", side_effect=AssertionError("batch concretized")), \
                mock.patch.object(T, "compile_state_reduction", side_effect=AssertionError("Fake compiled")):
            for plan in ("serial", "split"):
                out = T.run_rms_norm(x, gamma, plan=plan)
                assert isinstance(out.shape[0], torch.SymInt)
                assert out.shape == (batch, 3, 8)
        dynamic = torch.empty(2, width)
        dynamic_gamma = torch.empty(width)
        for plan in ("serial", "split"):
            with pytest.raises(NotImplementedError, match="static feature width"):
                T.run_rms_norm(dynamic, dynamic_gamma, plan=plan)


def test_rms_backend_hinted_dynamic_width_rejects_without_specialization():
    import torch
    import pytest
    import neptune_codegen.template as T

    class Source(torch.nn.Module):
        def forward(self, x, gamma):
            value = x.float()
            return (value * torch.rsqrt((value * value).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * gamma

    feature = torch.export.Dim("rms_feature_width", min=2, max=64)
    ep = torch.export.export(Source(), (torch.ones(2, 8), torch.ones(8)), dynamic_shapes={
        "x": {1: feature}, "gamma": {0: feature}})
    x, gamma = [node.meta["val"] for node in ep.graph.nodes if node.op == "placeholder"]
    assert isinstance(x.shape[-1], torch.SymInt) and x.shape[-1].node.has_hint()
    with mock.patch.object(torch.SymInt, "__int__", side_effect=AssertionError("width concretized")), \
            mock.patch.object(T, "compile_state_reduction", side_effect=AssertionError("dynamic width compiled")):
        for plan in ("serial", "split"):
            with pytest.raises(NotImplementedError, match="static feature width"):
                T.validate_rms_inputs(x, gamma, {"eps": 1e-6}, plan=plan, metadata=True)


def test_rms_backend_real_custom_op_fake_and_dynamic_batch_export():
    import torch
    import pytest
    import neptune_codegen.torch_op as TO
    import neptune_codegen.template as T
    from torch._subclasses.fake_tensor import FakeTensorMode

    if not hasattr(torch.library, "triton_op"):
        pytest.skip("real Triton bridge requires Torch >= 2.6; pure CPU metadata tests still run")
    with mock.patch.object(T, "compile_state_reduction", side_effect=AssertionError("export compiled Triton")), \
            mock.patch.object(T, "launch_rms_norm", side_effect=AssertionError("export launched GPU")), \
            mock.patch.object(torch.cuda, "get_device_name", side_effect=AssertionError("export queried CUDA")):
        for plan in ("serial", "split"):
            op = TO.as_state_op("rms_norm", {"eps": 1e-6}, plan=plan, lib="rms_cpu_metadata_tests")
            handle = TO.state_op_handle("rms_norm", {"eps": 1e-6}, plan=plan, lib="rms_cpu_metadata_tests")
            assert isinstance(handle, torch._ops.OpOverload)
            with FakeTensorMode():
                for shape in ((8,), (2, 3, 8), (0, 3, 8)):
                    out = op(torch.empty(shape, dtype=torch.bfloat16), torch.empty(8, dtype=torch.bfloat16))
                    assert out.shape == shape and out.dtype == torch.bfloat16 and out.is_contiguous()

            class Model(torch.nn.Module):
                def forward(self, x, gamma):
                    return op(x, gamma)

            ep = torch.export.export(Model(), (torch.empty(2, 3, 8), torch.empty(8)), dynamic_shapes={
                "x": {0: torch.export.Dim("rms_batch", min=1, max=16)}, "gamma": {}})
            call = next(node for node in ep.graph.nodes if node.op == "call_function")
            source = next(node for node in ep.graph.nodes if node.op == "placeholder")
            assert call.target == handle
            assert isinstance(call.meta["val"].shape[0], torch.SymInt)
            assert call.meta["val"].shape == source.meta["val"].shape


def test_rms_backend_import_and_metadata_without_triton():
    result = subprocess.run([sys.executable, "-c",
        "import sys; sys.modules['triton'] = None; import torch; "
        "import neptune_codegen as c; import neptune_codegen.template as t; "
        "t.compile_state_reduction=lambda *a,**k: (_ for _ in ()).throw(AssertionError('compiled')); "
        "x=torch.empty(2,3,8,device='meta'); g=torch.empty(8,device='meta'); "
        "assert c.run_rms_norm(x,g).shape == x.shape; "
        "assert 'libdevice.rsqrt_rn' in c.emit_rms_state_source(c.STATE_REDUCTIONS['rms_norm'],{'eps':1e-6})"],
        check=True, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0


def test_immutable_state_registry_and_nested_statistics_are_frozen():
    import dataclasses
    from types import MappingProxyType
    import pytest
    import neptune_codegen as CG
    import neptune_codegen.skeleton as S

    registry = S.STATE_REDUCTIONS
    assert isinstance(registry, MappingProxyType)
    assert CG.STATE_REDUCTIONS is registry is S._BUILTIN_STATE_REDUCTIONS
    with pytest.raises(TypeError):
        registry["rms_norm"] = registry["segment_sum"]
    for name, spec in registry.items():
        assert S.state_reduction_spec(name) is spec
        for value in (spec.inputs, spec.fields, spec.params, spec.merge, spec.numeric.stages):
            assert isinstance(value, tuple)
        with pytest.raises(TypeError):
            spec.fields[0] = spec.fields[1]
        with pytest.raises(TypeError):
            spec.numeric.stages[0] = "different_rounding"
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.fields[0].term = spec.inputs[0] ** 4
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.numeric.stages = ()
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.readout = spec.readout + 1
        assert S.state_reduction_spec(name) is spec


def test_immutable_state_rms_rebound_registry_refused_before_cached_compile_and_registration():
    import builtins
    import dataclasses
    import pytest
    import torch
    import neptune_codegen as CG
    import neptune_codegen.skeleton as S
    import neptune_codegen.template as T
    import neptune_codegen.torch_op as TO

    registry = S.STATE_REDUCTIONS
    spec = registry["rms_norm"]
    total, count = spec.fields
    changed_specs = (
        dataclasses.replace(spec, fields=(dataclasses.replace(total, term=spec.inputs[0] ** 4), count)),
        dataclasses.replace(spec, readout=spec.readout + 1),
        dataclasses.replace(spec, numeric=dataclasses.replace(spec.numeric, stages=())),
    )
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "triton" and not name.startswith("triton."), "authorization imported Triton"
        return original_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as isolated:
        isolated.setattr(T, "_GEN_RELEASED", False)
        isolated.setattr(T, "_STATE_CACHE", {})
        isolated.setattr(TO, "_STATE_OPS", {})
        with mock.patch.object(builtins, "__import__", side_effect=guarded_import), \
                mock.patch("builtins.open", side_effect=AssertionError("authorization wrote source")), \
                mock.patch.object(torch.library, "triton_op", create=True,
                                  side_effect=AssertionError("authorization registered an operator")):
            for plan in ("serial", "split"):
                params = {"eps": 1e-6}
                kernel = object()
                impl = object()
                key = (spec, autotune.parameter_identity(params), plan)
                op_key = ("immutable_state_cache", *key)
                T._STATE_CACHE[key] = kernel
                TO._STATE_OPS[op_key] = impl
                cached_kernels = dict(T._STATE_CACHE)
                cached_ops = dict(TO._STATE_OPS)
                source = CG.emit_rms_state_source(spec, params, plan=plan)
                assert T.compile_state_reduction("rms_norm", params, plan=plan) is kernel
                assert TO.as_state_op("rms_norm", params, plan=plan, lib=op_key[0]) is impl
                for changed in changed_specs:
                    with pytest.MonkeyPatch.context() as changed_registry:
                        changed_registry.setattr(S, "STATE_REDUCTIONS", {**registry, "rms_norm": changed})
                        # CG's original mapping is still immutable: it must not hide a rebound module registry.
                        assert CG.STATE_REDUCTIONS is registry
                        for call in (
                            lambda: S.state_reduction_spec("rms_norm"),
                            lambda: CG.emit_rms_state_source(spec, params, plan=plan),
                            lambda: CG.emit_rms_state_source(changed, params, plan=plan),
                            lambda: T.compile_state_reduction("rms_norm", params, plan=plan),
                            lambda: TO.as_state_op("rms_norm", params, plan=plan, lib=op_key[0]),
                            lambda: TO.state_op_handle("rms_norm", params, plan=plan, lib=op_key[0]),
                            lambda: TO.as_state_op("rms_norm", params, plan=plan, lib="immutable_state_uncached"),
                        ):
                            with pytest.raises(NotImplementedError, match="immutable|registered"):
                                call()
                        assert T._STATE_CACHE == cached_kernels
                        assert TO._STATE_OPS == cached_ops
                        assert T._STATE_CACHE[key] is kernel and TO._STATE_OPS[op_key] is impl
                    assert S.STATE_REDUCTIONS is registry
                    assert S.state_reduction_spec("rms_norm") is spec
                    assert CG.emit_rms_state_source(spec, params, plan=plan) == source
                    assert T.compile_state_reduction("rms_norm", params, plan=plan) is kernel
                    assert TO.as_state_op("rms_norm", params, plan=plan, lib=op_key[0]) is impl


def test_immutable_state_public_registry_cannot_replace_authorization_root():
    import dataclasses
    import pytest
    import neptune_codegen as CG
    import neptune_codegen.skeleton as S
    import neptune_codegen.template as T

    registry = S.STATE_REDUCTIONS
    spec = registry["rms_norm"]
    changed = dataclasses.replace(spec, readout=spec.readout + 1)
    params = {"eps": 1e-6}
    kernel = object()
    key = (spec, autotune.parameter_identity(params), "serial")
    source = CG.emit_rms_state_source(spec, params)
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(CG, "STATE_REDUCTIONS", {**registry, "rms_norm": changed})
        patched.setattr(T, "_GEN_RELEASED", False)
        patched.setattr(T, "_STATE_CACHE", {key: kernel})
        assert S.STATE_REDUCTIONS is registry
        assert S.state_reduction_spec("rms_norm") is spec
        assert CG.emit_rms_state_source(spec, params) == source
        with pytest.raises(NotImplementedError, match="registered"):
            CG.emit_rms_state_source(CG.STATE_REDUCTIONS["rms_norm"], params)
        with mock.patch("builtins.open", side_effect=AssertionError("cache hit wrote source")):
            assert T.compile_state_reduction("rms_norm", params) is kernel
    assert CG.STATE_REDUCTIONS is registry


def test_immutable_state_segment_compile_refuses_rebound_registry_before_existing_cache():
    import dataclasses
    import pytest
    import neptune_codegen.skeleton as S
    import neptune_codegen.template as T
    import neptune_codegen.torch_op as TO

    registry = S.STATE_REDUCTIONS
    module = types.ModuleType("immutable_state_segment_cached")
    module.calls = 0

    def cached_kernel(*args, **kwargs):
        module.calls += 1
        raise AssertionError("cached segment kernel executed")

    module.kernel = cached_kernel
    with pytest.MonkeyPatch.context() as isolated:
        isolated.setattr(T, "_GEN_RELEASED", False)
        isolated.setattr(TO, "_SEGMENT_KERNELS", {})
        isolated.setitem(sys.modules, module.__name__, module)
        with mock.patch("builtins.open", side_effect=AssertionError("cache validation wrote source")):
            for name in ("segment_sum", "segment_mean"):
                spec = registry[name]
                for initial in (None, -0.0, 2.125):
                    params = {"initial": initial, "offsets": True, "unsafe": False}
                    key = TO._segment_identity(name, TO._segment_params(name, params), "serial")
                    TO._SEGMENT_KERNELS[key] = module.kernel
                    assert TO.compile_segment_reduction(name, params) is module.kernel
                    for changed in (
                        dataclasses.replace(spec, fields=(dataclasses.replace(
                            spec.fields[0], term=spec.inputs[0] ** 4), spec.fields[1])),
                        dataclasses.replace(spec, readout=spec.readout + 1),
                        dataclasses.replace(spec, numeric=dataclasses.replace(spec.numeric, stages=())),
                    ):
                        with pytest.MonkeyPatch.context() as patched:
                            patched.setattr(S, "STATE_REDUCTIONS", {**registry, name: changed})
                            with pytest.raises(NotImplementedError, match="immutable|registered"):
                                TO.compile_segment_reduction(name, params)
                            assert TO._SEGMENT_KERNELS[key] is module.kernel
                        assert TO.compile_segment_reduction(name, params) is module.kernel
            assert module.calls == 0
    assert S.STATE_REDUCTIONS is registry


def test_immutable_state_segment_bridge_refuses_rebound_registry_before_existing_cache():
    import dataclasses
    import pytest
    import torch
    import neptune_codegen.skeleton as S
    import neptune_codegen.torch_op as TO

    registry = S.STATE_REDUCTIONS
    with pytest.MonkeyPatch.context() as isolated:
        isolated.setattr(TO, "_SEGMENT_OPS", {})
        isolated.setattr(TO, "_SEGMENT_NAMES", {})
        with mock.patch.object(torch.library, "triton_op", create=True,
                               side_effect=AssertionError("cache validation registered an operator")):
            for name in ("segment_sum", "segment_mean"):
                spec = registry[name]
                params = {"initial": -0.0, "offsets": False, "unsafe": False}
                lib = "immutable_state_segment_cache"
                key = (lib, TO._segment_identity(name, TO._segment_params(name, params), "serial"))
                impl = object()
                TO._SEGMENT_OPS[key] = impl
                assert TO.as_segment_op(name, params, lib=lib) is impl
                for changed in (
                    dataclasses.replace(spec, fields=(dataclasses.replace(
                        spec.fields[0], term=spec.inputs[0] ** 4), spec.fields[1])),
                    dataclasses.replace(spec, readout=spec.readout + 1),
                    dataclasses.replace(spec, numeric=dataclasses.replace(spec.numeric, stages=())),
                ):
                    with pytest.MonkeyPatch.context() as patched:
                        patched.setattr(S, "STATE_REDUCTIONS", {**registry, name: changed})
                        for call in (
                            lambda: TO.as_segment_op(name, params, lib=lib),
                            lambda: TO.segment_op_handle(name, params, lib=lib),
                            lambda: TO.as_state_op(name, params, lib=lib),
                            lambda: TO.state_op_handle(name, params, lib=lib),
                        ):
                            with pytest.raises(NotImplementedError, match="immutable|registered"):
                                call()
                        assert TO._SEGMENT_OPS[key] is impl
                    assert TO.as_segment_op(name, params, lib=lib) is impl
                    assert TO.as_state_op(name, params, lib=lib) is impl
    assert S.STATE_REDUCTIONS is registry


def test_immutable_state_count_domain_proof_is_not_int64_totality():
    import dataclasses
    import sympy as sp
    import pytest
    import neptune_codegen as CG

    spec = CG.STATE_REDUCTIONS["rms_norm"]
    total, count = spec.fields
    invalid_counts = (
        dataclasses.replace(count, term=-1),
        dataclasses.replace(count, identity=-1),
        dataclasses.replace(count, term=sp.Symbol("signed_count_input", integer=True)),
        dataclasses.replace(count, symbol=sp.Symbol("untyped_count", real=True)),
    )
    for invalid_count in invalid_counts:
        invalid = dataclasses.replace(spec, fields=(total, invalid_count))
        if sp.sympify(invalid_count.term).free_symbols:
            invalid = dataclasses.replace(invalid, inputs=spec.inputs + (invalid_count.term,))
        with pytest.raises(NotImplementedError, match="integer|nonnegative|int64"):
            CG.derive_state_certificate(invalid)
    for expression in (sp.Rational(1, 2), spec.inputs[0]):
        invalid = dataclasses.replace(spec, fields=(total, dataclasses.replace(count, term=expression)))
        with pytest.raises(NotImplementedError, match="integer|int64"):
            CG.derive_state_certificate(invalid)
    # Exact integer closure is conditional on representable reachable states, not a proof of no overflow.
    huge = dataclasses.replace(spec, fields=(total, dataclasses.replace(count, term=sp.Integer(2) ** 100)))
    certificate = CG.derive_state_certificate(huge)
    assert any("integer states fit" in assumption for assumption in certificate.assumptions)
    assert certificate.proof_kind == "exact-structured-real-algebra"
    assert certificate.ieee_associative is False
    with pytest.raises(NotImplementedError, match="registered"):
        CG.emit_rms_state_source(huge, {"eps": 1e-6})


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
