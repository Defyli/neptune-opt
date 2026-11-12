"""
Tests for the I4 lowering pass (`neptune_recognize/lowering.py`).

The lowering exists to give the region gate something to replace inside `log_softmax`, whose
decomposed form is `x - logsumexp(x)` -- two ops, where the frame needs a producer reduction whose
result is consumed inside a second reduction.

Three properties are tested here, and all three have been wrong at least once:

  1. the inserted nodes carry METADATA (the recogniser must read the output rank to know whether
     the operator keeps the reduced dimension; the first version inserted bare nodes and the
     rewrite refused every one of them)
  2. the commit gate is REWRITABILITY, not bare recognition (see `test_recognition_is_not_the_gate`)
  3. the pass is a NO-OP when it does not buy a fusion, byte-for-byte

CPU-only, and no Triton: everything asserted here is decided before a kernel is generated.

Run:  python tests/test_lowering.py     (or pytest tests/test_lowering.py)
"""
from __future__ import annotations

import os
import sys

from unittest.mock import patch

import torch
import torch.fx as fx
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neptune_recognize as NR  # noqa: E402
from neptune_recognize.lowering import (  # noqa: E402
    _fake_values,
    _half_to_float,
    _is_compound,
    lower_compound,
    _simulatable,
    lower_if_it_helps,
)

ROWS, COLS = 8, 16


class _LogSoftmax(nn.Module):
    def forward(self, x):
        return torch.log_softmax(x, dim=-1)


class _Softmax(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class _LogSoftmaxDim0(nn.Module):
    def forward(self, x):
        return torch.log_softmax(x, dim=0)


class _Plain(nn.Module):
    def forward(self, x):
        return torch.relu(x) + 1.0


def _export(module):
    x = torch.randn(ROWS, COLS)
    ep = torch.export.export(module, (x,)).run_decompositions({})
    return ep.graph_module, x


def _lowered_logsoftmax():
    gm, x = _export(_LogSoftmax())
    assert lower_if_it_helps(gm) == 1, "the lowering did not commit on log_softmax"
    return gm, x


# ---------------------------------------------------------------- 1. it lowers, and then it fuses
def test_lowering_makes_log_softmax_rewritable():
    gm, _x = _export(_LogSoftmax())
    ok, why = NR.would_rewrite(gm)
    assert ok is False, f"log_softmax was rewritable before lowering: {why!r}"
    assert "log_softmax" in why, f"unexpected refusal for the unlouvered graph: {why!r}"

    assert lower_if_it_helps(gm) == 1
    ok, why = NR.would_rewrite(gm)
    assert ok is True, f"still not rewritable after lowering: {why!r}"
    assert why == "logsumexp", f"lowered to {why!r}, expected the log-sum-exp operator"


def test_region_root_metadata_is_keepdim_and_real():
    """The exact bug the first version had: no metadata, so the rewrite refused the region."""
    gm, _x = _lowered_logsoftmax()
    root = NR.recognize_regions(gm)[0][0].region_root
    value = root.meta.get("val")
    assert value is not None, "the lowered region root carries no metadata"
    assert tuple(value.shape) == (ROWS, 1), (
        f"region root shape {tuple(value.shape)}: a keepdim reduction of {ROWS}x{COLS} is "
        f"({ROWS}, 1). A (rows,) shape would broadcast against the trailing dimension instead"
    )
    assert value.dtype == torch.float32
    assert tuple(value.stride()) == (1, 1), f"broadcast stride is {tuple(value.stride())}"


def test_lowered_graph_computes_numerically_close_log_softmax():
    gm, x = _lowered_logsoftmax()
    out = gm(x)
    out = out[0] if isinstance(out, (tuple, list)) else out
    want = _LogSoftmax()(x)
    assert out.shape == want.shape, f"{tuple(out.shape)} vs {tuple(want.shape)}"
    err = (out - want).abs().max().item()
    assert err < 1e-6, f"max|diff|={err:.3e}"


# ---------------------------------------------------------------- 2. the gate
def test_recognition_is_not_the_gate():
    """`lower_if_it_helps` gates on `would_rewrite`, and this is why the two differ.

    A region root with no shape metadata is RECOGNISED but must not be rewritten: whether the
    operator keeps the reduced dimension is then unknown, and guessing flips the broadcast. Gating
    the lowering on recognition alone committed a lowering that fused nothing -- more kernels than
    the operator it replaced.
    """
    gm, _x = _lowered_logsoftmax()
    root = NR.recognize_regions(gm)[0][0].region_root
    del root.meta["val"]

    from neptune_recognize.rewrite import region_contract

    # Test the contract directly: strict recognition may itself require metadata.
    from types import SimpleNamespace
    source = next(n for n in gm.graph.nodes if n.op == "placeholder")
    _, reason = region_contract(SimpleNamespace(input_node=source, region_root=root))
    assert "rank is unknown" in reason
    assert NR.would_rewrite(gm)[0] is False


def test_gate_rejects_a_lowering_that_does_not_rewrite():
    """The gate itself, tested against the bug it was written for.

    `lower_compound` here is replaced by a version that expands and then throws the metadata away
    -- exactly the first implementation. The expansion IS recognised, so a gate that asked only
    "does this now recognise?" would commit it and leave the model with more kernels than the
    operator it replaced.
    """
    from neptune_recognize import lowering as L

    real = L.lower_compound

    def stripped(gm, nodes=None):
        count = real(gm, nodes=nodes)
        for node in gm.graph.nodes:
            node.meta.pop("val", None)
        return count

    gm, _x = _export(_LogSoftmax())
    before = str(gm.graph)
    L.lower_compound = stripped
    try:
        assert L.lower_if_it_helps(gm) == 0, (
            "committed a lowering that enables recognition but not a rewrite")
    finally:
        L.lower_compound = real
    assert str(gm.graph) == before, "the graph was changed by a lowering that bought nothing"


def test_no_metadata_means_no_lowering():
    """A graph whose input has no fake value is left alone rather than lowered blind."""
    graph = fx.Graph()
    x = graph.placeholder("x")                       # no meta["val"]
    n = graph.call_function(torch.ops.aten._log_softmax.default, args=(x, -1))
    graph.output(n)
    gm = fx.GraphModule({}, graph)
    before = str(gm.graph)
    assert lower_if_it_helps(gm) == 0
    assert str(gm.graph) == before, "the graph was changed with nothing to gain"


# ---------------------------------------------------------------- 3. it is a no-op when it buys nothing
def test_softmax_is_not_lowered():
    """Measured 0.59x: Inductor's own lowering is a persistent reduction that beats our LSE."""
    gm, _x = _export(_Softmax())
    before = str(gm.graph)
    assert lower_if_it_helps(gm) == 0
    assert str(gm.graph) == before


def test_non_last_dim_is_not_lowered():
    """The frame emits one program per row, so a reduction over dim 0 is a different problem.

    Lowering it would change the graph and then be refused -- the worst of both.
    """
    gm, _x = _export(_LogSoftmaxDim0())
    before = str(gm.graph)
    assert lower_if_it_helps(gm) == 0
    assert str(gm.graph) == before


def test_unrelated_graph_is_not_lowered():
    gm, _x = _export(_Plain())
    before = str(gm.graph)
    assert lower_if_it_helps(gm) == 0
    assert str(gm.graph) == before


# ---------------------------------------------------------------- 4. semantics we do not implement
def test_half_to_float_is_refused():
    """`half_to_float=True` returns fp32 for an fp16 input: not `x - logsumexp(x)`."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.randn(ROWS, COLS)
    n = graph.call_function(torch.ops.aten._log_softmax.default, args=(x, -1, True))
    graph.output(n)
    gm = fx.GraphModule({}, graph)
    assert _half_to_float(n) is True
    assert lower_if_it_helps(gm) == 0, "half_to_float=True was lowered into a different function"
    assert _half_to_float(fx.Graph().call_function(
        torch.ops.aten._log_softmax.default, args=(x, -1, False))) is False
    assert _half_to_float(fx.Graph().call_function(
        torch.ops.aten._log_softmax.default, args=(x, -1))) is False


def test_simulatable_policy():
    assert _simulatable(None) is False
    assert _simulatable(torch.randn(8, 8)) is True              # tiny real tensor: cheap
    assert _simulatable(torch.randn(64, 4096)) is False         # would be real work
    assert _simulatable(torch.empty(64, 4096, device="meta")) is True


def _compound_graph(op, dtype=torch.float32, *, shape=(3, 17), dim=-1,
                    keepdim=False, output_dtype=None):
    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(shape, dtype=dtype, device="meta")
    if op == "lse":
        node = graph.call_function(torch.ops.aten.logsumexp.default,
                                   args=(x, [dim], keepdim))
        out_shape = shape[:-1] + ((1,) if keepdim else ()) if dim == -1 else shape
    else:
        target = (torch.ops.aten.log_softmax.int if op == "public"
                  else torch.ops.aten._log_softmax.default)
        node = graph.call_function(target, args=(x, dim, output_dtype if op == "public" else False))
        out_shape = shape
    node.meta["val"] = torch.empty(out_shape, dtype=output_dtype or dtype, device="meta")
    graph.output(node)
    return fx.GraphModule({}, graph)


def test_original_export_lse_and_dtype_boundaries():
    class LSE(nn.Module):
        def forward(self, x):
            return torch.logsumexp(x, [-1])

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(3, 17, dtype=dtype)
        gm = torch.export.export(LSE(), (x,)).graph_module
        old = next(n for n in gm.graph.nodes if _is_compound(n)).name
        assert lower_if_it_helps(gm) == 1
        matches, _ = NR.recognize_regions(gm)
        match = next(m for m in matches if m.region_root.meta.get("neptune_lowering") == old)
        assert NR.region_contract(match)[1] is None
        assert match.input_node.meta["val"].dtype == torch.float32
        assert match.region_root.meta["val"].dtype == torch.float32
        assert all(n.meta.get("neptune_lowering") == old for n in match.region_nodes)
        casts = [n for n in gm.graph.nodes if n.target == torch.ops.aten._to_copy.default]
        if dtype != torch.float32:
            assert len(casts) == 2
            assert match.input_node is casts[0]
            assert casts[-1].kwargs["dtype"] == dtype
            assert casts[-1] not in match.region_nodes
        actual = gm(x)[0]
        torch.testing.assert_close(actual, torch.logsumexp(x, -1),
                                   atol=0.04 if dtype == torch.bfloat16 else 0.004,
                                   rtol=0.008 if dtype == torch.bfloat16 else 0.001)
        assert actual.dtype == dtype


def test_compound_special_values_and_numerics():
    torch.manual_seed(41)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        finite = torch.randn(3, 17, dtype=dtype) * 4
        cases = [finite, torch.full_like(finite, float("-inf")),
                 torch.full_like(finite, float("inf")), finite.clone(), finite.clone()]
        cases[3][0, 0] = float("inf")
        cases[3][1, 0] = float("-inf")
        cases[4][0, 0] = float("nan")
        for op in ("lse", "log", "public"):
            for keepdim in ((False, True) if op == "lse" else (True,)):
                gm = _compound_graph(op, dtype, keepdim=keepdim)
                assert lower_compound(gm) == 1
                maximum = next(n for n in gm.graph.nodes if n.target == torch.ops.aten.amax.default)
                assert tuple(maximum.meta["val"].shape) == (3, 1)
                assert maximum.meta["val"].dtype == torch.float32
                for x in cases:
                    want = (torch.logsumexp(x, -1, keepdim=keepdim) if op == "lse"
                            else torch.log_softmax(x, -1))
                    actual = gm(x)
                    assert actual.dtype == want.dtype
                    assert torch.equal(torch.isnan(actual), torch.isnan(want))
                    assert torch.equal(torch.isposinf(actual), torch.isposinf(want))
                    assert torch.equal(torch.isneginf(actual), torch.isneginf(want))
                    torch.testing.assert_close(actual, want, equal_nan=True,
                                               atol=0.065 if dtype == torch.bfloat16 else 0.009,
                                               rtol=0.008 if dtype == torch.bfloat16 else 0.001)


def test_logsoftmax_math_is_not_bitwise_and_large_offsets_have_limits():
    # LSE readout loses the small log(s) when m is huge. Do not label real-algebra
    # matching a floating-point proof or silently promise arbitrary-input accuracy.
    gm = _compound_graph("log", shape=(3, 17))
    assert lower_compound(gm) == 1
    x = torch.full((3, 17), 1e8)
    assert (gm(x) - torch.log_softmax(x, -1)).abs().max() > 1
    for dtype in (torch.float16, torch.bfloat16):
        low = _compound_graph("log", dtype, shape=(3, 17))
        assert lower_compound(low) == 1
        x = torch.linspace(-2, 3, 51).reshape(3, 17).to(dtype)
        fp32_formula = (x.float() - torch.logsumexp(x.float(), -1, keepdim=True)).to(dtype)
        torch.testing.assert_close(low(x), fp32_formula, atol=0, rtol=0)
        assert not torch.equal(low(x), torch.log_softmax(x, -1))


def test_public_logsoftmax_dtype_argument_is_not_half_to_float():
    gm = _compound_graph("public", torch.float16, output_dtype=torch.float32)
    compound = next(n for n in gm.graph.nodes if _is_compound(n))
    assert not _half_to_float(compound)
    assert lower_if_it_helps(gm) == 1
    x = torch.randn(3, 17, dtype=torch.float16)
    torch.testing.assert_close(gm(x), torch.log_softmax(x, -1, dtype=torch.float32))


def test_per_node_gate_preserves_rejected_and_existing_regions():
    from neptune_recognize import lowering as L
    from neptune_recognize.match import recognize_regions as recognize

    class Mixed(nn.Module):
        def forward(self, x, y):
            return torch.log_softmax(x, -1), torch.logsumexp(y, -1), torch.softmax(y, -1)

    x, y = torch.randn(3, 17), torch.randn(3, 17)
    gm = torch.export.export(Mixed(), (x, y)).graph_module
    compounds = [n for n in gm.graph.nodes if _is_compound(n)]
    rejected, accepted = compounds[0].name, compounds[1].name
    original = {n.name: (n.op, n.target, n.args, dict(n.kwargs), dict(n.meta))
                for n in gm.graph.nodes}
    # Strip only the trial's rejected region root. Another region must not sponsor it.
    def selectively_refuse(trial, values=None):
        matches, reason = recognize(trial, values)
        for match in matches:
            if match.region_root.meta.get("neptune_lowering") == rejected:
                match.region_root.meta.pop("val", None)
        return matches, reason

    with patch("neptune_recognize.match.recognize_regions", selectively_refuse):
        assert L.lower_if_it_helps(gm) == 1
    assert any(n.name == rejected and _is_compound(n) for n in gm.graph.nodes)
    assert not any(n.name == accepted for n in gm.graph.nodes)
    assert not any(n.meta.get("neptune_lowering") == rejected for n in gm.graph.nodes)
    for n in gm.graph.nodes:
        if n.name in original and n.op != "output":
            op, target, args, kwargs, meta = original[n.name]
            assert (n.op, n.target, str(n.args), dict(n.kwargs)) == (op, target, str(args), kwargs)
            assert n.meta == meta
    # An already-eligible independent LSE must not prevent lowering the other node.
    assert L.lower_if_it_helps(gm) == 1
    for actual, want in zip(gm(x, y), Mixed()(x, y)):
        torch.testing.assert_close(actual, want)


def test_selected_expansion_and_unrelated_eligible_cannot_sponsor():
    class Pair(nn.Module):
        def forward(self, x, y):
            return torch.logsumexp(x, -1), torch.log_softmax(y, -1)

    gm = torch.export.export(Pair(), (torch.randn(3, 17), torch.randn(3, 17))).graph_module
    names = [n.name for n in gm.graph.nodes if _is_compound(n)]
    assert lower_compound(gm, nodes=names[0]) == 1
    before = (str(gm.graph), gm.code, gm.graph)
    real = NR.recognize_regions
    def unrelated_only(trial, values=None):
        matches, reason = real(trial, values)
        return [m for m in matches if m.region_root.meta.get("neptune_lowering") == names[0]], reason
    with patch("neptune_recognize.match.recognize_regions", unrelated_only):
        assert lower_if_it_helps(gm) == 0
    assert (str(gm.graph), gm.code, gm.graph) == before
    assert lower_compound(gm, nodes=[next(n for n in gm.graph.nodes if _is_compound(n))]) == 1


def test_exceptions_are_transactions():
    from neptune_recognize import lowering as L
    for path in ("_fake_values", "_resolve", "_commit"):
        gm = _compound_graph("lse")
        before = (str(gm.graph), gm.code, gm.graph)
        with patch.object(L, path, side_effect=RuntimeError("trial failure")):
            assert lower_compound(gm) == 0
            assert lower_if_it_helps(gm) == 0
        assert (str(gm.graph), gm.code, gm.graph) == before
    for path in ("neptune_recognize.match.recognize_regions",
                 "neptune_recognize.rewrite.region_contract",
                 "neptune_recognize.rewrite.would_rewrite"):
        gm = _compound_graph("lse")
        before = (str(gm.graph), gm.code, gm.graph)
        with patch(path, side_effect=RuntimeError("gate failure")):
            assert lower_if_it_helps(gm) == 0
        assert (str(gm.graph), gm.code, gm.graph) == before


def test_commit_compilation_exception_rolls_back():
    gm = _compound_graph("lse")
    before = (str(gm.graph), gm.code, gm.graph, gm.forward)
    with patch.object(gm, "recompile", side_effect=RuntimeError("commit failure")):
        assert lower_compound(gm) == 0
        assert (str(gm.graph), gm.code, gm.graph, gm.forward) == before
    assert gm.graph.owning_module is gm
    x = torch.randn(3, 17)
    torch.testing.assert_close(gm(x), torch.logsumexp(x, -1))


def test_required_metadata_dimensions_and_empty_width():
    for op in ("lse", "log", "public"):
        for shape, dim in (((3, 0), -1), ((3, 17), 0), ((), -1)):
            gm = _compound_graph(op, shape=shape, dim=dim)
            before = (str(gm.graph), gm.code, gm.graph)
            assert lower_compound(gm) == lower_if_it_helps(gm) == 0
            assert (str(gm.graph), gm.code, gm.graph) == before
        for missing in ("source", "output"):
            gm = _compound_graph(op)
            node = next(n for n in gm.graph.nodes if
                        (n.op == "placeholder" if missing == "source" else _is_compound(n)))
            node.meta.clear()
            before = (str(gm.graph), gm.code, gm.graph)
            assert lower_compound(gm) == lower_if_it_helps(gm) == 0
            assert (str(gm.graph), gm.code, gm.graph) == before
    gm = _compound_graph("lse")
    node = next(n for n in gm.graph.nodes if _is_compound(n))
    node.args = (node.args[0], [0, 1], False)
    assert lower_compound(gm) == 0


def test_fake_metadata_uses_torch_without_real_arithmetic():
    value = torch.randn(3, 17, dtype=torch.bfloat16).t().contiguous().t()
    result = _fake_values(value, -1, log_softmax=False, keepdim=False)
    assert all(v.is_meta for v in result.values() if isinstance(v, torch.Tensor))
    assert result["x"].dtype == result["lse"].dtype == torch.float32
    assert result["out"].dtype == torch.bfloat16
    assert result["out"].shape == (3,)
    assert result["m_safe"].shape == (3, 1)


def test_parameters_multiple_placeholders_and_opaque_ops_preserved():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(17, 17))

        def forward(self, x, residual, q, k, v):
            z = x @ self.weight + residual
            attention = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return torch.log_softmax(z, -1), attention

    model = Model().eval()
    args = (torch.randn(3, 17), torch.randn(3, 17),
            *(torch.randn(1, 1, 3, 17) for _ in range(3)))
    ep = torch.export.export(model, args).run_decompositions({})
    gm = ep.graph_module
    opaque = [(n.name, n.target, str(n.args)) for n in gm.graph.nodes
              if "matmul" in str(n.target) or "scaled_dot_product_attention" in str(n.target)]
    placeholders = [(n.name, n.target) for n in gm.graph.nodes if n.op == "placeholder"]
    assert len(opaque) == 2
    assert lower_if_it_helps(gm) == 1
    assert [(n.name, n.target, str(n.args)) for n in gm.graph.nodes
            if "matmul" in str(n.target) or "scaled_dot_product_attention" in str(n.target)] == opaque
    assert [(n.name, n.target) for n in gm.graph.nodes if n.op == "placeholder"] == placeholders
    for actual, want in zip(gm(model.weight, *args), model(*args)):
        torch.testing.assert_close(actual, want)
    # Also exercise get_attr parameters: graph trials must not deepcopy their storage.
    # A fresh unlifted program is used: gm above was already lowered in place.
    # Re-lowering that same graph is correctly a no-op.
    local = torch.export.export(model, args).run_decompositions({}).module()
    parameter = local.weight
    assert lower_if_it_helps(local) == 1
    assert local.weight is parameter


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            failed.append(name)
            print(f"  [FAIL] {name}: {e}")
        else:
            print(f"  [PASS] {name}")
    print(f"TOTAL: {len(tests) - len(failed)} passed, {len(failed)} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
