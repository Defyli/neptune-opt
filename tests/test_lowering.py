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

import torch
import torch.fx as fx
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neptune_recognize as NR  # noqa: E402
from neptune_recognize.lowering import (  # noqa: E402
    _half_to_float,
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
    ep = torch.export.export(module, (x,)).run_decompositions()
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
    root = NR.recognize(gm)[0].region_root
    value = root.meta.get("val")
    assert value is not None, "the lowered region root carries no metadata"
    assert tuple(value.shape) == (ROWS, 1), (
        f"region root shape {tuple(value.shape)}: a keepdim reduction of {ROWS}x{COLS} is "
        f"({ROWS}, 1). A (rows,) shape would broadcast against the trailing dimension instead"
    )
    assert value.dtype == torch.float32
    assert tuple(value.stride()) == (1, 1), f"broadcast stride is {tuple(value.stride())}"


def test_lowered_graph_computes_exactly_log_softmax():
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
    root = NR.recognize(gm)[0].region_root
    del root.meta["val"]

    assert NR.recognize(gm)[0] is not None, "recognition should still succeed without metadata"
    ok, why = NR.would_rewrite(gm)
    assert ok is False, "a region root with no output rank was accepted for rewriting"
    assert "rank is unknown" in why, f"unexpected refusal: {why!r}"


def test_gate_rejects_a_lowering_that_does_not_rewrite():
    """The gate itself, tested against the bug it was written for.

    `lower_compound` here is replaced by a version that expands and then throws the metadata away
    -- exactly the first implementation. The expansion IS recognised, so a gate that asked only
    "does this now recognise?" would commit it and leave the model with more kernels than the
    operator it replaced.
    """
    from neptune_recognize import lowering as L

    real = L.lower_compound

    def stripped(gm):
        count = real(gm)
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
