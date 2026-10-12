"""
Tests for the neptune_opt integration layer.

Most of these run on CPU with NO Triton installed, on purpose: the safety gates are the
part that must never be wrong, and they decide before any kernel is generated. A test suite
that can only run on the deployment machine is a test suite that gets skipped.

Run:  python tests/test_neptune_opt.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neptune_opt as NO  # noqa: E402
import neptune_recognize as NR  # noqa: E402

X = torch.randn(8, 32)


# ---------------------------------------------------------------- models
def _l2(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return (m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))).squeeze(-1)


def _l2_embedded(t):
    """The operator's result is combined with the data: NOT the whole graph.

    Replacing the graph output here would delete `+ t`, which is exactly the silent
    corruption the region gate exists to prevent.
    """
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    l2 = m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))
    return l2 + t


def _no_reduction(t):
    return torch.sin(t) * 2.0 + 1.0


def _two_inputs(a, b):
    return (a + b).sum(dim=-1)


class M(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


def _traced(fn, *args):
    from torch.fx.experimental.proxy_tensor import make_fx

    return make_fx(M(fn).eval())(*args)


# ---------------------------------------------------------------- tests
def test_region_root_is_not_the_graph_output():
    """The region must be the OPERATOR, not the whole graph.

    This replaces an earlier test that asserted embedded operators were REFUSED. That was the
    old, weaker design: once the tail is matched against the declared `finalize`, fusing an
    embedded operator is not merely possible but correct, because only the operator's own
    nodes are replaced. What must still hold is that the pass does not grab the output node,
    which would delete the surrounding computation.
    """
    gm = _traced(_l2_embedded, X)
    match, why = NR.recognize(gm)
    assert match is not None, why
    out = next(n for n in reversed(list(gm.graph.nodes)) if n.op == "output")
    leaves = []
    NR._flatten_nodes(out.args, leaves)
    assert match.region_root is not leaves[0], "region root must not be the graph output"


def test_region_gate_accepts_whole_graph_operator():
    """The positive control: when the operator IS the graph, the region gate passes.

    On a machine without Triton the rewrite stops at kernel generation, so the assertion is
    on the gate specifically, not on the whole rewrite succeeding.
    """
    gm = _traced(_l2, X)
    match, reason = NR.recognize(gm)
    assert match is not None, reason
    info = NR.interpret(gm)
    assert NR.region_root(gm, match, info[2]) is not None


def test_optimize_is_never_wrong():
    """Whatever optimize returns must be callable and agree with the original.

    On CPU (no Triton) it cannot fuse, so this checks the fallback; on the GPU the same
    assertion covers the fused result.
    """
    for fn in (_l2, _l2_embedded, _no_reduction):
        mod = M(fn).eval()
        opt, report = NO.optimize(mod, (X,))
        with torch.no_grad():
            a = mod(X)
            b = opt(X)
        assert a.shape == b.shape, f"{fn.__name__}: shape {b.shape} != {a.shape}"
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5), f"{fn.__name__} changed the result"
        assert isinstance(report.summary(), str)


def test_report_is_never_silent():
    """A pass that declines to fuse must say why; silence hides both bugs and missed work."""
    opt, report = NO.optimize(M(_no_reduction).eval(), (X,))
    assert not report.fusions
    assert report.skipped, "a refusal with no recorded reason is not acceptable"
    reason, detail = report.skipped[0]
    assert reason and detail


def test_multi_input_graph_is_refused():
    """The algebra is defined over ONE data symbol; a second input must not be guessed at."""
    a, b = torch.randn(8, 32), torch.randn(8, 32)
    gm = _traced(_two_inputs, a, b)
    match, reason = NR.recognize(gm)
    assert match is None, f"must not claim a fusion for a two-input graph: {match}"
    assert reason


def test_fusion_report_names_parameters():
    """The report must carry the values that were inferred, not just the operator name."""
    f = NO.Fusion("rmsnorm_max", {"N": 64.0, "eps": 1e-6})
    assert "N=64" in str(f) and "eps=1e-06" in str(f)


def _l2_scaled(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return ((m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))) * 2.0).squeeze(-1)


def test_region_root_excludes_surrounding_computation():
    """The region must stop at the operator's own `finalize`.

    Both cases here were WRONG before finalize matching existed: the guard only required the
    tail to be free of the data symbol, so it picked a node that also contained the model's
    own arithmetic. A rewrite would then have dropped it silently.
    """
    import sympy as sp

    from neptune_codegen import c_sym

    for fn, label in ((_l2_embedded, "data added after the operator"),
                      (_l2_scaled, "scaled after the operator")):
        gm = _traced(fn, X)
        match, why = NR.recognize(gm)
        assert match is not None, f"{label}: {why}"
        expr = NR.interpret(gm)[2][match.region_root]
        assert not expr.has(c_sym), f"{label}: region root still contains the data symbol"
        assert sp.Float(2.0) not in expr.atoms(sp.Float), f"{label}: region root contains the x2"


def test_embedded_operator_is_recognised():
    """An operator combined with other data must still be recognised (and fused)."""
    for fn in (_l2_embedded, _l2_scaled):
        gm = _traced(fn, X)
        match, why = NR.recognize(gm)
        assert match is not None, f"{fn.__name__}: {why}"
        assert match.skeleton == "stable_l2"


def test_every_public_emitter_is_callable():
    """Call each public emitter for every registered skeleton.

    Added after a refactor that split `neptune_codegen` into modules left one import out
    (`check_supported` in the template) and every generation silently raised NameError. A
    smoke test over the whole public surface is the cheapest net for that class of mistake.
    """
    import neptune_codegen as CG

    for name in CG.supported():
        sk = CG.SKELETONS[name]
        v = CG.param_defaults(name)
        CG.check_supported(sk)
        CG.repair_factor(sk)
        CG.term_expr(sk)
        CG.producer_operand(sk, v)
        CG.producer_init(sk)
        CG.consumer_init(sk)
        CG.guard_value(sk)
        CG.denominator_expr(sk)
        CG.producer_update(sk, v)
        CG.consumer_update(sk, v)
        CG.scale_source(sk, v)
        CG.term_source(sk, v)
        CG.finalize_source(sk, v)
        assert "def " + name + "(" in CG.emit_kernel_source(sk, v)


def _rmsnorm_max(t):
    r = (t * t).mean(dim=-1, keepdim=True)
    return (t / torch.sqrt(r + 1e-6)).amax(dim=-1)


def test_dynamic_shapes_are_supported():
    """A SYMBOLIC row length must still be recognised.

    This is precisely what used to fail: the row length was baked as a source literal, so with
    a symbolic shape there was no number to solve for and the operator was refused (measured:
    `param-unsolved [N]`). Declaring it intrinsic -- the kernel already receives `ncols` --
    removes the shape from the kernel entirely.

    `torch.export` + `dynamic_shapes` is used rather than a symbolic `make_fx` trace because
    it is the portable route: on torch 2.11 the symbolic make_fx trace raises
    "Expected Proxy, got [Proxy(arg0_1)]" while export works. The shape is asserted to be
    symbolic so this cannot silently degrade into re-testing the static case.
    """

    class Mod(torch.nn.Module):
        def forward(self, t):
            return _rmsnorm_max(t)

    try:
        ep = torch.export.export(Mod(), (X,),
                                 dynamic_shapes={"t": {1: torch.export.Dim("cols")}})
    except Exception as e:  # noqa: BLE001
        print(f"  [SKIP] dynamic export unavailable ({type(e).__name__}: {str(e)[:60]})")
        return
    shapes = [tuple(getattr(n.meta.get("val"), "shape", ()))
              for n in ep.graph_module.graph.nodes if n.op == "placeholder"]
    assert shapes and isinstance(shapes[0][-1], torch.SymInt), \
        f"the row length must be symbolic for this test to mean anything: {shapes}"
    match, why = NR.recognize(ep.graph_module)
    assert match is not None, why
    assert match.skeleton == "rmsnorm_max"
    assert set(match.params) == {"eps"}, f"only eps may be baked, got {match.params}"


def test_shape_coupled_kernel_is_shape_independent():
    """The generated kernel must not contain a row length."""
    import neptune_codegen as CG

    src = CG.emit_kernel_source(CG.SKELETONS["rmsnorm_max"], {"eps": 1e-6})
    assert "ncols" in src, "the row length must come from the kernel's own runtime argument"
    assert "1.0 / (32" not in src and "1.0 / (4096" not in src


def test_reduction_over_the_wrong_dim_is_refused():
    """The interpreter reads a reduction as the ROW reduction, so it must check the dim.

    Nothing verified this before. A `sum(dim=0)` would have been reinterpreted as a row
    reduction and fused into a kernel computing a different function.
    """
    def col_sum(t):
        return t.sum(dim=0)

    gm = _traced(col_sum, X)
    assert NR.interpret(gm) is None


def test_op_library_requires_only_undefaulted_parameters():
    """The explicit operator library must ask for what it needs and nothing more."""
    assert NO.op_parameters("rmsnorm_max") == ("eps",)
    assert NO.op_parameters("softcap_lse") == ("beta",)      # `tiny` has a declared default
    assert NO.op_parameters("stable_l2") == ()
    try:
        NO.rmsnorm_max(X)
    except TypeError as e:
        assert "eps" in str(e)
    else:
        raise AssertionError("a missing required parameter must be reported")


def test_apply_reports_instead_of_raising():
    """`apply` must never raise, and must always explain itself.

    On CPU it cannot generate a kernel (no Triton), so this exercises the failure gate; on the
    GPU the same call performs the fusion.
    """
    ep = torch.export.export(M(_l2).eval(), (X,))
    report = NO.apply(ep)
    assert isinstance(report.summary(), str)
    assert report.fusions or report.skipped, "a report must say something"


def test_export_forwards_kwargs_and_reports():
    """`export` is a drop-in for `torch.export.export`; extra kwargs must be forwarded."""
    ep = NO.export(M(_l2).eval(), (X,), strict=False)
    assert hasattr(ep, "graph_module")


def test_backend_refuses_unless_experimental():
    """The torch.compile backend is not verified effective; it must not be usable by accident."""
    try:
        NO.backend(None, [])
    except RuntimeError as e:
        assert "experimental" in str(e), e
        return
    raise AssertionError("neptune_opt.backend must refuse by default")


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed} passed, {failed} failed ({len(tests)} tests)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
