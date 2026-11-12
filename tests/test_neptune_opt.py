"""
Tests for the neptune_opt integration layer.

The transaction regressions mock kernel generation with CPU reference custom ops and
registered fake metadata, never launching a GPU or contacting a service. Historical
inline-fold source goldens below describe their correction to shared state semantics;
algebraic repair goldens and numerical bounds remain unchanged.

Most of these run on CPU with NO Triton installed, on purpose: the safety gates are the
part that must never be wrong, and they decide before any kernel is generated. A test suite
that can only run on the deployment machine is a test suite that gets skipped.

Run:  python tests/test_neptune_opt.py
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest
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
        if sk.two_pass:
            # No repair EXISTS for these -- that is what puts them in this plan -- so asking
            # for one is not a check that can pass, it is the wrong question. The plan's own
            # fold is what has to be callable.
            CG.consumer_fold_no_repair(sk, v)
        else:
            CG.repair_factor(sk)
            CG.repair_bias(sk)
            CG.repair_pair(sk)
            CG.scale_source(sk, v)
            CG.bias_source(sk, v)
            CG.bias_line(sk, v)
        CG.term_expr(sk)
        CG.producer_operand(sk, v)
        CG.producer_init(sk)
        CG.consumer_init(sk)
        CG.guard_value(sk)
        CG.denominator_expr(sk)
        CG.producer_update(sk, v)
        CG.term_source(sk, v)
        CG.finalize_source(sk, v)
        if not sk.two_pass:
            CG.consumer_update(sk, v)
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


def test_multiplicative_repairs_are_unchanged_by_the_affine_generalisation():
    """The affine repair must not disturb a single registered skeleton.

    Pinned as a golden table of the DERIVED slope, because the entire justification for
    replacing the `sp.solve` derivation was "same answers, wider reach". A refactor that
    silently changed a repair would change a kernel, and every GPU result recorded for this
    project is evidence about the kernels these expressions produce -- so a rewritten
    derivation has to be checked against what it replaced.
    """
    import neptune_codegen as CG

    golden = {
        "decay_renorm": "(N + s)/(N + s2)",
        "linear_attn": "s/s2",
        "logsumexp": "exp(s - s2)",
        "lse_biased": "exp(s - s2)",
        "rmsnorm_max": "sqrt(eps + s)/sqrt(eps + s2)",
        "softcap_lse": "exp(s - s2)",
        "stable_l2": "s**2/s2**2",
        "sumsq_div": "s/s2",
    }
    for name, want in golden.items():
        sk = CG.SKELETONS[name]
        got = str(CG.repair_factor(sk))
        assert got == want, f"{name}: repair slope changed from {want!r} to {got!r}"
        assert CG.repair_bias(sk) == 0, (
            f"{name}: gained a repair bias {CG.repair_bias(sk)}"
        )
        # A purely multiplicative repair must not emit the affine line at all, so no kernel
        # with existing GPU evidence gains a statement it was never measured with.
        src = CG.emit_kernel_source(sk, CG.param_defaults(name))
        # NB: test the emitted CODE, not the substring "bias" -- the lse_biased docstring
        # ("ALiBi-biased logsumexp") is copied into the generated header.
        assert "bias = tl.where" not in src, f"{name}: emitted an unexpected bias line"
        assert "+ bias" not in src, f"{name}: emitted an unexpected bias term"
        # Historical inline-fold golden is obsolete: serial now shares the certified
        # chunk/merge state with split. Preserve the repair expressions above, while
        # requiring explicit occupancy/status and eager-safe transport, not m == 0.
        assert f"def {name}_chunk(" in src and f"def {name}_merge_state(" in src
        assert "m, acc, seen, status = " + name + "_merge_state(" in src
        assert "both = (vA != 0) & (vB != 0)" in src
        assert "oldA = tl.where(both, mA, 1.0)" in src
        assert "oldB = tl.where(both, mB, 1.0)" in src
        assert "fresh = tl.where(both, combined_m, 1.0)" in src
        assert "safeA = tl.where(both, aA, 0.0)" in src
        assert "safeB = tl.where(both, aB, 0.0)" in src
        assert "return m, acc, vA | vB, statusA | statusB" in src
        assert "m ==" not in src

    src = CG.emit_kernel_source(CG.SKELETONS["rmsnorm_max"], {"eps": 1e-6})
    assert "combined_a = tl.maximum(safeA * (" in src, src
    assert "safeB * (" in src and "1e-30" not in src, src


def test_affine_repair_is_derived_for_a_max_consumer():
    """`g = c + s` needs a repair that a pure rescale cannot express.

    The ratio (c+s')/(c+s) keeps the data, so multiplicative separation fails; matching the
    c-derivatives of the two sides of condition (3) gives a = 1 and b = s' - s, and a max
    consumer admits an affine repair because h(t) = t + b is a max-morphism.
    """
    import neptune_codegen as CG
    from neptune_codegen.symbols import c_sym, s_sym, s2_sym  # noqa: F401

    sk = CG.Skeleton(name="affine_plus", g=c_sym + s_sym, finalize=CG._FINAL_ACC,
                     producer="signed", producer_reducer="max", consumer_reducer="max")
    a, b = CG.repair_pair(sk)
    assert a == 1, f"slope should be 1, got {a}"
    assert CG._simplify(b - (s2_sym - s_sym)) == 0, f"bias should be s'-s, got {b}"
    assert "+ bias" in CG.consumer_update(sk, {}), CG.consumer_update(sk, {})
    assert CG.bias_line(sk, {}) != "", "bias must be emitted for an affine repair"
    # and the bias must be zeroed on the first block, or the identity is broken there
    # A legitimate producer zero is not an empty-state sentinel.
    assert "bias = tl.where(seen, " in CG.bias_line(sk, {})
    assert "m ==" not in CG.bias_line(sk, {})


def test_bias_is_refused_for_an_additive_consumer():
    """Condition (4) is the reason a bias needs a max consumer, and it must be enforced.

    For f = x+y, `a*(x+y) + b == (a*x + b) + (a*y + b)` only holds when b == 0. Refusing
    here is what stops the generator from emitting a fold that does not compute f.
    """
    import neptune_codegen as CG
    from neptune_codegen.symbols import c_sym, s_sym, s2_sym  # noqa: F401

    sk = CG.Skeleton(name="affine_plus_sum", g=c_sym + s_sym, finalize=CG._FINAL_ACC,
                     producer="signed", producer_reducer="max", consumer_reducer="sum")
    try:
        CG.repair_pair(sk)
    except NotImplementedError as e:
        assert "condition 4" in str(e), e
        assert "b = 0" in str(e), e
        return
    raise AssertionError("a nonzero bias must be refused for an additive consumer")


def test_affine_fold_matches_the_reference_numerically():
    """Replay the emitted recurrence and compare with the row-wise reference.

    The generated kernel cannot run here (no Triton), so this validates the ALGEBRA the
    kernel is generated FROM: the recurrence below is transcribed from the template's
    `acc = fold(acc*scale + bias, block_reduce(term))`, with scale/bias/term taken from the
    real emitters. `g = c + s` with a max producer makes the reference exactly 2*max(c),
    which is non-degenerate -- unlike the additive-shift control family, whose value is
    identically zero and would pass a broken fold as easily as a correct one.
    """
    import numpy as np
    import sympy as sp
    import neptune_codegen as CG
    from neptune_codegen.symbols import c_sym, s_sym, s2_sym  # noqa: F401

    from neptune_codegen.symbols import _PARAM_N

    sk = CG.Skeleton(name="affine_plus", g=_PARAM_N * c_sym + s_sym,
                     finalize=CG._FINAL_ACC, producer="signed",
                     producer_reducer="max", consumer_reducer="max",
                     params=((_PARAM_N, "k"),))
    a, b = CG.repair_pair(sk)
    a_f = sp.lambdify((s_sym, s2_sym), a, "numpy")
    b_f = sp.lambdify((s_sym, s2_sym), b, "numpy")

    def replay(x, block, k):
        rows, ncols = x.shape
        out = np.empty(rows)
        for r in range(rows):
            m, acc = -np.inf, -np.inf
            for off in range(0, ncols, block):
                blk = x[r, off:off + block]
                m_new = max(m, blk.max())
                den = m_new                     # raw: nothing divides by the producer
                scale = 1.0 if m == -np.inf else float(a_f(m, den))
                bias = 0.0 if m == -np.inf else float(b_f(m, den))
                acc = max(acc * scale + bias, (k * blk + den).max())
                m = m_new
            out[r] = acc
        return out

    rng = np.random.default_rng(0)
    # The reference is `max_j(k*c_j) + max c`, and it must be written that way: the shorter
    # `(1+k)*max c` is correct only for k > 0, because max_j(k*c_j) selects the MAXIMUM of c
    # for positive k and the MINIMUM for negative k. Both signs are replayed here, since a
    # reference that is right for the sign one happens to test is the failure mode this
    # project keeps running into (a full GPU run was spent on exactly this).
    for k in (1.0, 2.0, -0.5, -3.0):
        for shape, block in (((5, 7), 3), ((4, 10), 4), ((3, 8), 8), ((6, 9), 2)):
            x = rng.normal(size=shape)
            got = replay(x, block, k)
            want = (k * x).max(axis=1) + x.max(axis=1)
            err = np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-12))
            assert err < 1e-12, (
                f"k={k} shape={shape} block={block}: max rel err {err:.3e}"
            )


def test_producer_floor_is_applied_only_where_the_source_divides():
    """The floor on the fresh producer value is a division guard, not a convention.

    It used to be keyed off `is_log`, which happens to be right for all eight registered
    skeletons but is the wrong test: an additive fold never divides, and flooring its
    producer is not a 1e-30 nicety -- for `g = c - s` on the all-negative row [-5, -1, -3]
    the floored producer is 1e-30, so the emitted term becomes `v - 1e-30` and the kernel
    returns -1 where the operator is identically 0.
    """
    import numpy as np
    import neptune_codegen as CG
    from neptune_codegen.symbols import c_sym, s_sym, s2_sym  # noqa: F401

    # Historical 'any denominator mentions s' predicate over-floored RMS: positive
    # eps already removes the zero singularity. A bare ratio still needs its floor.
    assert CG.denominator_expr(CG.SKELETONS["rmsnorm_max"]) == "m_new"
    assert CG.denominator_expr(CG.SKELETONS["linear_attn"]) == "tl.maximum(m_new, 1e-30)"
    # no division by the producer -> raw (the exponential folds, and now the affine ones)
    assert CG.denominator_expr(CG.SKELETONS["logsumexp"]) == "m_new"
    shift = CG.Skeleton(name="shift", g=c_sym - s_sym, finalize=CG._FINAL_ACC,
                        producer="signed", producer_reducer="max", consumer_reducer="max")
    assert CG.denominator_expr(shift) == "m_new"

    # The error the floor would introduce, replayed over the emitted recurrence.
    x = np.array([-5.0, -1.0, -3.0])
    a, b = CG.repair_pair(shift)

    def fold(floor):
        m, acc = -np.inf, -np.inf
        for c in x:
            m_new = max(m, c)
            den = max(m_new, 1e-30) if floor else m_new
            scale = 1.0 if m == -np.inf else float(a)
            bias = 0.0 if m == -np.inf else float(b.subs({s_sym: m, s2_sym: den}))
            acc = max(acc * scale + bias, c - den)
            m = m_new
        return acc

    assert abs(fold(floor=False)) < 1e-12, "the raw producer must reproduce the reference"
    # Measured -2.0 against a reference of exactly 0.0: an error of the order of the row's
    # own magnitude, which is what makes this a correctness bug rather than a rounding one.
    assert abs(fold(floor=True) - (-2.0)) < 1e-9, (
        "the floor is expected to break this family by ~|max|; if this changed, the "
        "justification for the per-skeleton predicate needs revisiting"
    )


def test_two_pass_plan_emits_two_loops_and_no_repair():
    """The fallback plan must look like a fallback plan.

    These operators have NO repair -- `repair_pair` refuses them, and the refusal is the
    reason the plan exists -- so the emitted kernel must contain no repair statement at all.
    Two loops, one producer finalize, and the term reading that finalized value.
    """
    import neptune_codegen as CG

    for name in ("variance", "std"):
        sk = CG.SKELETONS[name]
        assert sk.two_pass, f"{name} should be declared two_pass"
        # The repair genuinely does not exist; this is the premise of the whole plan.
        try:
            CG.repair_pair(sk)
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{name}: a repair exists, so it should not need two passes")
        src = CG.emit_kernel_source(sk, {})
        assert src.count("for off in range(0, ncols, BLOCK):") == 2, src
        # Test the emitted STATEMENTS, not the substrings: the std docstring says
        # "unbiased=False", so a bare `"bias" not in src` fails on prose.
        assert "scale = tl.where" not in src, "a two-pass kernel must not carry a repair"
        assert "bias = tl.where" not in src and "+ bias" not in src, src
        # The producer is complete before pass 2 starts and never moves again. The mean's
        # 1/N is inside pass 1's OPERAND (v * (1/ncols)) rather than applied afterwards,
        # because that is how the graph spells a mean and `producer_expr` is what the
        # recogniser compares against -- see the declaration in neptune_codegen/library.py.
        lines = src.splitlines()
        pass1 = next(i for i, line in enumerate(lines) if "pass 1" in line)
        den = next(i for i, line in enumerate(lines) if line.strip() == "den = m")
        assert "v * ((1.0 / (ncols)))" in src, src
        assert pass1 < den, src
        assert "den = m_new" not in src, src
        # No floor: a finalized producer is not a running one (see _needs_denominator_floor).
        assert "tl.maximum(m_new" not in src, src


def test_two_pass_matches_torch_variance_numerically():
    """Replay the emitted two-pass recurrence and compare with torch's own var/std.

    The kernel cannot run here (no Triton), so this checks the ARITHMETIC the kernel is
    generated from, transcribed from the template: finalize the producer, then accumulate
    (v - den)^2 over the whole row and apply the readout. The reference is `torch.var`/
    `torch.std` rather than a hand-written mean-centre-square, because an independent
    implementation is the only kind that can catch a self-consistent misreading.
    """
    import numpy as np
    import torch
    import neptune_codegen as CG

    rng = np.random.default_rng(0)
    # variance's readout is the identity; std's is the square root
    for name, readout in (("variance", lambda a, n: a), ("std", lambda a, n: np.sqrt(a))):
        for shape, block in (((5, 7), 3), ((4, 10), 4), ((3, 8), 8), ((6, 9), 2), ((2, 33), 16)):
            x = rng.normal(size=shape) * 3.0 + 1.0
            rows, ncols = shape
            got = np.empty(rows)
            for r in range(rows):
                # pass 1: the producer. The 1/N is INSIDE the operand, mirroring the emitted
                # source (`v * (1.0 / ncols)`) -- not applied afterwards -- because that is the
                # form the recogniser compares against a graph.
                m = 0.0
                for off in range(0, ncols, block):
                    blk = x[r, off:off + block]
                    m = m + (blk / ncols).sum()
                # pass 2: the consumer, with the producer constant and the same 1/N in the term
                acc = 0.0
                for off in range(0, ncols, block):
                    blk = x[r, off:off + block]
                    acc = acc + (((blk - m) ** 2) / ncols).sum()
                # `finalize` is the IDENTITY for both skeletons; `std`'s readout is the root
                got[r] = readout(acc, ncols)
            want = CG.reference(name, torch.tensor(x, dtype=torch.float32)).numpy()
            err = np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-6))
            assert err < 1e-6, f"{name} shape={shape} block={block}: rel err {err:.3e}"


def test_two_pass_refuses_a_term_that_ignores_the_producer():
    """Pass 1 must not be dead work, or the kernel reduces a producer nothing reads.

    `g = exp(c - |c|)` is the coverage study's out-of-framework control, and in the two-pass
    plan it separates TRIVIALLY (the term simply does not mention the producer), so nothing
    else stops it. This guard was silently dropped once, while the enclosing block was rewritten
    to remove a redundant field; the coverage probe caught it, and this pins it in the test suite
    so the next time does not depend on running that probe.
    """
    import sympy as sp
    import neptune_codegen as CG
    from neptune_codegen.symbols import _PARAM_N, c_sym

    dead = CG.Skeleton(name="dead_pass", g=sp.exp(c_sym - sp.Abs(c_sym)),
                       finalize=CG._FINAL_ACC, producer="signed", producer_reducer="sum",
                       two_pass=True, producer_expr=c_sym / _PARAM_N,
                       intrinsic_params=((_PARAM_N, "ncols"),))
    try:
        CG.check_supported(dead)
    except NotImplementedError as e:
        assert "dead work" in str(e), e
    else:
        raise AssertionError("a two-pass term that ignores its producer must be refused")

    # ...and the same expression is refused in the one-pass frame, for its own reason
    try:
        CG.repair_pair(dead)
    except NotImplementedError as e:
        assert "producer s" in str(e), e
    else:
        raise AssertionError("g ignoring the producer must be refused in the one-pass frame too")


def test_two_pass_declares_the_mean_the_way_the_graph_spells_it():
    """The declaration IS the recogniser's input, so a private spelling makes it unmatched.

    `torch.mean` is a sum over ncols, so the recogniser sees a mean as the OPERAND `c / N`.
    A skeleton may express the same operator with the 1/N applied afterwards instead --
    algebraically identical, and it was the first version -- but then `producer_expr` no
    longer equals what the graph reduces, and recognition silently fails. This pins the
    convention for both two-pass skeletons.
    """
    import neptune_codegen as CG
    from neptune_codegen.symbols import _PARAM_N, c_sym

    for name in ("variance", "std"):
        sk = CG.SKELETONS[name]
        assert sk.two_pass, f"{name} should be two_pass"
        assert sk.producer_expr == c_sym / _PARAM_N, (
            f"{name}: the producer operand must be the mean as the graph spells it "
            f"(c / N), got {sk.producer_expr}"
        )
        assert sk.finalize in (CG._FINAL_ACC, __import__("sympy").sqrt(CG._FINAL_ACC)), (
            f"{name}: the 1/N belongs in the term, not the readout; got {sk.finalize}"
        )
        # no producer_finalize step exists any more: it was redundant AND unrecognisable
        assert not hasattr(sk, "producer_finalize")


def test_all_registered_skeletons_have_a_usable_reference():
    """Every supported skeleton needs a reference, or the GPU suite silently cannot test it.

    Adding a skeleton to the registry automatically puts it in `CG.supported()`, into the
    explicit operator library, and into the GPU correctness sweep -- where a missing reference
    shows up as a KeyError far from the declaration that caused it.
    """
    import torch
    import neptune_codegen as CG

    x = torch.randn(4, 16)
    for name in CG.supported():
        for fn in (CG.reference, CG.graph_reference):
            try:
                out = fn(name, x, **CG.param_defaults(name))
            except KeyError as e:
                raise AssertionError(f"{name}: {fn.__name__} is not implemented ({e})") from None
            assert out.shape == (4,), f"{name}: {fn.__name__} returned {out.shape}"


def test_intrinsics_under_a_root_are_read_as_floats():
    """A shape value under `tl.sqrt` is a Triton COMPILE error, and it is easy to write.

    `std`'s readout is `sqrt(acc / ncols)`, but sympy stores that as `sqrt(acc) / sqrt(ncols)`,
    so an INTEGER shape value ends up under a square root -- Triton rejects it with
    "'math.sqrt' op operand #0 must be floating-point-like, but got 'i32'". The offender is not
    visible in the declaration, which is why this is checked on the emitted source.
    """
    import sympy as sp
    import neptune_codegen as CG
    from neptune_codegen.symbols import _FINAL_ACC, _PARAM_N, c_sym, s_sym

    # A skeleton whose READOUT takes a root of the row length. `std` used to look like this and
    # then stopped: writing every 1/N where the graph puts it (so the operator stays
    # recognisable) moved the division into the operand, leaving `sqrt(acc)`. The guard is
    # still needed -- it was found by exactly this shape -- so it is pinned with a synthetic
    # skeleton rather than left to whichever real operator happens to take a root of a shape.
    rooted = CG.Skeleton(name="rooted", g=(c_sym - s_sym) ** 2,
                         finalize=_FINAL_ACC / sp.sqrt(_PARAM_N),
                         producer="signed", producer_reducer="sum", two_pass=True,
                         producer_expr=c_sym, intrinsic_params=((_PARAM_N, "ncols"),))
    rooted_src = CG.emit_kernel_source(rooted, {})
    assert "tl.sqrt((1.0 * ncols))" in rooted_src, rooted_src
    assert "tl.sqrt(ncols)" not in rooted_src, rooted_src
    # `std` no longer takes a root of ncols at all, and must not gain a stray cast.
    std_src = CG.emit_kernel_source(CG.SKELETONS["std"], {})
    assert "1.0 * ncols" not in std_src, std_src
    assert "tl.sqrt(ncols)" not in std_src, std_src
    # A plain division needs no cast (Triton promotes it), so it must not silently gain one:
    # the cast is a targeted fix, not a blanket rule.
    var_src = CG.emit_kernel_source(CG.SKELETONS["variance"], {})
    assert "(1.0 / (ncols))" in var_src, var_src
    assert "1.0 * ncols" not in var_src, var_src
    assert "(1.0 / (ncols))" in CG.emit_kernel_source(CG.SKELETONS["rmsnorm_max"], {"eps": 1e-6})


def test_accumulation_stays_fp32_whatever_the_input_dtype():
    """The dtype support narrows the RESULT; the ACCUMULATION must stay fp32.

    Accumulating in bf16 would be a plausible "optimisation" -- half the register pressure -- and
    it would lose precision catastrophically on long rows, because summing thousands of bf16
    values has an effective precision far below even bf16. The emitted source must cast at the
    load, for every skeleton, and that is checkable without a GPU.
    """
    import neptune_codegen as CG

    for name in CG.supported():
        src = CG.emit_kernel_source(CG.SKELETONS[name], CG.param_defaults(name))
        assert ".to(tl.float32)" in src, f"{name}: the load must cast to fp32"
        # nothing may narrow the VALUE before or during the accumulation
        assert ".to(tl.bfloat16)" not in src, f"{name}: narrows the value before accumulating"
        assert ".to(tl.float16)" not in src, f"{name}: narrows the value before accumulating"


def test_dtype_tolerances_are_ordered_and_justified():
    """A bf16 result cannot be judged by an fp32 bound -- that is the whole point.

    The bounds are a few ulps of the dtype, so they must be ordered bf16 > fp16 > fp32, and the
    bf16 one must be large enough to accept a correct bf16 result (its eps is 2**-8).
    """
    import torch
    from neptune_codegen.references import rel_tolerance

    t32, tbf, t16 = (rel_tolerance(d) for d in
                     (torch.float32, torch.bfloat16, torch.float16))
    assert t32 < t16 < tbf, (t32, t16, tbf)
    assert tbf >= float(torch.finfo(torch.bfloat16).eps), tbf
    # ...and still tight enough to catch a wrong formula, which is off by O(1).
    assert tbf < 0.1, tbf


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


# Merge tests are intentionally appended: run with pytest (the legacy script main
# above runs before these definitions). No Triton import, compilation, or GPU launch.
def test_merge_exact_certificates_for_nine_single_pass_skeletons():
    import neptune_codegen as CG
    singles = [sk for sk in CG.SKELETONS.values() if not sk.two_pass]
    assert len(singles) == 9
    for sk in singles:
        cert = CG.derive_merge_certificate(sk, CG.param_defaults(sk.name))
        assert cert.proof_kind == "exact-symbolic"
        assert cert.requires_finite
        assert cert.assumptions
    assert CG.derive_merge_certificate(CG.SKELETONS["rmsnorm_max"], {"eps": 1e-6}).producer_domain == "nonnegative"


def test_merge_refuses_two_pass_unknown_reducers_and_unproved_signed_max():
    import dataclasses
    import pytest
    import sympy as sp
    import neptune_codegen as CG
    for name in ("variance", "std"):
        with pytest.raises(NotImplementedError, match="two_pass"):
            CG.emit_kernel_source(CG.SKELETONS[name], {}, plan="split")
    base = CG.SKELETONS["linear_attn"]
    for field in ("producer_reducer", "consumer_reducer"):
        with pytest.raises(NotImplementedError, match="unknown"):
            CG.derive_merge_certificate(dataclasses.replace(base, **{field: "mul"}))
    signed = dataclasses.replace(base, producer="signed", consumer_reducer="max")
    with pytest.raises(NotImplementedError, match="strictly positive"):
        CG.derive_merge_certificate(signed)
    false_domain = dataclasses.replace(signed, producer_domain="nonnegative")
    with pytest.raises(NotImplementedError, match="not proved"):
        CG.derive_merge_certificate(false_domain)
    shifted = CG.Skeleton("affine_max", CG.c_sym - CG.s_sym, CG._FINAL_ACC,
                          producer="signed", consumer_reducer="max")
    cert = CG.derive_merge_certificate(shifted)
    assert cert.factor == 1 and sp.simplify(cert.bias - CG.s_sym + CG.s2_sym) == 0


def test_merge_certificate_never_uses_numeric_proof_and_checks_params(monkeypatch):
    import pytest
    import neptune_codegen as CG
    import neptune_codegen.algebra as algebra
    # Even the legacy derivation's validation must stay on its symbolic branch here.
    def forbidden(*args, **kwargs):
        raise AssertionError("numeric sampling is not a proof")
    monkeypatch.setattr(algebra, "_numerically_zero", forbidden)
    for sk in CG.SKELETONS.values():
        if not sk.two_pass:
            CG.derive_merge_certificate(sk, CG.param_defaults(sk.name))
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(NotImplementedError):
            CG.derive_merge_certificate(CG.SKELETONS["rmsnorm_max"], {"eps": value})


def test_merge_floor_coordinates_only_for_zero_singular_nonnegative_max():
    import neptune_codegen as CG
    for name in ("stable_l2", "linear_attn", "sumsq_div"):
        sk = CG.SKELETONS[name]
        assert CG.coordinate_needs_floor(sk)
        src = CG.emit_kernel_source(sk, CG.param_defaults(name), plan="split")
        assert "qA = tl.maximum(oldA, 1e-30)" in src
        assert "qB = tl.maximum(oldB, 1e-30)" in src
        assert "qNew = tl.maximum(fresh, 1e-30)" in src
    for name in ("decay_renorm", "rmsnorm_max", "logsumexp"):
        sk = CG.SKELETONS[name]
        assert not CG.coordinate_needs_floor(sk)
        assert "1e-30" not in CG.emit_kernel_source(sk, CG.param_defaults(name))
    assert "tl.maximum(m, 1e-30)" in CG.finalize_source(CG.SKELETONS["stable_l2"])


def _merge_cpu_generated_helpers(sk, params, width, block):
    """Execute the actual generated chunk/combiner source against NumPy tl primitives.

    This validates source semantics, not Triton typing/codegen or GPU numerics.
    """
    import ast
    import types
    import numpy as np
    import neptune_codegen as CG

    class Pointer:
        def __init__(self, data, offset=0):
            self.data, self.offset = data, offset
        def __add__(self, offset):
            return Pointer(self.data, self.offset + offset)

    def load(ptr, mask, other):
        offsets = np.asarray(ptr.offset)
        safe = np.where(mask, offsets, 0)
        return np.where(mask, ptr.data[safe], other)

    tl = types.SimpleNamespace(
        float32=np.float32, int32=np.int32, constexpr=int,
        arange=np.arange, load=load, where=np.where, abs=np.abs,
        maximum=np.maximum, minimum=np.minimum, sqrt=np.sqrt,
        log=np.log, exp=np.exp, sigmoid=lambda x: 1 / (1 + np.exp(-x)),
        sum=lambda x, axis: np.asarray(np.sum(x, axis=axis)),
        max=lambda x, axis: np.asarray(np.max(x, axis=axis)),
        full=lambda shape, value, dtype: np.full(shape, value, dtype),
    )
    # Triton tensor.to is NumPy's astype for this CPU semantic replay.
    tree = ast.parse(CG.emit_state_helpers(sk, params))
    class ToNumpy(ast.NodeTransformer):
        def visit_Attribute(self, node):
            self.generic_visit(node)
            if node.attr == "to":
                node.attr = "astype"
            return node
    tree = ToNumpy().visit(tree)
    for fn in tree.body:
        fn.decorator_list = []
        for arg in fn.args.args:
            arg.annotation = None
    namespace = {"tl": tl, "ncols": width}
    exec(compile(ast.fix_missing_locations(tree), "<merge-source-cpu>", "exec"), namespace)
    chunk = namespace[sk.name + "_chunk"]
    merge = namespace[sk.name + "_merge_state"]
    def partial(data, off):
        return chunk(Pointer(np.asarray(data, np.float32)), 0, off, width, width, 1, block)
    return partial, merge


def test_merge_generated_source_partition_tree_zero_floor_and_global_n():
    import numpy as np
    import neptune_codegen as CG
    for name, sk in CG.SKELETONS.items():
        if sk.two_pass:
            continue
        data = np.array([-3., 0., 2., 1., -1., 0., 5.], np.float32)
        params = CG.param_defaults(name)
        partial, merge = _merge_cpu_generated_helpers(sk, params, len(data), 2)
        states = [partial(data, off) for off in range(0, len(data), 2)]
        left = states[0]
        for state in states[1:]:
            left = merge(*left, *state)
        balanced = merge(*merge(*states[0], *states[1]), *merge(*states[2], *states[3]))
        whole, _ = _merge_cpu_generated_helpers(sk, params, len(data), 8)
        expected = whole(data, 0)
        np.testing.assert_allclose(left[:2], expected[:2], rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(balanced[:2], expected[:2], rtol=2e-6, atol=1e-6)
        empty = (np.float32(0), np.float32(0), np.int32(0), np.int32(0))
        np.testing.assert_allclose(merge(*empty, *left)[:2], left[:2])
        np.testing.assert_allclose(merge(*left, *empty)[:2], left[:2])
    for name in ("stable_l2", "linear_attn", "sumsq_div"):
        sk = CG.SKELETONS[name]
        partial, merge = _merge_cpu_generated_helpers(sk, {}, 3, 1)
        data = np.array([0., 0.5e-30, 0.5e-30], np.float32)
        states = [partial(data, i) for i in range(3)]
        merged = merge(*merge(*states[0], *states[1]), *states[2])
        whole, _ = _merge_cpu_generated_helpers(sk, {}, 3, 4)
        np.testing.assert_allclose(merged[:2], whole(data, 0)[:2], rtol=2e-6, atol=0)


def test_merge_log_nonfinite_status_and_unclipped_transport():
    import numpy as np
    import neptune_codegen as CG
    sk = CG.SKELETONS["logsumexp"]
    for data, status, seen in (([-np.inf, -np.inf], 0, 0),
                                ([np.inf, 1.], 2, 1),
                                ([np.nan, np.inf], 3, 0),
                                ([-np.inf, 1.], 0, 1)):
        partial, merge = _merge_cpu_generated_helpers(sk, {"tiny": 0.0}, 2, 1)
        with np.errstate(invalid="raise", divide="raise"):
            result = merge(*partial(data, 0), *partial(data, 1))
        assert result[2] == seen and result[3] == status
        assert not np.isnan(result[1])
    source = CG.emit_kernel_source(sk, {"tiny": 0.0}, plan="split")
    assert "safe_phi" in source and "finite_mask" in source
    assert "-80.0" not in source and "80.0" not in source
    assert "(status & 1)" in source and "(status & 2)" in source


def test_merge_plan_capacity_dynamic_width_and_source_identity():
    import ast
    import pytest
    import neptune_codegen as CG
    assert CG.split_launch_shape(1025, 1024) == (2, 2)
    assert CG.split_launch_shape(4096, 1) == (4096, 4096)
    with pytest.raises(NotImplementedError, match="capacity"):
        CG.split_launch_shape(4097, 1)
    class SymbolicWidth:
        def __int__(self):
            raise AssertionError("must not concretise symbolic width")
    with pytest.raises(NotImplementedError, match="concrete width"):
        CG.split_launch_shape(SymbolicWidth(), 1024)
    with pytest.raises(ValueError, match="power of two"):
        CG.split_launch_shape(32, 3)
    for sk in CG.SKELETONS.values():
        params = CG.param_defaults(sk.name)
        ast.parse(CG.emit_kernel_source(sk, params))
        if not sk.two_pass:
            src = CG.emit_kernel_source(sk, params, plan="split")
            ast.parse(src)
            assert "tl.reduce((m, acc, seen, status)" in src
            assert "idx * stride_col" in src
            assert "row * stride_row" in src
            assert "m ==" not in src
            assert "plan='split'" in src
    p1, p2 = {"tiny": 1.23456701}, {"tiny": 1.23456702}
    assert CG.generated_path("logsumexp", p1) != CG.generated_path("logsumexp", p2)
    assert CG.generated_path("logsumexp", p1) != CG.generated_path("logsumexp", p1, plan="split")


def test_merge_log_finalizer_whole_math_nonfinite_and_large_range():
    import math
    import numpy as np
    import neptune_codegen as CG
    from neptune_codegen.emitters import state_finalize_source
    import types
    tl = types.SimpleNamespace(where=np.where, log=np.log, maximum=np.maximum)
    for name, sk in CG.SKELETONS.items():
        if not sk.is_log:
            continue
        params = CG.param_defaults(name)
        data_cases = ([np.nan, np.inf, 1.], [np.inf, -np.inf, 1.],
                      [-np.inf, -np.inf, -np.inf], [0., -100., -200.])
        for data in data_cases:
            partial, merge = _merge_cpu_generated_helpers(sk, params, 3, 1)
            with np.errstate(invalid="ignore", over="ignore", divide="raise"):
                states = [partial(data, i) for i in range(3)]
                m, acc, seen, status = merge(*merge(*states[0], *states[1]), *states[2])
                result = eval(state_finalize_source(sk, params), {"tl": tl},
                              {"m": m, "acc": acc, "seen": seen, "status": status})
            x = torch.tensor([data], dtype=torch.float32)
            expected = float(CG.reference(name, x, **params)[0])
            if math.isnan(expected):
                assert math.isnan(float(result))
            elif math.isinf(expected):
                assert float(result) == expected
            else:
                np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-6)


def test_merge_normalizer_singular_domain_is_not_certified():
    import pytest
    import sympy as sp
    import neptune_codegen as CG
    # A real signed sum can hit zero; no amount of positive-only sampling proves it safe.
    sk = CG.Skeleton("signed_sum_ratio", CG.c_sym / CG.s_sym, CG._FINAL_ACC,
                     producer="signed", producer_reducer="sum", producer_expr=CG.c_sym)
    with pytest.raises(NotImplementedError, match="nonzero"):
        CG.derive_merge_certificate(sk)
    # A nonnegative sum also cannot borrow the max-coordinate floor extension.
    sk = CG.Skeleton("zero_sum_ratio", CG.c_sym / CG.s_sym, CG._FINAL_ACC,
                     producer_reducer="sum", producer_expr=sp.Abs(CG.c_sym))
    with pytest.raises(NotImplementedError, match="nonzero"):
        CG.derive_merge_certificate(sk)


def test_merge_meta_api_shapes_and_fp32_scratch_launch_abi():
    import neptune_codegen as CG
    from neptune_codegen.template import launch_reduction
    for shape in ((17,), (3, 17), (2, 3, 17)):
        x = torch.empty(shape, device="meta", dtype=torch.bfloat16)
        for plan in ("serial", "split"):
            for keep in (False, True):
                out = CG.run("stable_l2", x, block=8, keepdim=keep, plan=plan)
                assert out.shape == shape[:-1] + ((1,) if keep else ())
                assert out.dtype == x.dtype
    records = []
    class RecordingKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                records.append((grid, args, kwargs))
            return launch
    x = torch.empty((2, 17), dtype=torch.bfloat16)[:, ::2]
    out = torch.empty((2,), dtype=x.dtype)
    launch_reduction((RecordingKernel(), RecordingKernel()), "split", x, out, 2, 9, 4, 1)
    assert len(records) == 2
    grid, args, kwargs = records[0]
    assert grid == (2, 3)
    assert args[1].shape == (2, 3) and args[1].dtype == torch.float32
    assert args[2].dtype == torch.float32
    assert args[3].dtype == args[4].dtype == torch.int32
    assert args[7:10] == (x.stride(0), x.stride(1), 3)
    assert records[1][1][-3:] == (2, 9, 3)
    assert records[1][2]["MERGE_BLOCK"] == 4


# Transaction tests use a real dispatcher op with CPU reference and fake metadata.
# Unlike aten.clone, it has the reduction ABI and passes ExportedProgram.validate().
@torch.library.custom_op("neptune_test::transaction_lse", mutates_args=())
def _transaction_lse(x: torch.Tensor, keepdim: bool) -> torch.Tensor:
    return torch.logsumexp(x, dim=-1, keepdim=keepdim)


@_transaction_lse.register_fake
def _transaction_lse_fake(x, keepdim):
    return x.new_empty(x.shape[:-1] + ((1,) if keepdim else ()))


@pytest.fixture
def cpu_lse_handle():
    import neptune_codegen as CG

    requests = []

    def handle(name, *, params, keepdim, **kwargs):
        assert name == "logsumexp", name
        assert params == {"tiny": 0.0}, params
        requests.append((name, keepdim, kwargs))
        # The FX target must be an OpOverload, not an arbitrary Python closure.
        return (_transaction_lse_keep if keepdim else _transaction_lse_drop)._opoverload

    with patch.object(CG, "op_handle", side_effect=handle):
        yield requests


@torch.library.custom_op("neptune_test::transaction_lse_keep", mutates_args=())
def _transaction_lse_keep(x: torch.Tensor) -> torch.Tensor:
    return _transaction_lse(x, True)


@_transaction_lse_keep.register_fake
def _transaction_lse_keep_fake(x):
    return _transaction_lse_fake(x, True)


@torch.library.custom_op("neptune_test::transaction_lse_drop", mutates_args=())
def _transaction_lse_drop(x: torch.Tensor) -> torch.Tensor:
    return _transaction_lse(x, False)


@_transaction_lse_drop.register_fake
def _transaction_lse_drop_fake(x):
    return _transaction_lse_fake(x, False)


class _TransactionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.5, 1.5, 32))
        self.register_buffer("offset", torch.tensor(0.25))

    def forward(self, x):
        data = x * self.weight + self.offset
        return {"lse": torch.logsumexp(data, dim=-1),
                "aux": (x > 0, torch.full((x.shape[0],), 2**55 + 1,
                                          dtype=torch.int64, device=x.device))}


class _TwoTransactionRegions(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(32))

    def forward(self, x):
        projected = x @ self.weight
        left = torch.logsumexp(projected, dim=-1, keepdim=True)
        # sin is an opaque region boundary: both candidates are plain LSE, not
        # the separately supported lse_biased(x, alpha) spelling of LSE(x + c).
        right = torch.logsumexp(torch.sin(x), dim=-1, keepdim=True)
        return (projected + x + left + right).clamp(-3.0, 4.0)


def _transaction_snapshot(ep):
    import copy

    return {"module": ep.graph_module, "graph": ep.graph_module.graph,
            "nodes": tuple(ep.graph_module.graph.nodes), "code": ep.graph_module.code,
            "signature": ep.graph_signature, "signature_value": copy.deepcopy(ep.graph_signature),
            "calls": ep.module_call_graph, "calls_value": copy.deepcopy(ep.module_call_graph),
            "state": ep.state_dict,
            "tensors": {key: (value, value.data_ptr(), value.detach().clone())
                        for key, value in ep.state_dict.items()}}


def _assert_transaction_unchanged(ep, before):
    assert ep.graph_module is before["module"]
    assert ep.graph_module.graph is before["graph"]
    assert tuple(ep.graph_module.graph.nodes) == before["nodes"]
    assert ep.graph_module.code == before["code"]
    assert ep.graph_signature is before["signature"]
    assert ep.graph_signature == before["signature_value"]
    assert ep.module_call_graph is before["calls"]
    assert ep.module_call_graph == before["calls_value"]
    assert ep.state_dict is before["state"]
    assert set(ep.state_dict) == set(before["tensors"])
    for key, (tensor, pointer, value) in before["tensors"].items():
        assert ep.state_dict[key] is tensor
        assert ep.state_dict[key].data_ptr() == pointer
        torch.testing.assert_close(ep.state_dict[key], value, rtol=0, atol=0)
    ep.validate()


def test_apply_transaction_generation_exception_rolls_back_after_first_region(cpu_lse_handle):
    import neptune_codegen as CG

    mod = _TwoTransactionRegions().eval()
    ep = torch.export.export(mod, (X,))
    before = _transaction_snapshot(ep)
    original_handle = CG.op_handle

    def fail_second(*args, **kwargs):
        if cpu_lse_handle:
            raise RuntimeError("injected second-region generation failure")
        return original_handle(*args, **kwargs)

    with patch.object(CG, "op_handle", side_effect=fail_second):
        report = NO.apply(ep)
    assert len(cpu_lse_handle) == 1, "exercise rollback after a successful candidate rewrite"
    assert not report.fusions and not report.graph_changed and not report.verified
    assert report.skipped[0][0] == "rewrite-failed"
    assert "injected second-region" in report.skipped[0][1]
    _assert_transaction_unchanged(ep, before)
    torch.testing.assert_close(ep.module()(X), mod(X))


def test_apply_transaction_verify_failure_never_publishes(cpu_lse_handle):
    mod = _TransactionModel().eval()
    ep = torch.export.export(mod, (X,))
    before = _transaction_snapshot(ep)
    candidates = []

    def reject(reference, candidate, args):
        candidates.append(candidate)
        assert reference is mod
        assert candidate is not ep.graph_module
        torch.testing.assert_close(candidate(*args), reference(*args))
        return False, "injected verification failure"

    with patch.object(NO, "_verify", side_effect=reject):
        report = NO.apply(ep, verify_with=(mod, (X,)))
    assert cpu_lse_handle and len(candidates) == 1
    assert not report.fusions and not report.graph_changed and not report.verified
    assert report.verification_cases == 1
    assert report.skipped == [("verify-failed", "case 0: injected verification failure")]
    _assert_transaction_unchanged(ep, before)


def test_apply_transaction_verified_binds_parameters_and_dict_pytree(cpu_lse_handle):
    mod = _TransactionModel().eval()
    ep = torch.export.export(mod, (X,))
    state = {key: (value, value.data_ptr(), value.detach().clone())
             for key, value in ep.state_dict.items()}
    probe = X.flip(-1) * 0.5
    report = NO.apply(ep, verify_with=(mod, (X,)), verification_inputs=[(probe,)])
    assert len(cpu_lse_handle) == len(report.fusions) == 1, report.summary()
    assert report.verified and report.graph_changed and report.verification_cases == 2
    for data in (X, probe):
        actual = ep.module()(data)
        torch.testing.assert_close(actual, mod(data), rtol=0, atol=0)
        assert actual["lse"].shape == (8,) and actual["lse"].dtype == X.dtype
        assert actual["aux"][0].dtype == torch.bool
        assert actual["aux"][1].dtype == torch.int64
        assert torch.equal(actual["aux"][1], torch.full((8,), 2**55 + 1, dtype=torch.int64))
    for key, (tensor, pointer, value) in state.items():
        assert ep.state_dict[key] is tensor and tensor.data_ptr() == pointer
        torch.testing.assert_close(tensor, value, rtol=0, atol=0)
    ep.validate()


def test_apply_transaction_verification_reference_cannot_mutate_original_state(cpu_lse_handle):
    """Shared trial storage must not let verification corrupt the caller's parameters.

    A read-only exported graph does not imply a supplied reference is read-only;
    rejecting or isolating that execution must leave both the EP and module intact.
    """
    mod = _TransactionModel().eval()
    ep = torch.export.export(mod, (X,))
    before = _transaction_snapshot(ep)

    def mutating_reference(x):
        mod.weight.add_(1.0)
        return mod(x)

    report = NO.apply(ep, verify_with=(M(mutating_reference), (X,)))
    assert cpu_lse_handle, "verification must reach a generated CPU candidate"
    torch.testing.assert_close(mod.weight, before["tensors"]["weight"][2], rtol=0, atol=0,
                               msg="verification mutated the original module's shared parameter")
    _assert_transaction_unchanged(ep, before)
    assert not report.fusions and not report.verified and not report.graph_changed
    assert any(reason == "verify-failed" for reason, _ in report.skipped), report.summary()


def test_apply_transaction_commit_rebinds_output_specs_to_new_nodes(cpu_lse_handle):
    mod = M(lambda x: torch.logsumexp(x, dim=-1)).eval()
    ep = torch.export.export(mod, (X,))
    old_root = ep.graph_signature.output_specs[0].arg.name
    old_graph = ep.graph_module.graph
    report = NO.apply(ep, verify_with=(mod, (X,)))
    assert report.verified and len(report.fusions) == 1, report.summary()
    assert ep.graph_module.graph is not old_graph
    output = next(node for node in ep.graph_module.graph.nodes if node.op == "output")
    root = output.args[0][0]
    assert root.op == "call_function"
    assert root.target == _transaction_lse_drop._opoverload
    assert root.name != old_root
    assert ep.graph_signature.output_specs[0].arg.name == root.name
    assert old_root not in {node.name for node in ep.graph_module.graph.nodes}
    for entry in ep.module_call_graph:
        if entry.signature:
            assert all(getattr(arg, "name", None) != old_root for arg in entry.signature.outputs)
    ep.graph_module.graph.lint()
    ep.validate()
    torch.testing.assert_close(ep.module()(X), mod(X))


@pytest.mark.parametrize("max_regions, expected", [(None, 2), (1, 1)])
def test_apply_transaction_multi_region_preserves_gemm_residual_clamp(
        cpu_lse_handle, max_regions, expected):
    mod = _TwoTransactionRegions().eval()
    ep = torch.export.export(mod, (X,))
    unsaturated = X * 0.05 - 1.5
    report = NO.apply(ep, max_regions=max_regions, verify_with=(mod, (X,)),
                      verification_inputs=[(unsaturated,)])
    assert len(report.fusions) == len(cpu_lse_handle) == expected, report.summary()
    assert report.verified and report.graph_changed and report.verification_cases == 2
    targets = [node.target for node in ep.graph_module.graph.nodes if node.op == "call_function"]
    assert targets.count(_transaction_lse_keep._opoverload) == expected
    assert torch.ops.aten.matmul.default in targets
    assert torch.ops.aten.clamp.default in targets
    assert targets.count(torch.ops.aten.add.Tensor) >= 3, "residual additions must survive"
    assert torch.ops.aten.sin.default in targets, "the second region input boundary must survive"
    # The bounded mode must leave the unused compound INTACT, not expand its internals.
    assert targets.count(torch.ops.aten.logsumexp.default) == 2 - expected
    ep.validate()
    torch.testing.assert_close(ep.module()(X), mod(X), rtol=1e-5, atol=1e-6)


def test_apply_no_fusion_keeps_state_storage_and_does_not_verify():
    class NoRegion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(32))
            self.register_buffer("offset", torch.zeros(32))

        def forward(self, x):
            return torch.sin(x * self.weight) + self.offset

    mod = NoRegion().eval()
    ep = torch.export.export(mod, (X,))
    before = _transaction_snapshot(ep)
    with patch.object(NO, "_verify", side_effect=AssertionError("no fusion must not verify")) as verify, \
            patch.object(torch.Tensor, "__deepcopy__", side_effect=AssertionError("no state copy")) as deepcopy:
        report = NO.apply(ep, verify_with=(mod, (X,)))
    verify.assert_not_called()
    deepcopy.assert_not_called()
    assert not report.fusions and not report.graph_changed and not report.verified
    assert report.verification_cases == 0 and report.skipped
    assert all(reason != "rewrite-failed" for reason, _ in report.skipped), report.summary()
    _assert_transaction_unchanged(ep, before)


@pytest.mark.parametrize("reference,candidate,detail", [
    ({"x": torch.tensor([float("inf")])}, {"x": torch.tensor([-float("inf")])}, "output"),
    ({"x": torch.tensor([float("inf")])}, {"x": torch.tensor([0.])}, "finite classification"),
    ({"x": torch.tensor([float("nan")])}, {"x": torch.tensor([float("inf")])}, "NaN classification"),
    ({"x": torch.tensor([2**55 + 1])}, {"x": torch.tensor([2**55])}, "integer/bool"),
    ({"x": torch.tensor([True])}, {"x": torch.tensor([False])}, "integer/bool"),
    ({"x": torch.ones(2)}, {"x": torch.ones(1, 2)}, "shape/dtype/device"),
    ({"x": torch.ones(2)}, {"x": torch.ones(2, dtype=torch.float64)}, "shape/dtype/device"),
    ({"x": (torch.ones(2),)}, {"x": [torch.ones(2)]}, "pytree structure"),
    ({"x": torch.ones(2)}, {"other": torch.ones(2)}, "pytree structure"),
    ({"x": 1}, {"x": True}, "non-tensor"),
])
def test_compare_trees_rejects_ieee_integer_dtype_and_structure_errors(reference, candidate, detail):
    ok, reason = NO._compare_trees(reference, candidate)
    assert ok is False and detail in reason, reason


def test_compare_trees_accepts_matching_nonfinite_and_exact_large_integers():
    value = {"float": torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0]),
             "aux": (torch.tensor([True, False]), torch.tensor([2**55 + 1]), 2**55 + 1)}
    assert NO._compare_trees(value, NO._clone_inputs(value)) == (True, "")


def test_verify_direct_handles_bound_parameter_pytree_and_exact_integer_bool():
    mod = _TransactionModel().eval()
    ep = torch.export.export(mod, (X,))
    assert NO._verify(mod, ep.module(), (X,)) == (True, "")
    for leaf in ("integer", "bool", "shape", "dtype", "structure"):
        def wrong(x):
            result = mod(x)
            if leaf == "integer":
                result["aux"] = (result["aux"][0], result["aux"][1] - 1)
            elif leaf == "bool":
                result["aux"] = (~result["aux"][0], result["aux"][1])
            elif leaf == "shape":
                result["lse"] = result["lse"].unsqueeze(-1)
            elif leaf == "dtype":
                result["lse"] = result["lse"].double()
            else:
                result["aux"] = list(result["aux"])
            return result

        ok, detail = NO._verify(mod, M(wrong), (X,))
        assert ok is False and detail, leaf


@pytest.mark.parametrize("mutating_side", ["reference", "candidate"])
def test_verify_direct_detects_input_mutation_without_mutating_caller(mutating_side):
    args = ({"x": X.clone(), "aux": [torch.tensor([2**55 + 1])]},)
    pristine = NO._clone_inputs(args)

    def read(data):
        return data["x"] * 2

    def mutate(data):
        output = read(data)
        data["aux"][0].add_(1)
        return output

    reference = M(mutate if mutating_side == "reference" else read)
    candidate = M(mutate if mutating_side == "candidate" else read)
    ok, detail = NO._verify(reference, candidate, args)
    assert ok is False and "inputs mutated" in detail, detail
    assert NO._compare_trees(args, pristine, rtol=0, atol=0) == (True, "")


def test_explicit_ulp_policy_is_opt_in_and_preserves_nonfinite_signs():
    value = torch.tensor([10.0], dtype=torch.bfloat16)
    adjacent = torch.nextafter(value, torch.full_like(value, float("inf")))
    two_steps = torch.nextafter(adjacent, torch.full_like(adjacent, float("inf")))
    assert not NO._compare_trees(value, adjacent)[0]
    assert NO._compare_trees(value, adjacent, max_ulps=1)[0]
    assert not NO._compare_trees(value, two_steps, max_ulps=1)[0]
    positive = torch.tensor([float("inf"), 10.0], dtype=torch.bfloat16)
    wrong = torch.tensor([-float("inf"), adjacent.item()], dtype=torch.bfloat16)
    assert not NO._compare_trees(positive, wrong, max_ulps=1)[0]


def test_skeleton_registration_cannot_reuse_a_name_for_different_algebra():
    import dataclasses
    import neptune_codegen as CG

    original = CG.SKELETONS["logsumexp"]
    assert CG.register(original) is original
    with pytest.raises(ValueError, match="different algebra"):
        CG.register(dataclasses.replace(original, finalize=original.finalize + 1))
    assert CG.SKELETONS["logsumexp"] is original


def test_bounded_fusion_does_not_expand_a_later_unused_compound(cpu_lse_handle):
    def fn(x):
        maximum = x.amax(-1, keepdim=True)
        first = maximum + (x - maximum).exp().sum(-1, keepdim=True).log()
        return first.squeeze(-1), torch.logsumexp(torch.sin(x), -1)

    mod = M(fn).eval()
    ep = torch.export.export(mod, (X,))
    report = NO.apply(ep, max_regions=1, verify_with=(mod, (X,)))
    assert report.verified and len(report.fusions) == 1
    assert sum(node.target == torch.ops.aten.logsumexp.default for node in ep.graph.nodes) == 1
    assert not any(node.meta.get("neptune_lowering") for node in ep.graph.nodes)


def test_default_lowering_declines_unstable_log_softmax(cpu_lse_handle):
    mod = M(lambda x: torch.log_softmax(x, -1)).eval()
    ep = torch.export.export(mod, (X,))
    before = _transaction_snapshot(ep)
    assert not NO.preview(ep)[0]
    report = NO.apply(ep)
    assert not report.fusions and not report.graph_changed
    _assert_transaction_unchanged(ep, before)
    explicit = NO.apply(ep, experimental_log_softmax=True, verify_with=(mod, (X,)))
    assert explicit.verified and len(explicit.fusions) == 1


def test_verify_direct_read_only_reference_does_not_mutate_original_parameters():
    mod = _TransactionModel().eval()
    candidate = torch.export.export(mod, (X,)).module()
    before = {key: (value.data_ptr(), value.detach().clone())
              for key, value in mod.state_dict().items()}
    assert NO._verify(mod, candidate, (X,)) == (True, "")
    for key, value in mod.state_dict().items():
        pointer, pristine = before[key]
        assert value.data_ptr() == pointer
        torch.testing.assert_close(value, pristine, rtol=0, atol=0)
