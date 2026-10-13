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
        assert "acc * scale +" in src or "acc * scale," in src

    # Pin the exact fold of the paper's own operator. The affine generalisation must have
    # added an OPTION to the emitted statement, not changed the statement itself -- the
    # max-consumer shape and the guarded scale are both load-bearing.
    src = CG.emit_kernel_source(CG.SKELETONS["rmsnorm_max"], {"eps": 1e-6})
    assert "scale = tl.where(m == 0.0, 1.0, " in src, src
    assert "acc = tl.maximum(acc * scale, tl.max(" in src, src


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
    assert "tl.where(m == " in CG.bias_line(sk, {})


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

    # divides by the producer -> floored
    assert CG.denominator_expr(CG.SKELETONS["rmsnorm_max"]) == "tl.maximum(m_new, 1e-30)"
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
        pass1 = next(i for i, l in enumerate(lines) if "pass 1" in l)
        den = next(i for i, l in enumerate(lines) if l.strip() == "den = m")
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
    from neptune_codegen.symbols import _PARAM_N, c_sym, s_sym

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
