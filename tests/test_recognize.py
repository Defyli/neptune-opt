"""
Recognition tests: do we recognise the operators from their DECOMPOSED torch forms?

This runs on CPU. Recognition is symbolic manipulation over an exported graph, so it needs
no GPU and no Triton -- only the generator's registry. That is worth stating because the
generation side cannot be tested without a GPU, and a slow feedback loop is how pattern
tables drift away from the registry they are supposed to mirror.

Each case is written the way a user would naturally write the operator in eager torch, i.e.
the multi-pass "decomposed" form, which is what a graph optimiser actually receives.

Run:  python recognize_tests.py
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest
import sympy as sp
import torch
from torch.fx.experimental.proxy_tensor import make_fx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.export import export  # noqa: E402

import neptune_recognize as NR  # noqa: E402

X = torch.randn(16, 64)   # 64 columns, so a .mean() over the last dim means /64


# ---------------------------------------------------------------- decomposed forms
def d_logsumexp(t):
    m = t.amax(dim=-1, keepdim=True)
    return (torch.log(torch.exp(t - m).sum(dim=-1, keepdim=True)) + m).squeeze(-1)


def d_stable_l2(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return (m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))).squeeze(-1)


def d_softcap(t):
    h = torch.tanh(0.5 * t)
    m = h.amax(dim=-1, keepdim=True)
    return (torch.log(torch.exp(h - m).sum(dim=-1, keepdim=True)) + m).squeeze(-1)


def d_sumsq_div(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return ((t * t) / m).sum(dim=-1, keepdim=True).squeeze(-1)


def d_linear_attn(t):
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return (t / m).sum(dim=-1, keepdim=True).squeeze(-1)


def d_decay_renorm(t):
    # The positive offset already makes the denominator safe at zero. Adding a floor
    # changes this operator (especially for a small offset), so it is not a free guard.
    m = t.abs().amax(dim=-1, keepdim=True)
    return (t / (m + 2.0)).sum(dim=-1, keepdim=True).squeeze(-1)


def d_lse_biased(t):
    z = t + 1.5
    m = z.amax(dim=-1, keepdim=True)
    return (torch.log(torch.exp(z - m).sum(dim=-1, keepdim=True)) + m).squeeze(-1)


def d_rmsnorm_max(t):
    # the natural eager form: .mean() hides the row length inside the op
    r = (t * t).mean(dim=-1, keepdim=True)
    return (t / torch.sqrt(r + 1e-6)).amax(dim=-1)


def d_variance(t):
    m = t.mean(dim=-1, keepdim=True)
    return ((t - m) ** 2).mean(dim=-1)


def d_std(t):
    m = t.mean(dim=-1, keepdim=True)
    return (((t - m) ** 2).mean(dim=-1)).sqrt()


def d_variance_keepdim(t):
    m = t.mean(dim=-1, keepdim=True)
    return ((t - m) ** 2).mean(dim=-1, keepdim=True)


def d_lse_scaled(t):
    """Temperature-scaled log-sum-exp: `logsumexp(t / 8)`, i.e. an attention denominator.

    The scale is a PARAMETER of the skeleton and is inferred from the graph: nothing tells the
    recogniser that 8 is the temperature.
    """
    return torch.logsumexp(t / 8.0, dim=-1)


def d_var_op(t):
    """`x.var(-1, unbiased=False)` -- a SINGLE aten op, not a decomposed pair."""
    return t.var(dim=-1, unbiased=False)


def d_std_op(t):
    return t.std(dim=-1, unbiased=False)


def d_var_correction0(t):
    """The same operator spelled with the newer `correction` argument."""
    return torch.var(t, dim=-1, correction=0)


def d_batchnorm_like(t):
    """Variance as an INTERMEDIATE of a larger computation, not as the graph output.

    The region gate must still find and replace the variance sub-region while leaving the
    normalisation around it alone. This is the case that would silently corrupt a model if
    the gate were sloppy.
    """
    m = t.mean(dim=-1, keepdim=True)
    v = ((t - m) ** 2).mean(dim=-1, keepdim=True)
    return (t - m) / (v + 1e-5).sqrt()


CASES = [
    ("logsumexp", d_logsumexp),
    ("stable_l2", d_stable_l2),
    ("softcap_lse", d_softcap),
    ("sumsq_div", d_sumsq_div),
    ("linear_attn", d_linear_attn),
    ("decay_renorm", d_decay_renorm),
    ("lse_biased", d_lse_biased),
    ("rmsnorm_max", d_rmsnorm_max),
    # The two-pass families. These were a NEGATIVE control until the two-pass plan existed
    # ("variance needs two accumulators and has no single-producer (f, g, h) form at all") --
    # which is still true of the ONE-PASS frame, and is exactly why the plan exists.
    ("variance", d_variance),
    ("std", d_std),
    ("variance", d_variance_keepdim),
    ("variance", d_batchnorm_like),
    ("variance", d_var_op),
    ("std", d_std_op),
    ("variance", d_var_correction0),
    ("lse_scaled", d_lse_scaled),
]


def d_prod(t):
    """A PRODUCT consumer. Refused on purpose: `prod + ratio` needs a carried count, and it is
    the one family in the coverage study that neither plan serves."""
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return (t / m).prod(dim=-1)


def d_reduce_first_dim(t):
    """Reduces the wrong axis: the frame emits one program per ROW."""
    return t.sum(dim=0)


def d_var_unbiased(t):
    """`x.var(-1)` -- the DEFAULT, which is the /(N-1) operator.

    Refused on purpose, and this is the most important negative in the file: the two forms
    differ by a factor of N/(N-1), so fusing this one into the biased kernel would silently
    change the model's numbers for the most common way anyone writes a variance.
    """
    return t.var(dim=-1)


def d_std_unbiased(t):
    return t.std(dim=-1)


NEGATIVE = [
    ("product consumer (needs a count)", d_prod),
    ("reduction over dim 0", d_reduce_first_dim),
    ("x.var(-1): unbiased by DEFAULT", d_var_unbiased),
    ("x.std(-1): unbiased by DEFAULT", d_std_unbiased),
]


class M(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, t):
        return self.fn(t)


def _run_cases():
    """Run every case once. Returns [(label, expected, got, ok, detail)].

    Shared by the pytest entry point and the printed report so the two cannot drift: the report
    a human reads and the assertion CI runs are the same computation.
    """
    out = []
    for want, fn in CASES:
        # The CORE-ATEN DECOMPOSITIONS, because that is what the pipeline sees: real modules
        # export as one high-level aten op (`aten.logsumexp`, `aten.var`), and the frame needs
        # the producer/consumer pair. `neptune_opt.export(..., decompose=True)` is the same
        # step, so a case that passes here is a case the pipeline can serve.
        ep = export(M(fn).eval(), (X,)).run_decompositions()
        match, reason = NR.recognize(ep.graph_module)
        got = match.skeleton if match else str(reason)
        detail = ""
        if match and match.inferred:
            detail = "inferred " + ", ".join(
                f"{k}={v:g}" for k, v in sorted(match.params.items()))
        out.append((fn.__name__, want, got, got == want, detail))
    for label, fn in NEGATIVE:
        ep = export(M(fn).eval(), (X,)).run_decompositions()
        match, got = NR.recognize(ep.graph_module)
        out.append((label, "refused",
                    f"matched {match.skeleton}" if match is not None else "refused",
                    match is None, str(got)[:60]))
    return out


def test_recognition_cases():
    """Every case matches the skeleton the registry says implements it.

    The pytest entry point. It runs the SAME `_run_cases()` as `main()`, because the previous
    version kept the whole suite inside a `main()` and pytest therefore collected ZERO tests from
    this file -- so `pytest tests/` (what CI runs, and what the README tells a user to run) would
    have silently skipped every recognition case.
    """
    failed = [f"{label}: expected {want}, got {got}  ({detail})"
              for label, want, got, ok, detail in _run_cases() if not ok]
    assert not failed, "recognition failures:\n  " + "\n  ".join(failed)


# ---------------------------------------------------------------- CPU region regressions
@pytest.mark.parametrize("raw_lse", [False, True], ids=["hand-lse", "raw-lse-lowered"])
def test_regions_after_opaque_gemm_sdpa_preserve_guard(raw_lse):
    """Multi-input model boundaries are not whole-graph recognition failures."""
    from neptune_recognize.lowering import lower_if_it_helps

    class Model(torch.nn.Module):
        def forward(self, x, weight, q, k, v):
            z = x @ weight
            torch.ops.aten._assert_tensor_metadata.default(z, dtype=torch.float32)
            attention = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            result = torch.logsumexp(z, dim=-1) if raw_lse else d_logsumexp(z)
            return result + 0.25, attention

    args = (torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12,
            torch.arange(20, dtype=torch.float32).reshape(4, 5) / 20,
            torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4))
    model = Model().eval()
    gm = export(model, args).run_decompositions({}).graph_module
    guard_target = torch.ops.aten._assert_tensor_metadata.default
    # Older upstream export consumes this assertion before Neptune sees the graph.
    # Seed an actual ATen guard when absent so both lowering and rewriting must
    # preserve it on every tested version, rather than silently weaken the check.
    if not any(node.target == guard_target for node in gm.graph.nodes):
        boundary = next(node for node in gm.graph.nodes if NR._op(node) in ("mm", "matmul"))
        value = boundary.meta["val"]
        with gm.graph.inserting_after(boundary):
            guard = gm.graph.call_function(guard_target, (boundary,),
                    {"size": tuple(value.shape), "stride": tuple(value.stride()), "dtype": value.dtype})
        guard.meta["val"] = None
        gm.graph.lint()
        gm.recompile()
    assert sum(node.target == guard_target for node in gm.graph.nodes) == 1
    if raw_lse:
        assert lower_if_it_helps(gm) == 1
    before = str(gm.graph)
    matches, reason = NR.recognize_regions(gm)
    assert len(matches) == 1, reason
    match = matches[0]
    assert match.skeleton == "logsumexp"
    assert NR._op(match.input_node) in ("mm", "matmul")
    assert len(match.input_node.all_input_nodes) == 2
    assert match.input_node not in match.region_nodes
    assert match.proof_kind == "exact-symbolic-real-algebra"
    assert NR.region_contract(match) == (False, None)
    preserved = [node for node in gm.graph.nodes
                 if "scaled_dot_product" in str(node.target)
                 or "_assert_tensor_metadata" in str(node.target)
                 or NR._op(node) in ("mm", "matmul")]
    assert len(preserved) == 3
    assert not set(preserved).intersection(match.region_nodes)
    assert str(gm.graph) == before, "recognition must be read-only"
    if raw_lse:
        from neptune_recognize.interp import _logsumexp_inf_guard

        guards = _logsumexp_inf_guard(gm)
        where = [node for node in gm.graph.nodes if NR._op(node) == "where"]
        assert len(where) == 1
        assert guards[where[0]] is where[0].args[2]
        assert NR._op(guards[where[0]]) == "amax"
        assert str(gm.graph) == before, "guard mapping must be read-only"
    # Only replace the discovered root with eager CPU math: no op registration/GPU.
    with patch("neptune_recognize.rewrite._launch_config", return_value=(32, 4)), \
            patch("neptune_codegen.op_handle", return_value=d_logsumexp):
        call, refusal = NR.rewrite_match(gm, match)
    assert refusal is None
    assert call.args == (match.input_node,)
    assert all(node in gm.graph.nodes for node in preserved)
    torch.testing.assert_close(gm(*args), model(*args))


def test_independent_region_roots_skeleton_matrix():
    """Different data/weights must never collapse into the first placeholder."""
    def model(x, y, weight):
        return d_logsumexp(x), d_logsumexp(y @ weight), d_stable_l2(y), d_rmsnorm_max(x)

    x = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 7
    y = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 5 - 1
    weight = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 9
    gm = make_fx(model)(x, y, weight)
    matches, reason = NR.recognize_regions(gm)
    assert len(matches) == 4, reason
    assert [match.skeleton for match in matches] == [
        "logsumexp", "logsumexp", "stable_l2", "rmsnorm_max"]
    roots = next(node for node in gm.graph.nodes if node.op == "output").args[0]
    assert [match.region_root for match in matches] == list(roots)
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    assert matches[0].input_node is placeholders[0]
    assert NR._op(matches[1].input_node) == "mm"
    assert matches[2].input_node is placeholders[1]
    assert matches[3].input_node is placeholders[0]
    for match in matches:
        assert match.input_node.meta["val"].dtype == torch.float32
        assert NR.region_contract(match) == (False, None)
        assert match.input_node not in match.region_nodes
        assert match.proof_kind == "exact-symbolic-real-algebra"
    torch.testing.assert_close(gm(x, y, weight), model(x, y, weight))


def test_business_clamp_is_exact_preserved_anchor():
    def model(x):
        bounded = x.clamp(min=-0.5, max=0.75)
        return d_logsumexp(bounded), bounded * 3

    x = torch.tensor([[-20., -1., 0., 2., 30.], [5., 4., 3., 2., 1.]])
    gm = make_fx(model)(x)
    matches, reason = NR.recognize_regions(gm)
    assert len(matches) == 1, reason
    match = matches[0]
    clamp = next(node for node in gm.graph.nodes if NR._op(node) == "clamp")
    original = next(node for node in gm.graph.nodes if node.op == "placeholder")
    assert match.input_node is clamp and match.input_node is not original
    assert clamp not in match.region_nodes
    assert match.region_root is next(node for node in gm.graph.nodes
                                     if node.op == "output").args[0][0]
    assert not torch.allclose(d_logsumexp(x), d_logsumexp(x.clamp(-0.5, 0.75)))
    with patch("neptune_recognize.rewrite._launch_config", return_value=(32, 4)), \
            patch("neptune_codegen.op_handle", return_value=d_logsumexp):
        call, refusal = NR.rewrite_match(gm, match)
    assert refusal is None and call.args == (clamp,)
    assert clamp in gm.graph.nodes
    torch.testing.assert_close(gm(x), model(x))


def test_constant_expansion_and_independent_dim0_boundary():
    def model(x, y):
        # A literal column expands data from [B, 1] to [B, N], not a no-op bias.
        z = x + torch.full((2, 5), 0.125, dtype=torch.float32)
        m0 = y.amax(dim=0, keepdim=True)
        wrong_axis = (torch.exp(y - m0).sum(dim=0, keepdim=True).log() + m0).squeeze(0)
        return d_logsumexp(z), wrong_axis, d_logsumexp(y)

    x, y = torch.tensor([[1.], [-2.]]), torch.arange(10, dtype=torch.float32).reshape(2, 5)
    gm = make_fx(model)(x, y)
    before = str(gm.graph)
    matches, reason = NR.recognize_regions(gm)
    assert len(matches) == 2, reason
    roots = next(node for node in gm.graph.nodes if node.op == "output").args[0]
    assert [match.region_root for match in matches] == [roots[0], roots[2]]
    match = matches[0]
    assert NR._op(match.input_node) == "add"
    assert tuple(match.input_node.meta["val"].shape) == (2, 5)
    assert match.input_node.args[0].meta["val"].shape == (2, 1)
    assert match.input_node not in match.region_nodes
    assert all(NR.region_contract(item) == (False, None) for item in matches)
    assert str(gm.graph) == before
    torch.testing.assert_close(gm(x, y), model(x, y))
    assert not torch.allclose(model(x, y)[0], d_logsumexp(x))


@pytest.mark.parametrize("dtype,upcast", [(torch.float16, False), (torch.bfloat16, False),
                                        (torch.bfloat16, True)],
                         ids=["fp16-rounding", "bf16-rounding", "bf16-fp32-cast"])
def test_lowprecision_rms_contract_preserves_cast(dtype, upcast):
    def model(x):
        z = x.to(torch.float32) if upcast else x
        return d_rmsnorm_max(z).to(dtype)

    x = torch.arange(15, dtype=torch.float32).reshape(3, 5).to(dtype) / 7
    gm = make_fx(model)(x)
    before = str(gm.graph)
    matches, reason = NR.recognize_regions(gm)
    assert len(matches) == 1, reason
    match = matches[0]
    assert match.skeleton == "rmsnorm_max"
    keepdim, refusal = NR.region_contract(match)
    if upcast:
        assert NR._op(match.input_node) == "_to_copy"
        assert match.input_node.kwargs["dtype"] == torch.float32
        casts = [node for node in gm.graph.nodes if NR._op(node) == "_to_copy"]
        assert len(casts) == 2
        assert not set(casts).intersection(match.region_nodes)
        assert (keepdim, refusal) == (False, None)
        with patch("neptune_recognize.rewrite._launch_config", return_value=(32, 4)), \
                patch("neptune_codegen.op_handle", return_value=d_rmsnorm_max):
            call, refusal = NR.rewrite_match(gm, match)
        assert refusal is None and call.args == (casts[0],)
        assert all(node in gm.graph.nodes for node in casts)
    else:
        assert keepdim is None and "low-precision intermediate rounding" in refusal
        assert NR.would_rewrite(gm)[0] is False
        assert str(gm.graph) == before
    torch.testing.assert_close(gm(x), model(x), rtol=0, atol=0)


def test_mutable_graph_refused_before_functionalization():
    def model(x, y):
        result = d_logsumexp(x)
        temporary = y.clone()
        temporary.add_(1.0)
        return result, d_logsumexp(temporary)

    x, y = torch.randn(2, 5), torch.randn(2, 5)
    gm = make_fx(model)(x, y)
    before = str(gm.graph)
    assert any(getattr(node.target, "_schema", None) is not None
               and node.target._schema.is_mutable for node in gm.graph.nodes)
    matches, reason = NR.recognize_regions(gm)
    assert matches == [] and "functionalized" in reason
    match, reason = NR.recognize(gm)
    assert match is None and "functionalized" in reason
    assert NR.would_rewrite(gm)[0] is False
    assert str(gm.graph) == before
    torch.testing.assert_close(gm(x, y), model(x, y))

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return model(x, y)

    ep = export(Model(), (x, y)).run_decompositions({})
    assert not any(getattr(node.target, "_schema", None) is not None
                   and node.target._schema.is_mutable for node in ep.graph_module.graph.nodes)
    matches, reason = NR.recognize_regions(ep.graph_module)
    assert len(matches) == 2, reason
    assert [match.skeleton for match in matches] == ["logsumexp", "lse_biased"]
    assert matches[1].params["alpha"] == 1.0
    assert all(NR.region_contract(match) == (False, None) for match in matches)
    torch.testing.assert_close(ep.module()(x, y), model(x, y))


@pytest.mark.parametrize("variant", ["add-alpha", "sub-alpha", "rounded-div", "integer-div",
                                     "sum-dtype"], ids=str)
def test_binary_alpha_division_and_dtype_kwargs(variant):
    def model(x):
        if variant == "add-alpha":
            z = torch.add(x, 1.25, alpha=-2)
        elif variant == "sub-alpha":
            z = torch.sub(x, 1.25, alpha=-2)
        elif variant == "rounded-div":
            m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
            return torch.div(x, m, rounding_mode="floor").sum(dim=-1)
        elif variant == "integer-div":
            m = x.abs().amax(dim=-1, keepdim=True)
            return torch.div(x, m, rounding_mode="trunc").sum(dim=-1)
        elif variant == "sum-dtype":
            m = x.amax(dim=-1, keepdim=True)
            return (torch.exp(x - m).sum(dim=-1, keepdim=True, dtype=torch.float64).log()
                    + m).squeeze(-1)
        return d_logsumexp(z)

    x = torch.tensor([[1., -2., 3., -4., 5.], [-5., 4., -3., 2., -1.]])
    if variant == "integer-div":
        x = x.to(torch.int64)
    gm = make_fx(model)(x)
    before = str(gm.graph)
    matches, reason = NR.recognize_regions(gm)
    if variant in ("add-alpha", "sub-alpha"):
        assert len(matches) == 1, reason
        match = matches[0]
        assert match.skeleton == "lse_biased"
        assert match.params["alpha"] == (-2.5 if variant == "add-alpha" else 2.5)
        assert match.input_node.op == "placeholder"
        assert NR.region_contract(match) == (False, None)
        torch.testing.assert_close(model(x), torch.logsumexp(x, dim=-1)
                                   + match.params["alpha"])
    else:
        assert matches == [], reason
        assert NR.recognize(gm)[0] is None
        assert NR.would_rewrite(gm)[0] is False
    assert str(gm.graph) == before
    torch.testing.assert_close(gm(x), model(x))


@pytest.mark.parametrize("boundary", ["floor-guard", "wrong-floor", "vector-floor",
                                      "decay-floor", "rank3", "unknown-meta"], ids=str)
def test_guard_floor_and_exact_safe_contours(boundary):
    from neptune_recognize.interp import S_RESULT, _logsumexp_inf_guard

    floor = 1e-30
    if boundary == "rank3":
        x = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5) / 11

        def model(t):
            result = d_logsumexp(t)
            return result, result.unsqueeze(-1)

        gm = make_fx(model)(x)
        matches, reason = NR.recognize_regions(gm)
        assert len(matches) == 1, reason  # Same algebraic root; largest safe keepdim tail.
        match = matches[0]
        assert match.input_node.meta["val"].shape == (2, 3, 5)
        assert match.region_root.meta["val"].shape == (2, 3, 1)
        assert NR.region_contract(match) == (True, None)
        # Equal element counts alone do not establish the correct output contour.
        match.region_root.meta["val"] = torch.empty(6, 1)
        assert "output shape" in NR.region_contract(match)[1]
        assert NR.rewrite_match(gm, match)[0] is None
        return
    if boundary == "unknown-meta":
        gm = make_fx(d_logsumexp)(torch.randn(2, 5))
        matches, reason = NR.recognize_regions(gm)
        assert len(matches) == 1, reason
        match = matches[0]
        match.input_node.meta.clear()
        before = str(gm.graph)
        assert "input metadata is unknown" in NR.region_contract(match)[1]
        assert NR.rewrite_match(gm, match)[0] is None
        assert NR.would_rewrite(gm)[0] is False
        assert str(gm.graph) == before
        return

    x = torch.tensor([[0., 0., 0.], [2e-31, -1e-31, 0.]], dtype=torch.float32)

    def model(t):
        raw = t.abs().amax(dim=-1, keepdim=True)
        if boundary == "vector-floor":
            guarded = torch.maximum(raw, torch.tensor([[1e-30], [1e-4]]))
        else:
            guarded = raw.clamp_min(1e-4 if boundary == "wrong-floor" else floor)
        if boundary == "decay-floor":
            return (t / (guarded + 2e-31)).sum(dim=-1)
        return (t / guarded).sum(dim=-1)

    gm = make_fx(model)(x)
    before = str(gm.graph)
    matches, reason = NR.recognize_regions(gm)
    assert _logsumexp_inf_guard(gm) == {}, "a floor clamp is not an infinity where-guard"
    if boundary == "floor-guard":
        assert len(matches) == 1, reason
        assert matches[0].skeleton == "linear_attn"
        assert any(NR._op(node) == "clamp_min" for node in matches[0].region_nodes)
        assert NR.region_contract(matches[0]) == (False, None)
        import neptune_codegen as CG

        certificate = CG.derive_merge_certificate(CG.SKELETONS["linear_attn"])
        assert certificate.coordinate_floor is True
        assert any("BOTH repair endpoints" in assumption
                   for assumption in certificate.assumptions)
    else:
        # A clamp can be a preserved anchor only for a NEW pair depending on its data;
        # a single sum after a reduction-coordinate clamp is not such a pair.
        assert matches == [], reason
        assert NR.would_rewrite(gm)[0] is False
    assert str(gm.graph) == before
    torch.testing.assert_close(gm(x), model(x), rtol=0, atol=0)
    torch.testing.assert_close(model(x)[0], torch.tensor(0.), rtol=0, atol=0)
    if boundary == "decay-floor":
        # For N << floor the previously erased guard changes even finite CPU answers.
        raw = x.abs().amax(dim=-1, keepdim=True)
        unguarded = (x / (raw + 2e-31)).sum(dim=-1)
        assert abs(model(x)[1] - unguarded[1]) > 0.1
        unguarded_gm = make_fx(lambda t: (t / (t.abs().amax(-1, keepdim=True)
                                                + 2e-31)).sum(-1))(x)
        decay, reason = NR.recognize_regions(unguarded_gm)
        assert len(decay) == 1 and decay[0].skeleton == "decay_renorm", reason
        assert decay[0].params["N"] == pytest.approx(2e-31, rel=1e-12, abs=0)
    if boundary in ("wrong-floor", "vector-floor"):
        # Numeric sampling must not prove a floor-dependent identity. The local gate
        # uses exact expressions, even when generic sampling misses the tiny domain.
        difference = 1 / sp.Max(S_RESULT, sp.Rational("1e-30")) - 1 / S_RESULT
        assert sp.simplify(difference) != 0
        assert difference.subs(S_RESULT, sp.Rational("1e-31")) != 0
        # The exact local gate refuses these vectors despite legacy numeric sampling.
        assert matches == []


# ---------------------------------------------------------------- opt-in structured regions

def _structured_rms(x, gamma, eps=1e-6):
    z = x.float()
    return (z * (z.square().mean(-1, keepdim=True) + eps).rsqrt()).to(x.dtype) * gamma.to(x.dtype)


_STRUCTURED_TEST_LIBRARIES = []


@pytest.fixture(scope="module")
def structured_cpu_library():
    # Keep dispatcher schemas alive across parametrized tests and cached OpOverloads.
    library = torch.library.Library("neptune_structured_recognize_tests", "FRAGMENT")
    library.define("rms(Tensor x, Tensor gamma, float eps) -> Tensor")
    library.define("segment(Tensor data, Tensor descriptor, str reduce, float? initial, bool offsets, bool unsafe) -> Tensor")
    library.impl("rms", _structured_rms, "CPU")

    def segment(data, descriptor, reduce, initial, offsets, unsafe):
        kwargs = {"offsets" if offsets else "lengths": descriptor}
        return torch.segment_reduce(data, reduce, initial=initial, unsafe=unsafe, **kwargs)

    library.impl("segment", segment, "CPU")
    library.impl("rms", lambda x, gamma, eps: x.new_empty(x.shape), "Meta")

    def segment_meta(data, descriptor, reduce, initial, offsets, unsafe):
        return data.new_empty((descriptor.shape[0] - int(offsets), *data.shape[1:]))

    library.impl("segment", segment_meta, "Meta")
    from itertools import count
    # Cached FX/dispatcher overloads may outlive module fixture teardown. Keep the
    # schema owner for the full process, as required by torch.library.Library.
    _STRUCTURED_TEST_LIBRARIES.append(library)
    yield library, count()


@pytest.fixture
def structured_cpu_handle(monkeypatch, structured_cpu_library):
    """Real dispatcher overloads with explicit schemas, also valid on Torch 2.4."""
    import neptune_codegen as CG

    library, sequence = structured_cpu_library
    # Bind parameters in a schema-defined two-input wrapper; no global annotation inference.
    handles = []

    def handle(name, params, plan="serial", lib="neptune"):
        operation = f"bound_{next(sequence)}"
        library.define(f"{operation}(Tensor x, Tensor auxiliary) -> Tensor")
        if name == "rms_norm":
            def implementation(x, auxiliary):
                return torch.ops.neptune_structured_recognize_tests.rms.default(x, auxiliary, params["eps"])

            def meta(x, auxiliary):
                return x.new_empty(x.shape)
        else:
            def implementation(x, auxiliary):
                return torch.ops.neptune_structured_recognize_tests.segment.default(
                    x, auxiliary, name.removeprefix("segment_"), params["initial"],
                    params["offsets"], params["unsafe"])

            def meta(x, auxiliary):
                return x.new_empty((auxiliary.shape[0] - int(params["offsets"]), *x.shape[1:]))
        library.impl(operation, implementation, "CPU")
        library.impl(operation, meta, "Meta")
        overload = getattr(torch.ops.neptune_structured_recognize_tests, operation).default
        handles.append((name, dict(params), plan, lib, overload))
        return overload

    monkeypatch.setattr(CG, "state_op_handle", handle, raising=False)
    yield handles
    # The Library owns registrations for the duration of each test.


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(512,), (2, 512), (2, 3, 128), (1, 1), (0, 32)])
@pytest.mark.parametrize("trace", ["fx", "export", "decompose-empty"])
def test_structured_rms_dtype_shape_export_matrix(dtype, shape, trace):
    from neptune_recognize.match import recognize_structured_regions

    x, gamma = torch.randn(shape).to(dtype), torch.randn(shape[-1]).to(dtype)

    class Model(torch.nn.Module):
        def forward(self, x, gamma):
            return _structured_rms(x, gamma)

    if trace == "fx":
        gm = make_fx(lambda x, gamma: _structured_rms(x, gamma))(x, gamma)
    else:
        ep = export(Model(), (x, gamma))
        gm = (ep.run_decompositions({}) if trace == "decompose-empty" else ep).graph_module
    before = str(gm.graph)
    matches, reason = recognize_structured_regions(gm, ("rms_norm",))
    assert len(matches) == 1, reason
    match = matches[0]
    assert match.kind == "structured" and match.operation == match.skeleton == "rms_norm"
    assert match.input_node is match.input_nodes[0]
    assert match.input_nodes[0].meta["val"].dtype == dtype
    assert match.params == {"eps": 1e-6}
    assert match.proof_kind == "exact-structured-real-algebra-with-explicit-rounding"
    assert NR.region_contract(match) == (False, None)
    assert not set(match.input_nodes).intersection(match.region_nodes)
    assert str(gm.graph) == before
    assert recognize_structured_regions(gm, ("segment_pool",))[0] == []
    assert recognize_structured_regions(gm, ())[0] == []
    assert NR.recognize_regions(gm)[0] == [], "scalar default must not gain structured capability"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("square", ["pow", "mul"])
def test_structured_rms_rewrite_shared_boundary_and_rounding(dtype, square, structured_cpu_handle):
    from neptune_recognize.match import recognize_structured_regions

    def model(x, weight, gamma, residual):
        bounded = (x @ weight).clamp(-0.5, 0.75)
        z = bounded.float()
        squared = z ** 2 if square == "pow" else z * z
        mean = squared.mean(-1, keepdim=True, dtype=torch.float32)
        inv = (mean + 1e-6).rsqrt()
        normalized = (inv * z).to(dtype)
        result = gamma.to(dtype) * normalized
        return result + residual, mean, bounded * 3, normalized

    args = (torch.randn(2, 7).to(dtype), torch.randn(7, 512).to(dtype),
            torch.randn(512), torch.randn(2, 512).to(dtype))
    gm = make_fx(model)(*args)
    before = gm(*args)
    matches, reason = recognize_structured_regions(gm)
    assert len(matches) == 1, reason
    match = matches[0]
    assert NR._op(match.input_node) == "clamp"
    if dtype != torch.float32:
        assert NR._op(match.input_nodes[1]) == "_to_copy", "gamma conversion stays a boundary"
    mean = match.consumer_node
    normalized = next(node for node in gm.graph.nodes if node.op == "output").args[0][3]
    guard_target = torch.ops.aten._assert_tensor_metadata.default
    with gm.graph.inserting_after(match.input_node):
        guard = gm.graph.call_function(guard_target, (match.input_node,), {"dtype": dtype})
    guard.meta["val"] = None
    old_name = match.region_root.name
    gm.meta["neptune_replaced_roots"] = {"existing": "untouched"}
    from neptune_recognize.rewrite import rewrite_match
    with patch("neptune_recognize.rewrite._launch_config", side_effect=AssertionError("no autotune")):
        call, refusal = rewrite_match(gm, match, plan="serial", lib="structured_test")
    assert refusal is None and call.args == match.input_nodes
    assert isinstance(call.target, torch._ops.OpOverload)
    assert structured_cpu_handle[0][:4] == ("rms_norm", {"eps": 1e-6}, "serial", "structured_test")
    assert mean in gm.graph.nodes and normalized in gm.graph.nodes
    assert guard in gm.graph.nodes and all(node in gm.graph.nodes for node in match.input_nodes)
    assert gm.meta["neptune_replaced_roots"] == {"existing": "untouched", old_name: call.name}
    torch.testing.assert_close(gm(*args), before, rtol=0, atol=0)
    # The registered fake rule must work without executing any backend compiler.
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode() as mode:
        fake_x, fake_gamma = mode.from_tensor(args[3]), mode.from_tensor(args[2]).to(dtype)
        assert call.target(fake_x, fake_gamma).shape == fake_x.shape


@pytest.mark.parametrize("variant", ["gamma-before-cast", "eps-after-rsqrt", "no-fp32", "axis0",
                                     "no-dims", "no-keepdim", "gamma-rank2", "gamma-wide",
                                     "mean-fp64", "eps-zero", "eps-negative", "eps-infinite"])
def test_structured_rms_semantic_negatives(variant):
    from neptune_recognize.match import recognize_structured_regions

    dtype = torch.float16
    x = torch.randn(8, 8).to(dtype)
    gamma = torch.randn((1, 8) if variant == "gamma-rank2" else (8,)).to(dtype)

    def model(x, gamma):
        z = x if variant == "no-fp32" else x.float()
        squared = z * z
        if variant == "no-dims":
            mean = squared.mean()
        else:
            mean = squared.mean(0 if variant == "axis0" else -1,
                                keepdim=variant != "no-keepdim",
                                dtype=torch.float64 if variant == "mean-fp64" else None)
        eps = {"eps-zero": 0., "eps-negative": -1., "eps-infinite": float("inf")}.get(variant, 1e-6)
        inv = mean.rsqrt() + eps if variant == "eps-after-rsqrt" else (mean + eps).rsqrt()
        if variant == "gamma-before-cast":
            return (z * inv * gamma.float()).to(dtype)
        weight = gamma.expand(8, 8) if variant == "gamma-wide" else gamma
        return (z * inv).to(dtype) * weight

    gm = make_fx(model)(x, gamma)
    before = str(gm.graph)
    assert recognize_structured_regions(gm)[0] == []
    assert str(gm.graph) == before


@pytest.mark.parametrize("defect", ["output-shape", "output-dtype", "gamma-dtype", "gamma-shape",
                                    "gamma-device", "layout", "unknown-meta"])
def test_structured_contract_refuses_metadata_before_handle(defect, monkeypatch):
    import neptune_codegen as CG
    from neptune_recognize.match import recognize_structured_regions
    from neptune_recognize.rewrite import rewrite_match

    gm = make_fx(lambda x, gamma: _structured_rms(x, gamma))(torch.randn(2, 512), torch.randn(512))
    matches, reason = recognize_structured_regions(gm)
    assert len(matches) == 1, reason
    match = matches[0]
    gamma = match.input_nodes[1]
    if defect == "output-shape":
        match.region_root.meta["val"] = torch.empty(2, 1, 512)
    elif defect == "output-dtype":
        match.region_root.meta["val"] = torch.empty(2, 512, dtype=torch.float64)
    elif defect == "gamma-dtype":
        gamma.meta["val"] = torch.empty(512, dtype=torch.float16)
    elif defect == "gamma-shape":
        gamma.meta["val"] = torch.empty(1, 512)
    elif defect == "gamma-device":
        gamma.meta["val"] = torch.empty(512, device="meta")
    elif defect == "layout":
        gamma.meta["val"] = torch.empty(512).to_sparse()
    else:
        gamma.meta.clear()
    monkeypatch.setattr(CG, "state_op_handle", lambda *a, **k: pytest.fail("must not compile"), raising=False)
    before = str(gm.graph)
    assert NR.region_contract(match)[0] is None
    assert rewrite_match(gm, match)[0] is None
    assert recognize_structured_regions(gm)[0] == []
    assert str(gm.graph) == before


@pytest.mark.parametrize("reduce", ["sum", "mean"])
@pytest.mark.parametrize("offsets", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(5,), (5, 7), (5, 2, 3)])
@pytest.mark.parametrize("initial", [None, 0.0, -2.5])
def test_structured_segment_cpu_matrix(reduce, offsets, dtype, shape, initial, structured_cpu_handle):
    from neptune_recognize.match import recognize_structured_regions
    from neptune_recognize.rewrite import rewrite_match

    data = torch.randn(shape).to(dtype)
    descriptor = torch.tensor([0, 2, 2, 5] if offsets else [2, 0, 3], dtype=torch.int64)

    def model(data, descriptor):
        kwargs = {"offsets" if offsets else "lengths": descriptor}
        pooled = torch.segment_reduce(data, reduce, initial=initial, unsafe=True, **kwargs)
        return pooled + 0.5, data * 2

    gm = make_fx(model)(data, descriptor)
    expected = gm(data, descriptor)
    matches, reason = recognize_structured_regions(gm, ("segment_pool",))
    assert len(matches) == 1, reason
    match = matches[0]
    assert match.skeleton == "segment_" + reduce
    assert match.operation == "segment_pool" and match.kind == "structured"
    assert match.params == {"initial": initial, "offsets": offsets, "unsafe": True}
    assert len(match.region_nodes) == 1 and NR.region_contract(match) == (False, None)
    call, refusal = rewrite_match(gm, match)
    assert refusal is None and call.args == match.input_nodes
    torch.testing.assert_close(gm(data, descriptor), expected, rtol=0, atol=0, equal_nan=True)
    if reduce == "mean" and initial is None:
        assert torch.isnan(gm(data, descriptor)[0][1]).all()
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode() as mode:
        result = call.target(mode.from_tensor(data), mode.from_tensor(descriptor))
        assert tuple(result.shape) == (3, *shape[1:]) and result.dtype == dtype


@pytest.mark.parametrize("trace", ["export", "decompose-empty", "fx-buffer"])
@pytest.mark.parametrize("descriptor_dtype", [torch.int32, torch.int64])
def test_structured_segment_exports_and_buffers(trace, descriptor_dtype):
    from neptune_recognize.match import recognize_structured_regions

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("lengths", torch.tensor([2, 0, 3], dtype=descriptor_dtype))

        def forward(self, data):
            return torch.segment_reduce(data, "mean", lengths=self.lengths)

    data, model = torch.randn(5, 512), Model()
    if trace == "fx-buffer":
        gm = make_fx(model)(data)
    else:
        ep = export(model, (data,))
        gm = (ep.run_decompositions({}) if trace == "decompose-empty" else ep).graph_module
    metadata_before = {node.name: dict(node.meta) for node in gm.graph.nodes}
    matches, reason = recognize_structured_regions(gm)
    assert len(matches) == 1, reason
    assert matches[0].params == {"initial": None, "offsets": False, "unsafe": False}
    from neptune_recognize.match import _structured_boundary_value
    assert _structured_boundary_value(gm, matches[0].input_nodes[1]).dtype == descriptor_dtype
    assert all(node.meta == metadata_before[node.name] for node in gm.graph.nodes), "recognition mutated source metadata"


@pytest.mark.parametrize("variant", ["axis", "indices", "both", "neither", "reduce", "data-dtype",
                                     "descriptor-dtype", "descriptor-rank", "output-shape", "initial-inf"])
def test_structured_segment_negatives(variant):
    from neptune_recognize.match import recognize_structured_regions

    data, lengths = torch.randn(5, 7), torch.tensor([2, 3])
    gm = make_fx(lambda x, lengths: torch.segment_reduce(x, "sum", lengths=lengths))(data, lengths)
    node = next(node for node in gm.graph.nodes if node.target == torch.ops.aten.segment_reduce.default)
    kwargs = dict(node.kwargs)
    if variant == "axis":
        kwargs["axis"] = 1
    elif variant == "indices":
        kwargs["indices"] = kwargs.pop("lengths")
    elif variant == "both":
        kwargs["offsets"] = kwargs["lengths"]
    elif variant == "neither":
        kwargs.pop("lengths")
    elif variant == "reduce":
        node.args = (node.args[0], "max")
    elif variant == "data-dtype":
        node.args[0].meta["val"] = torch.empty(5, 7, dtype=torch.float64)
    elif variant == "descriptor-dtype":
        kwargs["lengths"].meta["val"] = torch.empty(2, dtype=torch.float32)
    elif variant == "descriptor-rank":
        kwargs["lengths"].meta["val"] = torch.empty(1, 2, dtype=torch.int64)
    elif variant == "output-shape":
        node.meta["val"] = torch.empty(2, 1, 7)
    else:
        kwargs["initial"] = float("inf")
    node.kwargs = kwargs
    before = str(gm.graph)
    assert recognize_structured_regions(gm)[0] == []
    assert str(gm.graph) == before


def test_structured_multiple_regions_and_mutation_refusal():
    from neptune_recognize.match import Match, recognize_structured_regions

    def model(x, y, gamma):
        return _structured_rms(x, gamma), _structured_rms(y, gamma)

    args = (torch.randn(2, 512), torch.randn(3, 512), torch.randn(512))
    gm = make_fx(model)(*args)
    matches, reason = recognize_structured_regions(gm)
    assert len(matches) == 2, reason
    assert matches[0].input_node is not matches[1].input_node
    assert matches[0].input_nodes[1] is matches[1].input_nodes[1]
    assert Match(None, "legacy", {}).input_nodes == ()
    assert Match(None, "legacy", {}).kind == "scalar"
    assert "unsupported" in recognize_structured_regions(gm, ("unknown",))[1]

    def mutable(x, gamma):
        result = _structured_rms(x, gamma)
        gamma.add_(1)
        return result

    gm = make_fx(mutable)(args[0], args[2].clone())
    assert "functionalized" in recognize_structured_regions(gm)[1]
    assert recognize_structured_regions(gm)[0] == []


def main() -> None:
    print("=" * 100)
    print("RECOGNITION FROM THE ALGEBRA -- no per-operator pattern rows")
    print(f"registry: {', '.join(__import__('neptune_codegen').supported())}")
    print("=" * 100)
    passed = failed = 0
    for label, want, got, ok, detail in _run_cases():
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        shown = f"{got}" + (f"  {detail}" if detail and ok else "")
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:32s} -> {shown}")
        if not ok:
            print(f"         expected {want}")
    print("=" * 100)
    print(f"TOTAL: {passed} passed, {failed} failed")
    print("""Reading:
  * every row here is matched by COMPARING EXPRESSIONS against the generator's registry.
    There is no `if 'tanh' in chain` anywhere -- adding a skeleton to the registry extends
    recognition automatically, and recognition cannot drift from what can be generated.
  * parameters are INFERRED: nothing above tells the recogniser that the soft-cap uses
    beta=0.5, that the decay offset is 2.0, the ALiBi bias 1.5, or the attention temperature 8.
  * the NEGATIVE rows must stay refused, and the two most important are `x.var(-1)` and
    `x.std(-1)`: their DEFAULT is the /(N-1) form while the kernel computes /N, so fusing them
    would silently change the numbers.""")
    print("=" * 100)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
