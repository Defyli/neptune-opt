"""
Historical README limit-key inventory and explicit legacy-mode regressions.

The keys are documentation inventory, NOT a statement of current default capability.
Default apply now supports multiple regions and selective compound lowering without
core decomposition. Historical bounded behavior is tested with max_regions=1;
needs-decomposition is tested as an explicit lower=False no-fusion contract.

A limit that is described wrongly is worse than no limit, because it is believed. The README's
limits section had already gone stale twice by the time this file was written (it claimed
`nn.LogSoftmax` was unreachable after that stopped being true, and that any textual difference in
the operator's readout is refused, which was never true of arithmetic *outside* the region). So the
table in `README.md` carries a machine-readable key per row, and this file is the other half:

  * every historical key in the README must have an explicit-mode check here;
  * every inventory check here must appear in the README.

This guards key inventory, not the truth of historical prose about today's defaults;
additional tests below assert enhanced default behavior directly.

None of it needs a GPU or Triton: what is being asserted is what the pass DECIDES, and it decides
before any kernel exists.

The stub-kernel trick
---------------------
Two rows are about what the rewrite does to a graph rather than about what it recognises. Without
Triton, `rewrite` cannot build a kernel -- but building the kernel is not what those rows are about,
so `CG.op_handle` is replaced by a CPU logsumexp custom op with registered fake metadata.
The actual graph transformation, export verifier and numerical verification are exercised;
no GPU kernel, compilation or service is involved.

Run:  python tests/test_readme_limits.py     (or pytest tests/test_readme_limits.py)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from unittest.mock import patch

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import neptune_codegen as CG  # noqa: E402
import neptune_opt as NO  # noqa: E402
import neptune_recognize as NR  # noqa: E402
from neptune_recognize.lowering import lower_if_it_helps  # noqa: E402

X = torch.randn(8, 32)


def _validate_exported(program):
    # Torch 2.4 predates validate(), but must still run its complete EP verifier.
    if hasattr(program, "validate"):
        program.validate()
    else:
        program.verifier().check(program)


class M(nn.Module):
    """A module wrapper: `torch.export` needs a module, and lambdas are not one."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, t):
        return self.fn(t)


def _graph(fn, decompose: bool = True):
    ep = torch.export.export(M(fn), (X,))
    return ep.run_decompositions() if decompose else ep


def _verdict(fn, decompose: bool = True):
    """(skeleton | None, reason) for a model, recognised from its decomposed graph."""
    gm = _graph(fn, decompose).graph_module
    match, why = NR.recognize(gm)
    return (match.skeleton if match else None), str(why)


@torch.library.custom_op("neptune_readme_test::lse", mutates_args=(),
                         schema="(Tensor t) -> Tensor")
def _cpu_lse(t: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(t, dim=-1)


@_cpu_lse.register_fake
def _cpu_lse_fake(t):
    return t.new_empty(t.shape[:-1])


def _stub_kernel(fn):
    """Exercise real transaction/verification with a correct CPU reduction ABI."""
    def handle(name, *, keepdim, **kwargs):
        assert name == "logsumexp" and not keepdim, (name, keepdim)
        return _cpu_lse._opoverload

    with patch.object(CG, "op_handle", side_effect=handle):
        return fn()


# ---------------------------------------------------------------- the checks
def check_region_is_the_operator():
    """Only the operator's own region is replaced; a different readout is refused."""
    # outside the region: arithmetic on the result is preserved, not matched, not deleted
    outer = _graph(lambda t: torch.logsumexp(t, dim=-1) * 2.0)
    assert NR.recognize(outer.graph_module)[0] is not None, (
        "logsumexp * 2 stopped being recognised: the region gate got greedy")
    ep = _graph(lambda t: torch.logsumexp(t, dim=-1) * 2.0)
    mod = M(lambda t: torch.logsumexp(t, dim=-1) * 2.0)
    report = _stub_kernel(lambda: NO.apply(ep, verify_with=(mod, (X,))))
    assert len(report.fusions) == 1 and report.verified, report.summary()
    _validate_exported(ep)
    torch.testing.assert_close(ep.module()(X), mod(X))
    ops = [str(n.target) for n in ep.graph_module.graph.nodes if n.op == "call_function"]
    assert any("mul" in o for o in ops), (
        f"the `* 2` was deleted by the rewrite, not preserved: {ops}")

    # inside the region: an algebraically EQUAL but differently written log-sum-exp is refused
    def equal_but_different(t):
        m = t.amax(dim=-1, keepdim=True)
        return m + 0.5 * (2 * (t - m)).exp().sum(dim=-1).log()

    sk, why = _verdict(equal_but_different)
    assert sk is None, f"an algebraically rewritten readout was matched as {sk!r}"
    assert "no registered skeleton matches" in why, why


def check_one_operator_per_graph():
    """Historical key now means opt-in bounded mode, not a default capability limit."""
    def two(t):
        return torch.logsumexp(t, dim=-1) + torch.logsumexp(torch.sin(t), dim=-1)

    ep = _graph(two, decompose=False)
    report = _stub_kernel(lambda: NO.apply(ep, max_regions=1, verify_with=(M(two), (X,))))
    assert len(report.fusions) == 1 and report.verified, report.summary()
    targets = [n.target for n in ep.graph_module.graph.nodes if n.op == "call_function"]
    assert targets.count(_cpu_lse._opoverload) == 1
    # The unused compound must remain unexpanded, not merely remain numerically correct.
    assert targets.count(torch.ops.aten.logsumexp.default) == 1
    _validate_exported(ep)
    torch.testing.assert_close(ep.module()(X), two(X))


def check_row_reductions_only():
    for label, fn in (("sum(dim=0)", lambda t: (t * t).sum(dim=0)),
                      ("mean(dim=0)", lambda t: t.mean(dim=0)),
                      ("amax(dim=0)", lambda t: t.abs().amax(dim=0))):
        sk, why = _verdict(fn)
        assert sk is None, f"{label} was fused as {sk!r} -- a non-row reduction was reinterpreted"
        assert "reduction pair" in why or "dim" in why or "1 reduction" in why, why


def check_var_default_is_unbiased():
    for label, fn in (("var(dim=-1)", lambda t: t.var(dim=-1)),
                      ("std(dim=-1)", lambda t: t.std(dim=-1))):
        sk, why = _verdict(fn)
        assert sk is None, f"{label} was fused as {sk!r}: its default is the /(N-1) form"
    sk, _why = _verdict(lambda t: torch.var(t, dim=-1, unbiased=False))
    assert sk is not None, "var(unbiased=False) is no longer served"


def check_single_reduction_operators():
    class LN(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(32)

        def forward(self, t):
            return self.ln(t)

    class RMS(nn.Module):
        def __init__(self):
            super().__init__()
            self.r = nn.RMSNorm(32)

        def forward(self, t):
            return self.r(t)

    for label, mod in (("nn.LayerNorm", LN()), ("nn.RMSNorm", RMS())):
        gm = torch.export.export(mod, (X,)).run_decompositions().graph_module
        assert NR.recognize(gm)[0] is None, f"{label} became fusible: the frame is not what it says"
    sk, why = _verdict(lambda t: torch.softmax(t, dim=-1))
    assert sk is None and "softmax" in why, why


def check_softmax_internals_not_lowered():
    soft = _graph(lambda t: torch.softmax(t, dim=-1)).graph_module
    log_soft = _graph(lambda t: torch.log_softmax(t, dim=-1)).graph_module
    assert lower_if_it_helps(soft) == 0, "softmax was lowered: the 0.59x measurement was ignored"
    assert lower_if_it_helps(log_soft) == 1, "log_softmax is no longer lowered by the pass"


def check_needs_decomposition():
    ep = _graph(lambda t: torch.logsumexp(t, dim=-1), decompose=False)
    before = (ep.graph_module, ep.graph_module.graph, ep.graph_module.code, ep.graph_signature)
    with patch.object(CG, "op_handle", side_effect=AssertionError("lower=False must not generate")):
        report = NO.apply(ep, lower=False)
    assert not report.fusions and not report.graph_changed and not report.verified
    assert report.skipped and all(reason != "needs-decomposition" for reason, _ in report.skipped)
    assert any(reason == "no-fusion" and detail for reason, detail in report.skipped), report.summary()
    assert ep.graph_module is before[0] and ep.graph_module.graph is before[1]
    assert ep.graph_module.code == before[2] and ep.graph_signature is before[3]
    _validate_exported(ep)


def check_prod_ratio_needs_a_count():
    out = subprocess.run([sys.executable, os.path.join(_ROOT, "tools", "coverage_codegen.py"),
                          "--check"], capture_output=True, text=True, cwd=_ROOT)
    assert out.returncode == 0, out.stdout[-400:] + out.stderr[-400:]
    assert "prod + ratio" in out.stdout, "the refusing family is gone from the coverage report"
    assert "f-needs-carried-state" in out.stdout, (
        "the documented reason for refusing `prod + ratio` is no longer the one reported")


def check_lower_only_when_it_pays():
    gm = _graph(lambda t: torch.relu(t) + 1.0).graph_module
    before = str(gm.graph)
    assert lower_if_it_helps(gm) == 0
    assert str(gm.graph) == before, "the lowering changed a graph it cannot improve"


CHECKS = {
    "region-is-the-operator": check_region_is_the_operator,
    "one-operator-per-graph": check_one_operator_per_graph,
    "row-reductions-only": check_row_reductions_only,
    "var-default-is-unbiased": check_var_default_is_unbiased,
    "single-reduction-operators": check_single_reduction_operators,
    "softmax-internals-not-lowered": check_softmax_internals_not_lowered,
    "needs-decomposition": check_needs_decomposition,
    "prod-ratio-needs-a-count": check_prod_ratio_needs_a_count,
    "lower-only-when-it-pays": check_lower_only_when_it_pays,
}


def readme_keys() -> set:
    """The keys of the limits table in README.md: `| **`key`** | ... |` rows."""
    text = open(os.path.join(_ROOT, "README.md"), encoding="utf-8").read()
    section = text.split("## Known limits", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\|\s*\*\*`([a-z0-9-]+)`\*\*\s*\|", section, re.M))


def test_every_readme_limit_has_a_check():
    missing = readme_keys() - set(CHECKS)
    assert not missing, f"README rows with no check behind them: {sorted(missing)}"


def test_every_check_is_in_the_readme():
    extra = set(CHECKS) - readme_keys()
    assert not extra, f"checks that the README does not document: {sorted(extra)}"


def test_every_historical_key_has_a_valid_explicit_mode_check():
    """Keep historical inventory without claiming those limits apply to defaults."""
    for key, fn in sorted(CHECKS.items()):
        try:
            fn()
        except AssertionError as e:
            raise AssertionError(f"historical README key {key!r} regression failed: {e}") from None


def test_default_apply_selectively_lowers_multiple_regions_without_core_decomposition():
    def two(t):
        return {"first": torch.logsumexp(t, dim=-1),
                "second": torch.logsumexp(torch.sin(t), dim=-1)}

    ep = _graph(two, decompose=False)
    assert sum(n.target == torch.ops.aten.logsumexp.default
               for n in ep.graph_module.graph.nodes) == 2
    report = _stub_kernel(lambda: NO.apply(ep, verify_with=(M(two), (X,))))
    assert len(report.fusions) == 2 and report.verified and report.graph_changed, report.summary()
    targets = [n.target for n in ep.graph_module.graph.nodes if n.op == "call_function"]
    assert targets.count(_cpu_lse._opoverload) == 2
    assert torch.ops.aten.logsumexp.default not in targets
    _validate_exported(ep)
    torch.testing.assert_close(ep.module()(X), two(X))


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
