"""
The README's limits table, asserted against the code -- in both directions.

A limit that is described wrongly is worse than no limit, because it is believed. The README's
limits section had already gone stale twice by the time this file was written (it claimed
`nn.LogSoftmax` was unreachable after that stopped being true, and that any textual difference in
the operator's readout is refused, which was never true of arithmetic *outside* the region). So the
table in `README.md` carries a machine-readable key per row, and this file is the other half:

  * every key in the README must have a check here (a stale row fails);
  * every check here must appear in the README (a removed row fails).

None of it needs a GPU or Triton: what is being asserted is what the pass DECIDES, and it decides
before any kernel exists.

The stub-kernel trick
---------------------
Two rows are about what the rewrite does to a graph rather than about what it recognises. Without
Triton, `rewrite` cannot build a kernel -- but building the kernel is not what those rows are about,
so `CG.op_handle` is replaced by a plain aten op with the same `(Tensor) -> Tensor` shape. The graph
transformation is then exercised for real, with a result that is wrong numerically and irrelevant.

Run:  python tests/test_readme_limits.py     (or pytest tests/test_readme_limits.py)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import neptune_codegen as CG  # noqa: E402
import neptune_opt as NO  # noqa: E402
import neptune_recognize as NR  # noqa: E402
from neptune_recognize.lowering import lower_if_it_helps  # noqa: E402

X = torch.randn(8, 32)


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


def _stub_kernel(fn):
    """Run `fn` with kernel generation replaced by a no-op `(Tensor) -> Tensor` op."""
    real = CG.op_handle
    CG.op_handle = lambda *a, **k: torch.ops.aten.clone.default        # noqa: ARG005
    try:
        return fn()
    finally:
        CG.op_handle = real


# ---------------------------------------------------------------- the checks
def check_region_is_the_operator():
    """Only the operator's own region is replaced; a different readout is refused."""
    # outside the region: arithmetic on the result is preserved, not matched, not deleted
    outer = _graph(lambda t: torch.logsumexp(t, dim=-1) * 2.0)
    assert NR.recognize(outer.graph_module)[0] is not None, (
        "logsumexp * 2 stopped being recognised: the region gate got greedy")
    ep = _graph(lambda t: torch.logsumexp(t, dim=-1) * 2.0)
    report = _stub_kernel(lambda: NO.apply(ep))
    assert len(report.fusions) == 1, report.summary()
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
    def operator(t):
        m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
        return (m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True))).squeeze(-1)

    def two(t):
        return operator(t) + operator(t)          # two in-frame operators, one graph

    ep = _graph(two)
    report = _stub_kernel(lambda: NO.apply(ep))
    assert len(report.fusions) == 1, (
        f"expected exactly one fused region, got {[f.skeleton for f in report.fusions]}")
    # and the OTHER operator is still there: this limit is about reach, not about deleting work
    ops = [str(n.target) for n in ep.graph_module.graph.nodes if n.op == "call_function"]
    assert any("sqrt" in o for o in ops), f"the second operator was deleted as well: {ops}"


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
    report = NO.apply(ep)
    reasons = [r for r, _d in report.skipped]
    assert "needs-decomposition" in reasons, (
        f"a non-decomposed graph no longer says so: {report.summary()}")


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


def test_every_limit_holds():
    """Each check is a measurement; a failure here is a false claim in the README."""
    for key, fn in sorted(CHECKS.items()):
        try:
            fn()
        except AssertionError as e:
            raise AssertionError(f"README limit {key!r} is false: {e}") from None


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
