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

import torch

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
    m = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
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


def main() -> None:
    print("=" * 100)
    print("RECOGNITION FROM THE ALGEBRA -- no per-operator pattern rows")
    print(f"registry: {', '.join(__import__('neptune_codegen').supported())}")
    print("=" * 100)
    passed = failed = 0
    for want, fn in CASES:
        ep = export(M(fn).eval(), (X,))
        match, reason = NR.recognize(ep.graph_module)
        got = match.skeleton if match else reason
        ok = got == want
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        detail = ""
        if match and match.inferred:
            detail = "  inferred " + ", ".join(
                f"{k}={v:g}" for k, v in sorted(match.params.items()))
        print(f"  [{'PASS' if ok else 'FAIL'}] {fn.__name__:16s} -> {got}{detail}")
        if not ok:
            print(f"         expected {want}")
    for label, fn in NEGATIVE:
        ep = export(M(fn).eval(), (X,))
        match, got = NR.recognize(ep.graph_module)
        ok = match is None
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:24s} -> refused ({got[:58]})")
    print("=" * 100)
    print(f"TOTAL: {passed} passed, {failed} failed")
    print("""Reading:
  * every row here is matched by COMPARING EXPRESSIONS against the generator's registry.
    There is no `if 'tanh' in chain` anywhere -- adding a skeleton to the registry extends
    recognition automatically, and recognition cannot drift from what can be generated.
  * parameters are INFERRED: nothing above tells the recogniser that the soft-cap uses
    beta=0.5, that the decay offset is 2.0, or that the ALiBi bias is 1.5.""")
    print("=" * 100)


if __name__ == "__main__":
    main()
