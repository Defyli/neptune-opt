"""
The skeleton spec and its registry.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import random
import sys
import tempfile

import sympy as sp

from neptune_codegen.symbols import *  # noqa: F401,F403


# Powers with a precision-safe implementation. p >= 3 loses the small-sum precision that
# a plain cbrt preserves, and Triton has no fp32 `cbrt`; `pnorm3` below is kept only as a
# negative control for the test suite and is never part of the supported surface.
NEGATIVE_CONTROLS = {"pnorm3"}


@dataclasses.dataclass(frozen=True)
class Skeleton:
    """A fusable reduction skeleton, expressed as ALGEBRA rather than as source text.

    name       identifier, also the generated function name
    g          the term producer g(s, c) as a sympy expression -- the ONLY operator
               specific input. From it the generator derives:
                   the consumer term  T(c, s') = g(s', c)
                   the repair factor  R(s, s') = h/u = g(s',c)/g(s,c)
    finalize   sympy expression turning the accumulator back into the result,
               over (acc, m) -- the reduction's own scaling convention
    producer   "signed" (m = max v) or "abs" (m = max |v|); see the module docstring
    init       initial accumulator value
    doc        provenance note
    is_log     marks the exponential fold, whose producer must be signed
    """

    name: str
    g: sp.Expr
    finalize: sp.Expr
    producer: str = "abs"
    init: str = "0.0"
    doc: str = ""
    is_log: bool = False
    # Extra scalar parameters the algebra mentions (eps, alpha, p, ...). Each is a
    # (sympy symbol, kernel argument name) pair; the generator appends them to the
    # generated kernel's signature and binds them when translating. Without this, any
    # family whose `g` mentions a scalar is refused with "no Triton binding" -- which was
    # 2 of the EXACT families in the coverage study (ALiBi bias, decay renormalisation).
    # A tuple, not a dict: Skeleton is a frozen dataclass and must stay hashable.
    params: tuple = ()
    # Generalises `producer`: the expression whose row-maximum is the running producer
    # value, written over c. `producer="abs"` is just `Abs(c)` and `"signed"` is `c`, so
    # the two-valued enum was never fundamental.
    #
    # It stops being adequate for soft-capped attention, where g = exp(tanh(beta*c) - s)
    # and the producer must be max tanh(beta*c), NOT max c. Subtracting max c instead
    # underflows: with max c = 100 every term exp(tanh(beta*c) - 100) is 0 and the result
    # becomes 100 + log(1e-38) instead of log(sum(exp(tanh(beta*c)))).
    producer_expr: sp.Expr | None = None
    # The producer's and the consumer's reducers are INDEPENDENT axes. This stayed hidden
    # until RMSNorm+max was attempted: its producer is mean(x^2) -- an ADDITIVE reduction --
    # while its consumer is a max. Every earlier skeleton happened to be
    # (producer=max, consumer=sum), so both were hardcoded, and the paper's own headline
    # example was inexpressible for a reason that had nothing to do with the algebra.
    producer_reducer: str = "max"     # "max" | "sum"
    consumer_reducer: str = "sum"     # "sum" | "max"
    # Default VALUES for the declared scalars: ((sympy_symbol, float), ...). The generator
    # itself never invents a value -- `param_bindings` raises if one is missing -- because a
    # silently wrong constant is precisely the failure this project keeps finding. These are
    # for the test suite and the coverage probe, where the exact number does not matter.
    defaults: tuple = ()
    # Parameters the KERNEL ITSELF supplies at run time: ((sympy_symbol, intrinsic_name), ...)
    # where the name is one of `symbols.INTRINSICS`. A shape-coupled parameter belongs here
    # rather than in `params`: baking the row length as a literal makes the generated kernel
    # (and therefore the AOTI artifact) specific to one shape, and makes it impossible to
    # compile at all when the shape is symbolic.
    intrinsic_params: tuple = ()
    # ---- the two-pass plan: what to do when NO repair exists ---------------------------
    # The repair frame assumes the accumulator can be rewritten as the producer moves. Some
    # operators have no such rewrite, and then the accumulator is not the problem -- the TERM
    # is: for g = (c - mean)^2 the term needs the mean ITSELF, and no rescaling of a running
    # sum produces it (this is the paper's NEEDS_STATE group: five families in the coverage
    # study). The frame then re-reads the row instead of repairing it:
    #
    #     pass 1  reduce the producer       m = sum(c / ncols)     # the mean, as a graph
    #                                                               # spells it (see below)
    #     pass 2  reduce the consumer term  acc = sum((c - m)^2 / ncols)
    #
    # The producer is CONSTANT during pass 2, so there is nothing to repair and the whole
    # `repair_pair` derivation is skipped. The cost is a second read of the row; the gain is
    # that a two-kernel decomposed form becomes one kernel with no materialised intermediate.
    # What `check_supported` requires instead is that the term READS the producer (or pass 1
    # is dead work) and that `producer_expr` says what the graph reduces.
    two_pass: bool = False
    # NB: there is deliberately no `producer_finalize`. A first version had one, to turn the
    # raw reduction into the value the term reads (e.g. a mean). That was redundant, and worse,
    # it made the operator UNRECOGNISABLE: `producer_expr` is already the per-element operand
    # the graph reduces, so the mean belongs there, as `c / N` -- which is how `rmsnorm_max`
    # had been describing a mean of squares all along. The recogniser compares that operand
    # against the graph, so expressing the mean any other way cannot match. What cannot go in
    # `producer_expr` goes in `g`, which may use `s` however it likes (a root, a shift, ...).


SKELETONS: dict[str, Skeleton] = {}


def register(sk: Skeleton) -> Skeleton:
    SKELETONS[sk.name] = sk
    return sk


def supported() -> list[str]:
    """Skeletons whose algebra the generator can lower and that are validated."""
    # Imported lazily: `emitters` needs `Skeleton` from this module, so a module-level
    # import here would be a cycle.
    from neptune_codegen.emitters import check_supported

    out = []
    for name, sk in sorted(SKELETONS.items()):
        if name in NEGATIVE_CONTROLS:
            continue
        try:
            check_supported(sk)
        except NotImplementedError:
            continue
        out.append(name)
    return out


def all_registered() -> list[str]:
    return sorted(SKELETONS)


def unsupported_reason(name: str) -> str:
    """Why a registered skeleton is not emittable (useful for tests and diagnostics)."""
    from neptune_codegen.emitters import check_supported

    try:
        check_supported(SKELETONS[name])
        return ""
    except NotImplementedError as e:
        return str(e)


# ---------------------------------------------------------------- references
def param_defaults(name: str) -> dict:
    """Convenience values for a skeleton's scalars, so tests need not spell them out.

    Both the reference AND the kernel must use the same numbers, and they do: `run()` and
    `as_triton_op()` also default to these unless told otherwise. A skeleton may override the
    fallback per parameter (the log-family readout's additive constant defaults to 0, not 1,
    because the reference `torch.logsumexp` has no such constant).
    """
    sk = SKELETONS[name]
    declared = {sym: float(v) for sym, v in sk.defaults}
    intrinsic = {sym for sym, _name in sk.intrinsic_params}
    return {arg: declared.get(sym, 1.0)
            for sym, arg in sk.params if sym not in intrinsic}
