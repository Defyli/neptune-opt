"""
Every piece of Triton text the generator writes, derived not templated.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import random
import sys
import tempfile

import sympy as sp

from neptune_codegen.algebra import _marked, repair_factor, term_expr
from neptune_codegen.skeleton import Skeleton
from neptune_codegen.symbols import *  # noqa: F401,F403
from neptune_codegen.sym_to_triton import TranslationError, to_triton


# ---------------------------------------------------------------- derived emitters
def check_supported(sk: Skeleton) -> None:
    """Refuse to emit a skeleton the generator cannot lower correctly."""
    if sk.producer not in ("signed", "abs"):
        raise NotImplementedError(f"unknown producer mode {sk.producer!r}")
    if sk.producer_reducer not in ("max", "sum"):
        raise NotImplementedError(f"unknown producer reducer {sk.producer_reducer!r}")
    if sk.consumer_reducer not in ("sum", "max"):
        raise NotImplementedError(f"unknown consumer reducer {sk.consumer_reducer!r}")
    for _sym, intrinsic in sk.intrinsic_params:
        if intrinsic not in INTRINSICS:
            raise NotImplementedError(
                f"skeleton {sk.name!r}: unknown intrinsic {intrinsic!r}; have {INTRINSICS}"
            )
    if sk.consumer_reducer == "max" and sk.is_log:
        raise NotImplementedError(
            f"skeleton {sk.name!r}: an exponential fold needs a SUM consumer (a max of "
            "exponentiated terms is a different operator)"
        )
    if sk.producer_expr is not None:
        # must translate, and must not smuggle in the accumulator or the fresh value
        for bad, what in ((s2_sym, "the fresh producer value"), (u_sym, "u")):
            if sk.producer_expr.has(bad):
                raise NotImplementedError(
                    f"skeleton {sk.name!r}: producer_expr mentions {what}"
                )
        # Translatability only: bind parameters to their NAME, not to a literal, because
        # this check has no values and must not require them.
        names = {c_sym: "v", s_sym: "m", s2_sym: "den", u_sym: "acc",
                 _FINAL_ACC: "acc", _FINAL_M: "m"}
        names.update({sym: name for sym, name in sk.params})
        names.update(intrinsic_bindings(sk))
        to_triton(sk.producer_expr, names)
    if sk.is_log and sk.producer != "signed":
        raise NotImplementedError(
            f"skeleton {sk.name!r}: an exponential fold requires producer='signed' "
            "(exp(v - m) <= 1); max|v| collapses rows whose values are all negative"
        )
    repair_factor(sk)          # raises if the algebra is not translatable
    term_expr(sk)


def derived_producer_expr(sk: Skeleton) -> sp.Expr | None:
    """The expression whose row-maximum IS the producer value, derived where possible.

    For an exponential fold there is no choice to make: g = exp(phi(c) - s) forces the
    producer to be max phi(c), because any other choice makes the exponent underflow. And
    since log(exp(x)) = x, phi = log(g) + s -- so it need not be declared. This turns
    `producer="signed"` into a special case of a derived quantity.

    For the power families (g = (c/s)^k) the required producer is |c|, which is not
    recoverable this way (solving g = 1 for s is ambiguous for even k), so those keep the
    declared `producer` mode.
    """
    if sk.producer_expr is not None:
        return sk.producer_expr
    if sk.is_log:
        phi = sp.simplify(sp.log(sk.g) + s_sym)
        if not phi.has(s_sym) and not phi.has(s2_sym) and not phi.has(u_sym):
            return phi
    return None


def producer_nonneg(sk: Skeleton) -> bool:
    """Can the producer value be negative? Decides the identity element for the max.

    For a maximum, the neutral element is -inf unless every value is provably >= 0, in
    which case 0.0 is equivalent and cheaper. `Abs(c)` is nonnegative; `c` and
    `tanh(beta*c)` are not (sympy reports None, and None must mean "could be negative").
    """
    phi = derived_producer_expr(sk)
    if phi is not None:
        return phi.is_nonnegative is True
    return sk.producer != "signed"


def producer_init(sk: Skeleton) -> str:
    """The producer reducer's identity element.

    A sum producer always starts at 0; a max producer starts at -inf unless every value is
    provably non-negative, in which case 0 is equivalent and cheaper.
    """
    if sk.producer_reducer == "sum":
        return "0.0"
    return "0.0" if producer_nonneg(sk) else "-float('inf')"


def consumer_init(sk: Skeleton) -> str:
    """The consumer reducer's identity element."""
    return "-float('inf')" if sk.consumer_reducer == "max" else "0.0"


def producer_operand(sk: Skeleton, values: dict | None = None) -> str:
    """Operand of the running max: signed value for an exponential fold, magnitude else."""
    phi = derived_producer_expr(sk)
    if phi is not None:
        return to_triton(phi, _bindings(sk, values))
    return "v" if sk.producer == "signed" else "a"


def guard_value(sk: Skeleton) -> str:
    """The producer value meaning 'nothing consumed yet' (the repair is the identity).

    Must match `producer_init`: the guard is what makes the FIRST block's repair the
    identity, so it has to compare against the same neutral element.
    """
    return producer_init(sk)


def denominator_expr(sk: Skeleton) -> str:
    """Triton expression for the fresh producer value used in the repair and the term.

    Term/repair divide by it for power semantics (L2 and friends), so those need a floor
    to avoid 0/0 on an all-zero row. An exponential fold never divides, so it must keep
    the raw value -- flooring there would map a negative max to ~0 and shift every
    exponent.
    """
    return "m_new" if sk.is_log else "tl.maximum(m_new, 1e-30)"


def _bindings(sk: Skeleton, values: dict | None = None) -> dict:
    """Map skeleton symbols onto the Triton variables in scope inside the loop.

    Single source of truth for symbol -> source-name binding, so a new parameter cannot
    be bound in one place and forgotten in another.
    """
    names = {
        s_sym: "m",          # stale = running value before this block's update
        # Fresh producer value. An exponential fold must use the RAW value: flooring a
        # negative max with max(m, 1e-30) destroys it (measured 1e-3 relative error on
        # all-negative rows). Only a fold that divides needs the floor.
        s2_sym: "den",
        c_sym: "v",
        u_sym: "acc",
        _FINAL_ACC: "acc",
        _FINAL_M: "m",
    }
    names.update(param_bindings(sk, values))
    return names


def intrinsic_bindings(sk: Skeleton) -> dict:
    """Parameters the generated kernel already has in scope, e.g. `N -> ncols`."""
    return {sym: name for sym, name in sk.intrinsic_params}


def param_bindings(sk: Skeleton, values: dict | None = None) -> dict:
    """Bind each scalar parameter to a SOURCE LITERAL, not to a kernel argument.

    Passing them as runtime scalars actively broke. A Python float reaches the kernel as
    fp64, so `2.0 * v * beta` promoted the whole term to fp64 and Triton rejected the
    kernel with

        Loop-carried variable m has initial type fp32 but is re-assigned to fp64 in loop

    Emitting a literal instead is also what this module already does for every other
    knob (epsilon, the exp clamps, BLOCK) and for the stated reason: Triton rejects a
    module-level global inside @triton.jit, so constants must be written into the source.
    """
    values = values or {}
    # Intrinsic parameters are not in `sk.params`, so they must be seeded separately --
    # iterating only `params` silently left `N` unbound.
    out = dict(intrinsic_bindings(sk))
    for sym, arg_name in sk.params:
        if sym in out:
            continue
        if arg_name not in values:
            raise KeyError(
                f"skeleton {sk.name!r}: no value supplied for scalar parameter "
                f"{arg_name!r}; pass it to compile_skeleton/emit_kernel_source"
            )
        # repr(float(x)) round-trips exactly, so the kernel sees the value asked for.
        out[sym] = repr(float(values[arg_name]))
    return out


def producer_update(sk: Skeleton, values: dict | None = None) -> str:
    """Source for the producer reduction over one block."""
    operand = producer_operand(sk, values)
    if sk.producer_reducer == "sum":
        return f'm_new = m + tl.sum(tl.where(mask, {operand}, 0.0), axis=0)'
    return (f'm_new = tl.maximum(m, tl.max(tl.where(mask, {operand}, '
            f'-float("inf")), axis=0))')


def consumer_update(sk: Skeleton, values: dict | None = None) -> str:
    """Source for the consumer fold over one block.

    Both reducers keep the same shape, `fold(repair(acc), block_reduce(term))`; only the two
    functions change. For a MAX consumer the repaired accumulator must not win spuriously,
    which needs the repair to be a positive factor -- true for every repair the generator
    derives with a max consumer (a ratio of positive normalisers), and the reason a max
    consumer is only accepted together with such a repair.
    """
    if sk.consumer_reducer == "max":
        term = term_source(sk, values, fill='-float("inf")')
        return f'acc = tl.maximum(acc * scale, tl.max({term}, axis=0))'
    term = term_source(sk, values, fill="0.0")
    return f'acc = acc * scale + tl.sum({term}, axis=0)'


def scale_source(sk: Skeleton, values: dict | None = None) -> str:
    """Triton source for the repair factor R(m, safe_m)."""
    return to_triton(_marked(repair_factor(sk)), _bindings(sk, values))


def term_source(sk: Skeleton, values: dict | None = None, fill: str = "0.0") -> str:
    """Triton source for the consumer term, masked so out-of-range lanes contribute 0.

    The mask matters whenever ncols is not a multiple of BLOCK: those lanes load as 0.0,
    and for an exponential fold a bare exp(0 - m) would add a spurious 1.0 each.
    """
    return (f"tl.where(mask, {to_triton(_marked(term_expr(sk)), _bindings(sk, values))}, "
            f"{fill})")


def finalize_source(sk: Skeleton, values: dict | None = None) -> str:
    return to_triton(sk.finalize, _bindings(sk, values))
