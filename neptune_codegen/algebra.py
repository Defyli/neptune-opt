"""
Condition (3)/(4) as executable code: deriving the repair and the term.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import functools
import random
import sys
import tempfile

import sympy as sp

from neptune_codegen.sym_to_triton import _Q
from neptune_codegen.symbols import *  # noqa: F401,F403


# Functions whose "zero on the principal branch" does not mean "identically zero": the
# numeric fallback must not be trusted in their presence.
_BRANCH_CUT = (sp.log, sp.atanh, sp.acosh, sp.atan, sp.acos, sp.asin)


def _simplify(e: sp.Expr) -> sp.Expr:
    """sp.simplify, but rewrite hyperbolics through exp first.

    Without the rewrite, `h/u` for the soft-capped attention family stays as
        exp(-s2 + tanh(log(-sqrt(-(s + log(u) + 1)/(s + log(u) - 1)))))
    instead of `exp(s - s2)`: the same function, but it does not factor as u*R by
    inspection, so a perfectly good EXACT operator was reported unsupported. Rewriting in
    terms of exp lets the exponentials cancel. Mirrors repairlib.simplify deliberately --
    the generator must not contradict the classifier.
    """
    try:
        rewritten = e.rewrite(sp.exp)
    except (AttributeError, NotImplementedError):
        rewritten = e
    return sp.simplify(sp.simplify(rewritten))


# The derivation is pure and its inputs are hashable sympy expressions, so it is memoized.
# The authors' own `HExprSolver` does the same, and their comment gives the reason: the derivation
# "may be called thousands of times due to the autotuner replaying traces". The expensive part is
# the `simplify` calls, which is exactly what this avoids repeating.
@functools.lru_cache(maxsize=None)
def repair_factor(sk: Skeleton) -> sp.Expr:
    """R(s, s') such that  acc_new = acc_old * R + term.

    Derived, not declared: h = u*R by construction, so dividing the derived repair by `u`
    isolates the factor. Returns an expression in (s, s').
    """
    # A term producer may depend only on the value it is paired with (s) and the data
    # (c). `s'` is the value the producer takes AFTER this element is consumed, so a `g`
    # that mentions it is not a (producer, term) pair at all.
    #
    # This guard exists because without it the derivation does not fail -- it returns a
    # confident, meaningless answer. Measured on a deliberately invalid skeleton
    # `g = c * s'`: R came out as 1 and the emitted term as `(v*den)/1`, i.e. a kernel
    # that compiles, runs, and computes the wrong function. Silent wrongness is worse
    # than a refusal, so the precondition is checked before the inversion.
    for sym, which in ((s2_sym, "the FRESH producer value s'"), (u_sym, "the inversion variable u")):
        if sk.g.has(sym):
            raise NotImplementedError(
                f"skeleton {sk.name!r}: g mentions {which}. A term producer may only "
                "depend on the stale producer value s and the data c; a skeleton that "
                "needs s' has no (f, g, h) form. Refused because the derivation would "
                "otherwise return a plausible but meaningless repair."
            )
    inv = sp.solve(sp.Eq(sk.g, u_sym), c_sym)
    if not inv:
        raise NotImplementedError(
            f"skeleton {sk.name!r}: g is not invertible in c, so Eq.5 does not apply"
        )
    a = sp.Symbol("a", real=True)
    h = sk.g.xreplace({s_sym: s2_sym}).subs({c_sym: inv[0].xreplace({s_sym: a})})
    h = h.xreplace({a: s_sym})
    R = _simplify(h / u_sym)
    if R.has(u_sym):
        raise NotImplementedError(
            f"skeleton {sk.name!r}: the repair does not factor as u * R(s,s'); "
            f"got h={sp.simplify(h)}"
        )
    residual = R * sk.g - sk.g.xreplace({s_sym: s2_sym})
    if sp.simplify(residual) != 0 and not _numerically_zero(residual):
        raise NotImplementedError(
            f"skeleton {sk.name!r}: derived R does not satisfy R*g(s,c) = g(s',c)"
        )
    return R


def vanishes(expr: sp.Expr, samples: int = 9, tol: float = 1e-9) -> bool:
    """Is `expr` identically zero? Symbolic first, numeric fallback second.

    The symbolic attempt is not enough on its own: `sp.simplify` cannot prove identities
    involving Max (even for a concrete k, `Max(x,y)*k - Max(x*k,y*k)` stays unevaluated),
    and it cannot prove `(s/s')**p * (c/s)**p == (c/s')**p` for a symbolic exponent. Both
    are true identities that the simpler would reject, which turned correct repairs and
    correct matches into "unsupported".

    The fallback is a FILTER, not a proof, so it is deliberately fenced:
      * several random rational points, fixed seed, so a verdict never changes between runs
      * disabled outright when the residual contains a branch-cut function, because a
        residual can vanish on sympy's chosen branch and nowhere else (measured: the
        soft-capped attention residual evaluated to ~1e-152 at every sampled point while the
        two sides genuinely differ off-branch)
      * the symbolic attempt always runs first

    Mirrors `repairlib.vanishes` on purpose: the two modules are deployed independently
    (repairlib is CPU-only and never ships to the GPU host), and the generator must not
    contradict the classifier it is built on.
    """
    if expr == 0:
        return True
    if sp.simplify(expr) == 0:
        return True
    if expr.has(sp.I) or any(expr.has(f) for f in _BRANCH_CUT):
        return False
    return _numerically_zero(expr, samples=samples, tol=tol)


def _numerically_zero(expr: sp.Expr, samples: int = 7, tol: float = 1e-9) -> bool:
    """Does `expr` vanish at every one of several random rational points?

    A fallback for the identity check, NOT a replacement for it. `sp.simplify` cannot
    prove `(s/s')**p * (c/s)**p == (c/s')**p` for a SYMBOLIC exponent, so a correct repair
    for a p-norm skeleton was being rejected (measured: p=3 derives fine, symbolic p did
    not). Substituting random positive rationals and evaluating settles it in practice.

    Soundness: this is a filter, not a proof. A repair that is wrong would have to vanish
    at all sampled points to slip through, which is why several points are used and why
    the symbolic check is always tried first. Callers that need certainty should still
    compare against a reference implementation numerically -- which the test suite does.
    """
    free = sorted(expr.free_symbols, key=lambda z: z.name)
    if not free:
        return expr == 0
    # A fixed seed: a checker whose verdict changes between runs is not a checker.
    rng = __import__("random").Random(20251012)
    for _ in range(samples):
        point = {z: sp.Rational(rng.randint(1, 40), rng.randint(1, 17)) for z in free}
        try:
            val = abs(complex(sp.N(expr.subs(point), 30)))
        except (TypeError, ValueError):
            return False
        if val > tol * max(1.0, val):
            return False
    return True


def _marked(R: sp.Expr) -> sp.Expr:
    """Apply the quotient-power marker only at render time.

    It must not be applied to the expression used for the algebraic check: the marker is
    an inert function, so sympy cannot multiply it and the verification would fail.
    """
    return _as_quotient_power(R)


@functools.lru_cache(maxsize=None)
def term_expr(sk: Skeleton) -> sp.Expr:
    """The consumer term, evaluated at the current producer value: g(s', c).

    Plain `sp.simplify` on purpose -- NOT `_simplify`. The exp rewrite must be confined
    to isolating R from `h/u`: applying it here expands `tanh(beta*c)` into
    `(exp(2*beta*c) - 1)/(exp(2*beta*c) + 1)`, which is the same function but emitted as a
    huge expression with the exponential repeated several times. Keeping tanh intact lets
    the translator lower it to `2*sigmoid(2x) - 1`.
    """
    return sp.simplify(sk.g.xreplace({s_sym: s2_sym}))


def _as_quotient_power(expr: sp.Expr) -> sp.Expr:
    """Re-express num**k / den**k as a marked quotient power.

    Sympy stores (a/b)**2 as a**2/b**2. The two forms agree on real numbers but NOT in
    fp32: squaring both operands before dividing loses precision (measured ~9e-2 relative
    error versus ~2e-6). When numerator and denominator are the same integer power, the
    intended quotient is recoverable, so we mark it and let the renderer emit
    (a/b)*(a/b).
    """
    num, den = sp.fraction(sp.together(expr))
    if num == 1 and den == 1:
        return expr
    nb, ne = num.as_base_exp()
    db, de = den.as_base_exp()
    if ne.is_Integer and ne == de and int(ne) > 0:
        return _Q(nb, db, int(ne))
    return expr
