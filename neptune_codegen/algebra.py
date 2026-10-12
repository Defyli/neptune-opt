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


def _check_distributes(sk: Skeleton, a: sp.Expr) -> None:
    """Condition (4) of the paper, for the affine repair h(t) = a*t + b.

    For an additive consumer (f = x+y) condition (4) reads
        a*(x+y) + b == (a*x + b) + (a*y + b)
    which holds for every a but ONLY when b == 0 -- so a nonzero bias is refused by the
    caller, and there is nothing to check here.

    For a max consumer (f = max) it reads
        a*max(x,y) + b == max(a*x + b, a*y + b)
    which holds for ANY b provided a > 0. sympy can only ever REFUTE a > 0 here: the
    producer symbols are merely real, so `is_nonnegative` is None even for a correct
    factor. Do not demand `is_nonnegative is True` -- for rmsnorm_max the factor is
    sqrt(eps+s)/sqrt(eps+s2), whose sign is guaranteed by the runtime floor on the divisor
    in the emitted kernel (`tl.maximum(m_new, 1e-30)`), not by the algebra; requiring a
    proof would refuse a repair that is in fact correct.
    """
    if sk.consumer_reducer == "max" and a.is_negative is True:
        raise NotImplementedError(
            f"skeleton {sk.name!r}: repair factor {sp.simplify(a)} is negative, so it does "
            "not distribute over the max reducer (condition 4 fails)"
        )


# The derivation is pure and its inputs are hashable sympy expressions, so it is memoized.
# The authors' own `HExprSolver` does the same, and their comment gives the reason: the derivation
# "may be called thousands of times due to the autotuner replaying traces". The expensive part is
# the `simplify` calls, which is exactly what this avoids repeating.
@functools.lru_cache(maxsize=None)
def repair_pair(sk: Skeleton) -> tuple[sp.Expr, sp.Expr]:
    """Derive (a, b) with  acc_new = a(s,s') * acc_old + b(s,s').

    Two separations are attempted, in this order, and this is the whole derivation:

      * MULTIPLICATIVE -- the running producer s rescales every already-accumulated term
        g(s, c_j) at once, so the single factor that promotes them to g(s', c_j) is
        R = g(s', c)/g(s, c), taken when the data c cancels out of it.
      * AFFINE -- when the ratio still depends on c, the remaining repair shape is
        h(t) = a*t + b. Condition (3) `a*g(s,c) + b == g(s',c)` is solved for a by
        matching the c-DERIVATIVES, which is what makes both sides share one c-dependence:
        a = g'_c(s',c) / g'_c(s,c), then b = g(s',c) - a*g(s,c). `b` free of c is exactly
        the statement that no c-dependence is left over, so it is the acceptance test.

    The multiplicative case is the affine one with b = 0, and it is tried first because it
    is both cheaper and the shape the emitted kernel already assumed. Deleting the
    `sp.solve`-based version in favour of this did not change a single derived repair: all
    eight registered skeletons take the multiplicative branch and produce the identical R
    (checked for each one in tests/test_neptune_opt.py).
    """
    # A term producer may depend only on the value it is paired with (s) and the data
    # (c). `s'` is the value the producer takes AFTER this element is consumed, so a `g`
    # that mentions it is not a (producer, term) pair at all.
    #
    # This guard exists because without it the derivation does not fail -- it returns a
    # confident, meaningless answer. Measured on a deliberately invalid skeleton
    # `g = c * s'`: R came out as 1 and the emitted term as `(v*den)/1`, i.e. a kernel
    # that compiles, runs, and computes the wrong function. Silent wrongness is worse
    # than a refusal, so the precondition is checked before the separation.
    for sym, which in ((s2_sym, "the FRESH producer value s'"), (u_sym, "the term value u")):
        if sk.g.has(sym):
            raise NotImplementedError(
                f"skeleton {sk.name!r}: g mentions {which}. A term producer may only "
                "depend on the stale producer value s and the data c; a skeleton that "
                "needs s' has no (f, g, h) form. Refused because the derivation would "
                "otherwise return a plausible but meaningless repair."
            )
    # A term producer that never reads the running producer value is not a (producer,
    # term) pair: nothing couples the accumulator to `s`, so the repair collapses to the
    # identity and the emitted kernel silently ignores the reduction it was supposed to
    # repair. This is not a hypothetical -- `g = exp(c - |c|)` (the classifier's deliberate
    # out-of-framework control) separates perfectly, because a c-only g has the c-free
    # ratio 1.
    #
    # `R = 1` is vacuously a valid repair, so the algebra cannot reject it; the reason to
    # refuse is that the operator is not a fold at all, and emitting it would make the
    # generator contradict the classifier, which reports this family as NOT_INVERT. The
    # generator refuses rather than silently disagreeing.
    if not sk.g.has(s_sym):
        raise NotImplementedError(
            f"skeleton {sk.name!r}: g does not depend on the producer s at all, so the "
            "repair is the identity and the accumulator is unrelated to the reduction. "
            "This is not an (f, g, h) fold."
        )
    g_at_s2 = sk.g.xreplace({s_sym: s2_sym})
    raw = g_at_s2 / sk.g
    a = _simplify(raw)
    # The whole soundness of the multiplicative repair is this test. The accumulator is
    # sum_j g(s, c_j), and the factor that converts it to sum_j g(s', c_j) is a single
    # number only if the ratio does not depend on j -- i.e. only if c cancels. Separating
    # the variables is the entire obligation; there is nothing further to prove. (This is
    # the same test the authors' `arith.sympy.separate_vars` performs.)
    if not a.has(c_sym):
        # `_simplify` must not have changed the function while separating it: if it had,
        # the kernel would be wrong in a way the cancellation test cannot see.
        if not vanishes(a - raw):
            raise NotImplementedError(
                f"skeleton {sk.name!r}: separating g changed its value, so the derived "
                f"repair is not g(s',c)/g(s,c); got {sp.simplify(a)}"
            )
        _check_distributes(sk, a)
        return a, sp.Integer(0)
    # ---- affine branch: the ratio keeps a c-dependence that a shift can absorb --------
    # Match the c-derivatives of the two sides of a*g(s,c) + b == g(s',c). A `g` with no
    # c-dependence at all has both derivatives zero, and there is then no way to fix the
    # ratio's c-dependence -- that is a genuine refusal, not a division to force.
    d_fresh, d_stale = sp.diff(g_at_s2, c_sym), sp.diff(sk.g, c_sym)
    # A zero derivative means g is constant in c, so the ratio above was already c-free
    # and this branch is unreachable for such a g; treat it as unseparated rather than
    # dividing by zero.
    a_aff = _simplify(d_fresh / d_stale) if (d_stale != 0 and d_fresh != 0) else a
    if a_aff.has(c_sym):
        raise NotImplementedError(
            f"skeleton {sk.name!r}: g does not separate, so no repair of the form "
            f"a(s,s')*t + b(s,s') exists; the multiplicative factor is {sp.simplify(a)} and "
            f"the derivative-matched slope is {sp.simplify(a_aff)}"
        )
    b = _simplify(g_at_s2 - a_aff * sk.g)
    if b.has(c_sym):
        raise NotImplementedError(
            f"skeleton {sk.name!r}: g does not separate, so neither g(s',c)/g(s,c) nor the "
            f"affine repair leaves a data-free remainder; got b = {sp.simplify(b)}"
        )
    # Condition (4) for an additive consumer admits only b == 0 (see _check_distributes),
    # so a nonzero bias is usable only with a max consumer -- which is exactly where the
    # additive-shift families live.
    if b != 0 and sk.consumer_reducer != "max":
        raise NotImplementedError(
            f"skeleton {sk.name!r}: the repair needs a bias b = {sp.simplify(b)}, but an "
            f"additive {sk.consumer_reducer} reducer requires b = 0 (condition 4: "
            "a*(x+y)+b != (a*x+b)+(a*y+b) unless b = 0)"
        )
    _check_distributes(sk, a_aff)
    return _simplify(a_aff), b


def repair_factor(sk: Skeleton) -> sp.Expr:
    """a(s, s') from `repair_pair`:  acc_new = a * acc_old + b."""
    return repair_pair(sk)[0]


def repair_bias(sk: Skeleton) -> sp.Expr:
    """b(s, s') from `repair_pair`. Zero for every multiplicative repair."""
    return repair_pair(sk)[1]


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
