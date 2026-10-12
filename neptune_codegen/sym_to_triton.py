"""
Symbolic -> Triton source translation.

This is what makes the code generator an actual generator rather than a table of
hand-written templates. Until now a skeleton carried hand-written Triton snippets; now it
carries SYMPY EXPRESSIONS and the source is emitted from them.

Two facts make the translation small. Both follow from the repair framework itself rather
than from anything operator-specific:

  1. THE REPAIR FACTOR IS DERIVED, NOT DECLARED.
     repairlib derives h(u, s, s') with h(g(s,c), s, s') = g(s',c) and
     h(x f y) = h(x) f h(y). Writing h = u * R(s, s') and substituting t = g(s, c)
     gives g(s,c) * R = g(s',c), i.e. R = g(s',c) / g(s,c). So R depends only on the
     producer pair, and dividing the derived h by u produces it mechanically:

         logsumexp:  h = u*exp(s - s')      ->  R = exp(s - s')
         stable_l2:  h = s^2*u/s'^2         ->  R = (s/s')^2

     That is why no `power` field is needed: the exponent is an OUTPUT of the
     derivation, not an input to the generator.

  2. THE TERM IS JUST g.
     The consumer's per-element contribution is g evaluated at the current producer
     value, normalised by the running value. No operator-specific code is involved.

So a skeleton is (producer semantics, g, finalize) and everything else is compiled.
"""
from __future__ import annotations

import sympy as sp


class TranslationError(NotImplementedError):
    """Raised when an expression cannot be lowered, instead of emitting wrong code."""


# Triton's f32 limits: exp underflows below ~-87 and overflows above ~88. Both clamps are
# applied where the argument is an expression we do not control.
_EXP_LO, _EXP_HI = -80.0, 80.0

_BINOPS = {sp.Add: "+", sp.Mul: "*"}
# functions rendered as-is when Triton provides them
_DIRECT_FUNCS = {sp.sqrt: "tl.sqrt", sp.exp: "tl.exp", sp.log: "tl.log", sp.Abs: "tl.abs"}
_POW_SYMS = {"triton": None}  # placeholder to keep linters quiet


def _atom(text: str) -> bool:
    """True when `text` needs no surrounding parentheses as an operand."""
    if text.replace(".", "", 1).replace("_", "").isalnum():
        return True
    return False


def _wrap(text: str) -> str:
    return text if _atom(text) else f"({text})"


def _func_name(fn) -> str | None:
    for sym, name in _DIRECT_FUNCS.items():
        if fn is sym or fn == sym:
            return name
    if fn is sp.sin:
        return "tl.sin"
    if fn is sp.cos:
        return "tl.cos"
    return None


class _Q(sp.Function):
    """Marker for a quotient raised to an integer power.

    Sympy canonicalises (a/b)**2 to a**2/b**2, discarding the form that is numerically
    better in fp32. We carry the intended quotient explicitly and expand it to a product
    of quotients at render time.
    """

    nargs = 3


def _expand_q(expr: sp.Expr) -> sp.Expr:
    def repl(e):
        if isinstance(e, _Q):
            num, den, k = e.args
            k = int(k)
            return sp.Mul(*[sp.Mul(num, sp.Pow(den, -1))] * k)
        return e

    return expr.replace(lambda e: isinstance(e, _Q), repl)


def _require_positive_base(base, exp):
    """exp/log lowers only a positive base; refuse anything else.

    Triton has no `**` on tensors, so a non-integer power becomes exp(log(base)*exp).
    That is a valid rewrite on the positive reals and silently wrong outside them.
    """
    if not base.is_positive:
        raise TranslationError(
            f"cannot lower {base} ** {exp}: a non-integer exponent must go through "
            "exp/log, which requires a provably positive base. Use an integer exponent "
            "(emitted as repeated multiplication) instead."
        )


def to_triton(expr, names: dict, *, indent: int = 0) -> str:
    """Lower a sympy expression to a Triton source fragment.

    `names` maps each sympy Symbol to the Triton variable holding it.

    Deliberately conservative: anything unrecognised raises TranslationError rather than
    emitting source that might compute the wrong thing.
    """
    expr = sp.sympify(expr)

    if expr.is_Number:
        val = float(expr)
        # literal in the source: Triton rejects module-level globals inside @triton.jit
        return repr(int(val)) if val.is_integer() else repr(val)

    # Quotient power: emit (a/b)*(a/b) rather than a^2/b^2. Sympy stores the latter, but
    # it squares both operands first and loses fp32 precision.
    if isinstance(expr, _Q):
        num, den, k = expr.args
        q = (f"{_wrap(to_triton(num, names, indent=indent))} / "
             f"{_wrap(to_triton(den, names, indent=indent))}")
        return " * ".join([f"({q})"] * int(k))

    if isinstance(expr, sp.Symbol):
        if expr not in names:
            raise TranslationError(f"no Triton binding for symbol {expr}")
        return names[expr]

    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        exp_s = sp.simplify(exp)
        # sympy represents sqrt(x) as Pow(x, 1/2), so this must be handled before the
        # generic fractional-power path or the output becomes exp(log(x)*0.5).
        if exp_s == sp.Rational(1, 2):
            return f"tl.sqrt({to_triton(base, names, indent=indent)})"
        if exp_s == sp.Rational(-1, 2):
            # 1/sqrt(x) needs no positivity assumption: tl.sqrt already returns nan for a
            # negative argument, which is exactly x**(-1/2). Routing this through
            # exp(log(x)*-0.5) would be both slower and no safer.
            b = to_triton(base, names, indent=indent)
            return f"(1.0 / tl.sqrt({b}))"
        if exp_s == sp.Rational(1, 3):
            # NOTE: this is exp(log(x)/3), not a real cube root. A negative base gives nan
            # here, so only allow it where the base is provably positive.
            _require_positive_base(base, exp_s)
            b = to_triton(base, names, indent=indent)
            return f"tl.exp(tl.log({b}) / 3.0)"
        if exp_s.is_Integer and exp_s.is_number:
            k = int(exp_s)
            if k == 0:
                return "1.0"
            if k == 1:
                return to_triton(base, names, indent=indent)
            if k > 1:
                b = to_triton(base, names, indent=indent)
                if b.isidentifier() or b.replace(".", "").replace("_", "").isalnum():
                    return " * ".join([b] * k)
                return " * ".join([f"({b})"] * k)
            b = to_triton(base, names, indent=indent)
            denom = " * ".join([_wrap(b)] * (-k))
            return f"(1.0 / ({denom}))" if _atom(b) else f"(1.0 / {_wrap(denom)})"
        # Any remaining exponent (a symbolic `p`, or a fraction with no primitive) has to
        # go through exp/log, because Triton has no `**` on tensors. That rewrite is only
        # valid for a POSITIVE base: tl.log of a negative base is nan, while a negative
        # base with an integer exponent is an ordinary real number.
        #
        # Measured failure this guard prevents: g = (c/s)**p with producer='abs' used to
        # emit  term = tl.exp(tl.log(v * (1.0/den)) * p)  -- nan for every negative v,
        # which is exactly what an abs producer expects to see. Silent nan is worse than
        # a refusal.
        _require_positive_base(base, exp_s)
        b = to_triton(base, names, indent=indent)
        p = to_triton(exp_s, names, indent=indent)
        return f"tl.exp(tl.log({b}) * {p})"

    if isinstance(expr, sp.Mul):
        coeff, rest = expr.as_coeff_Mul()
        if coeff == -1:
            return f"-{_wrap(to_triton(rest, names, indent=indent))}"
        parts = [to_triton(a, names, indent=indent) for a in expr.args]
        return " * ".join(_wrap(p) for p in parts)

    if isinstance(expr, sp.Add):
        rendered = [to_triton(a, names, indent=indent) for a in expr.args]
        out = rendered[0]
        for r in rendered[1:]:
            out += f" - {_wrap(r[1:])}" if r.startswith("-") else f" + {_wrap(r)}"
        return out

    if isinstance(expr, sp.exp):
        arg = to_triton(expr.args[0], names, indent=indent)
        return f"tl.exp(tl.maximum(tl.minimum({arg}, {_EXP_HI}), {_EXP_LO}))"

    if isinstance(expr, sp.log):
        arg = to_triton(expr.args[0], names, indent=indent)
        return f"tl.log({arg})"

    # NOTE: there is deliberately no `isinstance(expr, sp.sqrt)` branch here. `sp.sqrt`
    # is a FUNCTION, not a class, so such a branch raises
    #     TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
    # for every expression that reaches it. That branch existed and was unreachable until
    # tanh support was added, at which point it fired. sqrt is always Pow(x, 1/2) and is
    # handled in the Pow branch above.

    if isinstance(expr, sp.Abs):
        arg = to_triton(expr.args[0], names, indent=indent)
        return f"tl.abs({arg})"

    if isinstance(expr, sp.tanh):
        # Triton 3.6 has no tanh: both `tl.tanh` and `tl.math.tanh` are absent (verified
        # on the target machine). tanh(x) = 2*sigmoid(2x) - 1 is exact and tl.sigmoid
        # exists. The previous mapping to "tl.math.tanh" was a LATENT bug: translation
        # succeeded, so the failure only surfaced when Triton compiled the kernel.
        arg = to_triton(expr.args[0], names, indent=indent)
        return f"(2.0 * tl.sigmoid(2.0 * {arg}) - 1.0)"

    # generic function application
    if isinstance(expr, sp.Function):
        name = _func_name(type(expr))
        if name is None:
            name = _func_name(expr.func)
        if name is None:
            raise TranslationError(f"unsupported function {type(expr).__name__}")
        args = ", ".join(to_triton(a, names, indent=indent) for a in expr.args)
        return f"{name}({args})"

    if isinstance(expr, sp.Max):
        args = [to_triton(a, names, indent=indent) for a in expr.args]
        out = args[0]
        for a in args[1:]:
            out = f"tl.maximum({out}, {a})"
        return out

    if isinstance(expr, sp.Min):
        args = [to_triton(a, names, indent=indent) for a in expr.args]
        out = args[0]
        for a in args[1:]:
            out = f"tl.minimum({out}, {a})"
        return out

    raise TranslationError(f"unsupported expression: {expr} ({type(expr).__name__})")


def roundtrip_check(expr, names: dict) -> bool:
    """True if the expression can be lowered (used by tests to probe coverage)."""
    try:
        to_triton(expr, names)
        return True
    except TranslationError:
        return False
