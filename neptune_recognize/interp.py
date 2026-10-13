"""
Interpreting an FX graph symbolically: the graph becomes the pair the algebra talks about.
"""
from __future__ import annotations

import dataclasses
import random

import sympy as sp

from neptune_codegen import c_sym


# A symbol standing for the max-like reduction's result. Named to be unmistakable in
# diagnostics and to avoid colliding with the data symbol.
S_RESULT = sp.Symbol("$producer", real=True)


# Pointwise ops lifted to sympy. Anything absent is a reason to refuse, not to guess.
_BINARY = {
    "mul": lambda a, b: a * b,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "div": lambda a, b: a / b,
    "truediv": lambda a, b: a / b,     # fx records `t / m` as operator.truediv
    "pow": lambda a, b: a ** b,
}


_UNARY = {
    "exp": sp.exp,
    "log": sp.log,
    "tanh": sp.tanh,
    "abs": sp.Abs,
    "sqrt": sp.sqrt,
    # `rsqrt(x) = 1/sqrt(x)`, a real op (it is how RMSNorm/LayerNorm are written), not a
    # tracing artifact. Understanding it does not make those operators fusable -- they have a
    # single reduction and are outside this frame -- but it lets the interpreter reach that
    # structural verdict instead of reporting `aten.rsqrt` as "unfamiliar".
    "rsqrt": lambda a: 1 / sp.sqrt(a),
    "neg": lambda a: -a,
    "reciprocal": lambda a: 1 / a,
}


# `amax`/`sum` are the reductions the algebra is about. `clamp_min` is a NUMERICAL GUARD
# that the reference decomposition adds (and that our generated kernel adds internally as
# `tl.maximum(value, 1e-30)`), so it must not be treated as part of the operator; treating
# it as the identity is what lets the same pattern match with and without the guard.
# The canonical stand-in for "the row length". The generated kernel knows it as `ncols`, so
# a skeleton can declare a parameter INTRINSIC and match against this symbol instead of a
# number -- which is what makes a shape-coupled operator work with symbolic shapes, where no
# number exists to solve for.
NCOLS_SYM = sp.Symbol("$ncols", positive=True)
_INTRINSIC_SYMBOLS = {"ncols": NCOLS_SYM, "rows": sp.Symbol("$rows", positive=True)}

_REDUCTIONS = ("amax", "sum", "max", "min")


_MEANS = ("mean",)

# `torch.var`/`torch.std` arrive as a SINGLE aten op, not as a producer/consumer pair, so the
# interpreter expands them here. They are mean((x - mean(x))^2) only for the BIASED form.
_VAR_STD = ("var", "std")


_IGNORED = ("clamp_min", "clamp", "squeeze", "unsqueeze", "getitem", "detach", "alias")

# Allocation and literal-tensor construction. An exported module materialises its parameters
# (weight, bias, eps) through these, and they carry no data path: `empty` produces a buffer, and
# `full_like(x, v)` is just the constant `v`. Choking on `aten.empty.memory_format` would hide the
# REAL verdict about an operator -- e.g. that RMSNorm has only one reduction -- behind a message
# about a tensor allocation.
_ALLOC = ("empty", "empty_like", "new_empty", "new_zeros", "new_ones")
_CONST_FILL = ("full_like", "full", "scalar_tensor", "zeros_like", "ones_like")

# Tracing wrappers that ExportedProgram inserts around a module's parameter handling. They carry
# no data path -- nothing in the reduction pattern consumes their result -- so they are skipped
# rather than reported as an "unfamiliar op".
_WRAPPERS = ("wrap_with_set_grad_enabled",)


def _is_zero(x) -> bool:
    """Is `x` the constant 0 -- either a literal or a `scalar_tensor(0.0)` NODE?

    The second form matters: `torch.logsumexp`'s guard passes `scalar_tensor(0.0)` as the
    `where`'s value operand, so checking only for a Python float found no guard at all (measured:
    the table came back empty and the graph was refused for an unrelated-looking reason).
    """
    try:
        return float(x) == 0.0
    except (TypeError, ValueError):
        pass
    if getattr(x, "op", None) == "call_function" and str(x.target).startswith("aten.scalar_tensor"):
        return any(_is_zero(a) for a in x.args)
    return False


def _logsumexp_inf_guard(gm) -> dict:
    """Map `torch.logsumexp`'s infinity guard onto the value it guards.

    `torch.logsumexp` decomposes to

        m = amax(t, -1, keepdim=True)
        m = where(eq(abs(m), inf), 0.0, m)      # <- this
        ...

    The guard replaces an INFINITE max with 0, so that `exp(t - inf)` cannot yield nan. For the
    finite inputs this frame assumes, it is the identity -- the same status as the `clamp_min`
    the reference decomposition adds and the interpreter already ignores -- so it is treated as a
    pass-through.

    Its operands (`abs`, `eq`, `scalar_tensor`) are accepted ONLY inside this shape. `abs` on its
    own is a real operator, and ignoring it in general would silently compute a different
    function; the guard is recognised structurally, from the graph, rather than by name.
    """
    table: dict = {}
    for node in gm.graph.nodes:
        if node.op != "call_function" or _op(node) != "where" or len(node.args) < 3:
            continue
        cond, low, high = node.args[:3]
        if not _is_zero(low) or getattr(cond, "op", None) != "call_function":
            continue
        # cond must be `eq(abs(m), inf)`
        if not str(cond.target).startswith("aten.eq") or not cond.args:
            continue
        inner = cond.args[0]
        if getattr(inner, "op", None) != "call_function":
            continue
        if not str(inner.target).startswith("aten.abs"):
            continue
        table[node] = high
        for structural in (cond, low):
            if getattr(structural, "op", None) == "call_function":
                table[structural] = None
                for sub in structural.args:
                    if getattr(sub, "op", None) == "call_function":
                        table[sub] = None
    return table


def _rank_of(node):
    """The rank of a value, if the graph records it (symbolic shapes still have a rank)."""
    value = node.meta.get("val")
    if value is None:
        value = node.meta.get("example_value")
    shape = getattr(value, "shape", None)
    return None if shape is None else len(shape)


def _var_is_biased(node) -> bool:
    """Is this a `/N` variance, i.e. `unbiased=False` (or `correction=0`)?

    Read, never assumed, because `torch.var`'s DEFAULT is `unbiased=True` -- the `/(N-1)`
    operator, which this registry does not implement. Treating an absent flag as "biased"
    would silently fuse the most common spelling (`x.var(-1)`) into a kernel computing a
    different number, which is precisely the failure this project refuses rather than ships.
    """
    if "correction" in node.kwargs:
        return float(node.kwargs["correction"]) == 0.0
    if "unbiased" in node.kwargs:
        return node.kwargs["unbiased"] is False
    # positional: (self, dim, unbiased[, keepdim])
    return len(node.args) > 2 and node.args[2] is False


def _reduces_last_dim_only(node, rank):
    """Does this reduction reduce exactly the last dimension?

    Nothing checked this before, which was a real hazard for using the pass on a model: the
    interpreter reads `sum`/`amax` as "the row reduction", so a `sum(dim=0)` or a mean over a
    middle dimension would have been silently reinterpreted as a row reduction and fused into
    a kernel computing something else.
    """
    dims = None
    for arg in node.args[1:]:
        if isinstance(arg, bool):
            continue
        if isinstance(arg, int):
            dims = [arg]
            break
        if isinstance(arg, (list, tuple)) and all(isinstance(d, int) for d in arg):
            dims = list(arg)
            break
    if dims is None and "dim" in node.kwargs:
        d = node.kwargs["dim"]
        dims = list(d) if isinstance(d, (list, tuple)) else ([d] if isinstance(d, int) else None)
    if not dims:
        return False                      # reduces everything, or no dim given
    if len(dims) != 1:
        return False
    d = dims[0]
    if d == -1:
        return True
    return rank is not None and d == rank - 1


def _last_dim_of(gm):
    """The concrete last dimension of the graph's first input, if it is known."""
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        # make_fx stores the shape under "val"; a Dynamo graph uses "example_value".
        # Explicit None checks: `a or b` truth-tests the value, and a FakeTensor in
        # `meta["val"]` raises "Boolean value of Tensor with more than one value is
        # ambiguous". Introduced and caught here; the failure gate turned it into a safe
        # refusal rather than a crash, which is why it was visible at all.
        value = node.meta.get("val")
        if value is None:
            value = node.meta.get("example_value")
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0:
            return None
        try:
            return sp.Integer(int(shape[-1]))
        except (TypeError, ValueError):
            return None
    return None


def _op(node) -> str:
    """The op name without namespace or overload.

    `str(node.target)` for an OpOverload is "aten.amax.default", so naively taking the last
    dot-separated component yields the OVERLOAD name ("default", "Tensor", "dim_IntList"),
    not the op. Reading the schema is the only reliable way.

    `call_method` nodes carry the method name as a plain string, which is already what we
    want; both shapes must be accepted because a pass may run on a symbolically traced
    module (methods + torch.* functions) or on an exported one (aten OpOverloads).
    """
    if node.op == "call_method":
        return node.target if isinstance(node.target, str) else str(node.target)
    tgt = getattr(node, "target", None)
    schema = getattr(tgt, "_schema", None)
    if schema is not None:
        return schema.name.split("::")[-1]              # aten::amax -> amax
    name = getattr(tgt, "__name__", None)               # operator.getitem -> getitem
    return name if name else str(tgt)


def _as_expr(a, env):
    """Symbolic value of one operand, or (None, False) if it is not a value operand."""
    if hasattr(a, "op"):
        return env.get(a), True
    if isinstance(a, bool):
        return None, False
    if isinstance(a, int):
        return sp.Integer(a), True
    if isinstance(a, float):
        return sp.Float(a), True
    return None, False          # dims, keepdim, lists: not values


def _flatten_nodes(arg, out: list) -> None:
    """Collect the graph Nodes inside a (possibly nested) output argument."""
    if hasattr(arg, "op"):
        out.append(arg)
    elif isinstance(arg, (tuple, list)):
        for item in arg:
            _flatten_nodes(item, out)
    elif isinstance(arg, dict):
        for item in arg.values():
            _flatten_nodes(item, out)


@dataclasses.dataclass
class Candidate:
    """One (producer, consumer) reduction pair found in the graph."""

    phi: object
    psi: object
    prod_kind: str
    cons_kind: str
    producer_node: object
    consumer_node: object
    consumer_sym: object = None
    producer_sym: object = None


def interpret(gm, why: list | None = None):
    """Symbolically evaluate `gm`.

    Returns (input_node, phi, psi) or None:
        phi  the expression the max-like reduction reduces over (in c)
        psi  the expression the sum-like reduction reduces over (in c and S_RESULT)
    """
    env: dict = {}
    input_node = None
    rank = None
    reductions: list = []
    guard_table = _logsumexp_inf_guard(gm)

    for node in gm.graph.nodes:
        if node in guard_table:
            guarded = guard_table[node]
            if guarded is not None:
                gv, g_is_val = _as_expr(guarded, env)
                if not g_is_val or gv is None:
                    return None
                env[node] = gv
            continue
        if node.op == "placeholder":
            if input_node is None:
                input_node = node
                rank = _rank_of(node)
                env[node] = c_sym
            continue
        if node.op not in ("call_function", "call_method"):
            continue
        name = _op(node)

        if name in _ALLOC or name in _WRAPPERS:
            continue                       # a buffer / a tracing wrapper; no data path

        if name in _CONST_FILL:
            # `full_like(x, v)` (and friends) is the constant v. `zeros_like`/`ones_like` fill
            # with 0/1, which is what makes an epsilon or a bias appear in a graph as a symbol.
            fill = 0.0 if name.startswith("zeros") else 1.0 if name.startswith("ones") else None
            if fill is None:
                for a in node.args[1:]:
                    if isinstance(a, (int, float, bool)):
                        fill = float(a)
                        break
            if fill is None:
                return None
            env[node] = sp.Float(fill)
            continue

        if name in _IGNORED:
            # pass through the first value operand (squeeze/getitem/clamp_min)
            for a in node.args:
                v, is_val = _as_expr(a, env)
                if is_val and v is not None:
                    env[node] = v
                    break
            continue

        if name in _MEANS:
            v, is_val = _as_expr(node.args[0], env) if node.args else (None, False)
            if not is_val or v is None or not _reduces_last_dim_only(node, rank):
                return None
            sym = sp.Symbol(f"$r{len(reductions)}", real=True)
            # A mean is a sum over ncols, and ncols is a RUNTIME value: using the canonical
            # symbol (rather than a number read off the shape) is what lets the same match
            # work for static and symbolic shapes alike.
            reductions.append(("sum", v / NCOLS_SYM, sym, node))
            env[node] = sym
            continue

        if name in _VAR_STD:
            # Expanded into the pair the algebra needs: a mean of c, then a mean of the
            # CENTERED squares. Both are `sum(operand)` with the mean's 1/ncols in the operand,
            # so the symbols this produces are exactly the ones a decomposed graph yields --
            # which is why recognition needs no special case beyond this expansion.
            v, is_val = _as_expr(node.args[0], env) if node.args else (None, False)
            if not is_val or v is None or not _reduces_last_dim_only(node, rank):
                return None
            if not _var_is_biased(node):
                return None                # /(N-1): a different operator, refuse
            prod_sym = sp.Symbol(f"$r{len(reductions)}", real=True)
            reductions.append(("sum", v / NCOLS_SYM, prod_sym, node))
            cons_sym = sp.Symbol(f"$r{len(reductions)}", real=True)
            reductions.append(("sum", (v - prod_sym) ** 2 / NCOLS_SYM, cons_sym, node))
            env[node] = sp.sqrt(cons_sym) if name == "std" else cons_sym
            continue

        if name in _REDUCTIONS:
            v, is_val = _as_expr(node.args[0], env) if node.args else (None, False)
            if not is_val or v is None or not _reduces_last_dim_only(node, rank):
                return None
            kind = "max" if name in ("amax", "max") else "sum"
            sym = sp.Symbol(f"$r{len(reductions)}", real=True)
            reductions.append((kind, v, sym, node))
            env[node] = sym
            continue

        if name in _BINARY:
            vals = []
            for a in node.args[:2]:
                v, is_val = _as_expr(a, env)
                if not is_val:
                    return None
                vals.append(v)
            if None in vals:
                return None
            env[node] = _BINARY[name](vals[0], vals[1])
            continue

        if name in _UNARY:
            v, is_val = _as_expr(node.args[0], env) if node.args else (None, False)
            if not is_val or v is None:
                return None
            env[node] = _UNARY[name](v)
            continue

        # Unknown op: refuse rather than guess -- and NAME it. "not a producer/consumer
        # reduction pair" is what a user sees otherwise, which says nothing about the fact that
        # their graph contains `aten.softmax.int` and would be understood after the core-aten
        # decompositions (`ep.run_decompositions()`). The refusal is the same; only the
        # explanation improves.
        if why is not None:
            why.append(f"unfamiliar op {str(node.target)!r} -- if this is a high-level aten op "
                       f"such as softmax/layer_norm, the core decompositions expose its "
                       f"reductions (ep.run_decompositions())")
        return None

    if input_node is None:
        if why is not None:
            why.append("no input placeholder found")
        return None
    if len(reductions) < 2:
        if why is not None:
            why.append(
                f"only {len(reductions)} reduction(s) found: this frame fuses a PRODUCER and a "
                f"CONSUMER reduction, so a single reduction with elementwise work around it "
                f"(RMSNorm, LayerNorm, softmax's normalisation) is outside it"
            )
        return None

    # Which reduction is the PRODUCER is decided by DEPENDENCY, not by order in the graph.
    # A repair pair has one reduction whose result is consumed inside the other's operand:
    # for logsumexp it is the max that feeds the sum, for RMSNorm+max it is the sum that
    # feeds the max. Assuming "max first" silently mislabelled the second case.
    syms = [r[2] for r in reductions]
    candidates = []
    for i, (ki, vi, si, ni) in enumerate(reductions):
        for j, (kj, vj, sj, nj) in enumerate(reductions):
            if i == j or not vj.has(si) or vi.has(sj):
                continue
            if any(vi.has(other) for other in syms) or any(vj.has(other) for other in syms
                                                          if other is not si):
                continue
            psi = vj.subs({si: S_RESULT})
            if not psi.has(S_RESULT):
                continue
            candidates.append(Candidate(vi, psi, ki, kj, ni, nj, sj, si))
    if not candidates:
        if why is not None:
            why.append(f"{len(reductions)} reductions found, but none depends on another")
        return None
    return input_node, candidates, env
