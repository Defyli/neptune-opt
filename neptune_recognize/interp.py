"""
Interpreting an FX graph symbolically: the graph becomes the pair the algebra talks about.
"""
from __future__ import annotations

import dataclasses
import math

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

_REDUCTIONS = ("amax", "sum", "max")


_MEANS = ("mean",)

# `torch.var`/`torch.std` arrive as a SINGLE aten op, not as a producer/consumer pair, so the
# interpreter expands them here. They are mean((x - mean(x))^2) only for the BIASED form.
_VAR_STD = ("var", "std")


_IGNORED = ("squeeze", "unsqueeze", "getitem", "detach", "alias")

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
_WRAPPERS = (
    "wrap_with_set_grad_enabled",
    # A pure assertion about metadata, inserted by export. It computes nothing and its result is
    # consumed by nobody, so it is skipped rather than reported as an unfamiliar op.
    "_assert_tensor_metadata",
)


def _no_value(node, why, where: str) -> None:
    """Record why a value could not be interpreted.

    Every `return None` in the walk used to be silent about WHICH node defeated it, and the
    caller then reported the generic "graph is not a producer/consumer reduction pair" -- which
    says nothing. Measured: a graph whose reduction was fine but whose parameter plumbing was not
    produced exactly that message, with no clue in it.
    """
    if why is not None:
        why.append(f"{where} could not be interpreted at {str(node.target)!r}")


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
        # Check the entire guard, not merely the names of its first two operations.
        if not str(cond.target).startswith("aten.eq") or len(cond.args) != 2:
            continue
        inner, infinity = cond.args
        if not _positive_infinity(infinity):
            continue
        if getattr(inner, "op", None) != "call_function":
            continue
        if not str(inner.target).startswith("aten.abs") or inner.args != (high,):
            continue
        if getattr(high, "op", None) != "call_function" or _op(high) != "amax":
            continue
        table[node] = high
        for structural in (cond, low):
            if getattr(structural, "op", None) == "call_function":
                table[structural] = None
                for sub in structural.args:
                    if getattr(sub, "op", None) == "call_function":
                        table[sub] = None
    return table


def _positive_infinity(value) -> bool:
    if isinstance(value, float):
        return value == float("inf")
    if getattr(value, "op", None) == "call_function" and _op(value) == "scalar_tensor":
        return bool(value.args) and _positive_infinity(value.args[0])
    return False


def _tensor_value(node):
    value = node.meta.get("val") if hasattr(node, "meta") else None
    return node.meta.get("example_value") if value is None and hasattr(node, "meta") else value


def _same_shape(a, b) -> bool:
    # Symbol names/expressions are compared without truth-testing or concretising SymInts.
    return len(a) == len(b) and all(str(x) == str(y) for x, y in zip(a, b))


def _argument(node, name, default=None):
    if name in node.kwargs:
        return node.kwargs[name]
    schema = getattr(node.target, "_schema", None)
    if schema is not None:
        for index, argument in enumerate(schema.arguments):
            if argument.name == name and index < len(node.args):
                return node.args[index]
    positions = {"dim": 1, "keepdim": 2, "min": 1, "max": 2}
    index = positions.get(name)
    return node.args[index] if index is not None and index < len(node.args) else default


def _exact_floor_guard(node, env) -> bool:
    if _op(node) not in ("clamp", "clamp_min") or not node.args:
        return False
    lower = _argument(node, "min")
    upper = _argument(node, "max") if _op(node) == "clamp" else None
    if not isinstance(lower, (int, float)) or lower != 1e-30 or upper is not None:
        return False
    source = node.args[0]
    expr = env.get(source)
    if expr is None or not getattr(expr, "is_Symbol", False):
        return False
    # Only the documented max(abs(x)) zero guard is interpreted as a coordinate guard.
    if getattr(source, "op", None) != "call_function" or _op(source) != "amax":
        return False
    operand = source.args[0] if source.args else None
    return env.get(operand) == sp.Abs(c_sym)


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
        correction = node.kwargs["correction"]
        return isinstance(correction, (int, float)) and correction == 0
    if "unbiased" in node.kwargs:
        return node.kwargs["unbiased"] is False
    # positional: (self, dim, unbiased[, keepdim])
    return len(node.args) > 2 and node.args[2] is False


def _is_row_flatten(node, env, rank) -> bool:
    """Is this `view`/`reshape` a no-op for a 2-D row-reduction?

    `x.reshape(x.shape[0], -1)` is how a lot of real code normalises a batch into rows before a
    row-wise reduction, and it appears in the project's own reference helpers. On a 2-D input it
    is the identity, so the interpreter can pass through it.

    The rule is deliberately narrow, because a reshape in general is NOT harmless: viewing
    `(B, T, C)` as `(B, T*C)` turns a per-position reduction into a per-sequence one, and treating
    that as the identity would silently compute a different operator. Accepted only when the input
    is 2-D and the target's first entry is the SAME integer as the input's first dimension with
    `-1` for the rest.
    """
    if rank != 2 or not node.args or len(node.args) < 2:
        return False
    if not isinstance(node.args[1], (list, tuple)) or len(node.args[1]) != 2:
        return False
    first, second = node.args[1]
    if second != -1 or not isinstance(first, int):
        return False
    src = node.args[0]
    shape = None
    value = src.meta.get("val") if hasattr(src, "meta") else None
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 2:
        return False
    return isinstance(shape[0], int) and shape[0] == first


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


def _scalar_expr(value):
    return sp.Rational(str(value)) if isinstance(value, float) and math.isfinite(value) else sp.sympify(value)


def _as_expr(a, env):
    """Symbolic value of one operand, or (None, False) if it is not a value operand."""
    if hasattr(a, "op"):
        return env.get(a), True
    if isinstance(a, bool):
        return None, False
    if isinstance(a, int):
        return sp.Integer(a), True
    if isinstance(a, float):
        return _scalar_expr(a), True
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
    producer_guarded: bool = False


class _Refusal(ValueError):
    """An unsupported semantic boundary, not an optimization failure."""


def _value_operand(node, env, index=0):
    value, is_value = _as_expr(node.args[index], env) if len(node.args) > index else (None, False)
    if not is_value or value is None:
        raise _Refusal(f"a value operand could not be interpreted at {str(node.target)!r}")
    return value


def _check_binary_broadcast(node, env):
    operands = [(arg, env.get(arg)) for arg in node.args[:2] if hasattr(arg, "op")]
    data = [arg for arg, expr in operands if expr is not None and expr.has(c_sym)]
    for arg, expr in operands:
        if expr is None or expr.has(c_sym) or not expr.free_symbols or not data:
            continue
        data_value, reduced_value = _tensor_value(data[0]), _tensor_value(arg)
        if data_value is None or reduced_value is None:
            raise _Refusal("reduction broadcast metadata is unknown")
        expected = data_value.shape[:-1] + (1,)
        scalar_row = len(data_value.shape) == 1 and len(reduced_value.shape) == 0
        if not scalar_row and not _same_shape(reduced_value.shape, expected):
            raise _Refusal("a reduction is broadcast across columns instead of its own row")


def _check_data_shape(node, env, anchor):
    expr = env.get(node)
    if expr is None or not expr.has(c_sym):
        return
    source, output = _tensor_value(anchor), _tensor_value(node)
    if source is not None and output is not None and hasattr(output, "shape"):
        if not _same_shape(output.shape, source.shape):
            raise _Refusal("pointwise broadcasting changes row shape; preserve it as a boundary")


def _interpret_node(node, env, reductions, guarded, rank):
    name = _op(node)
    schema = getattr(node.target, "_schema", None)
    if schema is not None and (schema.is_mutable or not schema.name.startswith("aten::")):
        raise _Refusal(f"unsupported or mutable operation {node.target!s}")
    if name in _ALLOC or name in _WRAPPERS:
        return
    if name in ("view", "reshape"):
        source = node.args[0] if node.args else None
        before, after = _tensor_value(source), _tensor_value(node)
        identical = (before is not None and after is not None
                     and _same_shape(before.shape, after.shape))
        if not identical and not _is_row_flatten(node, env, rank):
            raise _Refusal(f"a {name} changes the row structure")
        env[node] = _value_operand(node, env)
        return
    if name in _CONST_FILL:
        fill = 0.0 if name.startswith("zeros") else 1.0 if name.startswith("ones") else None
        if fill is None:
            position = 0 if name == "scalar_tensor" else 1
            fill = node.args[position] if len(node.args) > position else None
        if not isinstance(fill, (int, float)) or isinstance(fill, bool):
            raise _Refusal("a fill value is not a scalar constant")
        env[node] = _scalar_expr(fill)
        return
    if name in ("clamp", "clamp_min"):
        if not _exact_floor_guard(node, env):
            raise _Refusal("a clamp is real computation, not the exact max-abs zero guard")
        value = _value_operand(node, env)
        guarded[value] = sp.Max(value, _scalar_expr(1e-30))
        env[node] = guarded[value]
        return
    if name in _IGNORED:
        if name == "getitem":
            source = node.args[0] if node.args else None
            if (len(node.args) != 2 or node.args[1] != 0
                    or getattr(source, "op", None) != "call_function" or _op(source) != "max"):
                raise _Refusal("getitem may select data or reduction indices")
        value = _value_operand(node, env)
        if name in ("squeeze", "unsqueeze") and value.has(c_sym):
            before, after = _tensor_value(node.args[0]), _tensor_value(node)
            if before is None or after is None or not _same_shape(before.shape, after.shape):
                raise _Refusal("a data reshape changes the row structure")
        env[node] = value
        return
    if name in (*_MEANS, *_VAR_STD, *_REDUCTIONS):
        value = _value_operand(node, env)
        source = node.args[0]
        source_rank = _rank_of(source)
        if not _reduces_last_dim_only(node, source_rank if source_rank is not None else rank):
            raise _Refusal("only a last-dimension row reduction is supported")
        requested_dtype = _argument(node, "dtype")
        source_value = _tensor_value(source)
        if requested_dtype is not None and (source_value is None or requested_dtype != source_value.dtype):
            raise _Refusal("a reduction changes dtype; preserve the conversion as a region boundary")
        if name in _VAR_STD:
            if not _var_is_biased(node):
                raise _Refusal("unbiased variance/std requires a carried count")
            producer = sp.Symbol(f"$r{len(reductions)}", real=True)
            reductions.append(("sum", value / NCOLS_SYM, producer, node))
            consumer = sp.Symbol(f"$r{len(reductions)}", real=True)
            reductions.append(("sum", (value - producer) ** 2 / NCOLS_SYM, consumer, node))
            env[node] = sp.sqrt(consumer) if name == "std" else consumer
        else:
            symbol = sp.Symbol(f"$r{len(reductions)}", real=True)
            kind = "max" if name in ("amax", "max") else "sum"
            operand = value / NCOLS_SYM if name in _MEANS else value
            reductions.append((kind, operand, symbol, node))
            env[node] = symbol
        return
    if name in _BINARY:
        left, right = _value_operand(node, env), _value_operand(node, env, 1)
        if name in ("add", "sub"):
            alpha = _argument(node, "alpha", 1)
            if not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
                raise _Refusal("alpha is not a finite scalar constant")
            right *= _scalar_expr(alpha)
        if name in ("div", "truediv") and _argument(node, "rounding_mode") is not None:
            raise _Refusal("rounded division is not real division")
        _check_binary_broadcast(node, env)
        env[node] = _BINARY[name](left, right)
        return
    if name in _UNARY:
        env[node] = _UNARY[name](_value_operand(node, env))
        return
    raise _Refusal(f"unfamiliar op {str(node.target)!r}; compound reductions may need lowering/decomposition")


def interpret(gm, why: list | None = None, *, input_node=None, local=False):
    """Interpret one row-data boundary.

    Strict mode retains the original diagnostic API. Local mode stops each unsupported
    dataflow branch without ignoring the operation: its output may be a DIFFERENT boundary
    in a later invocation, so GEMMs, casts, clamps, masks and other inputs stay in the graph.
    """
    env, reductions, guarded = {}, [], {}
    anchor, rank = input_node, _rank_of(input_node) if input_node is not None else None
    guard_table = _logsumexp_inf_guard(gm)
    # A functionalized graph has an explicit effect/data dependency contract. Without
    # that guarantee, a write can invalidate any aliased region input at a different time.
    for node in gm.graph.nodes:
        schema = getattr(node.target, "_schema", None)
        mutable = schema is not None and schema.is_mutable
        mutable = mutable or (node.op == "call_method" and str(node.target).endswith("_"))
        if mutable:
            if why is not None:
                why.append("mutable graph must be functionalized before reduction fusion")
            return None
    for node in gm.graph.nodes:
        if node is anchor:
            env[node] = c_sym
            continue
        if node.op == "placeholder":
            if anchor is None:
                anchor, rank = node, _rank_of(node)
                env[node] = c_sym
            continue
        if node.op not in ("call_function", "call_method"):
            continue
        try:
            if node in guard_table:
                source = guard_table[node]
                if source is not None:
                    value, known = _as_expr(source, env)
                    if not known or value is None:
                        raise _Refusal("infinity guard source is outside the region")
                    env[node] = value
                continue
            _interpret_node(node, env, reductions, guarded, rank)
            _check_data_shape(node, env, anchor)
        except _Refusal as error:
            env.pop(node, None)
            if not local:
                if why is not None:
                    why.append(str(error))
                return None
    if anchor is None:
        if why is not None:
            why.append("no input placeholder found")
        return None
    if len(reductions) < 2:
        if why is not None:
            why.append(f"only {len(reductions)} reduction(s) found; a dependent reduction pair is required")
        return None
    symbols = [r[2] for r in reductions]
    candidates = []
    for i, (kind_p, phi, producer, pnode) in enumerate(reductions):
        for j, (kind_c, operand, consumer, cnode) in enumerate(reductions):
            if i == j or not operand.has(producer) or phi.has(consumer):
                continue
            if any(phi.has(other) for other in symbols) or any(
                    operand.has(other) for other in symbols if other != producer):
                continue
            coordinate = guarded.get(producer)
            uses_guard = coordinate is not None and operand.has(coordinate)
            coordinate = coordinate if uses_guard else producer
            psi = operand.subs({coordinate: S_RESULT})
            if psi.has(S_RESULT) and not psi.has(producer):
                candidates.append(Candidate(phi, psi, kind_p, kind_c, pnode, cnode,
                                            consumer, coordinate, uses_guard))
    if not candidates:
        if why is not None:
            why.append(f"{len(reductions)} reductions found, but none depends on another")
        return None
    return anchor, candidates, env


def region_inputs(gm):
    """Bound analysis by reduction ancestors, not by every node times every other node."""
    known_data = set(_BINARY) | set(_UNARY) | {"view", "reshape", "detach", "alias"}
    anchors = set()
    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method") or _op(node) not in (*_REDUCTIONS, *_MEANS, *_VAR_STD):
            continue
        frontier = list(node.args[:1])
        visited = set()
        while frontier:
            value = frontier.pop()
            if not hasattr(value, "op") or value in visited:
                continue
            visited.add(value)
            meta = _tensor_value(value)
            if meta is None or not hasattr(meta, "shape") or not meta.is_floating_point() or len(meta.shape) < 1:
                continue
            anchors.add(value)
            # Unknown operators are kept as opaque tensor-producing boundaries. In
            # particular do not symbolically erase casts, clamps, GEMMs or other inputs.
            if value.op in ("call_function", "call_method") and _op(value) in known_data:
                frontier.extend(value.all_input_nodes)
    return [node for node in gm.graph.nodes if node in anchors]
