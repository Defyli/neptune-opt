"""Targeted compound lowering, without whole-graph ATen decomposition.

The structural gate certifies a complete fp32 LSE region, not floating-point
bitwise equivalence or a speedup. Low-precision log-softmax uses fp32 LSE plus
an elementwise readout and final cast; PyTorch rounds its intermediates
differently. Even fp32 ``x - (m + log(s))`` can lose accuracy for large offsets
compared with PyTorch's ``(x - m) - log(s)``. Numeric verification belongs to
the caller; failed kernel generation is also the caller's outer transaction.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext

_LOG_SOFTMAX = ("_log_softmax", "log_softmax")


def _name_of(target) -> str:
    text = str(target)
    if text.startswith("aten."):
        return text[len("aten."):].split(".")[0]
    return text.split(".")[0]


def _is_compound(node) -> bool:
    """Pure CPU preview predicate; applicability still requires metadata checks."""
    return (node.op == "call_function" and str(node.target) in (
        "aten._log_softmax.default", "aten.log_softmax.int",
        "aten.log_softmax.default", "aten.logsumexp.default"))


def _argument(node, name, position, default=None):
    return node.kwargs.get(name, node.args[position] if len(node.args) > position else default)


def _last_dim(node, rank: int):
    dim = _argument(node, "dim", 1)
    if _name_of(node.target) == "logsumexp" and isinstance(dim, (tuple, list)):
        if len(dim) != 1:
            return None
        dim = dim[0]
    if type(dim) is not int or rank < 1:
        return None
    return dim if dim in (-1, rank - 1) else None


def _rank_of(node) -> int | None:
    shape = getattr(_value(node), "shape", None)
    return None if shape is None else len(shape)


def _value(node):
    meta = getattr(node, "meta", {})
    value = meta.get("val")
    return meta.get("example_value") if value is None else value


def _resolve(path: str):
    import torch

    out = torch.ops.aten
    for part in path.split("."):
        out = getattr(out, part)
    return out


def _expansion(dtype, output_dtype, *, log_softmax=True, keepdim=True):
    """One specification for FX nodes and Torch-computed Fake/meta values.

    Always guard a shape (..., 1) max, including for keepdim=False. Casts are
    explicit boundaries, never interpreter identities. The LSE root remains
    fp32; squeezing and the final output cast stay outside its fusion region.
    """
    import torch

    steps = []
    accumulation = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    if dtype != accumulation:
        steps.append(("_to_copy.default", "x", lambda a: (a["x"],), {"dtype": accumulation}))
    steps.extend((
        ("amax.default", "m", lambda a: (a["x"], [a["dim"]], True), {}),
        ("abs.default", "abs_m", lambda a: (a["m"],), {}),
        ("eq.Scalar", "infinite", lambda a: (a["abs_m"], float("inf")), {}),
        ("where.ScalarSelf", "m_safe", lambda a: (a["infinite"], 0.0, a["m"]), {}),
        ("sub.Tensor", "centred", lambda a: (a["x"], a["m_safe"]), {}),
        ("exp.default", "e", lambda a: (a["centred"],), {}),
        ("sum.dim_IntList", "s", lambda a: (a["e"], [a["dim"]], True), {}),
        ("log.default", "lg", lambda a: (a["s"],), {}),
        ("add.Tensor", "lse", lambda a: (a["m_safe"], a["lg"]), {}),
    ))
    result = "lse"
    if log_softmax:
        # Preserve PyTorch's row-wide NaN classification for +inf inputs. A plain
        # x-LSE gives -inf on the finite entries of that row instead of NaN.
        steps.append(("eq.Scalar", "posinf_row", lambda a: (a["lse"], float("inf")), {}))
        steps.append(("sub.Tensor", "readout", lambda a: (a["x"], a["lse"]), {}))
        steps.append(("where.ScalarSelf", "log_out",
                      lambda a: (a["posinf_row"], float("nan"), a["readout"]), {}))
        result = "log_out"
    elif not keepdim:
        steps.append(("squeeze.dim", "squeezed", lambda a: (a["lse"], a["dim"]), {}))
        result = "squeezed"
    if output_dtype != accumulation:
        steps.append(("_to_copy.default", "out", lambda a, key=result: (a[key],),
                      {"dtype": output_dtype}))
        result = "out"
    return steps, result


def _fake_values(value, dim: int, *, output_dtype=None, log_softmax=True, keepdim=True) -> dict:
    """Use Torch Fake/meta operations, never real arithmetic or hand-inferred shapes."""
    import torch

    if type(value).__name__ != "FakeTensor" and not value.is_meta:
        value = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device="meta")
    steps, result = _expansion(value.dtype, output_dtype or value.dtype,
                              log_softmax=log_softmax, keepdim=keepdim)
    env = {"x": value, "dim": dim}
    with getattr(value, "fake_mode", None) or nullcontext():
        for path, name, build, kwargs in steps:
            env[name] = _resolve(path)(*build(env), **kwargs)
    env["out"] = env[result]
    return env


def _simulatable(value) -> bool:
    if value is None:
        return False
    if type(value).__name__ == "FakeTensor" or getattr(value, "is_meta", False):
        return True
    # Real metadata is accepted only for tiny CPU examples. No GPU work in this pass.
    if getattr(getattr(value, "device", None), "type", None) != "cpu":
        return False
    numel = getattr(value, "numel", None)
    return callable(numel) and numel() <= 4096


def _half_to_float(node) -> bool:
    """Only _log_softmax has this flag: log_softmax's third arg is a dtype."""
    return (_name_of(node.target) == "_log_softmax"
            and _argument(node, "half_to_float", 2, False) is not False)


def _trial_graph(gm):
    """Copy only the FX graph: large parameters/buffers are shared, never cloned."""
    import torch.fx as fx

    return fx.GraphModule(gm, copy.deepcopy(gm.graph))


def _commit(gm, trial):
    """Publish an already linted/compiled graph, rolling back compilation exceptions."""
    state = gm.__dict__.copy()
    cls = type(gm)
    generated = {key: cls.__dict__.get(key) for key in ("forward", "_wrapped_call")}
    try:
        gm.graph = trial.graph
    except Exception:
        gm.__dict__.clear()
        gm.__dict__.update(state)
        gm.graph.owning_module = gm
        trial.graph.owning_module = trial
        for key, value in generated.items():
            if value is not None:
                setattr(cls, key, value)
            elif key in cls.__dict__:
                delattr(cls, key)
        raise


def _expand_node(gm, node) -> bool:
    import torch

    if not _is_compound(node) or _half_to_float(node):
        return False
    x = _argument(node, "self", 0)
    source, output = _value(x), _value(node)
    allowed = (torch.float32, torch.float16, torch.bfloat16)
    if (source is None or output is None or getattr(source, "dtype", None) not in allowed
            or getattr(output, "dtype", None) not in allowed or not _simulatable(source)):
        return False
    rank = _rank_of(x)
    dim = _last_dim(node, rank) if rank is not None else None
    if dim is None or (type(source.shape[-1]) is int and source.shape[-1] <= 0):
        return False
    is_log = _name_of(node.target) in _LOG_SOFTMAX
    keepdim = True if is_log else _argument(node, "keepdim", 2, False)
    if type(keepdim) is not bool:
        return False
    # log_softmax(dtype=...) converts its input BEFORE doing any arithmetic.
    requested_dtype = _argument(node, "dtype", 2) if _name_of(node.target) == "log_softmax" else None
    effective_dtype = requested_dtype or source.dtype
    if effective_dtype not in allowed or output.dtype != effective_dtype:
        return False
    fake_source = source
    if requested_dtype is not None:
        with getattr(source, "fake_mode", None) or nullcontext():
            if type(source).__name__ != "FakeTensor" and not source.is_meta:
                fake_source = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype,
                                                 device="meta")
            fake_source = _resolve("_to_copy.default")(fake_source, dtype=requested_dtype)
    fake = _fake_values(fake_source, dim, output_dtype=output.dtype,
                        log_softmax=is_log, keepdim=keepdim)
    if tuple(map(str, fake["out"].shape)) != tuple(map(str, output.shape)):
        return False
    steps, result = _expansion(effective_dtype, output.dtype,
                              log_softmax=is_log, keepdim=keepdim)
    graph = gm.graph
    with graph.inserting_before(node):
        env = {"x": x, "dim": dim}
        if requested_dtype is not None:
            cast = graph.call_function(_resolve("_to_copy.default"), args=(x,),
                                       kwargs={"dtype": requested_dtype})
            cast.meta.update(val=fake_source, neptune_lowering=node.name)
            env["x"] = cast
        for path, name, build, kwargs in steps:
            call = graph.call_function(_resolve(path), args=build(env), kwargs=kwargs)
            call.meta.update(val=fake[name], neptune_lowering=node.name)
            env[name] = call
    node.replace_all_uses_with(env[result])
    graph.erase_node(node)
    return True


def lower_compound(gm, nodes=None) -> int:
    """Explicit legacy expansion of all supported nodes, or selected nodes/names.

    No benefit gate here. A selected Node is resolved by name on the copied graph.
    Exceptions leave the original untouched, as does unsupported metadata.
    """
    if nodes is None:
        names = {node.name for node in gm.graph.nodes if _is_compound(node)}
    elif isinstance(nodes, str) or hasattr(nodes, "op"):
        names = {nodes if isinstance(nodes, str) else nodes.name}
    else:
        names = {node if isinstance(node, str) else node.name for node in nodes}
    if not names:
        return 0
    try:
        trial = _trial_graph(gm)
        lowered = sum(_expand_node(trial, node) for node in list(trial.graph.nodes)
                      if node.name in names)
        if not lowered:
            return 0
        trial.graph.lint()
        trial.recompile()
        _commit(gm, trial)
        return lowered
    except Exception:  # noqa: BLE001 -- unsupported trials must not mutate the caller
        return 0


def lower_if_it_helps(gm, values=None) -> int:
    """Commit each node only if ITS marked LSE closure passes the structural gate.

    Existing regions or another compound's eligible region cannot sponsor a
    trial. No Triton generation, global decomposition or dead-code elimination.
    """
    from neptune_recognize.match import recognize_regions
    from neptune_recognize.rewrite import region_contract, would_rewrite

    names = [node.name for node in gm.graph.nodes if _is_compound(node)]
    lowered = 0
    for name in names:
        try:
            trial = _trial_graph(gm)
            if not lower_compound(trial, nodes=name):
                continue
            matches, _ = recognize_regions(trial, values)
            eligible = any(
                match.region_root.meta.get("neptune_lowering") == name
                and any(node.meta.get("neptune_lowering") == name for node in match.region_nodes)
                and region_contract(match)[1] is None
                for match in matches
            )
            if not eligible or not would_rewrite(trial, values)[0]:
                continue
            _commit(gm, trial)
            lowered += 1
        except Exception:  # noqa: BLE001 -- every single-node trial is a transaction
            continue
    return lowered
