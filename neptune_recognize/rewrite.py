"""Replace exact row-reduction regions without changing the surrounding dataflow."""
from __future__ import annotations

import copy

import neptune_codegen as CG
from neptune_recognize.interp import _same_shape, _tensor_value
from neptune_recognize.match import recognize_regions


def _wants_keepdim(node, input_node=None):
    value = _tensor_value(node)
    if value is None or not hasattr(value, "shape"):
        return None
    if input_node is None:
        # Retained for older callers; the rewrite always supplies its input boundary.
        return len(value.shape) >= 2
    source = _tensor_value(input_node)
    if source is None or not hasattr(source, "shape"):
        return None
    if _same_shape(value.shape, source.shape[:-1]):
        return False
    if _same_shape(value.shape, source.shape[:-1] + (1,)):
        return True
    return None


def _launch_config(node):
    value = _tensor_value(node)
    if value is None or not hasattr(value, "shape") or not value.shape:
        return None, None
    rows = 1
    for dimension in value.shape[:-1]:
        rows *= dimension
    return CG.autotune.best_config(rows, value.shape[-1], value.dtype, value.device)


def region_contract(match):
    """Check dtype and exact output shape before creating a custom operation."""
    import torch

    source, output = _tensor_value(match.input_node), _tensor_value(match.region_root)
    if source is None or not hasattr(source, "shape"):
        return None, "region input metadata is unknown"
    if output is None or not hasattr(output, "shape"):
        return None, "operator output rank is unknown"
    if source.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None, f"unsupported region dtype {source.dtype}"
    if output.dtype != source.dtype:
        return None, "output dtype differs from the region input; preserve the cast boundary"
    # Hand-decomposed low precision arithmetic rounds after every operation. Algebra
    # alone cannot certify removing those roundings. Real low-precision compound ops
    # are lowered with explicit fp32 arithmetic and a preserved final cast instead.
    if source.dtype != torch.float32:
        return None, "low-precision intermediate rounding needs explicit fp32 lowering"
    if len(source.shape) < 1:
        return None, "row-reduction input must have rank >= 1"
    if type(source.shape[-1]) is int and source.shape[-1] <= 0:
        return None, "empty row width is unsupported"
    keepdim = _wants_keepdim(match.region_root, match.input_node)
    if keepdim is None:
        return None, "operator output shape is not the input's last-dimension reduction"
    return keepdim, None


def _locate(gm, values=None):
    matches, reason = recognize_regions(gm, values)
    for match in matches:
        keepdim, refusal = region_contract(match)
        if refusal is None:
            return match, match.region_root, keepdim, None
        reason = refusal
    return (matches[0] if matches else None), None, None, reason


def would_rewrite(gm, values=None):
    match, _, _, reason = _locate(gm, values)
    return (True, match.skeleton) if reason is None else (False, reason)


def rewrite_match(gm, match, *, lib="neptune", plan="serial"):
    """Mutate a candidate graph only; the public caller owns the transaction."""
    keepdim, reason = region_contract(match)
    if reason is not None:
        return None, reason
    block, warps = _launch_config(match.input_node)
    if plan == "split":
        block, warps = 1024, 4
        CG.split_launch_shape(_tensor_value(match.input_node).shape[-1], block)
    params = {**CG.param_defaults(match.skeleton), **match.params}
    op = CG.op_handle(match.skeleton, params=params, lib=lib, keepdim=keepdim,
                      block=block, num_warps=warps, plan=plan)
    graph = gm.graph
    root = match.region_root
    with graph.inserting_before(root):
        call = graph.call_function(op, args=(match.input_node,))
        call.meta = copy.copy(root.meta)
    # Own the mapping: GraphModule metadata may be shallow-copied by FX callers.
    # Keep internal roots too, since preserved module signatures can refer to
    # values that are not final graph outputs. Consumers resolve chains later.
    replacements = copy.deepcopy(gm.meta.get("neptune_replaced_roots", {}))
    replacements[root.name] = call.name
    gm.meta["neptune_replaced_roots"] = replacements
    root.replace_all_uses_with(call)
    graph.erase_node(root)
    # Only remove nodes from this complete region, and never remove side effects or
    # values shared by other consumers. Global dead-code deletion may erase assertions.
    for node in reversed(match.region_nodes):
        if node is root or node.op in ("placeholder", "output", "get_attr"):
            continue
        if not node.users and not node.is_impure():
            graph.erase_node(node)
    graph.lint()
    gm.recompile()
    return call, None


def rewrite(gm, values=None, ops=None, lib="neptune", *, plan="serial"):
    """Compatibility API: commit one region only after the entire rewrite succeeds."""
    trial = copy.deepcopy(gm)
    match, _, _, reason = _locate(trial, values)
    if reason is not None:
        return gm, None, reason
    call, reason = rewrite_match(trial, match, lib=lib, plan=plan)
    if call is None:
        return gm, None, reason
    gm.graph = trial.graph
    gm.recompile()
    return gm, match.skeleton, "rewritten"
