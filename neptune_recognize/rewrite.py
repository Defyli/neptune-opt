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


def _canonical_metadata(value):
    """Infer output metadata without allocating real CPU/GPU output storage."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    if isinstance(value, FakeTensor):
        return value.new_empty(value.shape)
    if value.device.type == "meta":
        return value.new_empty(value.shape)
    mode = FakeTensorMode()
    with mode:
        return torch.empty(value.shape, device=value.device, dtype=value.dtype)


def _structured_contract(match):
    """Metadata-only structured ABI validation; never compile or launch a kernel."""
    import math
    import torch

    if len(match.input_nodes) != 2 or match.input_node is not match.input_nodes[0]:
        return None, "structured operation requires two ordered input boundaries"
    boundaries = getattr(match, "metadata", {}).get("boundary_values", ())
    inputs = [_tensor_value(node) if _tensor_value(node) is not None else
              (boundaries[index] if node.op == "get_attr" and index < len(boundaries) else None)
              for index, node in enumerate(match.input_nodes)]
    values = [*inputs, _tensor_value(match.region_root)]
    if any(value is None or not hasattr(value, "shape") for value in values):
        return None, "structured input/output metadata is unknown"
    source, auxiliary, output = values
    if any(value.layout != torch.strided for value in values):
        return None, "structured operation requires strided tensor layout"
    if any(value.device != source.device for value in values):
        return None, "structured inputs and output must share a device"
    if source.device.type not in ("cpu", "cuda"):
        return None, "unsupported structured device"
    if source.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None, f"unsupported structured dtype {source.dtype}"
    if output.dtype != source.dtype:
        return None, "structured output dtype differs from input"
    if len(source.shape) < 1:
        return None, "structured input must have rank >= 1"
    # These first backends return fresh contiguous outputs. Preserve layout-sensitive
    # downstream semantics rather than copying stale noncontiguous Fake metadata.
    canonical = _canonical_metadata(output)
    if not _same_shape(output.stride(), canonical.stride()):
        return None, "structured output layout must be canonical contiguous"
    if match.operation == "rms_norm" and match.skeleton == "rms_norm":
        eps = match.params.get("eps")
        if type(eps) not in (float, int) or not math.isfinite(eps) or eps <= 0:
            return None, "RMSNorm requires finite positive eps"
        if type(source.shape[-1]) is int and source.shape[-1] <= 0:
            return None, "RMSNorm requires nonempty row width"
        if auxiliary.dtype != source.dtype or len(auxiliary.shape) != 1:
            return None, "RMSNorm gamma must be a same-dtype 1D vector"
        if not _same_shape(auxiliary.shape, source.shape[-1:]):
            return None, "RMSNorm gamma width differs from input"
        if not _same_shape(output.shape, source.shape):
            return None, "RMSNorm output shape must exactly equal input shape"
    elif match.operation == "segment_pool" and match.skeleton in ("segment_sum", "segment_mean"):
        if auxiliary.dtype not in (torch.int32, torch.int64) or len(auxiliary.shape) != 1:
            return None, "segment lengths/offsets must be a 1D int32/int64 tensor"
        if type(match.params.get("offsets")) is not bool or type(match.params.get("unsafe")) is not bool:
            return None, "segment offsets/unsafe parameters must be explicit bools"
        initial = match.params.get("initial")
        if initial is not None and (type(initial) not in (int, float) or not math.isfinite(initial)):
            return None, "segment initial must be None or a finite scalar"
        count = auxiliary.shape[0] - int(match.params["offsets"])
        if type(count) is int and count < 0:
            return None, "segment offsets must contain at least one entry"
        if not _same_shape(output.shape, (count, *source.shape[1:])):
            return None, "segment output shape differs from axis-0 segment contour"
    else:
        return None, "unsupported structured operation/skeleton"
    return False, None


def region_contract(match):
    """Check dtype and exact output shape before creating a custom operation."""
    import torch

    if getattr(match, "kind", "scalar") == "structured":
        return _structured_contract(match)
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
    if getattr(match, "kind", "scalar") == "structured":
        inputs = match.input_nodes
        from neptune_recognize.match import _structured_boundary_value
        values = tuple(_structured_boundary_value(gm, node) for node in inputs)
        # Verify the real shared backend ABI BEFORE registration or graph mutation,
        # including static width, resident/split capacity and initial conversion.
        if match.operation == "rms_norm":
            from neptune_codegen.template import validate_rms_inputs
            validate_rms_inputs(*values, match.params, plan=plan, metadata=True)
        else:
            from neptune_codegen.torch_op import validate_segment_inputs
            validate_segment_inputs(*values, reduce=match.skeleton.removeprefix("segment_"),
                                    plan=plan, metadata=True, **match.params)
        op = CG.state_op_handle(match.skeleton, params=match.params, plan=plan, lib=lib)
    else:
        block, warps = _launch_config(match.input_node)
        if plan == "split":
            block, warps = 1024, 4
            CG.split_launch_shape(_tensor_value(match.input_node).shape[-1], block)
        params = {**CG.param_defaults(match.skeleton), **match.params}
        op = CG.op_handle(match.skeleton, params=params, lib=lib, keepdim=keepdim,
                          block=block, num_warps=warps, plan=plan)
        inputs = (match.input_node,)
    graph = gm.graph
    root = match.region_root
    with graph.inserting_before(root):
        call = graph.call_function(op, args=inputs)
        call.meta = copy.copy(root.meta)
        if getattr(match, "kind", "scalar") == "structured":
            value = _canonical_metadata(_tensor_value(root))
            call.meta["val"] = value
            if "example_value" in call.meta:
                call.meta["example_value"] = value
            if "tensor_meta" in call.meta:
                from torch.fx.passes.shape_prop import _extract_tensor_metadata
                call.meta["tensor_meta"] = _extract_tensor_metadata(value)
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
