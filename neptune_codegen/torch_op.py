"""
Exposing a generated kernel to torch.compile / AOTInductor.
"""
from __future__ import annotations

from neptune_codegen import autotune
from neptune_codegen.skeleton import param_defaults
from neptune_codegen.template import (compile_skeleton, _row_last_layout, _validate_row_last,
    launch_reduction, validate_plan, split_launch_shape)


# ---------------------------------------------------------------- compiler entry point
_TRITON_OPS: dict[str, object] = {}


def as_triton_op(name: str, block: int | None = None, num_warps: int | None = None,
                 lib: str = "neptune", params: dict | None = None, keepdim: bool = False, plan: str = "serial"):
    """Expose one generated kernel to torch.compile / AOTInductor as a custom op.

    This is the bridge that makes the generated source part of the compiled artifact
    rather than a private launch. `torch.library.triton_op` is the supported entry point:

        generated source -> @triton_op  --export--> triton_kernel_wrapper_functional
                                        --Inductor--> ir.UserDefinedTritonKernel
                                        --AOTI-->  .cubin inside the .pt2

    Every step of that chain is existing compiler machinery; nothing here hand-writes IR.
    The kernel is generated from the skeleton's algebra, so a new operator needs a new
    `g` and nothing else.
    """
    # The op name must encode the parameter values: torch ops are global, so registering
    # `neptune::softcap_lse` twice with different betas would silently reuse the first
    # kernel. Distinct values get distinct ops.
    # The name carries the parameters, the output rank and the launch configuration: a package
    # bakes the launch in, and torch ops are global, so two shapes with different configurations
    # must not share a name. When the configuration is not known (the explicit operator library has
    # no shape), the body resolves it per call instead and it stays out of the name -- correct
    # either way. `autotune.op_key` is shared with the tests so this cannot drift from the claim.
    validate_plan(plan)
    if plan == "split":
        # Serial launch evidence does not tune a two-kernel split plan.
        block = 1024 if block is None else block
        num_warps = 4 if num_warps is None else num_warps
    params = {**param_defaults(name), **(params or {})}
    key = autotune.op_key(lib, name, params, keepdim, block, num_warps)
    if plan != "serial":
        key += "_plan_split"
        from neptune_codegen.algebra import derive_merge_certificate
        from neptune_codegen.skeleton import SKELETONS
        derive_merge_certificate(SKELETONS[name], params)
    if key in _TRITON_OPS:
        return _TRITON_OPS[key]

    import torch
    from torch.library import triton_op, wrap_triton

    # torch 2.11's triton_op has no usable `schema` argument: it always calls
    # infer_schema, which `eval`s the annotation string against the function's module
    # globals. Our annotations are strings (PEP 563) and torch is imported lazily here
    # to keep this module importable without a GPU, so bind the name where eval will
    # look for it. Without this, registration dies with
    # "Unsupported type annotation torch.Tensor".
    globals().setdefault("torch", torch)

    kernel = (compile_skeleton(name, params) if plan == "serial" else
              compile_skeleton(name, params, plan=plan))
    # The registered name IS the key: one place computes it (`autotune.op_key`), so the name a graph
    # carries and the name the op is registered under cannot drift apart. They did once -- a
    # refactor left a stale local behind, and every registration raised NameError while the CPU
    # tests, which stub `op_handle`, stayed green.
    op_name = key
    impl_name = key.split("::", 1)[1]

    # An explicit schema is required: this module uses `from __future__ import
    # annotations`, so the annotation is the string 'torch.Tensor' and torch's schema
    # inference cannot resolve it (torch is a function-local import here).
    @triton_op(op_name, mutates_args={}, schema="(Tensor x) -> Tensor")
    def _impl(x: torch.Tensor) -> torch.Tensor:
        _validate_row_last(x)
        x2, rows, ncols = _row_last_layout(x)
        shape = x.shape[:-1] + ((1,) if keepdim else ())
        out = torch.empty(shape, device=x.device, dtype=x.dtype)
        # A concrete empty batch is valid. Dynamic batches retain a symbolic grid; the
        # kernel's row guard remains applicable without int(rows) or a batch-size guard.
        if isinstance(rows, int) and rows == 0:
            return out
        cfg = (block, num_warps)
        if block is None or num_warps is None:
            chosen = autotune.best_config(rows, ncols, x.dtype, x.device)
            cfg = (chosen[0] if block is None else block, chosen[1] if num_warps is None else num_warps)
        # A `(rows, 1)` and a `(rows,)` result broadcast differently against the rest of a
        # model, so the rank is part of the operator, not a detail of the caller.
        # The result takes the INPUT's dtype, which is what torch's own reductions do
        # (`torch.logsumexp(bf16)` is bf16): the graph downstream is typed, and returning fp32
        # from a bf16 node is a dtype mismatch rather than a precision improvement. The
        # ACCUMULATION still happens in fp32 inside the kernel -- only the result is narrowed.
        launch_reduction(kernel, plan, x2, out, rows, ncols, cfg[0], cfg[1], wrapper=wrap_triton)
        return out

    # Export/Fake needs only metadata, not a CUDA launch or the physical device's tuning
    # table. Override triton_op's default fake implementation, but keep the real body fully
    # traceable so functional decomposition still exposes reshape/copy and the Triton HOP.
    def _fake(x):
        _validate_row_last(x, metadata=True)
        if plan == "split":
            split_launch_shape(x.shape[-1], block if block is not None else 1024)
        shape = x.shape[:-1] + ((1,) if keepdim else ())
        return torch.empty(shape, device=x.device, dtype=x.dtype)

    if hasattr(_impl, "register_fake"):
        _impl.register_fake(_fake)
    _impl.__name__ = f"{lib}_{impl_name}"
    _TRITON_OPS[key] = _impl
    return _impl


def op_handle(name: str, params: dict | None = None, lib: str = "neptune",
              keepdim: bool = False, block: int | None = None, num_warps: int | None = None,
              plan: str = "serial"):
    """The OpOverload for a compiled skeleton, for building graph nodes.

    Registration tags the op with its parameter values and output rank (see `as_triton_op`),
    so the caller must not guess the name -- ask for the handle.
    """
    import torch

    impl = as_triton_op(name, block=block, num_warps=num_warps, params=params, lib=lib,
                        keepdim=keepdim, plan=plan)
    packet = getattr(getattr(torch.ops, lib), impl.__name__[len(lib) + 1:])
    # Return the OVERLOAD, not the packet: an FX graph carries `torch.ops.aten.amax.default`,
    # not `torch.ops.aten.amax`, and Inductor's post-grad graph rejects a packet outright
    # ("neptune.stable_l2 is not an OpOverload"). The pre-export path happened to tolerate the
    # packet; a pass running inside Inductor does not.
    return getattr(packet, "default", packet)


def triton_op_name(name: str, lib: str = "neptune") -> str:
    """The OpOverload the compiler pass should call, e.g. 'neptune.stable_l2'."""
    return f"{lib}.{name}"


_STATE_OPS: dict[tuple, object] = {}


def as_state_op(name, params=None, plan="serial", lib="neptune"):
    """Explicit two-input structured bridge, separate from the scalar registry."""
    import torch
    from neptune_codegen.algebra import derive_state_certificate
    from neptune_codegen.skeleton import state_reduction_spec
    from neptune_codegen.template import compile_state_reduction, validate_rms_inputs, launch_rms_norm

    spec = state_reduction_spec(name)
    derive_state_certificate(spec)
    if name in ("segment_sum", "segment_mean"):
        return as_segment_op(name, params=params, plan=plan, lib=lib)
    validate_plan(plan)
    params = dict(params or {})
    # Metadata-independent checks before registration, even when the op is cached.
    from neptune_codegen.emitters import emit_rms_state_source
    emit_rms_state_source(spec, params, plan=plan)
    key = (lib, spec, autotune.parameter_identity(params), plan)
    if key in _STATE_OPS:
        return _STATE_OPS[key]
    from torch.library import triton_op, wrap_triton
    globals().setdefault("torch", torch)
    tag = autotune.parameter_tag(params)
    impl_name = f"state_{name}{tag}_plan_{plan}"

    @triton_op(f"{lib}::{impl_name}", mutates_args={}, schema="(Tensor x, Tensor gamma) -> Tensor")
    def _impl(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        shape = validate_rms_inputs(x, gamma, params, plan=plan)
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        rows = 1
        for size in x.shape[:-1]:
            rows *= size
        if isinstance(rows, int) and rows == 0:
            return out
        kernel = compile_state_reduction(name, params, plan=plan)
        launch_rms_norm(kernel, x, gamma, out, plan=plan, wrapper=wrap_triton)
        return out

    def _fake(x, gamma):
        shape = validate_rms_inputs(x, gamma, params, plan=plan, metadata=True)
        return torch.empty(shape, dtype=x.dtype, device=x.device)

    if hasattr(_impl, "register_fake"):
        _impl.register_fake(_fake)
    _impl.__name__ = f"{lib}_{impl_name}"
    _STATE_OPS[key] = _impl
    return _impl


def state_op_handle(name, params=None, plan="serial", lib="neptune"):
    import torch

    impl = as_state_op(name, params=params, plan=plan, lib=lib)
    packet = getattr(getattr(torch.ops, lib), impl.__name__[len(lib) + 1:])
    return getattr(packet, "default", packet)


# Segment state is additive (sum, count) over exact reals, NOT an IEEE monoid.
# The numerical contract is an ordered storage-dtype fold; split/tree is refused.
_SEGMENT_KERNELS: dict[tuple, object] = {}
_SEGMENT_OPS: dict[tuple, object] = {}
_SEGMENT_NAMES: dict[str, tuple] = {}


def _segment_params(name, params=None, plan="serial"):
    import math

    if name not in ("segment_sum", "segment_mean"):
        raise KeyError(f"unknown segment reduction {name!r}")
    from neptune_codegen.skeleton import state_reduction_spec
    state_reduction_spec(name)  # Validate authorization before kernel/operator cache hits.
    validate_plan(plan)
    if plan != "serial":
        raise NotImplementedError("segment split cannot preserve ordered storage-dtype rounding; use serial")
    params = dict(params or {})
    unknown = params.keys() - {"initial", "offsets", "unsafe"}
    if unknown:
        raise ValueError(f"unknown segment parameters: {sorted(unknown)}")
    initial = params.get("initial")
    if initial is not None:
        if isinstance(initial, bool) or not isinstance(initial, (int, float)):
            raise TypeError("segment initial must be a finite Python scalar or None")
        initial = float(initial)
        if not math.isfinite(initial):
            raise ValueError("segment initial must be finite")
    offsets, unsafe = params.get("offsets", False), params.get("unsafe", False)
    if type(offsets) is not bool or type(unsafe) is not bool:
        raise TypeError("segment offsets and unsafe must be bool")
    return {"initial": initial, "offsets": offsets, "unsafe": unsafe}


def _segment_identity(name, params, plan):
    numeric = {k: v for k, v in params.items() if v is not None}
    return (name, params["initial"] is None, autotune.parameter_identity(numeric), plan, 64, 4)


def _segment_module_name(identity):
    import hashlib

    stem = f"neptune_gen_{identity[0]}_serial_h" + hashlib.sha256(repr(identity).encode()).hexdigest()[-10:]
    candidate = stem
    collision = 0
    # A short digest is only a name, never the cache identity. Resolve even hash collisions.
    while candidate in _SEGMENT_NAMES and _SEGMENT_NAMES[candidate] != identity:
        collision += 1
        candidate = f"{stem}_{collision}"
    _SEGMENT_NAMES[candidate] = identity
    return candidate


def emit_segment_source(name, *, initial=None, offsets=False, plan="serial"):
    """Emit named sum/mean from additive state with strict serial numeric semantics."""
    import struct

    from neptune_codegen.skeleton import state_reduction_spec
    from neptune_codegen.algebra import derive_state_certificate

    spec = state_reduction_spec(name)
    derive_state_certificate(spec)
    params = _segment_params(name, {"initial": initial, "offsets": offsets}, plan)
    initial = params["initial"]
    # Native CUDA scalar conversion uses fp32 before narrowing to storage dtype.
    value = 0.0 if initial is None else initial
    try:
        value = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        value = float("-inf") if value < 0 else float("inf")
    # Decimal -0.0 constants can be canonicalised by Triton; an IEEE bitcast keeps its sign.
    initial_bits = struct.unpack("I", struct.pack("f", value))[0]
    readout = "result = acc"
    if name == "segment_mean":
        empty = 'float("nan")' if initial is None else "acc"
        readout = ("denominator = (end - start).to(storage_dtype).to(tl.float32)\n"
                   "        result = tl.div_rn(acc, denominator)\n"
                   f"        result = tl.where(end == start, {empty}, result)")
    return f'''# AUTO-GENERATED segment state=(sum,count); exact-real additive certificate only.
# Numeric contract: ordered per-token storage-dtype RTNE; no IEEE reassociation.
# name={name!r}, offsets={offsets!r}, plan={plan!r}; no custom merges assumed.
import triton
import triton.language as tl


@triton.jit
def {name}(data_ptr, spans_ptr, out_ptr, nrows, channels, nsegments,
           stride_row, stride_channel, stride_span, BLOCK: tl.constexpr):
    segment = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    storage_dtype: tl.constexpr = out_ptr.dtype.element_ty
    if segment < nsegments:
        start = tl.load(spans_ptr + segment * stride_span).to(tl.int64)
        end = tl.load(spans_ptr + (segment + 1) * stride_span).to(tl.int64)
        valid = (start >= 0) & (end >= start) & (end <= nrows)
        acc = tl.full((BLOCK,), {initial_bits}, tl.uint32).to(tl.float32, bitcast=True).to(storage_dtype).to(tl.float32)
        # Defense in depth: malformed spans never issue data loads, even under unsafe.
        if valid:
            for token in range(start, end):
                v = tl.load(data_ptr + token * stride_row + channel * stride_channel,
                            mask=channel < channels, other=0).to(tl.float32)
                acc = (acc + v).to(storage_dtype, fp_downcast_rounding="rtne").to(tl.float32)
        {readout}
        result = tl.where(valid, result, float("nan"))
        tl.store(out_ptr + segment * channels + channel, result, mask=channel < channels)
'''


def compile_segment_reduction(name, params=None, plan="serial"):
    """Keep real generated module files under template's existing temporary owner."""
    import importlib.util
    import os
    import sys
    from neptune_codegen import template

    params = _segment_params(name, params, plan)
    if template._GEN_RELEASED:
        raise RuntimeError("generated source directory has been released")
    identity = _segment_identity(name, params, plan)
    if identity in _SEGMENT_KERNELS:
        return _SEGMENT_KERNELS[identity]
    source = emit_segment_source(name, initial=params["initial"], offsets=params["offsets"], plan=plan)
    module_name = _segment_module_name(identity)
    path = os.path.join(template._GEN_DIR, module_name + ".py")
    with open(path, "w") as file:
        file.write(source)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        kernel = getattr(module, name)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    _SEGMENT_KERNELS[identity] = kernel
    return kernel


def validate_segment_inputs(data, segments, *, metadata=False, offsets=False,
                            initial=None, unsafe=False, reduce="sum", plan="serial"):
    """Metadata-only contract; returns output shape, without reading segment contents.

    Main's state contract can call this on CPU/Fake tensors with metadata=True.
    Runtime value assertions are traceable ATen ops in _segment_spans, not host reads.
    """
    import torch

    _segment_params("segment_" + reduce, {"initial": initial, "offsets": offsets, "unsafe": unsafe}, plan)
    if data.layout != torch.strided or segments.layout != torch.strided:
        raise TypeError("segment requires strided data and segments")
    if data.ndim < 1 or segments.ndim != 1:
        raise ValueError("segment data rank must be >= 1 and segments rank must be 1")
    if data.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("segment data dtype must be float16/bfloat16/float32")
    if initial is not None and abs(initial) > torch.finfo(data.dtype).max:
        raise RuntimeError("segment initial cannot be converted to input dtype without overflow")
    if segments.dtype not in (torch.int32, torch.int64):
        raise TypeError("segment index dtype must be int32/int64")
    if data.device != segments.device:
        raise ValueError("segment tensors must be on the same device")
    if not metadata and data.device.type != "cuda":
        raise ValueError("segment execution requires CUDA device")
    if offsets:
        torch._check(segments.shape[0] >= 1, lambda: "segment offsets needs at least one boundary")
    count = segments.shape[0] - 1 if offsets else segments.shape[0]
    return (count,) + data.shape[1:]


def _segment_spans(data, segments, *, offsets, unsafe):
    import torch

    nrows = data.shape[0]
    if offsets:
        spans = segments
    else:
        # Native lengths need exact coverage unless unsafe. Range checks remain mandatory.
        torch._assert_async(torch.all((segments >= 0) & (segments <= nrows)),
                            "segment lengths must be nonnegative and within data bounds")
        spans = torch.cat((segments.new_zeros((1,), dtype=torch.int64),
                           torch.cumsum(segments, dim=0, dtype=torch.int64)))
        if not unsafe:
            torch._assert_async(spans[-1] == nrows, "segment lengths must sum to data axis0 size")
    torch._assert_async(torch.all((spans >= 0) & (spans <= nrows)),
                        "segment boundaries out of data bounds")
    torch._assert_async(torch.all(spans[1:] >= spans[:-1]), "segment boundaries must be monotonic")
    return spans


def _launch_segment(name, data, segments, params, plan, *, wrapper=None):
    import torch

    shape = validate_segment_inputs(data, segments, reduce=name.removeprefix("segment_"),
                                    plan=plan, **params)
    spans = _segment_spans(data, segments, offsets=params["offsets"], unsafe=params["unsafe"])
    channels = 1
    for size in data.shape[1:]:
        channels = channels * size
    out = torch.empty(shape, dtype=data.dtype, device=data.device)
    if (type(shape[0]) is int and shape[0] == 0) or (type(channels) is int and channels == 0):
        return out
    flat = data.reshape(data.shape[0], channels)
    kernel = compile_segment_reduction(name, params, plan)
    if wrapper is not None:
        kernel = wrapper(kernel)
    kernel[(shape[0], (channels + 63) // 64)](
        flat, spans, out, data.shape[0], channels, shape[0],
        flat.stride(0), flat.stride(1), spans.stride(0), BLOCK=64, num_warps=4,
        enable_fp_fusion=False)
    return out


def run_segment_pool(data, segments, *, reduce="sum", offsets=False, initial=None,
                     unsafe=False, plan="serial"):
    """Direct launch bridge; Fake/meta calls allocate metadata only, without Triton."""
    import torch
    from torch._subclasses.fake_tensor import FakeTensor

    name = "segment_" + reduce
    params = _segment_params(name, {"initial": initial, "offsets": offsets, "unsafe": unsafe}, plan)
    if data.device.type == "meta" or isinstance(data, FakeTensor):
        shape = validate_segment_inputs(data, segments, metadata=True, reduce=reduce, plan=plan, **params)
        return torch.empty(shape, dtype=data.dtype, device=data.device)
    return _launch_segment(name, data, segments, params, plan)


def as_segment_op(name, params=None, plan="serial", lib="neptune"):
    """Return CustomOpDef for two Tensor inputs; constants live in exact op identity.

    Registration/Fake do not compile or import Triton. Execution exposes ATen prefix,
    assertions, reshape and wrap_triton to functional decomposition on supported Torch.
    """
    import torch

    params = _segment_params(name, params, plan)
    identity = (lib, _segment_identity(name, params, plan))
    if identity in _SEGMENT_OPS:
        return _SEGMENT_OPS[identity]
    from torch.library import triton_op, wrap_triton

    globals().setdefault("torch", torch)
    impl_name = _segment_module_name(identity[1]).removeprefix("neptune_gen_")

    @triton_op(f"{lib}::{impl_name}", mutates_args={}, schema="(Tensor data, Tensor segments) -> Tensor")
    def _impl(data: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        return _launch_segment(name, data, segments, params, plan, wrapper=wrap_triton)

    def _fake(data, segments):
        shape = validate_segment_inputs(data, segments, metadata=True,
                                        reduce=name.removeprefix("segment_"), plan=plan, **params)
        return torch.empty(shape, dtype=data.dtype, device=data.device)

    if hasattr(_impl, "register_fake"):
        _impl.register_fake(_fake)
    _impl.__name__ = f"{lib}_{impl_name}"
    _SEGMENT_OPS[identity] = _impl
    return _impl


def segment_op_handle(name, params=None, plan="serial", lib="neptune"):
    """Return the OpOverload for main's CG.state_op_handle dispatch."""
    import torch

    impl = as_segment_op(name, params=params, plan=plan, lib=lib)
    packet = getattr(getattr(torch.ops, lib), impl.__name__[len(lib) + 1:])
    return getattr(packet, "default", packet)
