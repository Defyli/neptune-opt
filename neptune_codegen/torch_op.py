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
