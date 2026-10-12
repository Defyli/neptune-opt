"""
Exposing a generated kernel to torch.compile / AOTInductor.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import random
import sys
import tempfile

import sympy as sp

from neptune_codegen.skeleton import SKELETONS
from neptune_codegen.template import compile_skeleton


# ---------------------------------------------------------------- compiler entry point
_TRITON_OPS: dict[str, object] = {}


def as_triton_op(name: str, block: int = 1024, lib: str = "neptune",
                 params: dict | None = None, keepdim: bool = False):
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
    tag = "_keep" if keepdim else ""
    for arg_name, val in sorted((params or {}).items()):
        tag += "_" + arg_name + f"{float(val):g}".replace(".", "p").replace("-", "m")
    key = f"{lib}::{name}{tag}"
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

    kernel = compile_skeleton(name, params)
    op_name = f"{lib}::{name}{tag}"

    # An explicit schema is required: this module uses `from __future__ import
    # annotations`, so the annotation is the string 'torch.Tensor' and torch's schema
    # inference cannot resolve it (torch is a function-local import here).
    @triton_op(op_name, mutates_args={}, schema="(Tensor x) -> Tensor")
    def _impl(x: torch.Tensor) -> torch.Tensor:
        # x is (rows, ncols); the generated kernel reduces the last dimension.
        rows = x.shape[0]
        ncols = x.numel() // rows
        # A `(rows, 1)` and a `(rows,)` result broadcast differently against the rest of a
        # model, so the rank is part of the operator, not a detail of the caller.
        shape = (rows, 1) if keepdim else (rows,)
        out = torch.empty(shape, device=x.device, dtype=torch.float32)
        wrap_triton(kernel)[(rows,)](x, out, rows, ncols, x.stride(0), BLOCK=block)
        return out

    _impl.__name__ = f"neptune_{name}{tag}"
    _TRITON_OPS[key] = _impl
    return _impl


def op_handle(name: str, params: dict | None = None, lib: str = "neptune",
              keepdim: bool = False):
    """The OpOverload for a compiled skeleton, for building graph nodes.

    Registration tags the op with its parameter values and output rank (see `as_triton_op`),
    so the caller must not guess the name -- ask for the handle.
    """
    import torch

    impl = as_triton_op(name, params=params, lib=lib, keepdim=keepdim)
    packet = getattr(getattr(torch.ops, lib), impl.__name__[len(lib) + 1:])
    # Return the OVERLOAD, not the packet: an FX graph carries `torch.ops.aten.amax.default`,
    # not `torch.ops.aten.amax`, and Inductor's post-grad graph rejects a packet outright
    # ("neptune.stable_l2 is not an OpOverload"). The pre-export path happened to tolerate the
    # packet; a pass running inside Inductor does not.
    return getattr(packet, "default", packet)


def triton_op_name(name: str, lib: str = "neptune") -> str:
    """The OpOverload the compiler pass should call, e.g. 'neptune.stable_l2'."""
    return f"{lib}.{name}"
