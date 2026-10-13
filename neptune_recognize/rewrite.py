"""
Replacing a recognised operator's own region with one generated kernel call.
"""
from __future__ import annotations

import dataclasses
import random

import sympy as sp

import torch

import neptune_codegen as CG
from neptune_recognize.match import recognize


def _wants_keepdim(node):
    """Does the operator's final value keep the reduced dimension?

    Needed because the generated kernel can produce either `(rows,)` or `(rows, 1)`, and the
    surrounding model's broadcasting depends on which. A `(rows,)` result broadcast against
    `(rows, cols)` aligns on the TRAILING dimension and silently computes something else, so
    an unknown rank is a reason to refuse rather than to guess.
    """
    value = node.meta.get("val")
    if value is None:
        value = node.meta.get("example_value")
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return len(shape) >= 2


def _locate(gm, values=None):
    """(match, root, keepdim, reason). `reason` is set exactly when the region cannot be replaced.

    Shared by `rewrite` and `would_rewrite` so the two can never disagree about what is
    replaceable -- a disagreement here is what let a lowering be committed for a region the
    rewrite then refused.
    """
    match, reason = recognize(gm, values)
    if match is None:
        return None, None, None, reason

    root = match.region_root
    if root is None:                       # cannot happen: the matcher requires it
        return match, None, None, f"{match.skeleton!r} matched but no region root was located"

    keepdim = _wants_keepdim(root)
    if keepdim is None:
        return match, root, None, (
            f"{match.skeleton!r} matched but the operator's output rank is unknown; refusing "
            "rather than guessing (broadcasting depends on it)")
    return match, root, keepdim, None


def would_rewrite(gm, values=None) -> tuple[bool, str]:
    """Can `rewrite` replace a region in this graph? (Yes/no, skeleton-or-reason).

    Pure Python: no kernel source is generated and no Triton is imported, so this is usable on a
    CPU-only machine and inside the lowering trial.
    """
    match, _root, _keepdim, reason = _locate(gm, values)
    if reason is not None:
        return False, reason
    return True, match.skeleton


def rewrite(gm, values=None, ops=None, lib="neptune"):
    """Replace a recognised operator with ONE call to its generated kernel.

    Returns (graph_module, skeleton_name_or_None, reason). Only the operator's own region is
    replaced; whatever the surrounding model does with the result is preserved.
    """
    import torch

    import neptune_codegen as CG

    match, root, keepdim, reason = _locate(gm, values)
    if reason is not None:
        return gm, None, reason

    graph = gm.graph
    op = CG.op_handle(match.skeleton, params=match.params or None, lib=lib, keepdim=keepdim)
    with graph.inserting_before(root):
        call = graph.call_function(op, args=(match.input_node,))
    root.replace_all_uses_with(call)
    graph.erase_node(root)

    # Erase whatever is now dead. This removes the decomposed computation, but only up to
    # the point where a value is still needed elsewhere: if the producer reduction feeds
    # something outside the operator, it stays.
    changed = True
    while changed:
        changed = False
        for node in reversed(list(graph.nodes)):
            if node.op == "output" or node.op == "placeholder" or node is call:
                continue
            if node.op in ("call_function", "call_method") and len(node.users) == 0:
                graph.erase_node(node)
                changed = True
    graph.lint()
    gm.recompile()
    return gm, match.skeleton, "rewritten"
