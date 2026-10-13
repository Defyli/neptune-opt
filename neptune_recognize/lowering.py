"""
Lowering: expand the compound ops the core-aten decomposition table leaves alone.

`aten._log_softmax` / `aten.log_softmax` arrive as ONE node, so the log-sum-exp inside them has no
node of its own and the region gate has nothing to replace. Expanding it gives the gate a node.

Scope is decided by measurement, not by reach. `experiments/gpu_m20_i4_ceiling.py` built what this
pass would produce and raced it against Inductor's own lowering:

    softmax        our LSE + elementwise   0.59x   Inductor's is a PERSISTENT reduction -> skip
    log_softmax    our LSE                 1.41x   -> this pass exists
    layer_norm     mean + our variance     0.41x   Inductor's dedicated lowering wins -> skip

Reach is not value: fusing softmax would split one good kernel into a kernel plus elementwise work.

The lowering is semantics-preserving (`log_softmax(t) = t - logsumexp(t)`), but it is still only
KEPT when it enables a fusion -- a decomposed graph that does not fuse compiles to more kernels
than the operator it replaced, so committing it blindly would make a model slower while reporting
no fusion.
"""
from __future__ import annotations

import copy

_LOG_SOFTMAX = ("_log_softmax", "log_softmax")


def _name_of(target) -> str:
    """The op name without namespace or overload: 'aten._log_softmax.default' -> '_log_softmax'."""
    text = str(target)
    if text.startswith("aten."):
        return text[len("aten."):].split(".")[0]
    return text.split(".")[0]


def _last_dim(node, rank: int):
    """The reduced dim if it is the LAST one, else None.

    Only the last dimension: the frame emits one program per row, and a log-softmax over any other
    axis would be lowered into a shape the recogniser then refuses -- but after having already
    changed the graph.
    """
    dim = None
    if len(node.args) > 1 and isinstance(node.args[1], int):
        dim = node.args[1]
    elif "dim" in node.kwargs and isinstance(node.kwargs["dim"], int):
        dim = node.kwargs["dim"]
    if dim is None:
        return None
    return dim if dim in (-1, rank - 1) else None


def _rank_of(node) -> int | None:
    value = node.meta.get("val") if hasattr(node, "meta") else None
    shape = getattr(value, "shape", None)
    return None if shape is None else len(shape)


def _resolve(path: str):
    """'amax.default' -> `torch.ops.aten.amax.default`."""
    import torch

    out = torch.ops.aten
    for part in path.split("."):
        out = getattr(out, part)
    return out


# `log_softmax(x, dim) = x - logsumexp(x, dim, keepdim=True)`, written as the ops the interpreter
# understands and in the order it recognises: amax -> sub -> exp -> sum -> log -> add.
#
# ONE spec, TWO interpreters -- FX nodes (the graph) and fake-tensor values (the metadata). Keeping
# them in lockstep matters: the shapes the recogniser then reads off the inserted nodes are
# torch's own, not arithmetic this pass did on the side.
_EXPANSION = (
    ("amax.default", "m", lambda a: (a["x"], [a["dim"]], True)),
    ("sub.Tensor", "centred", lambda a: (a["x"], a["m"])),
    ("exp.default", "e", lambda a: (a["centred"],)),
    ("sum.dim_IntList", "s", lambda a: (a["e"], [a["dim"]], True)),
    ("log.default", "lg", lambda a: (a["s"],)),
    ("add.Tensor", "lse", lambda a: (a["lg"], a["m"])),          # keepdim: the broadcast operand
    ("sub.Tensor", "out", lambda a: (a["x"], a["lse"])),
)


def _fake_values(value, dim: int) -> dict:
    """Run the expansion on the input's FAKE tensor, to get the inserted nodes' metadata.

    A FakeTensor costs no memory and yields the exact shape/dtype/stride the real nodes produce,
    including the broadcast in `x - lse`. Without this, the inserted nodes carry no `val`, and the
    recogniser -- which must know whether the operator keeps the reduced dimension -- refuses.

    Doing it on real arithmetic rather than by hand is deliberate: the earlier version of this
    function *did* hand-set the shapes and got the edge case wrong.
    """
    env = {"x": value, "dim": dim}
    for path, name, build in _EXPANSION:
        env[name] = _resolve(path)(*build(env))
    return env


def _simulatable(value) -> bool:
    """Is running the expansion on this value cheap and side-effect free?"""
    if value is None:
        return False
    if type(value).__name__ == "FakeTensor" or getattr(value, "is_meta", False):
        return True
    numel = getattr(value, "numel", None)
    return callable(numel) and numel() <= 4096        # a real value: allowed, but only if tiny


def _half_to_float(node) -> bool:
    """`log_softmax(..., half_to_float=True)` returns fp32 for an fp16 input -- not the semantics
    of `x - logsumexp(x)`, so it is not this pass's to rewrite."""
    if len(node.args) > 2 and node.args[2]:
        return True
    return bool(node.kwargs.get("half_to_float", False))


def lower_compound(gm) -> int:
    """Expand every supported compound op in place. Returns how many were lowered.

    `torch` is imported inside `_resolve`, not at module scope: this module is imported by the
    recognition package, which is usable without a GPU, and torch is a heavy import.
    """
    graph = gm.graph
    lowered = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or _name_of(node.target) not in _LOG_SOFTMAX:
            continue
        x = node.args[0] if node.args else None
        if x is None or _half_to_float(node):
            continue
        rank = _rank_of(x)
        if rank is None:
            continue
        dim = _last_dim(node, rank)
        if dim is None:
            continue
        fake = None
        if _simulatable(x.meta.get("val")):
            try:
                fake = _fake_values(x.meta["val"], dim)
            except Exception:  # noqa: BLE001 -- no metadata means no lowering, not a broken pass
                fake = None
        if fake is None:
            continue                    # refuse rather than insert nodes the recogniser cannot read
        with graph.inserting_before(node):
            env = {"x": x, "dim": dim}
            for path, name, build in _EXPANSION:
                call = graph.call_function(_resolve(path), args=build(env))
                call.meta["val"] = fake[name]
                env[name] = call
        out = env["out"]
        node.replace_all_uses_with(out)
        graph.erase_node(node)
        lowered += 1
    if lowered:
        graph.lint()
        gm.recompile()
    return lowered


def lower_if_it_helps(gm, values=None) -> int:
    """Lower, but only commit when a REWRITE then succeeds.

    The trial runs on a COPY, because a lowering that does not enable a fusion is a pessimisation:
    the decomposed form of `log_softmax` is more kernels than `aten._log_softmax` itself. So the
    graph is left byte-identical unless the expansion actually buys something.

    The gate is `would_rewrite`, not bare recognition, and that distinction is load-bearing:
    recognition can succeed on a graph the rewrite still refuses (its region root carries no shape
    metadata, so whether the operator keeps the reduced dimension is unknown). Gating on
    recognition alone committed a lowering that then fused NOTHING -- the exact pessimisation this
    function exists to prevent. `would_rewrite` is pure Python, so this still needs no Triton.
    """
    from neptune_recognize.rewrite import would_rewrite

    if would_rewrite(gm, values)[0]:
        return 0                      # already rewritable; do not touch the graph
    trial = copy.deepcopy(gm)
    trial_graph = getattr(trial, "graph_module", trial)
    if not lower_compound(trial_graph) or not would_rewrite(trial_graph, values)[0]:
        return 0                      # the lowering did not help: leave the real graph alone
    return lower_compound(gm)
