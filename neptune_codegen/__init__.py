"""
Automatic codegen for break-and-repair fused reductions.

Turns a reduction skeleton `(f, g, h)` -- as derived and classified by `repairlib` --
into a single-pass Triton kernel.

Why this layer exists
---------------------
Milestone 8 measured that a *graph-level* rewrite of the same operator buys 1.00x: ATen
has no loops, so it cannot express "both reductions in one traversal", and Inductor
re-canonicalises the result back to what it already had. Milestone 9 measured that a
hand-written fused loop buys 1.25x-2.69x with input traversals dropping 4 -> 1. This
module makes the second thing automatic.

The uniform form
----------------
Every validated skeleton reduces to the same update, so one template covers all of them:

    m_new = producer_max(m, |v| or v)      running producer value
    scale = m == initial ? 1 : (s/s')^k    the repair h(t, s, s') = t*(s/s')^k
    acc   = acc*scale + term               consumer: one fused fold
    m     = m_new
    out   = finalize(acc, m)

    skeleton     producer    term            finalize
    ----------   ---------   -------------   -------------------
    logsumexp    signed      exp(v - m)      m + log(acc)
    stable_l2    abs         (|v| / m)^2     m * sqrt(acc)

A new skeleton is therefore *data* (a power, a producer mode, a finalize snippet), not
new codegen. The template branches on nothing.

The two producer modes are not interchangeable, and this is the most important thing
learned while building this file (measured in `numeric_logsumexp_design.py`):

  * logsumexp NEEDS the SIGNED max: every term requires exp(v - m) <= 1. Using max|v|
    is actively wrong -- for a row of negative numbers max|v| is large and positive, so
    all terms underflow to 0 and the result collapses. Against a float64 reference,
    max|v| gave 4.4e-2 relative error on large inputs and inf on all-negative rows; the
    signed max gave 7.8e-12 and 6.6e-08.
  * L2-type terms NEED the max of |v|: their magnitude scales with |v|, so a signed max
    would let (|v|/m) exceed 1 and overflow.

Other traps encoded here, each of which cost a debugging round:
  * The repair guard must test the PREVIOUS producer value, not the updated one:
    guarding on `m_new == 0` zeroes the accumulator on every zero-valued block, so an
    all-zero row loses all but one block and returns log(BLOCK) instead of log(ncols).
  * Out-of-range lanes load as 0.0 and must be excluded from both the producer max and
    the terms (fill the max with -inf, wrap the terms in `tl.where(mask, ...)`).
  * `safe_m = max(m_new, 1e-30)` is used ONLY where a division happens, never in the
    exponent and never in the repair guard: an all-zero row divides 0/0 for L2 terms and
    yields nan, but clamping the value that feeds an exponent shifts every term instead
    (that combination cost log(BLOCK) instead of log(ncols)).
  * Triton rejects a module-level global inside @triton.jit ("Cannot access global
    variable ... from within @jit'ed function") and has no `**` on tensors, so powers are
    written as explicit multiplications and every knob is a source literal.
  * The generated module must be written to a REAL file: Triton reads the function source
    with `inspect.getsourcelines`, so an exec'd function fails with "could not get source
    code".
"""
from __future__ import annotations

from neptune_codegen.symbols import u_sym, s_sym, s2_sym, c_sym, _FINAL_ACC, _FINAL_M, _PARAM_TINY, _PARAM_N, _PARAM_EPS, _PARAM_ALPHA, _PARAM_BETA  # noqa: F401
from neptune_codegen.skeleton import NEGATIVE_CONTROLS, Skeleton, SKELETONS, register, supported, all_registered, unsupported_reason, param_defaults  # noqa: F401
from neptune_codegen.algebra import _BRANCH_CUT, _simplify, repair_factor, repair_bias, repair_pair, vanishes, _numerically_zero, _marked, term_expr, _as_quotient_power  # noqa: F401
from neptune_codegen.emitters import check_supported, derived_producer_expr, producer_nonneg, producer_init, consumer_init, producer_operand, guard_value, denominator_expr, _bindings, param_bindings, producer_update, consumer_update, scale_source, bias_source, bias_line, term_source, finalize_source  # noqa: F401
from neptune_codegen.template import _GEN_DIR, _CACHE, _TEMPLATE, emit_kernel_source, generated_path, compile_skeleton, run  # noqa: F401
from neptune_codegen.torch_op import _TRITON_OPS, as_triton_op, op_handle, triton_op_name  # noqa: F401
from neptune_codegen.references import reference, graph_reference  # noqa: F401

# Importing the library is what populates the registry.
from neptune_codegen import library  # noqa: F401,E402

from neptune_codegen import sym_to_triton  # noqa: F401,E402
# Re-exported because the pre-split module imported them deliberately, so they were part of
# its public surface even though nothing here needs them.
from neptune_codegen.sym_to_triton import TranslationError, _Q, to_triton  # noqa: F401

__all__ = [
    "NEGATIVE_CONTROLS",
    "SKELETONS",
    "Skeleton",
    "_BRANCH_CUT",
    "_CACHE",
    "_FINAL_ACC",
    "_FINAL_M",
    "_GEN_DIR",
    "_PARAM_ALPHA",
    "_PARAM_BETA",
    "_PARAM_EPS",
    "_PARAM_N",
    "_PARAM_TINY",
    "_TEMPLATE",
    "_TRITON_OPS",
    "_as_quotient_power",
    "TranslationError",
    "_Q",
    "to_triton",
    "_bindings",
    "_marked",
    "_numerically_zero",
    "_simplify",
    "all_registered",
    "as_triton_op",
    "c_sym",
    "check_supported",
    "compile_skeleton",
    "consumer_init",
    "consumer_update",
    "denominator_expr",
    "derived_producer_expr",
    "emit_kernel_source",
    "finalize_source",
    "generated_path",
    "graph_reference",
    "guard_value",
    "op_handle",
    "param_bindings",
    "param_defaults",
    "producer_init",
    "producer_nonneg",
    "producer_operand",
    "producer_update",
    "reference",
    "register",
    "repair_factor",
    "repair_bias",
    "repair_pair",
    "run",
    "s2_sym",
    "s_sym",
    "scale_source",
    "bias_source",
    "bias_line",
    "supported",
    "term_expr",
    "term_source",
    "triton_op_name",
    "u_sym",
    "unsupported_reason",
    "vanishes",
]
