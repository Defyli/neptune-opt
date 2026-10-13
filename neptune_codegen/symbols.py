"""
The sympy symbols the algebra is written over.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import random
import sys
import tempfile

import sympy as sp


# ---------------------------------------------------------------- skeleton spec
# Symbols used by the algebra. A skeleton describes itself with expressions over these;
# the generator derives the repair and emits Triton for both.
u_sym = sp.Symbol("u", positive=True)      # the accumulator value fed to the repair


s_sym = sp.Symbol("s", real=True)          # stale producer value


s2_sym = sp.Symbol("s2", real=True)        # fresh producer value


c_sym = sp.Symbol("c", real=True)          # the non-reduction input (one element)


# ---------------------------------------------------------------- algebra -> source
_FINAL_ACC = sp.Symbol("acc", real=True)


_FINAL_M = sp.Symbol("m", real=True)


# The log-family readout needs an additive constant. It is DECLARED rather than hardcoded
# so that recognition can match the operator's tail against the graph's: a literal 1e-38 in
# `m + log(acc + 1e-38)` can never equal a graph that computes `m + log(acc)`, whereas a
# declared `tiny` solves to 0 and matches exactly.
_PARAM_TINY = sp.Symbol("tiny", nonnegative=True)


# ---------------------------------------------------------------- more skeletons
# These come from `repairlib/coverage.py`, where the ALGEBRA (not this file) decided they
# are EXACT. Adding one is a declaration; repair, term and source all follow from it, and
# so does recognition (see neptune_recognize.py) -- no pattern row is written anywhere.
_PARAM_N = sp.Symbol("N", positive=True)


_PARAM_EPS = sp.Symbol("eps", positive=True)


_PARAM_ALPHA = sp.Symbol("alpha", real=True)


_PARAM_BETA = sp.Symbol("beta", positive=True)
# Temperature. Real models scale logits before a log-sum-exp (`logits / T`), so the producer of
# that fold is max(c/T) -- and because the fold is exponential, `phi = log(g) + s` derives `c/T`
# without it having to be declared. See `lse_scaled`.
_PARAM_T = sp.Symbol("T", positive=True)


# Explicit, because the readout symbols are underscore-prefixed and `from ... import *`
# skips those by default -- which silently broke every module that imported them.
# Names the generated kernel already has in scope. A skeleton parameter may bind to one of
# these instead of to a baked literal, which is what lets a SHAPE-COUPLED operator work with
# dynamic shapes: the row length is `ncols`, a runtime argument, not a compile-time constant.
INTRINSICS = ("rows", "ncols", "stride_row")

__all__ = [
    "INTRINSICS",
    "u_sym", "s_sym", "s2_sym", "c_sym",
    "_FINAL_ACC", "_FINAL_M",
    "_PARAM_TINY", "_PARAM_N", "_PARAM_EPS", "_PARAM_ALPHA", "_PARAM_BETA", "_PARAM_T",
]
