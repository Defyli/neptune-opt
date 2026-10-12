"""
The registered operators. A new one is a declaration, not new codegen.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import random
import sys
import tempfile

import sympy as sp

from neptune_codegen.skeleton import Skeleton, register
from neptune_codegen.symbols import *  # noqa: F401,F403


register(Skeleton(
    name="logsumexp",
    g=sp.exp(c_sym - s_sym),
    # acc holds sum(exp(v - m)), so logsumexp = m + log(acc). The 1e-38 keeps log()
    # finite when every term underflowed.
    finalize=_FINAL_M + sp.log(_FINAL_ACC + _PARAM_TINY),
    producer="signed",
    is_log=True,
    params=((_PARAM_TINY, "tiny"),),
    defaults=((_PARAM_TINY, 0.0),),
    doc="softmax's max+sumexp fold, log-softmax, cross-entropy",
))

register(Skeleton(
    name="stable_l2",
    g=(c_sym / s_sym) ** 2,
    finalize=_FINAL_M * sp.sqrt(_FINAL_ACC),
    producer="abs",
    doc="numerically stable L2 norm / RMS numerator (no torch equivalent)",
))

register(Skeleton(
    name="linear_attn",
    # ratio producer, sum reducer: the accumulator holds sum_j x_j / max|x|
    g=c_sym / s_sym,
    finalize=_FINAL_ACC,
    producer="abs",
    doc="linear attention normaliser: sum(x)/max|x|",
))

register(Skeleton(
    name="decay_renorm",
    # additive denominator offset: the producer is |x| but the term divides by (N + s)
    g=c_sym / (_PARAM_N + s_sym),
    finalize=_FINAL_ACC,
    producer="abs",
    params=((_PARAM_N, "N"),),
    doc="duration/decay renormalisation: sum(x/(N+max|x|))",
))

register(Skeleton(
    name="softcap_lse",
    # Gemma-2 attention soft-capping. The producer operand is NOT declared: an exponential
    # fold forces it to be max tanh(beta*c), and phi = log(g) + s is derived.
    g=sp.exp(sp.tanh(_PARAM_BETA * c_sym) - s_sym),
    finalize=_FINAL_M + sp.log(_FINAL_ACC + _PARAM_TINY),
    producer="signed",
    is_log=True,
    params=((_PARAM_BETA, "beta"), (_PARAM_TINY, "tiny")),
    defaults=((_PARAM_TINY, 0.0),),
    doc="Gemma-2 soft-capped attention: logsumexp(tanh(beta*x))",
))

register(Skeleton(
    name="rmsnorm_max",
    # THE PAPER'S OWN EXAMPLE, and the reason the two reducers had to be separated.
    # producer  = mean(x^2)          -> an ADDITIVE reduction
    # consumer  = max of x/sqrt(.)   -> a MAX
    # This is not multi-producer: there is exactly one producer value, it is just produced
    # by a different reducer than the consumer's.
    g=c_sym / sp.sqrt(s_sym + _PARAM_EPS),
    finalize=_FINAL_ACC,
    producer="abs",
    producer_expr=c_sym ** 2 / _PARAM_N,     # the running MEAN of squares
    producer_reducer="sum",
    consumer_reducer="max",
    # N is the row length, which the kernel already receives as `ncols`: declaring it
    # intrinsic instead of baking it is what makes this operator work with dynamic shapes.
    params=((_PARAM_EPS, "eps"),),
    intrinsic_params=((_PARAM_N, "ncols"),),
    doc="RMSNorm + row max: max_j x_j / sqrt(mean(x^2) + eps)",
))

register(Skeleton(
    name="sumsq_div",
    g=c_sym ** 2 / s_sym,
    finalize=_FINAL_ACC,
    producer="abs",
    doc="sum(x^2)/max|x| -- a scale-normalised energy statistic",
))

register(Skeleton(
    name="lse_biased",
    # the same exponential fold with a constant bias inside: ALiBi-style attention.
    # NOTE what is NOT declared: the producer operand. An exponential fold forces it to be
    # max(c + alpha), and the generator derives phi = log(g) + s.
    g=sp.exp(c_sym + _PARAM_ALPHA - s_sym),
    finalize=_FINAL_M + sp.log(_FINAL_ACC + _PARAM_TINY),
    producer="signed",
    is_log=True,
    params=((_PARAM_ALPHA, "alpha"), (_PARAM_TINY, "tiny")),
    defaults=((_PARAM_TINY, 0.0),),
    doc="ALiBi-biased logsumexp: logsumexp(x + alpha)",
))
