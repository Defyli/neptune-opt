# Comparison with the authors' own implementation

The paper's authors released an implementation: **<https://github.com/uiuc-arc/neptune>**
(Apache-2.0). It is a **fork of Apache TVM**, and it targets TVM only.

This document records what I found reading it, what is worth borrowing, and what it says about
this repository's design. **No code was copied** — everything below is either an idea or a
statement about their code, and any idea adopted here would be re-implemented against PyTorch.

## What their implementation actually does

Their fusion is a **TIR schedule primitive**, not a graph rewrite. A user writes the operator in
the ordinary decomposed form — `softmax()` in `scripts/neptune_bench/ours/softmax.py` is literally
`max`, then `exp(x - max)`, then `sum` — and the schedule transforms it:

```python
b2rf = sch.rolling_update(b2, j0, factor_axis=0)   # the max
b6rf = sch.rolling_update(b6, j0, factor_axis=0)   # the sum
sch.decompose_reduction(b5, j0)
```

`rolling_update` is described in `src/tir/schedule/primitive.h` as *"Generalized operator fusion.
Fuses and r-factors a reduction block under a reduction loop"*. Its six steps are:

1. walk the DFG from the consumer to find the **reduce producers** (blocks with incomplete outputs)
2. fuse the intermediate blocks under the target loop
3. convert each producer reduction into a **scan** block
4. **mock-inline** everything into the consumer, then match the concentrated expression against a
   self-reduction pattern `out[i…] = f(out[i…], in[i…, j])` and, from it, **derive a
   repair function**
5. `RFactor` the consumer
6. apply the derived repair to the write-back block's store expression

So the two designs agree on the essential point: **the repair is derived, not tabulated.** Their
derivation lives in `ReduceRepairer::DeriveFromBlockExpr` → `HExprSolver`:

```
1. separate vars:   g(u…, x…) = gcomb(gu(u…), gx(x…))      via arith.sympy.separate_vars
2. invert:          gcomb(a, b) = r  ->  a = gcomb^-1(r, b) via arith.sympy.inverse
3. h(r, x, x')  =   gcomb(gcomb^-1(r, gx(x)), gx(x'))
4. prove:           h(f(y1, y2), x, x') == f(h(y1), h(y2), x, x')   "If so, H is just h"
```

and, like this repository, they **reach for sympy** (over an FFI) and they **check the foldability
condition explicitly** before using the repair.

## The mapping

| theirs (TVM) | this repository (PyTorch) |
|---|---|
| `MatchSelfReduction` — match `out = f(out, in)` | symbolic interpreter + registry comparison |
| `ReduceRepairer::HExprSolver` | `neptune_codegen.algebra.repair_factor` |
| step 4 above: prove `h` distributes over `f` | `repairlib`'s condition (4) check / `vanishes` |
| `RewriteBlockExpr` + `load_condition` | the template's `scale`, `tl.where(mask, …)` |
| `MockInlineBlocks` + `RFactorInternal` | `neptune_recognize.rewrite` (replace the region) |
| `TritonBuildKernel` | `sym_to_triton` + the source template |
| `rolling_update` / `split_k_update` | `neptune_opt.apply(ep)` |
| a schedule pass inside TVM | a graph rewrite before `aoti_compile_and_package` |

## What is worth borrowing

### 1. Variable separation — this is the big one

`arith.sympy.separate_vars` splits `g` into a function of the *dependent* variable and a function
of everything else: `g(u…, x…) = gcomb(gu(u…), gx(x…))`, and then a *pair* of producers is
handled by a single inverse because only the `x` part has to be inverted.

This repository did not do that: it inverted `g` in `c` with `sympy.solve` and read the repair off
Eq. 5. **Adopted** — the derivation is now the separation `g(s', c) / g(s, c)`, accepted only when
`c` cancels, with a golden table in `tests/test_neptune_opt.py` pinning that every registered
skeleton derives the identical repair it did before. See
[design.md](design.md#how-the-repair-is-derived).

Two corrections to what this document predicted, both worth recording:

* **Separation did not unlock the two-producer row.** That family is still refused, and the
  measurement says why: its placeholder declares `g = c · s'`, which mentions the *fresh* producer
  value, so it is not a `(producer, term)` pair at all (`g-depends-on-fresh-value`). Separation
  could not have helped; the declared family was malformed. A real two-producer operator — the
  RMSNorm + max case, where the maximum is over a different quantity than the sum of squares — is
  already expressible through `producer_expr` + `producer_reducer`, which is the axis that
  actually mattered.
* **What separation *did* unlock is the affine repair.** Writing the condition as a ratio rather
  than as an inversion makes it obvious when the residual is a *shift* rather than a factor:
  `h(t) = a·t + b`, with `a` from matching the `c`-derivatives and `b = g(s',c) - a·g(s,c)`
  accepted when free of `c`. Condition (4) then splits by reducer — a bias is illegal for an
  additive consumer and legal for a max consumer, because `t ↦ a·t + b` is a max-morphism. That
  took the generator from 17 to **19 of 28** families, i.e. to 100% of what the algebra admits.

So the borrowed idea paid off on a different axis than predicted: the separation *formulation* is
worth more than the inversion it replaces, even where the two give identical answers.

### 2. A fallback when the foldability condition fails  ✅ solved, by a different mechanism

Their comment ends with *"This covers a lot of common cases"*, and there is a second mechanism:
`ReduceToScan` converts a producer reduction into a scan block, which combines partial results
without needing the repair to distribute over the reducer at all. The paper's `SplitKUpdate` is the
same idea for split-K.

This repository had no such fallback: when condition (4) fails, `repairlib` classifies the operator
`NEEDS_STATE` and the pipeline refused it. **Those were 6 of the 28 families** (the variance /
Welford group).

**Solved — but not by adopting `ReduceToScan`.** That primitive is a TVM *scheduling*
transformation: it turns the producer into a scan block so that `HasIncompleteBuffer` holds and the
producer can be fused under the loop, which is what makes the tiling/split-K machinery work. This
repository rewrites an ATen graph and emits one program per row, so it has no loop nest to schedule
and no partial results to combine — adopting the primitive would mean adopting TVM's execution
model. The fallback it needed is instead a **second traversal inside the same kernel**: pass 1
finalizes the producer, pass 2 reduces the term with the producer held constant, and the repair is
the identity *by construction*. See
[design.md](design.md#when-no-repair-exists-the-two-pass-plan).

That took the generator from 20 to **26 of 28** families, and it is worth being precise about what
it does *not* show: it is not evidence that this repository now does what `ReduceToScan` does. The
two mechanisms serve the same families from opposite directions — theirs avoids the repair by
combining partial results across tiles, this one avoids it by reading the row again — and the
second pass is the one that fits a single-program-per-row frame.

Nor is the variance group something they serve: their own benchmark scripts use `rolling_update`
only for softmax and attention (`scripts/neptune_bench/ours/{softmax,attn_plain,attn_gqa,attn_fp8}.py`),
and no variance or standard-deviation schedule exists in the repository.

### 3. Memoize the derivation by printed expression  ✅ adopted

`HExprSolver` caches on `SExprPrinter::Print(g_expr)`, with the reason spelled out: *"this function
may be called thousands of times due to the autotuner replaying traces"*. This repository cached
compiled kernels but recomputed `repair_factor` and its `simplify` calls on every invocation,
which is the expensive part. **Adopted**: `repair_pair` (and `term_expr`,
`derived_producer_expr`) are `functools.lru_cache`d. Same pressure, same fix, and free.

### 4. The mask belongs to the program, not to the compiler

In their pipeline the mask is already in the user's graph — `attn_plain.py` writes
`tir.if_then_else(mask_cond(...), score, tir.min_value("float32"))` — so the identity element falls
out of the data flow and no separate mask concept is needed. The only masking left over is the
block tail, which the lowering handles with `LoopPartition`.

This repository invented a second mask concept (`tl.where(mask, …, identity)` for the out-of-range
block tail) and also relies on the graph's own mask. That is not wrong, but it means the same
concern is handled in two places at two different levels, which is worth simplifying.

## What their code confirms about this repository's choices

**The region gate is not paranoia.** Their own doc comment carries this warning:

> `\warning` RollingUpdate changes the output of the `Bt` spatial blocks by fusing them under the
> loop `l`. A fully correct RollingUpdate impl should inline all `Bt` blocks, but inlining can be
> bad for performance or break codegen. In this implementation, **if any block that is not
> roll-updated uses one of the `Bt` blocks, the output of that block will be wrong.** A correct
> implementation should make a copy of each of `Bt` blocks … and redirect `b0` to use the copied
> buffer.

That is exactly the failure mode this repository's region gate exists to refuse: an operator whose
producer value is also consumed elsewhere. This repository replaces only the operator's own region,
and its dead-code elimination *keeps the producer alive* when it has other users, so the result
stays correct (at the cost of a missed fusion). **The stricter behaviour should be kept.**

Two more of their notes match findings here:

* *"the algebraic rewrite … always requires `b0` itself to be a reduction block"* — this
  repository likewise only fuses when the consumer is a reduction it understands.
* *"this rewrite is not always possible. RollingUpdate fails if a rewrite is not found."* — they
  refuse rather than emit something plausible, which is the same policy as this repository's
  `check_supported` and its register of explicit refusals.

## Where this repository is structurally weaker

Their fusion happens **inside the loop nest**, so a whole operator — matmul, softmax, matmul — is
tiled and fused into one kernel. That is why they can produce FlashAttention-equivalent kernels.

This repository rewrites an **ATen graph**, so it can replace one reduction region with one
generated kernel and must leave everything around it to Inductor. **A `QK^T → softmax → PV` chain
cannot become a single kernel here**, no matter how much of the algebra is covered.

That is the honest headline of this comparison: the two projects are not competing
implementations of the same thing. This one makes a *single reduction* one-pass over the input and
delivers it through AOTInductor; theirs makes a *whole dependent-operator chain* one kernel and
delivers it through TVM. Coverage inside the reduction is comparable; coverage around it is not.

## Suggested order of work, if this is taken further

1. ~~**Variable separation** (item 1)~~ — **done**, and the payoff was the affine repair rather
   than the two-producer case this document expected.
2. ~~**Memoization** (item 3)~~ — **done**.
3. ~~**A fallback for the failed foldability condition** (item 2)~~ — **done**, by a two-pass plan
   rather than by `ReduceToScan`; see the note in item 2. This closed the `NEEDS_STATE` group.
4. **Anything loop-nest-level** — still not attempted, and still a different architecture.

The current state is reproducible with `python tools/coverage_codegen.py`: 26 of 28 families emit,
and both refusals are labelled by cause, so the remaining work is a list rather than a feeling.
The two that remain are honest limits, not oversights: a `mul` consumer needs a carried count, and
the last control family declares a term that ignores its producer.
