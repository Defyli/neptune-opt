# The paper: Neptune (arXiv:2510.08726, PLDI 2026)

**Neptune: A Compilation Framework for Optimizing Machine Learning Programs with Reduction
Dependencies** — <https://arxiv.org/abs/2510.08726>

This page summarises what the paper establishes and which parts of it this repository builds on.
Read the paper for the full treatment; anything below is a summary, and the repository's own
claims are kept separate in [design.md](design.md).

## The problem it attacks

Many operators need **two reductions of the same data, where the second depends on the first**.
A numerically stable softmax is the canonical example:

```
m = max_j x_j                    # pass 1
y = sum_j exp(x_j - m)           # pass 2, needs m
```

and a numerically stable L2 norm, RMSNorm, Welford variance and attention all have the same
shape. The dependency forces either two traversals of the input or materialising an intermediate,
so the operator does not fuse into a single pass the way an elementwise chain does.

The usual fix is to special-case the operator. PyTorch does exactly this: it ships a hand-written
"online softmax" and no general mechanism.

## The idea: break the dependency, then repair it

Neptune's core observation is that you can consume the data with a **stale** producer value and
later correct the accumulated result when the producer value changes. It formalises this as a
triple `(f, g, h)`:

| symbol | meaning |
|---|---|
| `f` | the consumer's reduction operator (`+`, `max`, ...) |
| `g` | the **term producer**: how a single element is turned into a term, given the current producer value `s` — `g(s, c)` |
| `h` | the **repair**: how an already-accumulated value must be adjusted when the producer value moves from `s` to `s'` |

The "break" is that element `c` is consumed using whatever `s` was at the time. The "repair" is
`h`, and the paper derives it (Eq. 5) rather than asking the user for it:

```
h(t, s, s') = g( s', g_c^{-1}(s, t) )
```

Read it as: recover the element `c` that would have produced the accumulator value `t` under the
stale `s`; then re-apply `g` with the fresh `s'`. The result is the accumulator value as it would
have been had `s'` been known all along.

## The two conditions that decide everything

The repair is only useful if it can be **folded into the accumulator**, which needs two
conditions:

* **(3)** `h(g(s, c), s, s') = g(s', c)` — the repair actually undoes the staleness. This is what
  makes the derivation meaningful at all.
* **(4)** `h(x f y, s, s') = h(x, s, s') f h(y, s, s')` — the repair distributes over the
  consumer's reducer, so it can be applied to the *accumulator* instead of to every term.

**Condition (4) is the decisive one, and this repository treats it as the primary classification
criterion** (the paper presents (5) as the construction; the classification below is the
engineering reading of it). If (4) holds, the accumulator can be fixed with `O(1)` work per
update and the operator folds into one pass. If it fails, a per-term correction may still exist,
but it cannot be propagated through the fold.

## The classification this repository uses

| verdict | meaning |
|---|---|
| `EXACT` | (3) and (4) both hold → foldable into a single-pass accumulator |
| `AFFINE` | (3) holds, (4) fails → needs carried state |
| `NEEDS_STATE` | (3) itself fails and the residual still contains the data symbol → no `h(t, s, s')` can exist; extra state must be carried (this is the Welford situation) |
| `NOT_INVERT` | Eq. 5 cannot be instantiated at all |

`NEEDS_STATE` is a refinement introduced here rather than in the paper: when (3) fails, whether
the residual still mentions the data symbol `c` is the informative distinction. If it does, no
repair of the form `h(t, s, s')` can ever work and something about `c` must be carried across
iterations. If it does not, the construction simply cannot repair this `g`.

## What the paper reports, and what the conditions imply

The paper's reported examples are the max + sum-of-exp fold (softmax, log-softmax,
cross-entropy), numerically stable L2 / RMSNorm-style normalisation, and an RMSNorm + max
combination. It notes that the same repair is shared across surface variants of an operator
(causal masking, biases, and so on do not change `g`).

Two consequences that this repository leans on heavily:

1. **The interesting quantity is not the number of operators but the number of distinct
   algebras.** Every softmax variant collapses to one `(f, g)` pair and therefore one repair.
2. **The conditions are decidable.** Given `f` and `g` as symbolic expressions, (3) and (4) are
   statements about expressions, so the classification can be computed rather than asserted. That
   is what makes a general mechanism possible instead of a table of hand-written kernels.

## What the paper does *not* give you

The paper is about the algebra and its exploitation in a compiler. It does not ship:

* a mapping from "an operator expressed in eager PyTorch" to the corresponding `(f, g)` — i.e.
  **recognition**;
* an integration path into a specific production compiler stack (**AOTInductor**, here);
* a statement of which of the two conditions is the one that matters in practice.

Those three gaps are what this repository is about; see [design.md](design.md).
