# The design: how this repository engineers the paper

[paper.md](paper.md) covers what Neptune establishes. This page covers what had to be built to
turn it into something a PyTorch user can actually run, and it is deliberately explicit about
which parts are verified and which are not.

## The shape of the thing

```
        an eager PyTorch model
                 │
    (1) RECOGNITION          neptune_recognize/
        which (f, g) is this graph implementing?  -- compared against the registry,
        not against a hand-written table
                 │
    (2) ALGEBRA + CODEGEN    neptune_codegen/
        derive h, apply conditions (3)/(4), derive the term, emit Triton source
                 │
    (3) INTEGRATION          neptune_opt/
        rewrite the graph and let AOTInductor carry the kernel into the .pt2
```

`repairlib/` is the classifier from step (2) kept as a standalone, dependency-free study: it is
the thing that answers "how far does this reach at all?" across 28 operator families, and it runs
on CPU with nothing but sympy.

## (1) Recognition is derived, not tabulated

This was the first thing that had to be solved and the least obvious. The paper gives you `(f, g)`
for an operator it names; it does not tell you how to find `(f, g)` in an arbitrary graph.

The approach here is a small **symbolic interpreter over the graph**:

1. Walk the FX graph assigning each node a sympy expression, starting from one data symbol `c`.
   Every reduction installs a fresh symbol standing for its result.
2. That yields the two things the algebra actually talks about: `phi(c)`, what the max-like
   reduction reduces over, and `psi(c, s)`, what the sum-like reduction reduces over.
3. Compare those against **every registered skeleton**, using the skeleton's own producer and term
   expressions.

Three properties follow, and they are the reason for doing it this way rather than with a pattern
table:

* **Registering a skeleton extends recognition automatically.** There is no second place to update,
  so recognition cannot drift away from what the generator can emit.
* **Scalar parameters are inferred from the graph.** Recognition solves for them, so a soft-cap
  `beta`, an ALiBi `alpha` and a decay offset `N` are read off the model rather than configured.
* **Adding an operator costs one declaration and no codegen.** A skeleton is a sympy `g`, a
  `finalize`, and two reducer kinds.

The interpreter is strict on purpose: an unknown op, a reduction over anything but the last
dimension, or more than one independent producer/consumer pair is a refusal, not a guess.

## (2) Code generation is derived, not templated

A skeleton declares only:

```python
Skeleton(name="rmsnorm_max",
         g=c_sym / sp.sqrt(s_sym + eps),         # the ONLY operator-specific input
         finalize=_FINAL_ACC,                    # what the operator MEANS
         producer_expr=c_sym ** 2 / N,           # what the running producer value is
         producer_reducer="sum",                 # <- independent axis
         consumer_reducer="max")                 # <- independent axis
```

and everything else is computed: the repair `h`, the consumer term, the identity elements, the
division guard, and the entire Triton source.

Two axes turned out to be genuinely independent, and finding that out required an experiment
rather than insight: every early skeleton happened to be *(producer = max, consumer = sum)*, so
both were hardcoded — which made the paper's own **RMSNorm + max** example inexpressible, for a
reason that had nothing to do with the algebra. Its producer `mean(x^2)` is additive while its
consumer is a maximum.

### How the repair is derived

The paper writes the repair as `h(t, r, r') = g(r', g_c^{-1}(r, t))` (Eq. 5) — a formula that
reads as "invert the term producer in `c`, then re-apply it at the new producer value". The first
implementation took that literally and used `sympy.solve` for the inversion.

The authors' own implementation does something different and better, and it was worth borrowing:
it never inverts. It **separates the variables** — asking whether `g(s', c) / g(s, c)` is
independent of the data `c` — and then *proves* that the resulting `h` distributes over the
reducer. The two agree on every operator here (pinned as a golden table in
`tests/test_neptune_opt.py`), and the separation form is strictly more general, because it also
exposes the case where the ratio still depends on `c` but a *shift* absorbs it:

```
h(t) = a(s, s') * t + b(s, s')
a = g'_c(s', c) / g'_c(s, c)          # match the c-derivatives of the two sides
b = g(s', c) - a * g(s, c)            # accepted when b is free of c
```

Condition (4) then decides which repairs are legal, and it is not the same answer for both
reducers: for `f = x + y` the bias must be **zero** (`a(x+y)+b ≠ (ax+b)+(ay+b)`), while for
`f = max` any bias is fine, because `t ↦ a·t + b` is a max-morphism for `a > 0`. So an additive
repair is admissible exactly on a max consumer — which is where the two "shift" families live.
Adding this took the generator from 17 to **19 of 28** families, i.e. from 89% to 100% of what the
algebra admits.

Writing the derivation this way also exposed a latent bug: the fresh producer value was floored
(`tl.maximum(m_new, 1e-30)`) for every non-exponential fold, because the flag used as a proxy
(`is_log`) happened to give the right answer for all eight registered skeletons. The floor is a
*division* guard, and an additive fold never divides — flooring its producer is not a rounding
nicety but a real error, since the value is used as an offset. The guard is now decided by
inspecting the derived expressions (`_needs_denominator_floor`), and the emitted **code** for all
eight registered kernels is unchanged (verified by comparing the generated source with comments
stripped, before and after).

### The one thing the generator cannot derive

`finalize` — what the operator *means* — has to be declared. `g` decides whether the fusion is
**legal** (conditions (3) and (4) are decidable); `finalize` decides what the operator *is*, and no
amount of algebra recovers it from `g`.

This is not a theoretical caveat. Declaring `finalize` wrongly for one operator produced a 3.4
relative error, and — more instructively — the *test* passed, because the reference implementation
had been written from the same misreading. A self-consistent error is invisible; only a reference
written from the operator's documentation catches it.

## (3) Integration: where the rewrite has to happen

This is the part with no paper to follow, and it is where the measurements matter most.

**There is no hook to plug into.** AOTInductor's entry point is
`aoti_compile_and_package(exported_program)`; `torch.export.export` has no pass argument and
`torch/_export/` contains no pass registry. Of the five `*_custom_pass` hooks that Inductor
exposes, all run *after* AOTAutograd:

| placement | kernel occurrences in the `.pt2` |
|---|---|
| present during tracing | 41 |
| `export` → rewrite `ep.graph_module` → AOTI | **41** (4/4 runs) |
| `make_fx` → rewrite → `export` → AOTI | 41 |
| inside Inductor's `post_grad_custom_pre_pass` | 1, and **cubins = 0** |

The generated op is a `torch.library.triton_op`, and it becomes a Triton kernel only during
AOTAutograd's functionalisation. That is why a pass running inside Inductor cannot work however
idiomatic it looks: the op has already missed the only window in which it could become a kernel.
The reason the second row works is that **AOTI re-runs `aot_autograd` on the exported program**,
so rewriting the exported graph is enough — no re-trace, no re-export.

This repository's own history is a cautionary tale about measuring carefully. An earlier version
claimed the rewrite *had* to precede `export`, on the strength of a single 0x measurement. That
was wrong: the 0x came from the then-current rewrite replacing the graph's **output** with the op,
whereas replacing only the operator's own region gives 41x reproducibly. The mechanism behind that
difference is still not explained, and it is recorded as an open question rather than papered over.

## Safety model

A fusion pass that is occasionally wrong is worse than no pass, so there are three gates:

1. **Region gate.** The recognised operator must be the operator: the rewrite replaces the node
   computing the operator's own `finalize`, found by matching that expression, and nothing else.
   This exists because of a real bug — the first version replaced the graph output, which for
   `add(fused_result, residual)` silently deleted the residual.
2. **Verify gate.** The rewritten program is executed and compared against the original module on
   the example inputs; on mismatch the fusion is reported as failed rather than passed on.
3. **Failure gate.** Nothing raises. Any exception at any stage leaves the graph alone and records
   the reason in a report that always says something.

Everything runs on CPU without Triton where possible: recognition, region extraction and the gates
are all pure graph and algebra work, so the tests that matter most do not need a GPU.

## Scope, honestly

Recall from [paper.md](paper.md) that the paper's classification admits an operator only if
condition (4) holds. Across 28 operator families, **19 are `EXACT`** and the generator emits **all
19** of them. The remaining 9 are enumerated with reasons in `tools/coverage_codegen.py`, and the
structural limits are:

* `NEEDS_STATE` families (the variance / Welford group) need state carried across iterations,
  which the accumulator-repair frame does not express;
* a `mul` consumer needs a carried count.

Both are stated as limits rather than worked around, because "the algebra does not reach here" is
a useful thing for a reader to know.

## Verification map

| claim | where it is checked |
|---|---|
| recognition of every registered skeleton | `tests/test_recognize.py` (CPU) |
| region extraction, gates, operator library | `tests/test_neptune_opt.py` (CPU) |
| dynamic shapes (symbolic row length) | `tests/test_neptune_opt.py` (CPU + GPU) |
| the derived repair, for every registered skeleton | `tests/test_neptune_opt.py` (CPU, golden table) |
| the affine repair, derived and replayed numerically | `tests/test_neptune_opt.py` (CPU) |
| the affine kernel compiles and is exact on hardware | `experiments/gpu_m16_affine.py` (GPU) |
| the generated kernel inside the `.pt2` | `experiments/gpu_m15_integration.py` (GPU) |
| one kernel serving several widths | `experiments/gpu_m15_integration.py` §E (GPU) |
| generator correctness across skeletons | `experiments/gpu_m10_codegen_tests.py` (GPU) |
| how much of the algebra is covered | `tools/coverage_codegen.py` (CPU) |

Raw logs for these are under [evidence/](evidence/).
