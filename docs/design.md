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

### The frame needs TWO reductions, and real modules need decomposing

Measured against real `torch.nn` modules (`experiments/real_modules.py`), which is a much harsher
test than the family study because that study is a list *this project* wrote:

| module | as exported | after core decompositions |
|---|---|---|
| `torch.logsumexp` | `aten.logsumexp.default` | **`logsumexp`** |
| `t - logsumexp(t, -1, keepdim=True)` | `aten.logsumexp.default` | **`logsumexp`** |
| `logsumexp(t / 8)` | `aten.logsumexp.default` | **`lse_scaled`** (`T` inferred = 8) |
| `nn.RMSNorm` | `aten.rms_norm.default` | one reduction — **out of frame** |
| `nn.LayerNorm` | `aten.layer_norm.default` | `native_layer_norm` — needs its own expansion |
| `nn.Softmax` | `aten.softmax.int` | `aten._softmax.default` — not decomposed by core |
| RMSNorm + row max | `rmsnorm_max` | `rmsnorm_max` |

Two things follow, and the second is the more important one.

**1. Real models must be decomposed first.** A module exports as ONE high-level aten op, not as a
producer/consumer pair, so the frame cannot see it. `neptune_opt.export(..., decompose=True)` (the
default) applies the core-aten decompositions via `ExportedProgram.run_decompositions()`; on an
already-exported program, `ep = ep.run_decompositions()` first. This is not a detour — the
decomposed form is what AOTAutograd hands Inductor anyway. When `apply` refuses and decomposition
would have helped, the report says so by name.

**2. Some operators are out of frame structurally, not for want of algebra.** RMSNorm, LayerNorm
and softmax's normalisation have a **single** reduction with elementwise work around it. This frame
fuses a producer reduction *with a consumer reduction*; a lone reduction feeding elementwise
arithmetic cannot be served by adding more algebra, however much of it is added. That is worth
knowing before investing in coverage: the right response is to say so (the interpreter reports
"only 1 reduction(s) found ... outside it") rather than to keep counting families.

**Where the frame does win** is the overlap: two dependent reductions over the same row, which is
what `rmsnorm_max`, `stable_l2`, `decay_renorm` and the log-sum-exp fold are — and Inductor
compiles each of those into *two* kernels, because it cannot fuse two dependent reductions.

### Single ops that ARE a reduction pair

`x.var(-1, unbiased=False)` does not export as a producer/consumer pair — it is one `aten.var.dim`
node — so the interpreter *expands* it into the pair the algebra needs (`mean(c)`, then
`mean((c - mean)^2)`), producing exactly the symbols a decomposed graph would. That matters because
real models call `.var()` and `.std()` rather than writing out the mean and the centering. Same for
`aten.std.dim`, and for the `correction=0` spelling.

The flag is **read, never assumed**, and this is a correctness boundary rather than a nicety:

| graph | fused? |
|---|---|
| `x.var(-1, unbiased=False)` | yes — `/N` |
| `x.var(-1, correction=0)` | yes — `/N` |
| `x.var(-1)` (the DEFAULT) | **refused** — `/(N-1)` |
| `x.std(-1)` (the DEFAULT) | **refused** — `/(N-1)` |

The two forms differ by a factor of `N/(N-1)`, so treating an absent flag as "biased" would
silently change the numbers of the most common way anyone writes a variance. Both refusals are
pinned in `tests/test_recognize.py`, which is also where the fused positive cases live
(19 cases, including variance as an *intermediate* of a batch-norm-like computation).

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

### When no repair exists: the two-pass plan

Everything above assumes a repair exists. For some operators it cannot, and the reason is worth
stating precisely, because it is not about the accumulator at all — it is about the **term**. For

```
g(s, c) = (c - s)^2          # variance, standard deviation, batch-norm stats
```

the term needs the **mean itself**. Repairing a running `sum((c - s)^2)` as `s` moves requires the
running *count* as well (`sum((c-s)^2) = sum((c-s')^2) + n*(s'-s)^2`), and the count is state this
frame does not carry. That is the paper's `NEEDS_STATE` group: 5 of the 28 families in the study.

The answer is not more algebra. It is a second traversal, and declaring it is the whole change:

```python
register(Skeleton(
    name="variance",
    g=(c_sym - s_sym) ** 2 / N,   # what torch.mean() of the centered squares sums
    finalize=_FINAL_ACC,          # identity: the accumulator IS the operator's value
    producer_reducer="sum",
    two_pass=True,
    producer_expr=c_sym / N,      # the mean, as the graph spells it
    intrinsic_params=((N, "ncols"),),
))
```

which emits

```
# pass 1: the producer, whose 1/N is already in the OPERAND
m = 0.0
for off in range(0, ncols, BLOCK):
    ...  m = m + tl.sum(v * (1.0 / ncols))
# pass 2: the producer is constant, so there is NO repair to apply
acc = 0.0
for off in range(0, ncols, BLOCK):
    ...  acc = acc + tl.sum((1.0 / ncols) * (v - m) * (v - m))
```

The producer never moves during pass 2, so the repair is the identity **by construction** — which is
why `repair_pair` is never consulted and `check_supported` asks a different question instead: the
producer must be *declared* (operand and reducer) and the **term must read it**, or pass 1 is dead
work.

**Where every `1/N` is written is not cosmetic.** A first version of this skeleton applied the 1/N
afterwards — a `producer_finalize` step plus `finalize=acc/N` — which is algebraically identical and
*unrecognisable*: `torch.mean` is a sum over `ncols`, so the recogniser sees a mean as the operand
`c / N` and compares that against `producer_expr`. The field it needed therefore did not exist, and
the operator could never be matched from a graph — which defeats the purpose of the plan, since its
whole point is to serve operators that appear in real models. `rmsnorm_max` had been describing a
mean of squares this way all along (`producer_expr=c**2/N`). For a max-abs producer
declared with `two_pass` and an integer shape value under a root, the intrinsic is bound as a float;
see `_intrinsics_needing_float`.

Two guards had to be re-derived for this frame, and the asymmetry is instructive:

| | one-pass frame | two-pass frame |
|---|---|---|
| `g` mentions the fresh producer symbol | **refused** — "fresh" is a *running* value, so the term would be order-dependent | **required** — "fresh" is the single final value, and reading it is the point |
| `g` ignores the producer | refused — the repair collapses to the identity | refused — pass 1 would be dead work |

Same symbols, two frames, opposite verdicts, and both are correct. The second row is the one that
matters for honesty: without it the out-of-framework control `g = exp(c - |c|)` "separates" in this
plan and gets reported as covered, while the emitted kernel reduces a producer that nothing reads.

**What it costs.** This plan buys capability, not speed, and that distinction is worth stating
plainly. It reads the row twice, so it wins only where the decomposed form's second kernel and its
materialised intermediate cost more than a second read. Measured on an RTX 5090 D v2
(`experiments/gpu_m17_two_pass.py`, which measures a ONE-PASS kernel at the same shapes so the two
causes can be told apart):

| shape | decomposed | two-pass `variance` | one-pass `stable_l2` |
|---|---|---|---|
| `(8, 4096)` | 0.0347 ms | **1.52x** | 1.52x |
| `(8, 16384)` | 0.0563 ms | **2.07x** | 2.09x |
| `(8, 65536)` | 0.0571 ms | 0.97x | 1.17x |
| `(32, 65536)` | 0.0551 ms | 0.93x | 1.13x |
| `(1, 262144)` | 0.0516 ms | 0.30x | **0.39x** |

(One run's medians; repeated runs move these by a few percent, which is why the suite does not
assert a speedup here.)

The last row is the important one: the one-pass kernel loses there too, so that is a property of the
**frame** (one program per row, a serial scan of the columns) rather than of this plan. The gap
between the two columns is what the second pass actually costs. This is why the GPU suite asserts a
>1.05x speedup only for the one-pass plan and merely "not pathological" here.

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

Across 28 operator families the generator now emits **26**, by two plans:

* **one-pass** (repair exists): 20 `EXACT` families. The accumulator is repaired as the producer
  moves, so the row is read once.
* **two-pass** (no repair exists): the 5 `NEEDS_STATE` families plus the "term reads the final
  producer" family. Pass 1 finalizes the producer, pass 2 reduces the term with the producer
  constant.

The 2 refusals are enumerated with reasons in `tools/coverage_codegen.py`:

* a `mul` consumer (`prod + ratio`) needs a carried count, which neither plan provides;
* `g = exp(c - |c|)` is a deliberate out-of-framework control: its term ignores the producer, so
  pass 1 would be dead work and there is no fold at all. (That guard was lost once during a
  refactor of this very plan, and the coverage probe reported the control as *covered* on the next
  run — which is what the probe is for.)

The distinction that survives from the original framing: the algebra's condition (4) decides
whether a **one-pass** fold exists, and the coverage study's `NEEDS_STATE` verdict is exactly
"no one-pass fold exists". That verdict was never a statement that the operator cannot be
served -- only that it cannot be served in one pass -- and the two-pass plan is what that
observation turns into.

## Verification map

| claim | where it is checked |
|---|---|
| recognition of every registered skeleton | `tests/test_recognize.py` (CPU) |
| region extraction, gates, operator library | `tests/test_neptune_opt.py` (CPU) |
| dynamic shapes (symbolic row length) | `tests/test_neptune_opt.py` (CPU + GPU) |
| the derived repair, for every registered skeleton | `tests/test_neptune_opt.py` (CPU, golden table) |
| the affine repair, derived and replayed numerically | `tests/test_neptune_opt.py` (CPU) |
| the affine kernel compiles and is exact on hardware | `experiments/gpu_m16_affine.py` (GPU) |
| the two-pass plan's kernel, correctness and speed | `experiments/gpu_m17_two_pass.py` (GPU) |
| the generated kernel inside the `.pt2` | `experiments/gpu_m15_integration.py` (GPU) |
| one kernel serving several widths | `experiments/gpu_m15_integration.py` §E (GPU) |
| generator correctness across skeletons | `experiments/gpu_m10_codegen_tests.py` (GPU) |
| how much of the algebra is covered | `tools/coverage_codegen.py` (CPU) |

Raw logs for these are under [evidence/](evidence/).

**Every GPU script exits non-zero when a check fails**, and the numbers in the table above are
asserted rather than printed. They were not always: `gpu_m10` printed `449 passed, 44 failed` and
exited 0, `gpu_m12`/`gpu_m13`/`gpu_m15` printed verdicts and exited 0, and `gpu_m13` was crashing in
section `[1]` — so the section `[4]` AOTI evidence it exists to produce had never actually run. One
`raise SystemExit(1 if fails else 0)` per script is what surfaced all of it. `gpu_m10` also seeded
its test data with `hash((name, shape))`, which is salted per process, so the suite tested different
random data on every run. If you add a check, assert it, and if you need randomness, seed it
deterministically.
