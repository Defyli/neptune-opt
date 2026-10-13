# Roadmap: research prototype → usable product

The mechanism works and is verified end to end. What makes it a *prototype* rather than a product
is reach, reliability and packaging — not the algebra. This file is the plan, and it is written so
that each iteration can be judged by a script with a real exit code rather than by a claim.

## What "usable product" means here

A user should be able to `pip install` this, add **one line** to an existing AOTI flow, and get a
fused kernel — with a report that explains every refusal — on the operators in the frame, in the
dtypes real inference uses, on the torch versions in the field, with the limits documented and
true.

### Definition of done

| # | criterion | how it is judged |
|---|---|---|
| 1 | `pip install -e .` then `import neptune_opt` works in a clean venv | scripted, in CI |
| 2 | the CPU half of the test suite runs on every push | GitHub Actions job, must be green |
| 3 | it serves the real ops in its frame: log-sum-exp (with and without temperature), the cross-entropy pattern, and the normaliser+reduction family | `experiments/real_modules.py` recognition table, no unexplained refusals |
| 4 | the standard inference dtypes work: **fp32, bf16, fp16** | GPU log with per-dtype errors; references computed in the same dtype |
| 5 | block size is chosen per shape rather than fixed | measured, with the search printed |
| 6 | every claim has a script that exits non-zero on failure | every script has `raise SystemExit`; CI runs the CPU ones |
| 7 | the documented limits are true | a test asserts the README's refusal list against the code |

### Non-goals, with reasons

* **Fusing a whole attention chain** (`QK^T → softmax → PV`). The official implementation does this
  by scheduling inside the loop nest; this one rewrites an ATen graph and emits one program per
  row. Different architecture, not more algebra. Recorded, not attempted.
* **Single-reduction operators with elementwise work around them** (`nn.RMSNorm`, `nn.LayerNorm`,
  softmax's normalisation). Structurally outside the frame: it fuses two *dependent reductions*.
  Already reported as such by the interpreter.
* **Training / backward.** This is an inference-pass optimisation.
* **Multiple GPUs, CUDA graphs, distributed.**

## Iterations

Each iteration ends with: tests updated, GPU verification with real exit codes, docs updated, one
commit. An iteration is not "done" until the previous sentence is true.

### I1 — close the known verification gap  ✅ done

*Deliverable*: dynamic (symbolic) row length verified for **all** in-frame skeletons, not just
`rmsnorm_max` (which was all `gpu_m15_integration.py` §E covered). See
[evidence/exp34_dynamic_shapes.txt](evidence/exp34_dynamic_shapes.txt): all 11 skeletons fuse from
a symbolic export, the `.pt2` carries the generated kernel, and ONE kernel is correct at widths
1024 / 999 / 333 / 7 with relative errors ~1e-7. `experiments/gpu_m18_dynamic_shapes.py`, 36
checks, exit 0.

Two blockers had to be removed, and one of them was hiding a *vacuous test*:

* the symbolic decomposed graph contains `x.reshape(x.shape[0], -1)`, which the interpreter
  refused. Accepted now, but **only** for a 2-D input whose target is the same leading integer and
  `-1` -- any other reshape changes which elements share a row, and treating it as the identity
  would silently compute a different operator. `aten._assert_tensor_metadata` (a pure assertion
  that computes nothing) and `view`/`reshape` are now understood;
* `lse_scaled` declared no default for `T`, and `param_defaults` fills undeclared scalars with 1.0
  -- so `T=1`, which makes the skeleton the PLAIN log-sum-exp. Every earlier sweep of it, including
  the "542 checks" run, was exercising a degenerate case. It even matched the `logsumexp` skeleton
  in the recognition suite, which is how it surfaced. `T` is now declared as 2.0.

*Risk*: low. *Exit criterion*: one kernel serves several widths for each skeleton, with the
artifact carrying the generated kernel and the numbers unchanged.

### I2 — reliability infrastructure  ✅ done

`.github/workflows/ci.yml` runs the CPU half on push and PR across **torch 2.4.1 / 2.6.0 /
latest** — green on all three (run #2, `fe29279`). It was red first, which is the point: the
suite passed locally on 2.14.1 and the tests' `numpy` dependency was declared nowhere, so
`pip install -e ".[test]" && pytest` failed for anyone who tried it. CI installs the test EXTRA
now, so an undeclared test dependency fails in CI rather than on a user's machine.

*Deliverable*: GitHub Actions running the CPU half (recognition, gates, coverage tool, compile
check, plus the packaging check from I5's criterion 1 once it exists); pytest usable, so the
README's instruction is real; `docs/verification.md` mapping every claim → script → last result.

*Why first*: every later iteration adds surface area. Right now the only thing standing between a
regression and `main` is that I remember to run the suites.

*Risk*: low, mostly mechanics. *Exit criterion*: CI green on a push that breaks a test (verified by
deliberately breaking one).

### I3 — dtypes: bf16 and fp16  ✅ done

`experiments/gpu_m19_dtypes.py`: **134 checks, 0 failed** (see
[evidence/exp35_dtypes.txt](evidence/exp35_dtypes.txt)). Every skeleton returns the INPUT's dtype
and is numerically right in all three; measured worst relative error vs the bound:

| dtype | worst measured | bound | what dominates |
|---|---|---|---|
| fp32 | 2.9e-07 | 3.1e-05 | summation order over the row |
| bf16 | 3.7e-03 | 3.1e-02 | bf16's eps/2 = 3.9e-03 |
| fp16 | 4.6e-04 | 3.9e-03 | fp16's eps/2 = 4.9e-04 |

A control pins that the dtype is not decoration: `1 + 2**-9` is kept by fp16 and rounded away by
bf16, so a kernel that secretly up-cast would be caught.

Two corrections along the way, both worth keeping: the OUTPUT was allocated fp32 regardless of
the input (a dtype mismatch for the graph, not a precision question), and the tolerance must
scale with the reduction LENGTH -- `linear_attn` over 2048 terms showed 6.2e-6, which is 13 ulps
of fp32 and entirely correct, so `summation_tolerance` adds `eps_f32*sqrt(n)`. And an early
version of the control asserted fp16 could represent `1 + 2**-12`; it cannot (fp16 has 11 mantissa
bits, not 23). The test was wrong, not the kernel -- the third time in this project that a
reference/expectation written from a misreading had to be corrected.

The accumulation still happens in fp32 whatever the input dtype, and that is pinned by a CPU test
(`test_accumulation_stays_fp32_whatever_the_input_dtype`) because accumulating in bf16 is a
plausible "optimisation" that would lose catastrophic precision on long rows.

### I4 — reach: expose the reductions hidden inside compound ops  ✅ done

The plan was to expand `aten._softmax` / `aten._log_softmax` / `aten.native_layer_norm` so their
log-sum-exp (and layer norm's variance) becomes a node the region gate can replace. Before writing
a graph-mutating pass, `experiments/gpu_m20_i4_ceiling.py` measured its **ceiling**: build by hand
what the lowering would produce, and race it against Inductor's own lowering.

```
    variant                        ms   vs torch's own
    softmax (torch)            0.0583   --
    softmax via our LSE        0.0985   0.59x   <- 2x SLOWER
    log_softmax (torch)        0.0588   --
    log_softmax via our LSE    0.0417   1.41x   <- a real win
    layer_norm (torch)         0.0498   --
    mean + our variance        0.1221   0.41x   <- 2.4x SLOWER
```

**So I4 is `_log_softmax` alone**, and the other two are recorded as out of scope with the reason:
Inductor has a dedicated *persistent* lowering for last-dim softmax and for layer norm, and fusing
the reduction ourselves means splitting one good kernel into our kernel plus elementwise work.
Reach is not value, and this is the measurement that distinguishes them.

Shipped as `neptune_recognize/lowering.py`, wired into `apply(lower=True)` — which is now the
default, so `neptune_opt.export(...)` and `neptune_opt.apply(ep)` both get it, while
`apply(ep, lower=False)` remains the control. Three decisions in it are load-bearing, and each was
wrong once:

* **the inserted nodes carry metadata, produced by running the same expansion on the input's fake
  tensor** (one spec, two interpreters: FX nodes and FakeTensors). The first version inserted bare
  nodes, so the recogniser could not tell whether the operator keeps the reduced dimension, and
  every single rewrite was refused with *"matched but the operator's output rank is unknown"*. The
  hand-written shape arithmetic that replaced it was replaced in turn — it is what got the edge
  case wrong.
* **the expansion has to be ours.** `get_decompositions([aten.log_softmax.int])` returns an
  **empty table** — and that is the overload `torch.export` actually produces — while
  `aten._log_softmax.default` has one entry. On torch 2.11 `ep.run_decompositions()` merely
  RENAMES `log_softmax.int` to `_log_softmax.default` and stops (`gpu_m21` §1 prints all three
  facts). No amount of calling the core decompositions exposes the log-sum-exp.
* **the commit gate is rewritability, not recognition** (`would_rewrite`, not `recognize`). A
  region can be recognised and still refused by the rewrite; a lowering gated on recognition alone
  would then be committed and fuse nothing, leaving the model with more kernels than the operator
  it replaced. `tests/test_lowering.py::test_gate_rejects_a_lowering_that_does_not_rewrite`
  reproduces exactly that bug against the gate.

*Verified*: `experiments/gpu_m21_i4_log_softmax.py`, **37 checks, 0 failed, exit 0** — raw log in
[evidence/exp36_i4_log_softmax_gpu.txt](evidence/exp36_i4_log_softmax_gpu.txt). End to end
(`torch.export` → AOTI package → load → run, 256x16384, launches pipelined), the generated kernel
appears **41 times** in the fused `.pt2` and **0** times in the baseline:

| spelling | fp32 | fp16 |
|---|---|---|
| `torch.log_softmax` | 1.17x | 1.27x |
| `F.log_softmax` | 1.11x | 1.42x |
| `t - logsumexp(t, keepdim=True)` (fused before this iteration) | 1.21x | refused (see below) |

Numbers match Inductor's own lowering to a few ULPs (fp32 `1.9e-06` against a `8.3e-06` bound;
fp16 `7.8e-03` against `6.0e-02`).

Two honest readings of that table. The win is **1.11x–1.42x**, not the 1.41x the ceiling
experiment reported: that number was a hand-built best case, and it is the top of this range, not
its middle. And the mechanism is visible in the GPU log, because Inductor says so itself —
*"Online softmax is disabled on the fly since Inductor decides to split the reduction"* — which is
also the reason this frame wins here and loses on tall-thin shapes (see I6).

*New limit found while verifying, recorded rather than papered over*: the hand-written spelling in
**fp16** does not fuse. `torch.logsumexp`'s core decomposition inserts an explicit fp32 upcast
(`aten._to_copy`) for half precision, and the symbolic interpreter refuses an op it cannot model
instead of matching through it. The refusal is pinned as an expectation in `gpu_m21`, so if the
interpreter later learns to see through a pure dtype cast the experiment fails and points here.
This is a refusal, not a wrong answer, and it does not affect the operators this iteration serves.

*Risk*: was medium, is now low — one operator, the ceiling measured first, the pass committed only
when it enables a rewrite. *Exit criterion*: met, and `nn.LogSoftmax`, `F.log_softmax` and
`torch.log_softmax` all carry the generated kernel in the `.pt2` in both fp32 and fp16.

### I5 — packaging  ✅ done

`docs/api.md` documents the public surface, and **`tests/test_api_docs.py` asserts it against
`neptune_opt.__all__` in both directions** — a phantom `neptune_opt.thing` fails, and an
undocumented public name fails. Both directions were verified by mutation (adding a phantom name,
and deleting `std` from the list), because a documentation test that cannot fail is decoration.
The README's limits section had already gone stale once.

`CHANGELOG.md` records 0.2.0; the version is bumped. CI installs `-e ".[test]"`, imports the
package, and runs the README's quickstart, which covers the clean-install criterion on every push
and across torch 2.4.1 / 2.6.0 / latest.

*Risk*: low. *Exit criterion*: criterion 1 of the definition of done, in CI.

### I6 — performance: block autotuning, then decide on split reduction  ✅ done

**Deliverable (a), shipped.** `neptune_codegen/autotune.py` chooses `(BLOCK, num_warps)` per shape;
`neptune_codegen/launch_table.py` holds the measurements (a generated file, with the command and the
machine in its header); the configuration is part of the operator's IDENTITY, so
`apply`/`export` bake it into the registered op name — two shapes get two ops instead of whichever
registered first deciding for both. `experiments/gpu_m22_autotune.py`, **40 checks, exit 0**, log in
[evidence/exp37_autotune_gpu.txt](evidence/exp37_autotune_gpu.txt).

The rule the table encodes is the interesting part, and the first attempt got it wrong in a way
worth recording. Measured across the candidate set, most shapes spread by a few percent run to run:
for a row of 4096 the best four configurations are 0.0098 / 0.0099 / 0.0099 / 0.0101 ms. Recording
that minimum would have been recording noise, in a file that then decides a compiled artifact's
launch. So the policy is `autotune.choose`: **keep the default unless a candidate beats it by more
than the measured noise margin** (5%), and for shapes sharing a bucket take the best WORST CASE
(`choose_many`), not the best of whichever shape was timed first — the first version flapped at the
margin, choosing two different configurations for one bucket off 1.0500x vs 1.0501x.

What the table then holds is short, and that is the finding:

| bucket | chosen | gain vs the old fixed default |
|---|---|---|
| `('single', 262144)` | `(8192, 8)` | **3.04x** (0.1252 → 0.0411 ms) |
| `('many', 262144)` | `(8192, 8)` | 1.07x |
| `('many', 1024)` | `(512, 2)` | 1.23x |
| `('few', 4096)` | `(512, 2)` | 1.05x |
| everything else measured | the default | within noise |

End to end through AOTI at the shape where the choice is not noise (a single row of 262144, both
sides packaged and loaded): **2.74x** (0.1315 → 0.0481 ms), and the measurement is only meaningful
because the configuration is visible in the package's op name (`_b8192w8`), which is asserted.

Three things had to be fixed for that number to exist at all, and all three were silent:

* `CG.run("logsumexp", x)` raised `KeyError: no value supplied for scalar parameter` — the skeleton's
  scalars had to be recited by the caller. Every candidate in the first search therefore "failed to
  compile", the sort returned the first of seven `inf` timings, and a table was written from them.
  `run` now fills the skeleton's declared defaults, and `autotune.best_of` REFUSES to name a winner
  when every measurement failed;
* the bucket key was too coarse: with `MAX_BUCKET` at 16384, a row of 262144 and a row of 16384 shared
  a key, and one row is not four rows. Buckets now reach `1<<20`, and one row is its own class
  (`single` / `few` / `many`) because measurement says it wants a different block;
* the op registration raised `NameError` for every operator after a refactor, because the shared
  key builder left a stale local behind — and every CPU test stubbed the very function that broke.
  The name construction is now shared (`autotune.op_key`) and tested without a GPU.

**Deliverable (b), the written decision: do NOT build split reduction.** The ceiling was measured
rather than argued — a chunked kernel (grid of `k` programs) plus a combine is exactly what a split
reduction costs — against the now-tuned single-program kernel at `(1, 262144)`:

```
    Inductor's own lowering                0.0275 ms   1.00x
    our single program, tuned (8192, 8)    0.0411 ms   0.67x
    our emulated split, best k=16          0.0423 ms   0.65x
```

Note what changed from the 0.30x this file used to record: **3x of that gap was our own default**,
not the architecture, and autotuning closed it. What remains is that Inductor splits this reduction
across programs and does it better than a chunked two-launch emulation does. So the decision is not
"split reduction is impossible" but "our split reduction would have to beat an emulation of itself
that is already 1.5x behind Inductor's" — and for a single row of 262144 the honest advice is to let
Inductor lower it. The frame wins where rows are long *and* numerous, which is where models are.

*Risk*: was medium, is now low. *Exit criterion*: met — the chosen configuration is never slower than
the default (9 direct launches, plus 2.74x end to end through AOTI), and the split question has a
number behind its answer in both directions.

### I7 — close out  ✅ done

*Deliverable*: the README leads with the one line and the report it returns; "what this is not" is
stated next to what it is; and the limits are a table whose every row is **asserted by
`tests/test_readme_limits.py` in both directions** — a row with no check fails, and a check with no
row fails. Nine rows, each a real measurement made on CPU with no Triton:

| key | how it is checked |
|---|---|
| `region-is-the-operator` | `logsumexp(x) * 2` keeps its `* 2` (the graph still multiplies after the rewrite), while an algebraically equal but differently written readout is refused |
| `one-operator-per-graph` | a graph with two in-frame operators yields exactly one fusion, and the other operator is still in the graph |
| `row-reductions-only` | `sum(dim=0)`, `mean(dim=0)`, `amax(dim=0)` are all refused |
| `var-default-is-unbiased` | `var(-1)`/`std(-1)` refused, `unbiased=False` served |
| `single-reduction-operators` | `nn.LayerNorm`, `nn.RMSNorm`, `nn.Softmax` refused after decomposition |
| `softmax-internals-not-lowered` | `lower_if_it_helps(softmax) == 0` and `== 1` for `log_softmax` |
| `needs-decomposition` | a fresh `torch.logsumexp` export refuses with that reason by name |
| `prod-ratio-needs-a-count` | `tools/coverage_codegen.py --check` still reports `f-needs-carried-state` for it |
| `lower-only-when-it-pays` | the lowering leaves an unrelated graph byte-identical |

Both directions were verified by mutation (de-keying a row, and adding a phantom row), as was the
one check that could have passed vacuously: the stub-kernel trick that makes the graph-level rows
testable on CPU was first applied at the wrong scope, so `apply` failed with `triton-missing` and the
assertion about the surviving `* 2` passed for the wrong reason.

### I7 — close out

*Deliverable*: the README's leads (one line, one report), a short "what this is not" section, and
the limits table asserted by a test so it cannot go stale — which it already did once.

## Order and why

I1 and I2 first: one closes a known hole for free, the other is what makes the rest of the plan
survivable. I3 before I4 because a reach increase in a dtype nobody uses is not a product. I4 is
the biggest single value item and the riskiest, so it goes after the cheap ones. I5 is cheap and
makes the thing installable while I4 is being worked. I6 last, because performance work on a
narrow reach is not worth much.

## Backlog: measured, deliberately not scheduled

Each of these has a number or a reproduction behind it, and each is recorded so that "not done" is
a decision rather than an oversight. None of them is in the frame of a single iteration above.

| item | state of the evidence | what would settle it |
|---|---|---|
| `aten._to_copy` transparency | fp16 `torch.logsumexp` decomposes with an explicit fp32 upcast and is refused as an unfamiliar op (`gpu_m21` §2, pinned there). The cast is pure algebra; seeing through it is a small change *plus* a decision about casts that narrow precision mid-region | one experiment over casts that widen, narrow and change device, with the dtype of the result pinned |
| `prod + ratio` | refused as `f-needs-carried-count`: the repair needs a count carried across the reduction, which is a different program shape, not more algebra. 26 of 28 probed families are served | a skeleton whose repair carries a second accumulator, and a measurement that it beats two passes |
| `torch.compile` backend | the rewrite lands and is numerically right, but searching the Inductor cache finds the generated kernel **0** times, against **41** through AOTI. Cause unknown; the entry point is gated behind `experimental=True` until it is understood | a diagnostic that shows where the op is dropped on the Dynamo route (the AOTI route is verified and is the documented one) |
| operator as the graph output | measured 3 times at 0x — the operator is rewritten, and the result is not faster. Unexplained | a profile of that shape |
| one region per graph | the pass rewrites the first recognisable region and stops; a model with two such operators is not covered | a loop over regions, with the interaction between them measured |
| a stack without `torch.export` | measured on torch 2.0.1 (Tesla V100): `import neptune_opt`, the gates and the launch configuration all work, but `test_recognize.py` fails at IMPORT and `test_lowering.py`/`test_readme_limits.py` at the first export, so the suite crashes where it could skip with a reason. `make_fx` — which recognition actually needs — exists on 2.0.1 | guard the export-dependent tests on availability, and build the recognition cases with `make_fx` so that half is testable on the old stack |
| the launch table is per GPU | measured: 4 deviations from the default on the 5090, 1 on the 4090D, both agreeing on the ~3x single-long-row case ([exp38](evidence/exp38_cross_gpu_and_old_stack.md)) | nothing to build — it is why the table is a generated file — but a machine without a table silently uses the default, which is worth a warning |
| fp8 / integer dtypes | not supported; fp32, bf16 and fp16 are (I3) | the accumulation story for fp8 has to be decided first — this is not a dtype table entry |
