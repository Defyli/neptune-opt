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

### I4 — reach: expose the reductions that are hidden inside compound ops

*Deliverable*: a **lowering pre-pass** that expands `aten._softmax` / `aten._log_softmax` /
`aten.native_layer_norm` into decomposed subgraphs, so the log-sum-exp (and, for layer norm, the
mean/variance) becomes a real node that the region gate can replace.

*Why*: these are the most common real ops, and today they are refused because the core
decomposition table does not expand them. This is the single biggest reach increase available
without changing the architecture.

*Risk*: **high** — it mutates the graph before recognition, so it must be semantics-preserving and
must only fire when it actually enables a fusion, with a numeric self-check. Expect a dedicated
adversarial test set (masked logits, `-inf` rows, `keepdim`, non-trailing dims).

*Exit criterion*: `nn.Softmax`/`nn.LogSoftmax` models gain the generated kernel in the `.pt2`, with
numbers unchanged, **and** a model where the expansion would not fuse is left byte-identical.

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

### I6 — performance: block autotuning, then decide on split reduction

*Deliverable (a)*: `BLOCK` (and `num_warps`) chosen per shape by a measured search, with the search
logged, instead of a fixed default.

*Deliverable (b)*: an honest evaluation of tall-thin shapes, where the frame currently loses badly
(0.30x at `(1, 262144)`, and the one-pass kernel loses too — so that is the frame, not the plan).
Split reduction is the fix and it is a real architectural step; this iteration ends with a
measurement and a decision, not with code in the other direction.

*Risk*: medium. *Exit criterion*: autotuning demonstrably not slower than the fixed default, and a
written decision on split reduction with the numbers behind it.

### I7 — close out

*Deliverable*: the README's leads (one line, one report), a short "what this is not" section, and
the limits table asserted by a test so it cannot go stale — which it already did once.

## Order and why

I1 and I2 first: one closes a known hole for free, the other is what makes the rest of the plan
survivable. I3 before I4 because a reach increase in a dtype nobody uses is not a product. I4 is
the biggest single value item and the riskiest, so it goes after the cheap ones. I5 is cheap and
makes the thing installable while I4 is being worked. I6 last, because performance work on a
narrow reach is not worth much.
