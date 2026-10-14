# Changelog

Notable changes. The project is a research prototype being turned into a usable module; see
[docs/roadmap.md](docs/roadmap.md) for the plan and what remains.

## 0.2.0 — in development

Productisation, in the order the roadmap took it. Every entry is verified by a script that exits
non-zero on failure; the GPU ones are `experiments/gpu_*.py`, and the CPU ones run in CI on every
push.

### Verification

* **Unverified:** `experiments/gpu_m22_autotune.py` was refactored to share the measurement code
  with `measure_launch_configs.py`; the shared code is verified on two GPUs, the refactored
  experiment's own GPU run is not (it died after §1, cause unidentified).
* **Cross-GPU and old-stack verification** (`docs/evidence/exp38_cross_gpu_and_old_stack.md`): the
  same stack on an RTX 4090 D reproduces the ~3x single-long-row win (2.57x end to end) while
  choosing a DIFFERENT set of deviations, and on Tesla V100 / torch 2.0.1 the algebra, recognition,
  gates and launch configuration work while `torch.export`, `triton_op` and
  `aoti_compile_and_package` do not exist.
* Symbolic (dynamic) row length verified for **every** in-frame skeleton, not just `rmsnorm_max`:
  one compiled kernel is correct at widths 1024 / 999 / 333 / 7, and the `.pt2` carries it
  (`gpu_m18_dynamic_shapes.py`).
* fp32 / **bf16** / **fp16** support: the result takes the input's dtype, the accumulation stays
  fp32, and the tolerances are dtype-aware and reduction-length-aware (`gpu_m19_dtypes.py`).
* Recognition reaches real modules at all: they must be decomposed first, which is now the default
  in `export` and is diagnosed by name when missing (`real_modules.py`).
* **`log_softmax` is reachable now, in both directions.** The core decomposition table has no
  entry for the overload `torch.export` produces, so the log-sum-exp inside it is invisible without
  the lowering pass; with it, `torch.log_softmax` / `F.log_softmax` / `nn.LogSoftmax` fuse at
  **1.11–1.42x** in fp32 and fp16, with the generated kernel 41 times in the fused `.pt2` and 0
  times in the baseline (`gpu_m21`, 37 checks, exit 0).
* End-to-end, on real modules, both sides AOTI-packaged and loaded: **1.19–1.57x** for
  `torch.logsumexp`, `logsumexp(t/8)`, `rmsnorm_max`, `stable_l2` and a hand-written
  `log_softmax`. At small rows the same comparison is 0.9–1.05x, which is recorded rather than
  hidden (`exp33`).

### Added

* **A performance fallback, `compile_best(model, args)`.** Every other gate protects correctness; this
  one protects speed. It compiles the unmodified package AND the fused one, times both on the example
  inputs (interleaved rounds, minimum kept), and ships the faster — because a numerically correct
  fusion can still be slower (measured: `x.var(-1)` at `(1, 262144)` is 0.30x). A slower, tied,
  numerically different or non-compiling fusion returns the **unmodified baseline** with the
  measurement in `reason`. It costs two compilations, needs a CUDA device and static shapes, and
  skips the second compilation when there is nothing to fuse.
* **The launch configuration is a per-GPU registry.** `experiments/measure_launch_configs.py`
  measures one machine, prints every candidate timing, and MERGES its entry, leaving the others
  alone; it needs only torch and Triton, so a machine whose stack is too old for the AOTI path can
  still contribute (the V100 on torch 2.0.1 / Triton 2.0.0 did). A GPU with no entry uses the
  default instead of another machine's numbers — measured: the same search chose four deviations on
  an RTX 5090 D v2 and one on an RTX 4090 D.
* **A measured launch configuration** instead of the fixed `BLOCK=1024` / 4 warps: chosen per shape
  from a generated table, part of the op's identity (so it is baked into the package), and kept at
  the default unless a candidate beats it by more than the measured noise margin. Worth **3.04x** at
  a single row of 262144 columns, and **2.74x** end to end through AOTI at that shape
  (`gpu_m22_autotune.py`, 40 checks, exit 0).
* `tests/test_readme_limits.py`: the README's limits are a table whose every row is asserted against
  the code, in both directions, so a stale limit fails the build.
* A **lowering pass** (`neptune_recognize/lowering.py`, on by default in `apply`) that expands
  `log_softmax` into `x - logsumexp(x, keepdim=True)` so the region gate has a node to replace. It
  is committed **only** when it enables a rewrite, and `lower=False` is the control. `softmax`,
  `layer_norm` and a reduction over a non-last dim come out byte-identical, because the ceiling
  experiment measured them at 0.59x and 0.41x against Inductor's dedicated lowerings.
* `neptune_recognize.would_rewrite`: whether a rewrite is possible, decided without generating a
  kernel, so the question can be asked on a CPU machine and inside the lowering's trial.
* A **two-pass plan** for operators that have no repair, because their term reads the *final*
  producer (variance, standard deviation). Coverage 20 → 26 of 28 families.
* **Affine repairs**: when the ratio keeps the data but a shift absorbs it, `h(t) = a*t + b`, with
  a bias admissible only for a max consumer (condition (4)). Two families, 100% of what the
  algebra admits.
* `logsumexp(x/T)` (`lse_scaled`), found by running the recogniser over real modules; `T` is
  inferred from the graph.
* Recognition of `torch.var` / `torch.std` as a single `aten` op — and **refusal** of their
  defaults, which are the `/(N-1)` form.
* CI on push and PR across torch 2.4.1 / 2.6.0 / latest: the CPU tests, a coverage **floor**, a
  compile check, a clean install, and the README's quickstart.

### Fixed

* `best_config` never consulted the table when the GPU could not be identified, which made the
  pre-registry (flat `TABLE`) fallback dead code: a valid old-format entry was silently ignored.
  Caught by a test that asserted on `table()` succeeding while `best_config` did not look.
* Test isolation: `from package import submodule` binds an ATTRIBUTE on the package as well as an
  entry in `sys.modules`, so a test that injected a fake data module leaked into the next one.
* The autotuning search compared candidates against a SINGLE timing round, so a noisy measurement of
  the default made ordinary candidates look like wins: the first 4090 D run chose four deviations
  from the default, one of them on a shape whose entire candidate spread is ~2%. Every candidate is
  now timed in three interleaved rounds and the MINIMUM is kept, which removes the baseline's
  variance (a stall only ever adds time). Re-measured, the same machine chose one deviation. (My
  first write-up of this attributed the effect to a specific 0.1358 ms stall at `(1, 4096)`; that was
  a mis-reading of grep output and is corrected in `docs/evidence/exp38`.)
* `CG.run("logsumexp", x)` raised `KeyError` unless the caller recited the skeleton's scalars; the
  first autotuning search therefore "measured" seven failures, sorted them, and wrote a table from
  the first `inf`. `run` now fills the declared defaults, and a search in which everything failed
  reports no winner instead of one.
* Op registration raised `NameError` for **every** operator after a refactor left a stale local in
  the shared name construction — and every CPU test stubbed the function that broke, so nothing
  caught it. The name is built in one place now (`autotune.op_key`) and tested without a GPU.
* The lowering's first version inserted nodes with **no shape metadata**, so every rewrite was
  refused with *"the operator's output rank is unknown"* — the recogniser needs to know whether the
  operator keeps the reduced dimension. The metadata now comes from running the same expansion on
  the input's fake tensor (one spec, two interpreters), so the shapes are torch's own.
* The lowering's commit gate asked whether the graph was *recognised*, not whether it was
  *rewritable*. Recognition can succeed on a region the rewrite then refuses, so the pass committed
  expansions that fused nothing — strictly more kernels than the operator it replaced.
* The output was allocated fp32 regardless of the input dtype — a graph-level dtype mismatch for
  any bf16 model.
* `std` did not compile at all: sympy stores `sqrt(acc/N)` as `sqrt(acc)/sqrt(N)`, putting an
  integer shape value under a square root.
* The fresh producer was floored for every non-exponential fold; for an additive fold that is a
  real error (measured: `-2.0` where the operator is identically `0.0`).
* `lse_scaled` declared no default for `T`, and undeclared scalars default to 1.0 — so every sweep
  of it was testing the *unscaled* operator.
* Every GPU script printed verdicts and exited 0. The exit codes immediately revealed that
  `gpu_m13` was crashing before the section carrying its evidence, and that `gpu_m10`'s data was
  seeded with a per-process-salted `hash`.
* `pytest tests/` collected zero recognition tests (they lived inside a `main()`), and the tests'
  `numpy` dependency was declared nowhere.

### Documented

* A limit found while verifying I4, pinned as a test expectation rather than left as a surprise:
  fp16 `torch.logsumexp` (hand-written) decomposes with an explicit fp32 upcast
  (`aten._to_copy`) and is refused rather than matched through. The documented spellings —
  `torch.log_softmax` / `F.log_softmax` / `nn.LogSoftmax`, fp32 and fp16 — are unaffected.
* What the frame **cannot** do, with the reason: operators with a single reduction and elementwise
  work around it (RMSNorm, LayerNorm, softmax's normalisation) are outside it structurally, not for
  want of algebra.
* The two-pass plan's cost: it reads the row twice, so it wins on long rows and is at parity on
  short ones.

## 0.1.0

Initial prototype: break-and-repair reduction fusion for AOTInductor, generated from algebra
(`g`, `finalize`, two reducer kinds), recognised from a graph by comparing expressions against the
registry, and delivered through `aoti_compile_and_package`.
