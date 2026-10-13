# Changelog

Notable changes. The project is a research prototype being turned into a usable module; see
[docs/roadmap.md](docs/roadmap.md) for the plan and what remains.

## 0.2.0 — in development

Productisation, in the order the roadmap took it. Every entry is verified by a script that exits
non-zero on failure; the GPU ones are `experiments/gpu_*.py`, and the CPU ones run in CI on every
push.

### Verification

* Symbolic (dynamic) row length verified for **every** in-frame skeleton, not just `rmsnorm_max`:
  one compiled kernel is correct at widths 1024 / 999 / 333 / 7, and the `.pt2` carries it
  (`gpu_m18_dynamic_shapes.py`).
* fp32 / **bf16** / **fp16** support: the result takes the input's dtype, the accumulation stays
  fp32, and the tolerances are dtype-aware and reduction-length-aware (`gpu_m19_dtypes.py`).
* Recognition reaches real modules at all: they must be decomposed first, which is now the default
  in `export` and is diagnosed by name when missing (`real_modules.py`).
* End-to-end, on real modules, both sides AOTI-packaged and loaded: **1.19–1.57x** for
  `torch.logsumexp`, `logsumexp(t/8)`, `rmsnorm_max`, `stable_l2` and a hand-written
  `log_softmax`. At small rows the same comparison is 0.9–1.05x, which is recorded rather than
  hidden (`exp33`).

### Added

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

* What the frame **cannot** do, with the reason: operators with a single reduction and elementwise
  work around it (RMSNorm, LayerNorm, softmax's normalisation) are outside it structurally, not for
  want of algebra.
* The two-pass plan's cost: it reads the row twice, so it wins on long rows and is at parity on
  short ones.

## 0.1.0

Initial prototype: break-and-repair reduction fusion for AOTInductor, generated from algebra
(`g`, `finalize`, two reducer kinds), recognised from a graph by comparing expressions against the
registry, and delivered through `aoti_compile_and_package`.
