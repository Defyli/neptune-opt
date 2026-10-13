# neptune-opt

[![ci](https://github.com/Defyli/neptune-opt/actions/workflows/ci.yml/badge.svg)](
    https://github.com/Defyli/neptune-opt/actions/workflows/ci.yml)

**Break-and-repair reduction fusion for PyTorch AOTInductor, generated from algebra.**

## One line, one report

```python
import neptune_opt

ep = torch.export.export(model, args)      # your existing AOTI flow
report = neptune_opt.apply(ep)             # <- the line you add
print(report.summary())
#   fused   : stable_l2
#   verified: fused output matches the original on the example inputs

torch._inductor.aoti_compile_and_package(ep, package_path="model.pt2")
```

`apply` returns a report, not a boolean, and it never raises. Every refusal comes back with its
reason (`skipped : no-fusion: unfamiliar op 'aten.layer_norm.default' …`), and the graph is left
untouched whenever the pass cannot help — so the safe way to try it is to try it.

A numerically stable softmax, L2 norm, RMSNorm or attention all need *two reductions of the same
data where the second depends on the first*. That dependency normally costs two passes over the
input. This library applies [Neptune](#based-on) — break the dependency, then repair the
accumulated result — to find such operators in a graph, derive a single-pass Triton kernel for
them from algebra, and carry that kernel into an AOTInductor `.pt2`.

The kernel is not chosen from a table. An operator is described by one symbolic expression, and the
repair, the term and all of the Triton source follow from it.

## Based on

This project is an independent engineering effort built on the following paper. **It is not
affiliated with the paper's authors and ships none of their code.**

> **Neptune: A Compilation Framework for Optimizing Machine Learning Programs with Reduction
> Dependencies**
> arXiv:2510.08726 — <https://arxiv.org/abs/2510.08726> (PLDI 2026)

```bibtex
@inproceedings{neptune2026,
  title  = {Neptune: A Compilation Framework for Optimizing Machine Learning Programs
            with Reduction Dependencies},
  year   = {2026},
  note   = {PLDI 2026},
  eprint = {2510.08726},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2510.08726}
}
```

The paper establishes the `(f, g, h)` formulation, the derived repair of Eq. 5 and conditions (3)
and (4). What this repository adds — recognition, code generation, and the AOTInductor path — is
described in [docs/design.md](docs/design.md). See [docs/paper.md](docs/paper.md) for a summary of
the paper's own findings, and
[docs/paper-notes.zh.md](docs/paper-notes.zh.md) for reading notes in Chinese.

## Status

A verified research prototype, **not a production library**. Every claim in this README is
measured; the measurements, including the ones that contradict earlier versions of this document,
are in [docs/evidence/](docs/evidence/).

| | |
|---|---|
| operator families classified (algebra) | 28 → 20 `EXACT`, 1 `AFFINE`, 5 `NEEDS_STATE`, 2 `NOT_INVERT` |
| families the generator emits | **26 / 28**, via two plans (see `docs/design.md`) |
| registered skeletons recognised from a graph | **11 / 11**, with no per-operator pattern rows |
| scalar parameters inferred from the graph | yes (`beta`, `alpha`, `N`, `eps`, …) |
| dynamic shapes | supported; one kernel serves every width |
| dtypes | **fp32, bf16, fp16** — the result takes the input's dtype, accumulation stays fp32 |
| generated kernel present in the `.pt2` | verified (41 kernel occurrences, `cubins ≥ 1`) |
| launch configuration (`BLOCK`, `num_warps`) | **measured per shape**, and part of the op's identity |
| `torch.compile` backend | **does not work** — gated behind `experimental=True` |
| real `torch.nn` modules | fuse **only after `ep.run_decompositions()`** — see below |

### What it fuses, and what it cannot

The frame fuses a **producer reduction together with a consumer reduction over the same row** —
`max` then `sum(exp(x - max))`, `mean(x²)` then `max(x / sqrt(...))`. That pairing is the win,
because Inductor compiles two dependent reductions into two kernels and cannot fuse them.

It cannot serve an operator whose consumer is **elementwise**, and that is a structural limit
rather than a missing pattern:

| | |
|---|---|
| two dependent reductions (`rmsnorm_max`, `stable_l2`, `logsumexp`, `decay_renorm`, …) | fused, 1.1–2.5x kernel-level |
| `torch.logsumexp`, `logsumexp(t / T)` | fused after decomposition; `T` inferred from the graph |
| `torch.log_softmax`, `F.log_softmax`, `nn.LogSoftmax` | fused after the lowering pass, fp32 and fp16, **1.11–1.42x** |
| `nn.RMSNorm`, `nn.LayerNorm`, softmax's normalisation | **out of frame** — one reduction |
| a `mul` consumer (`prod + ratio`) | refused — needs a carried count |

End to end on real modules (256×16384, both sides packaged with AOTI and loaded):
**1.19–1.57x** for `torch.logsumexp`, `logsumexp(t/8)`, `rmsnorm_max`, `stable_l2` and a
hand-written `log_softmax`. At small shapes (64×4096) the same comparison is 0.9–1.05x, because
the kernel is no longer the bottleneck — the win needs rows long enough to matter. Both numbers,
and the two measurement bugs that produced the misleading one, are in
[docs/evidence/exp33_real_modules.md](docs/evidence/exp33_real_modules.md).

Real models also export as one high-level aten op per module, so they must be decomposed first:
`neptune_opt.export(model, args)` does that by default, and on an existing `ExportedProgram`,
`ep = ep.run_decompositions()` comes first. `experiments/real_modules.py` measures all of this
against real modules and prints the reason for every refusal.

## Install

```bash
git clone https://github.com/Defyli/neptune-opt.git
cd neptune-opt
pip install -e .            # or: pip install -e ".[gpu]" on a CUDA machine
```

See [docs/roadmap.md](docs/roadmap.md) for what is done and what remains, and
[docs/api.md](docs/api.md) for the public surface.

Requires PyTorch and sympy. Triton is only needed to *generate* kernels, so recognition and the
whole test suite work on a CPU-only machine:

```bash
pytest tests/ -v            # no GPU, no Triton required
python examples/01_quickstart.py
```

## Entry points

Pick the row that fits how much you want to change.

| form | what you write | model edits |
|---|---|---|
| **`apply(ep)`** | one line in an existing AOTI flow | none |
| `export(model, args, **kw)` | replaces `torch.export.export`; kwargs forwarded | none |
| `neptune_opt.rmsnorm_max(x, eps=…)` | rename the operation in your model | yes |
| `neptune_opt.backend` | — | **not functional**, see [Status](#status) |

`apply` and `export` never raise: on any failure the graph is left untouched and the reason lands
in the report. Add the numeric self-check with `apply(ep, verify_with=(model, args))`.

`neptune_opt.preview(ep)` reports what *would* be fused without generating a kernel, so it needs no
Triton — handy in CI.

### The explicit operator library

When recognition declines, name the operator yourself:

```python
import neptune_opt

neptune_opt.op_parameters("rmsnorm_max")     # ('eps',)
y = neptune_opt.rmsnorm_max(x, eps=1e-6)
neptune_opt.stable_l2(x)
neptune_opt.softcap_lse(x, beta=0.5)
```

These are ordinary `torch.library` ops, so they export and package like anything else. Same
algebra, same code generation — you choose the operator instead of the compiler.

## What it fuses

| skeleton | operator |
|---|---|
| `logsumexp` | softmax / log-softmax / cross-entropy's max + sum-of-exp fold |
| `stable_l2` | overflow-safe `‖x‖₂` / RMSNorm numerator |
| `rmsnorm_max` | the paper's own example — `max_j x_j / sqrt(mean(x²) + eps)` |
| `softcap_lse` | Gemma-2 style soft-capped attention (logsumexp of `tanh(βx)`) |
| `lse_biased` | ALiBi-biased logsumexp |
| `linear_attn` | linear attention's normaliser |
| `decay_renorm` | duration / decay renormalisation |
| `sumsq_div` | scale-normalised energy |

## Known limits

Every row of this table is asserted by [`tests/test_readme_limits.py`](tests/test_readme_limits.py),
which runs on CPU with no Triton and fails when this README and the code disagree — in **either**
direction, so a limit that stops being true is caught as well as one that is described wrongly.

| limit | what it means here |
|---|---|
| **`region-is-the-operator`** | only the operator's own dataflow is replaced, so `logsumexp(x) * 2` keeps its `* 2`. A readout that is algebraically equal but written differently — `m + ½·log Σexp(2(x−m))` — is **refused**, not matched: this is a frame, not an algebra solver |
| **`one-operator-per-graph`** | `apply` fuses one region; a second in-frame operator in the same graph is left alone |
| **`row-reductions-only`** | a reduction over any dimension but the last is refused, `mean(0)` included |
| **`var-default-is-unbiased`** | `x.var(-1)` and `x.std(-1)` are refused because their default is the `/(N−1)` form while the kernel computes `/N`; `unbiased=False` is served |
| **`single-reduction-operators`** | `nn.LayerNorm`, `nn.RMSNorm` and `nn.Softmax` are out of frame: their consumer is elementwise, so there is no second reduction to fuse. More algebra cannot fix a shape mismatch |
| **`softmax-internals-not-lowered`** | the log-sum-exp inside `aten._softmax` is not extracted, by measurement (0.59x); the one inside `aten._log_softmax` is (1.11–1.42x) |
| **`needs-decomposition`** | a freshly exported `torch.logsumexp` is refused *by name*, with a diagnostic saying the core decompositions would expose it; `export(..., decompose=True)` is the default |
| **`prod-ratio-needs-a-count`** | the one probed family of 28 the generator refuses, with the reason `f-needs-carried-state` |
| **`lower-only-when-it-pays`** | the lowering pass is a no-op — byte-for-byte on the graph — whenever it would not enable a fusion |

Stated so that nobody assumes otherwise; these are **not** machine-checked, and they are the ones
most likely to have gone stale:

* **Verified on three GPUs, not on one stack.** End to end on torch 2.11.0+cu128 / Triton 3.6.0 on
  an RTX 5090 D v2 and an RTX **4090 D** (the same stack, a different GPU — and the measured launch
  table differs between them, which is why it is a generated file that carries its machine), plus
  torch 2.14.1 on CPU and CI on 2.4.1 / 2.6.0 / latest. On an **old** stack (Tesla V100, torch
  **2.0.1**, Triton 2.0.0) the algebra, recognition, gates and launch configuration all work
  (`test_autotune.py`: 11 passed; `test_neptune_opt.py`: 27 passed, 2 failed), while the integration
  surface does not exist at all: no `torch.export`, no `torch.library.triton_op`, no
  `aoti_compile_and_package`. Measured, not assumed — see
  [docs/evidence/exp38](evidence/exp38_cross_gpu_and_old_stack.md).
* **Inference only.** No gradients, and nothing here has been trained through.
* **dtypes:** fp32 / bf16 / fp16. fp8, integer and complex are not supported, and the accumulation
  always happens in fp32.
* **A runtime-variable exponent** is refused, because the only safe lowering is `exp(log(x)·p)` and
  that is `nan` for negative `x`.
* **One operator family per fused region**, and no attention-chain fusion (`QKᵀ → softmax → PV`) or
  multi-GPU: those are architectural non-goals, with the reasons in
  [docs/roadmap.md](docs/roadmap.md).

## Repository layout

```
neptune_codegen/     algebra -> Triton source
  symbols.py           the sympy symbols the algebra is written over
  skeleton.py          the skeleton spec and the registry
  algebra.py           conditions (3)/(4) as executable code: derive the repair and the term
  emitters.py          every piece of Triton text, derived rather than templated
  library.py           the registered operators (one declaration each)
  autotune.py          (BLOCK, num_warps): measured table, with a documented fallback
  launch_table.py      the measurements themselves -- generated, not hand-edited
  template.py          render the kernel, compile it into a real Triton function
  torch_op.py          expose a generated kernel to torch.compile / AOTInductor
  references.py        the references the kernels are checked against
neptune_recognize/   graph -> "which skeleton is this?"
  interp.py            interpret an FX graph symbolically
  params.py            recover a skeleton's scalar parameters from the graph
  match.py             compare expressions against the registry
  rewrite.py           replace the operator's own region with one kernel call
  lowering.py          expand a compound op the core decompositions leave whole
neptune_opt/         the user-facing layer: apply / export / operator library
repairlib/           the standalone algebra classifier (28 families, CPU + sympy only)
tests/               CPU test suite
examples/            runnable sample models
experiments/         GPU scripts that produced the evidence
tools/               coverage report
docs/                paper summary, design notes, evidence logs
```

## Reproducing the evidence

```bash
pytest tests/ -v                       # recognition + gates, CPU
# the same tests without pytest installed:
python tests/test_neptune_opt.py
python tests/test_recognize.py
python tests/test_lowering.py              # the log_softmax expansion, and its no-op guarantee
python tests/test_readme_limits.py         # this README's limits table, asserted in both directions
python tests/test_autotune.py              # the launch-configuration chooser
python tools/coverage_codegen.py --check   # coverage, and FAILS if it regresses (CI runs this)
python -m pytest tests/ -q              # recognition + gates, CPU
# on a CUDA machine:
python experiments/gpu_m10_codegen_tests.py   # generator correctness across skeletons
python experiments/gpu_m16_affine.py          # the affine repair, compiled and checked
python experiments/gpu_m17_two_pass.py        # the two-pass plan, correctness and speed
python experiments/real_modules.py            # what real torch.nn modules do, and why they don't
python experiments/gpu_m18_dynamic_shapes.py  # symbolic row length, every in-frame skeleton
python experiments/gpu_m19_dtypes.py          # fp32 / bf16 / fp16, every skeleton
python experiments/gpu_m21_i4_log_softmax.py  # log_softmax reach, speed, and the no-op guarantee
python experiments/gpu_m22_autotune.py        # the launch-configuration search, and the split decision
python experiments/gpu_m15_integration.py     # the .pt2 evidence, end to end
```

## License

MIT — see [LICENSE](LICENSE). The paper this work builds on is cited above and is not covered by
this license.
