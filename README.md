# neptune-opt

**Break-and-repair reduction fusion for PyTorch AOTInductor, generated from algebra.**

A numerically stable softmax, L2 norm, RMSNorm or attention all need *two reductions of the same
data where the second depends on the first*. That dependency normally costs two passes over the
input. This library applies [Neptune](#based-on) — break the dependency, then repair the
accumulated result — to find such operators in a graph, derive a single-pass Triton kernel for
them from algebra, and carry that kernel into an AOTInductor `.pt2`.

The kernel is not chosen from a table. An operator is described by one symbolic expression, and the
repair, the term and all of the Triton source follow from it.

```python
import neptune_opt

ep = torch.export.export(model, args)      # your existing AOTI flow
report = neptune_opt.apply(ep)             # <- the line you add
print(report.summary())                    # always says what it did, and what it skipped

torch._inductor.aoti_compile_and_package(ep, package_path="model.pt2")
```

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
| generated kernel present in the `.pt2` | verified (41 kernel occurrences, `cubins ≥ 1`) |
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

* **One operator per graph.** `apply` fuses a single region; a model with two RMSNorms gets one
  fused. Lifting this needs the interpreter to skip already-fused ops.
* **Only row reductions.** A reduction over any other dimension is refused outright, as is a
  `.mean()` over anything but the last dimension. Deliberate — nothing verified the reduction dims
  before, so `sum(dim=0)` would have been silently reinterpreted as a row reduction.
* **The operator's tail must match the declared `finalize`.** An algebraically equivalent but
  textually different readout will not match. Declared constants are inferred, so those are fine.
* **Out of frame structurally:** an operator with a single reduction and elementwise work around
  it — `nn.RMSNorm`, `nn.LayerNorm`, softmax's normalisation — cannot be served by more algebra,
  because the frame fuses two *dependent reductions* (see above).
* **Refused, with reasons:** a `mul` consumer (`prod + ratio`) needs a carried count; a
  runtime-variable exponent is refused because the only safe lowering is `exp(log(x)·p)` and that
  is nan for negative `x`; `x.var(-1)` is refused because its default is the `/(N-1)` form while
  the kernel computes `/(N)`.
* **A decomposition pass may be required.** Real modules export as one high-level aten op, so
  `apply` on a freshly exported program will refuse and say so; `export(..., decompose=True)` (the
  default) handles it. `aten._softmax` / `aten.native_layer_norm` are not decomposed by the core
  table at all, so those modules are still out of reach.
* **The LSE *inside* softmax is not extracted.** `nn.Softmax`/`nn.LogSoftmax` would need a graph
  expansion that materialises the log-sum-exp as a node before the region gate can replace it.
  Not implemented.
* **Verified on one stack:** torch 2.11.0+cu128 with Triton 3.6.0 on an RTX 5090, plus torch 2.14.1
  on CPU. The `triton_op` API differs between releases, so portability across versions is untested.
* **fp32 inference only.** No gradients, and `BLOCK` is not autotuned.

## Repository layout

```
neptune_codegen/     algebra -> Triton source
  symbols.py           the sympy symbols the algebra is written over
  skeleton.py          the skeleton spec and the registry
  algebra.py           conditions (3)/(4) as executable code: derive the repair and the term
  emitters.py          every piece of Triton text, derived rather than templated
  library.py           the registered operators (one declaration each)
  template.py          render the kernel, compile it into a real Triton function
  torch_op.py          expose a generated kernel to torch.compile / AOTInductor
  references.py        the references the kernels are checked against
neptune_recognize/   graph -> "which skeleton is this?"
  interp.py            interpret an FX graph symbolically
  params.py            recover a skeleton's scalar parameters from the graph
  match.py             compare expressions against the registry
  rewrite.py           replace the operator's own region with one kernel call
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
python tools/coverage_codegen.py       # how much of the algebra the generator covers, CPU
# on a CUDA machine:
python experiments/gpu_m10_codegen_tests.py   # generator correctness across skeletons
python experiments/gpu_m16_affine.py          # the affine repair, compiled and checked
python experiments/gpu_m17_two_pass.py        # the two-pass plan, correctness and speed
python experiments/real_modules.py            # what real torch.nn modules do, and why they don't
python experiments/gpu_m15_integration.py     # the .pt2 evidence, end to end
```

## License

MIT — see [LICENSE](LICENSE). The paper this work builds on is cited above and is not covered by
this license.
