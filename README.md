# neptune-opt

[![ci](https://github.com/Defyli/neptune-opt/actions/workflows/ci.yml/badge.svg)](
    https://github.com/Defyli/neptune-opt/actions/workflows/ci.yml)

**Algebra-derived reduction fusion for PyTorch AOTInductor: break-and-repair folds,
explicit chunk merging, and opt-in structured RMSNorm / segment pooling.**

## One line, one report

```python
import torch
import neptune_opt

ep = torch.export.export(model, args)                 # your existing AOTI flow
report = neptune_opt.apply(ep, verify_with=(model, args))
print(report.summary())                               # fusions, refusals, verification status

torch._inductor.aoti_compile_and_package(ep, package_path="model.pt2")
```

`apply` returns a report and transactionally replaces eligible **local regions**. GEMMs,
residuals, clamps, casts, shared inputs and unrelated graph nodes remain outside those regions.
The default has no one-region budget: use `max_regions=1` only when you explicitly want one.
Eligible last-dimension `torch.logsumexp` compounds are selectively lowered by default; a
whole-graph `ep.run_decompositions()` is **not a prerequisite** for that path.

Without `verify_with`, a fusion is explicitly **numerically unverified**. Real-algebra matching
is a legality check, not a proof of IEEE bitwise equivalence, all dynamic inputs, or a speedup.
Ordinary rewrite / generation / verification failures leave the original graph and state
untouched and return a refusal. **Fatal CUDA context errors are different:** device-side asserts
and illegal memory accesses propagate as `FatalCudaError` from `neptune_opt.ab`; stop that worker
process rather than attempting an in-process baseline fallback. `export` can also propagate
errors from the underlying PyTorch export operation.

The scalar frame derives a fold for two dependent reductions over the same row, such as
`max(x)` followed by `sum(exp(x - max(x)))`. A separate structured-state path describes sufficient
statistics `(sum, count)` and a readout, so a complete vector RMSNorm or a segmented sum / mean
need not pretend to have a second scalar reduction. Kernel source follows from these
specifications; the **launch configuration**, not the operator implementation, may come from
GPU-specific measurements.

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
and (4). This repository adds recognition, code generation, transactional rewriting, certified
chunk merging, structured-state backends and an AOTInductor integration path. The structured
extension is this project's engineering work, not a claim about code shipped by the paper.

[docs/design.md](docs/design.md), [docs/roadmap.md](docs/roadmap.md) and the older logs in
[docs/evidence/](docs/evidence/) describe earlier iterations. This README describes the current
surface; historical statements such as "split not built" or "RMSNorm is out of frame" do not
apply to all current paths. See [docs/paper.md](docs/paper.md) for the paper's own findings and
[docs/paper-notes.zh.md](docs/paper-notes.zh.md) for Chinese reading notes.

## Status

A tested research prototype, **not a production library**. Capabilities are guarded by tests;
performance numbers are observations on the named hardware, inputs and execution path, not
universal guarantees. Verification snapshot: **2026-11-13**.

| Area | Current scope |
|---|---|
| Legacy algebra family study | 28 families: 20 `EXACT`, 1 `AFFINE`, 5 `NEEDS_STATE`, 2 `NOT_INVERT` |
| Legacy codegen coverage | **26 / 28** emitted; this study is separate from the three structured specifications |
| Scalar registry | **11** supported skeletons, algebra-based discovery without per-skeleton pattern rows |
| Structured registry | **3** immutable specifications: `rms_norm`, `segment_sum`, `segment_mean`; exact ATen stage / access-contract recognition |
| Local graph rewriting | Multiple eligible regions; transactional graph, signature and module-call-graph publication |
| Execution plans | `serial` by default; explicit experimental `split` for nine single-pass scalar skeletons and full RMSNorm |
| Dynamic shapes | Scalar serial supports symbolic row width; RMSNorm requires static feature width; split requires concrete width and bounded scratch |
| Dtypes / rounding | fp32, bf16, fp16 kernels; scalar / RMS arithmetic uses fp32, while segment pooling preserves ordered storage-dtype rounding |
| Performance selection | `compile_best()` compares the original baseline with eligible serial / split candidates over every supplied sample |
| Recommendation-model evidence | Six segment regions in real `emb_preprocess`: **1.92x on RTX 4090 D / 1.83x on RTX 5090 D v2**, whole-node native timing |
| Full-model integration | **Not established**: node experiments do not prove whole-DAG / Infer_server performance, precision or deployment readiness |
| `torch.compile` backend | Still not functional; experimental gate is not a supported backend |

### Opt-in recommendation-model capabilities

The default `capabilities=()` keeps the scalar path unchanged. Enable only the domains you want:

```python
matches, reason = neptune_opt.preview(
    ep, capabilities=("rms_norm", "segment_pool")
)
print(reason)  # metadata-only analysis; no kernel generation or CUDA launch

report = neptune_opt.apply(
    ep,
    capabilities=("rms_norm", "segment_pool"),
    verify_with=(model, args),
    verification_inputs=additional_samples,  # a sequence of argument tuples
    rtol=1e-5,
    atol=1e-5,
)
print(report.summary())
```

The first RMSNorm backend serves a **complete canonical vector output**, with this exact order:
input promotion to fp32 → square / mean / epsilon / rsqrt / normalization in fp32 → round back
to the input dtype → multiply by a same-dtype 1D `gamma`. For example:

```python
class RMSBlock(torch.nn.Module):
    def forward(self, x, gamma):
        value = x.float()
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * gamma
```

`gamma` may be a readonly tensor input or parameter. Epsilon must be explicitly finite and
positive. Serial resident width is at most **32768**; the explicit split backend supports at
most **4096 partials × 1024 elements**. Batch dimensions may remain symbolic. This is not a
promise to replace every `nn.RMSNorm` / `aten.rms_norm` spelling or a different rounding order.

Segment pooling recognizes `torch.segment_reduce(data, "sum" / "mean", lengths=...)` or
`offsets=...`, **axis 0**, with a readonly 1D int32 / int64 descriptor. It supports `initial=None`
or a finite representable scalar, including the source's empty-segment behavior. Lengths use
traceable prefix sums and coverage checks; offsets use bounds and monotonicity checks.
`unsafe=True` is **not** a proof that descriptors are valid. The backend preserves ordered
per-token storage-dtype rounding; **segment split / tree reassociation is refused**. Its empty
mean with `initial=None` returns NaN rather than silently clamping it to zero.

Structured specifications have exact-real identity / associativity certificates, but
`ieee_associative=False`: a certificate does not authorize arbitrary floating-point
reassociation or an arbitrary monoid backend. These first backends return fresh contiguous
outputs and refuse incompatible output-layout metadata. Supplied-case value / readonly-state
verification does not prove output aliasing or every internal constant in a native runner.

## Benchmarks: AOTI's own lowering vs neptune

**Baseline means Inductor-generated code for the unmodified graph, not eager PyTorch.** The
current A/B path exports that baseline once and clones it independently for each candidate;
it does not globally decompose the baseline just to make Neptune easier to match.

### Real recommendation node — native AOTI, 2026-11-12

Model package `1794139980`, **`emb_preprocess` only**, BF16, six `segment_mean` regions with
`capabilities=("segment_pool",)` and `plan="serial"`:

| GPU | Baseline ms/call | Neptune ms/call | Speedup | A/B choice | Source |
|---|---|---|---|---|---|
| RTX 4090 D | 0.67208 | 0.34976 | **1.92x** | candidate; no critical per-case regression | M4090 |
| RTX 5090 D v2 | 0.46525 | 0.25383 | **1.83x** | candidate; no critical per-case regression | M5090 |

Both model JSONs record PyTorch **2.11.0+cu128** and CUDA runtime 12.8; Triton **3.6.0** is
recorded in the corresponding GPU's native sweep (S4090 / S5090). The model runs compile `model.so`
and load it with `torch._C._aoti.AOTIModelContainerRunnerCuda` through its Python binding.
There is **no Python Neptune operator callback** in the compiled execution path. Timings include
native binding / host overhead, allocations and descriptor validation: 20 warmup calls per case
and variant, **50 calls × 5 interleaved rounds**, then the equally weighted mean of case medians.
This is not a production request-weighted latency or a kernel-only number.

**Each GPU validated all 24 captured real samples and 4 shape probes** against the baseline / source
with `rtol=atol=1e-5`, without ULP slack. Input values / pointers and original exported / module
state were checked. This is supplied-case numerical agreement, **not a bitwise or full-domain
proof**. Full-DAG performance / precision, cold-move deployment and native output aliasing were
not validated; temporary native artifacts were not retained.

Preview also found **17 RMSNorm regions in `dense`** (3 feature-tower regions, 12 backbone
RMSNorm regions across 6 blocks, and 2 task-tower regions). That is coverage evidence only:
there is no measured dense-node or whole-model speedup attached to that count.

### Structured operators: wins, losses and fallback

The following are **synthetic operators**, not additional real-model results. Native timings
include host / allocation / descriptor work; 200 calls × 5 rounds, 100 warmup calls per sample
and variant. The full sweeps cover 28 workloads / 88 supplied samples per GPU.

| Workload | GPU | Baseline → candidate ms/call | Candidate speedup | Selection | Source |
|---|---|---|---|---|---|
| Full RMSNorm `(1, 65536)`, fp32 | 4090 D | 0.01676 → 0.01424 | **1.18x** | split; serial exceeds resident width | S4090 |
| Full RMSNorm `(1, 65536)`, fp32 | 5090 D v2 | 0.01262 → 0.01025 | **1.23x** | split; serial exceeds resident width | S5090 |
| Full RMSNorm `(1125, 512)`, bf16 | 4090 D | 0.01012 → 0.01077 | **0.94x** | baseline; serial and split both lose | F4090 |
| Full RMSNorm `(1125, 512)`, bf16 | 5090 D v2 | 0.00855 → 0.00910 | **0.94x** | baseline; serial and split both lose | S5090 |
| Segment mean `(4096, 64)`, 64 segments, bf16 | 4090 D | 0.13123 → 0.09351 | **1.40x** | serial | F4090 |
| Segment mean `(4096, 64)`, 64 segments, bf16 | 5090 D v2 | 0.10470 → 0.10185 | **1.03x** | baseline: ragged-tail cases regress | S5090 |

In both sweeps, ten of twelve RMSNorm workloads kept the baseline; only the two `(1, 65536)`
FP32 / BF16 workloads selected split. All sixteen segment workloads selected serial on the
4090 D, but only eight did on the 5090 D v2. Some 5090 mean wins were still rejected because
empty / long-tail samples regressed. **Mean speedup alone is not sufficient.** Segment serial
checks in these runs recorded zero maximum absolute error; RMSNorm is not universally exact,
and some synthetic BF16 cases explicitly allowed 1 ULP. That synthetic policy was **not** used
for the real-model node above.

The implemented scalar chunk-merge path also changes the old "do not build split" conclusion:
for synthetic FP32 `logsumexp` at `(1, 262144)`, the forced explicit split measured
**0.01876 → 0.01599 ms (1.17x)** on 4090 D and **0.01517 → 0.01258 ms (1.21x)** on 5090 D v2
(A4090 / A5090): 100 warmup calls per sample / variant, **500 calls × 7 rounds**, and an equally
weighted mean of sample medians. These are actual native compiled variants, not the historical
emulated split ceiling or records of `compile_best` selecting a production artifact.

**Provenance of the new tables.** These JSON files are existing workspace evidence **outside
this Git checkout**; they are not included by `git clone`. Paths below are relative to this
repository directory, not portable GitHub download links. Each run records source fingerprints,
input / sample identity, correctness checks and round timings. The tables above are the summary;
reproduction also requires the trusted model package and its capture dependencies.

* M4090: `../perf_1794139980/neptune_eval/structured/results/model_emb_unique_verified_4090d/native_results.json`
* M5090: `../perf_1794139980/neptune_eval/structured/results/model_emb_5090d/native_results.json`
* S4090: `../perf_1794139980/neptune_eval/structured/results/native_sweep_4090d/native_results.json`
* S5090: `../perf_1794139980/neptune_eval/structured/results/native_sweep_5090d/native_results.json`
* F4090: `../perf_1794139980/neptune_eval/structured/results/native_final_4090d/native_results.json`
* Preview: `../perf_1794139980/neptune_eval/structured/results/model_preview.json`
* A4090: `../perf_1794139980/neptune_eval/enhanced/results/native_4090d/native_results.json`
* A5090: `../perf_1794139980/neptune_eval/enhanced/results/native_5090d.json`

### Historical `.pt2` measurements — do not mix timing protocols

The following older RTX 5090 D v2 tables remain useful evidence, not current multi-plan or
real-recommendation-model results. Both sides used
`torch.export.export(model, args).run_decompositions()`, then packaged and loaded `.pt2` files.
Current `compile_best` deliberately does **not** impose that decomposition on its baseline.

The exp33 / exp36 fused packages recorded 41 occurrences of the generated kernel name and zero
in their baselines; those counts belong to those runs, not every package format. Launches were
pipelined to reduce Python dispatch / per-call launch overhead. These numbers must not be
combined with the synchronized native wall-time tables above.

#### Scalar fusion / experimental log-softmax — 256×16384

| Model | AOTI plain | Neptune | Speedup | Error (metric) | Log |
|---|---|---|---|---|---|
| `torch.logsumexp`, fp32 | 0.0281 ms | 0.0183 ms | **1.54x** | 9.4e-08 relative | [exp33](docs/evidence/exp33_real_modules_gpu.txt) |
| `logsumexp(t/8)`, fp32 | 0.0286 ms | 0.0194 ms | **1.47x** | 9.8e-08 relative | [exp33](docs/evidence/exp33_real_modules_gpu.txt) |
| RMSNorm + row max, fp32 | 0.0277 ms | 0.0182 ms | **1.52x** | 4.5e-07 relative | [exp33](docs/evidence/exp33_real_modules_gpu.txt) |
| Stable L2 norm, fp32 | 0.0288 ms | 0.0184 ms | **1.57x** | 9.5e-07 relative | [exp33](docs/evidence/exp33_real_modules_gpu.txt) |
| Log-softmax by hand, fp32 | 0.0311 ms | 0.0261 ms | 1.19x | 2.4e-07 relative | [exp33](docs/evidence/exp33_real_modules_gpu.txt) |
| `torch.log_softmax` / `nn.LogSoftmax`, fp32 | 0.0317 ms | 0.0271 ms | 1.17x | 1.9e-06 max absolute | [exp36](docs/evidence/exp36_i4_log_softmax_gpu.txt) |
| `F.log_softmax`, fp32 | 0.0318 ms | 0.0285 ms | 1.11x | 1.9e-06 max absolute | [exp36](docs/evidence/exp36_i4_log_softmax_gpu.txt) |
| `nn.LogSoftmax`, fp16 | 0.0305 ms | 0.0241 ms | 1.27x | 7.8e-03 max absolute | [exp36](docs/evidence/exp36_i4_log_softmax_gpu.txt) |
| `F.log_softmax`, fp16 | 0.0330 ms | 0.0232 ms | 1.42x | 7.8e-03 max absolute | [exp36](docs/evidence/exp36_i4_log_softmax_gpu.txt) |

The current default **declines log-softmax lowering**. `apply(...,
experimental_log_softmax=True, verify_with=...)` opts into its `x - LSE` readout; cancellation
at large offsets and intermediate rounding mean the historical 1.11–1.42x rows are not an
all-input correctness guarantee. The old fp16 bound was four ULPs of the largest magnitude
present (about 0.06), not the current opt-in per-value ULP-distance check.

#### Variance's two-traversal serial plan, fp32

| Shape | AOTI plain | Neptune `variance` | Speedup | Row reads | `stable_l2` at the same shape (different operator) |
|---|---|---|---|---|---|
| `(8, 4096)` | 0.0347 ms | 0.0228 ms | **1.52x** | 2 | 0.0229 ms / 1.52x |
| `(8, 16384)` | 0.0563 ms | 0.0272 ms | **2.07x** | 2 | 0.0269 ms / 2.09x |
| `(8, 65536)` | 0.0571 ms | 0.0586 ms | 0.97x | 2 | 0.0487 ms / 1.17x |
| `(32, 65536)` | 0.0551 ms | 0.0591 ms | 0.93x | 2 | 0.0489 ms / 1.13x |
| `(1, 262144)` | 0.0516 ms | 0.1715 ms | **0.30x** | 2 | 0.1332 ms / 0.39x |

[exp31](docs/evidence/exp31_two_pass_gpu.txt) uses one program per row and two traversals for
variance. `stable_l2 = sqrt(sum(x²))` is a **different operator**, shown for a same-shape
comparison, not an alternative way to compute variance. Variance / std still have no supported
split plan in this implementation.

#### Launch configuration: two already-fused packages, `(1, 262144)`

| Configuration | Latency | Relative to forced default |
|---|---|---|
| Neptune fused, forced `(BLOCK=1024, warps=4)` | 0.1315 ms | 1.00x |
| Same fusion, measured `(8192, 8)` | 0.0481 ms | **2.74x** |

[exp37](docs/evidence/exp37_autotune_gpu.txt) compares **two fused packages**, not an unmodified
Inductor baseline. The corresponding historical 4090 D result is 2.57x over its forced default
([exp38](docs/evidence/exp38_cross_gpu_and_old_stack.md)); neither number is a native-baseline
speedup. GPU-keyed launch evidence must not silently transfer to another GPU.

#### Historical losses still explain the defaults

* Softmax via an LSE expansion: **0.59x**; LayerNorm via mean + variance: **0.41x**. Dedicated
  Inductor lowerings are left alone, not decomposed by default ([roadmap §I4](docs/roadmap.md)).
* Small scalar shapes such as 64×4096 were **0.9–1.05x** in
  [exp33](docs/evidence/exp33_real_modules.md), despite larger-shape wins.
* The tuned single-program `(1, 262144)` kernel and **emulated** split ceiling in exp37 lost to
  Inductor. The later actual chunk-merge results above supersede the old "split not built"
  capability statement, not the measurements recorded in that log.
* Historical `compile_best` LSE was **1.41x on 5090 / 0.96x on 4090 D**, with the latter keeping
  the baseline ([4090](docs/evidence/exp39_ab_fallback_4090d.txt),
  [5090](docs/evidence/exp39_ab_fallback_5090.txt)). A win is GPU- and input-dependent.

## Install

```bash
git clone https://github.com/Defyli/neptune-opt.git
cd neptune-opt
pip install -e ".[test]"     # CPU analysis and verification tests
# On a compatible CUDA / PyTorch stack, additionally:
pip install -e ".[gpu]"
```

Declared package requirements are Python >=3.10, PyTorch >=2.4 and sympy >=1.12. Those floors
are **not** a claim that every GPU API exists on every version. Real custom-op integration needs
`torch.library.triton_op` / `wrap_triton` (introduced in 2.6), and packaging needs the relevant
AOTInductor APIs. The complete GPU paths here were validated on **2.11.0+cu128 / Triton 3.6.0**
on RTX 4090 D and RTX 5090 D v2. Triton is needed to compile / execute kernels, not to emit source
text, inspect algebra or run metadata-only `preview`.

**CPU verification snapshot (2026-11-13, completed 06:15 UTC):** the local Python 3.12.11 /
PyTorch 2.11.0+cu128 acceptance, with CUDA disabled, Triton blocked and pytest caching disabled,
passed **562 tests / 1 skipped**, including the four new README contract checks. The skip is the
existing GPU segment-correctness test. Codegen coverage remained 26 emitted / 20 `EXACT`.
CI retains Python 3.11 with official CPU wheels **2.4.1 / 2.6.0 / latest**. In the last inspected
hosted run (`a74af56`), the two fixed versions passed and `latest` failed at tests. The local
`ContextVar` config-fixture compatibility fix needs an official rerun; official CPU `latest`
was not fully reproduced locally because downloads / dependencies were unavailable. Check the
badge / run for current status rather than treating the matrix definition as a green result.

The old V100 / PyTorch 2.0.1 experiment in [exp38](docs/evidence/exp38_cross_gpu_and_old_stack.md)
is historical portability evidence **below the supported package floor**, not validation of
today's complete integration on that stack.

## Entry points

| Form | Purpose | Model edits |
|---|---|---|
| `apply(ep, ...)` | Transactional local rewrites, optional supplied-case numerical checks | None |
| `preview(ep, ...)` | Metadata-only `(matches, reason)`; does not compile or mutate the original | None |
| `export(model, args, decompose=True, capabilities=...)` | Export + apply; core decomposition is explicit and optional | None |
| `neptune_opt.rmsnorm_max(x, eps=...)` | Explicit scalar operation, not complete RMSNorm | Yes |
| `compile_best(model, args, ...)` | Compile the original baseline and select among eligible candidates | None |
| `neptune_codegen.as_state_op(...)` | Explicit structured custom-op bridge | Yes |
| `neptune_opt.backend` | Experimental, not a functional supported compiler backend | — |

`export` keeps `decompose=True` for compatibility; `decompose=False` avoids global decomposition.
Its extra kwargs go to **PyTorch export**, not to every `apply` option. For plans, tolerances and
additional samples, call `apply` directly or use `compile_best` with its documented parameters.
The older scalar API reference is [docs/api.md](docs/api.md); current signatures are in
`neptune_opt/__init__.py` and `neptune_opt/ab.py`.

### If the fusion is slower

`apply` checks legality and optionally supplied-case correctness, **not speed**. Use the A/B gate:

```python
choice = neptune_opt.compile_best(
    model,
    args,
    capabilities=("rms_norm", "segment_pool"),
    plans=("serial", "split"),
    verification_inputs=additional_samples,
    margin=0.03,
    regression_margin=0.03,
    rtol=1e-5,
    atol=1e-5,
)
print(choice)
print(choice.selected_plan, choice.sample_metrics)
torch._inductor.aoti_load_package(choice.package_path)
```

The original baseline is compiled once. Each requested candidate is isolated, verified and
compiled only if eligible; requesting both plans costs up to three compilations, not always two.
A graph containing segment pooling will reject its split candidate; the serial candidate and
baseline remain independent. Default plans are `("serial",)`.

Selection uses an equally weighted sample mean, a minimum win margin and per-input-key
regression checks. Segment pooling also checks **every supplied sample** independently, because
ragged descriptor values can differ despite identical metadata keys. No supplied critical
regression is hidden by an average win. CUDA is required for actual timing. Pass PyTorch export
`dynamic_shapes` if needed, but validation / performance coverage remains the supplied cases,
not the entire shape range. Baseline export / compilation errors raise; ordinary candidate
failures fall back. Fatal CUDA failures propagate. `verify=False` explicitly opts out and must
not be described as a verified result.

`max_ulps=None` is the default. An explicit FP16 / BF16 `max_ulps` permits representable-value
ULP-distance slack only after the normal tolerance check fails; it does not relax finite / NaN /
infinity-sign classification. Signed zero compares numerically equal, so this is not a bitwise
check. Actual package comparisons in `compile_best` still reject nonfinite tensor outputs.

Structured AOTI packaging uses scoped standard options: `triton.unique_user_kernel_names=True`
(to keep specialized user cubins distinct) and, when available,
`triton.autotune_at_compile_time=False` (to avoid tuning with fabricated integer intermediates).
`compile_best` applies these to both sides and restores them on success or failure. If packaging
an `apply` result yourself, use the same scoped options on a supported stack; unique-user-name
support is required for structured packages. Packaging may still execute / tune using real export
inputs; it is not a no-execution metadata operation.

### The explicit operator library

```python
import neptune_codegen

neptune_opt.op_parameters("rmsnorm_max")  # ('eps',)
y = neptune_opt.rmsnorm_max(x, eps=1e-6)

rms_op = neptune_codegen.as_state_op("rms_norm", {"eps": 1e-6})
y = rms_op(x, gamma)
pooled = neptune_codegen.run_segment_pool(data, lengths, reduce="mean", initial=0.0)
```

The scalar wrappers and structured bridges use ordinary `torch.library` custom ops / Triton
integration. Explicit naming chooses an implementation; it does not waive its dtype, shape,
resource, rounding or descriptor contract.

## What it fuses

### Scalar skeletons

| Skeleton | Reduction / readout |
|---|---|
| `logsumexp` | Signed max + sum-of-exp LSE fold, not an entire softmax / attention chain |
| `stable_l2` | Overflow-aware scaled L2 norm |
| `rmsnorm_max` | RMS normalization + **row max**, not the complete vector output |
| `softcap_lse` | `logsumexp(tanh(beta*x))` |
| `lse_biased` | `logsumexp(x + alpha)` |
| `lse_scaled` | `logsumexp(x / T)`; scalar temperature inferred from eligible graphs |
| `linear_attn` | `sum(x) / max(abs(x))` normalizer, not full attention |
| `decay_renorm` | `sum(x / (N + max(abs(x))))` |
| `sumsq_div` | `sum(x²) / max(abs(x))` |
| `variance` | Population variance; two-traversal serial plan |
| `std` | Population standard deviation; two-traversal serial plan |

### Structured specifications

| Specification | Capability | Access / output |
|---|---|---|
| `rms_norm` | `rms_norm` | Last-dimension statistics + complete canonical vector map with `gamma` |
| `segment_sum` | `segment_pool` | Axis-0 runtime lengths / offsets; per-channel sum |
| `segment_mean` | `segment_pool` | Axis-0 runtime lengths / offsets; per-channel mean and source empty policy |

Direct scalar kernels accept fp32 / bf16 / fp16, but scalar **graph rewriting** does not erase
low-precision intermediate roundings. Such graphs need eligible explicit fp32 stages / compound
lowering; arbitrary casts are boundaries, not symbolic identities. Structured recognition checks
its own ordered stages and two-input ABI separately.

## Known limits

[`tests/test_readme_limits.py`](tests/test_readme_limits.py) guards the nine historical keys below
and their explicit-mode regressions; it also checks enhanced default behavior. Key inventory is
machine-checked in both directions, **not an automatic proof of every prose claim or benchmark**.
Historical key names are retained so that default and opt-in bounded / disabled modes are clear.

| Key | Current meaning / mode |
|---|---|
| **`region-is-the-operator`** | Only the exact registered region is replaced. `logsumexp(x) * 2` keeps its `* 2`; a differently written unregistered readout is refused rather than guessed from sampling. |
| **`one-operator-per-graph`** | Only the explicit `max_regions=1` mode stops at one fusion. Default `max_regions=None` can fuse multiple eligible local regions. |
| **`row-reductions-only`** | Scalar and RMSNorm row reductions require the last dimension; `mean(0)` is not reinterpreted. Opt-in segment pooling has a separate axis-0 access contract. |
| **`var-default-is-unbiased`** | Default `var` / `std` use `/(N-1)` and are refused by population-variance skeletons; `unbiased=False` is eligible when the other contracts hold. |
| **`single-reduction-operators`** | The default scalar frame does not fuse elementwise RMSNorm / LayerNorm / softmax consumers. Complete canonical RMSNorm is separately opt-in via `capabilities=("rms_norm",)`; LayerNorm and full softmax remain unsupported. |
| **`softmax-internals-not-lowered`** | Softmax / LayerNorm / SDPA are left alone. Log-softmax lowering is disabled by default; only `experimental_log_softmax=True` opts into the cancellation-prone readout with supplied-case verification. |
| **`needs-decomposition`** | Eligible fresh LSE is selectively lowered by default, without core decomposition. Explicit `lower=False` leaves a compound LSE unfused with a no-fusion report; global decomposition is not required by default. |
| **`prod-ratio-needs-a-count`** | `prod + ratio` remains one of two refused families in the legacy 28-family study, with `f-needs-carried-state`; structured additive state does not implement this product monoid. |
| **`lower-only-when-it-pays`** | Targeted lowering is a no-op when it enables no eligible region. “Pays” means structural eligibility, not proven latency: `compile_best` owns the measured speed gate. |

Additional boundaries, covered by the implementation / relevant regression tests rather than
the historical-key inventory:

* **Inference only:** no gradient / training integration. Source state or input mutation is refused.
* **Floating-point scope:** real-algebra certificates have assumptions about finite reachable
  arithmetic. Supplied-case checks do not prove universal IEEE equivalence. Business clamps,
  nonfinite classification and required dtype conversions must not be silently removed.
* **Resources / layouts:** scalar split needs static width and at most 4096 partials; full RMSNorm
  has the bounds above. Incompatible structured output layout is refused; required input copies
  and scratch are included in execution, not hidden from latency measurements.
* **Dtypes:** fp8, integer / complex data and mixed-dtype structured inputs are outside these
  backends. Segment descriptors / counts are integer; that is not integer data reduction support.
* **No arbitrary domain expansion:** no arbitrary monoid backend, gather embedding-bag fusion,
  complete LayerNorm, attention-chain fusion (`QKᵀ → softmax → PV`) or multi-GPU execution.
* **Safety is not speed:** readonly inputs / state and numerical checks do not establish all
  output aliasing, all native internal constants, whole-DAG behavior or cold-move deployment.

## Repository layout

```text
neptune_codegen/     scalar repair + chunk merge + structured state -> Triton source
  skeleton.py          immutable specifications and registries
  algebra.py           repair / merge / structured certificates and domain assumptions
  emitters.py          derived helpers and RMSNorm source
  template.py          execution plans, resource / layout contracts, source ownership
  torch_op.py          custom-op bridges and ordered segment backend
  autotune.py          GPU-keyed launch evidence with conservative fallback
  launch_table.py      measured launch configurations
neptune_recognize/   local graph discovery, exact structured stages, targeted lowering
  rewrite.py           candidate-only rewrites and metadata contracts
neptune_opt/         apply / preview / export / operator library
  ab.py                isolated baseline, multi-plan / multi-sample performance selection
repairlib/           legacy algebra classifier (28 families, CPU + sympy)
tests/               CPU safety / numeric replay / documentation checks; optional GPU check
examples/            sample models
experiments/         GPU / native AOTI benchmark scripts
tools/               codegen coverage report
docs/                historical design, paper notes and older committed evidence
```

## Reproducing the evidence

CPU checks need the test extra only. **All native commands below additionally require** CUDA,
the compatible AOTI stack and `infer_compiler` with its precision / compilation dependencies;
they are not standalone tests of the base package. Install those dependencies first. In the
benchmark workspace layout, make the sibling checkout importable for all native modes with
`export PYTHONPATH="$PWD:../infer_compiler/src${PYTHONPATH:+:$PYTHONPATH}"`.

```bash
python -m pytest tests/ -q -ra
python tools/coverage_codegen.py --check
python examples/01_quickstart.py

# Current synthetic scalar native comparison; needs CUDA + the AOTI stack:
python experiments/gpu_m23_ab_fallback.py --native --calls 500 --repeats 7 \
  --output-dir results/native_current

# Current full RMSNorm / segment pooling sweep:
python experiments/gpu_m23_ab_fallback.py --native --structured --calls 200 --repeats 5 \
  --output-dir results/structured_current
```

The real-node experiment also requires a **trusted** parameter package with its captured
samples; neither the package nor the external result JSONs are shipped in this repository.
From the benchmark workspace layout:

```bash
PYTHONPATH="$PWD:../infer_compiler/src${PYTHONPATH:+:$PYTHONPATH}" \
python experiments/gpu_m23_ab_fallback.py --native --model-package ../1794139980 \
  --node emb_preprocess --model-precision bfloat16 --capabilities segment_pool --plan serial \
  --calls 50 --repeats 5 --output-dir results/model_emb_current
```

Run independently on each target GPU; do not reuse another GPU's launch evidence. Native
experiments remove temporary artifacts by default; `--keep-artifacts` retains them for separate
cold-move inspection but does not itself validate that deployment.

The old scripts (`real_modules.py`, `gpu_m17_two_pass.py`, `gpu_m21_i4_log_softmax.py`,
`gpu_m22_autotune.py`) document historical experiments. Re-running them on current code / hardware
may produce different fusion decisions or timings; keep the old raw logs as observations,
not promises that the current default reproduces an old policy. Use pytest for the current
suite: standalone `python tests/test_*.py` runners do not cover today's fixtures and parametrization.

## License

MIT — see [LICENSE](LICENSE). The paper this work builds on is cited above and is not covered by
this license.
