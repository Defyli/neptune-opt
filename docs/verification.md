# Verification map

Every claim this repository makes, the script that decides it, and the last recorded result. The
point of the file is that a reader never has to take a sentence on trust: a claim without a script
is deleted, and a claim whose script has no committed log says so.

Conventions:

* **CPU** scripts run on every push in CI (torch 2.4.1 / 2.6.0 / latest) and need no GPU and no
  Triton. **GPU** scripts need the GPU host and exit non-zero on failure, like the CPU ones.
* `log` names the raw output committed under `docs/evidence/`. A dash means the result is recorded
  in the script's own printed summary and in `docs/roadmap.md`, not as a committed log — rerun the
  script to reproduce it.

## The algebra and the code generator

| claim | script | last result | log |
|---|---|---|---|
| the generator emits a kernel for every registered skeleton, and the algebra's verdicts are right | `experiments/gpu_m10_codegen_tests.py` | 542 checks, 0 failed | — |
| the operators have no repair are served by the two-pass plan (variance, std) | `experiments/gpu_m17_two_pass.py` | 129 checks, 0 failed | `exp31_two_pass_gpu.txt` |
| a shift is absorbed by an affine repair, and a bias is admissible only for a max consumer | `experiments/gpu_m16_affine.py` | 145 checks, 0 failed | `exp30_affine_gpu.txt` |
| coverage over the probed operator families, with floors enforced in CI | `tools/coverage_codegen.py --check` | 26 of 28 emitted, 20 `EXACT`, 1 `AFFINE`, 5 `NEEDS_STATE` | — |

## Recognition

| claim | script | last result | log |
|---|---|---|---|
| recognition is derived from the registry by comparing expressions, and refuses what the algebra refuses (`var`/`std` defaults, the `g`-ignores-producer control) | `tests/test_recognize.py` | 20 cases passed | — |
| the interpreter and the generator agree on every registered skeleton | `experiments/gpu_m14_recognition.py` | exit 0 | `exp32_recognition_gpu.txt` |
| real modules are recognised only after the core decompositions, and the refusal names the reason | `experiments/real_modules.py` | see the table in the log | `exp33_real_modules_gpu.txt` |
| the lowering exposes the log-sum-exp inside `log_softmax`, which the core decompositions leave whole | `experiments/gpu_m21_i4_log_softmax.py` §1 | 37 checks, 0 failed | `exp36_i4_log_softmax_gpu.txt` |

## Fusion, correctness and speed

| claim | script | last result | log |
|---|---|---|---|
| the safety gates (region / verify / failure) never fire wrongly, and the API surface is stable | `tests/test_neptune_opt.py` | 35 tests passed | — |
| the lowering is a no-op unless it enables a rewrite | `tests/test_lowering.py` | 11 tests passed | — |
| the public API and `docs/api.md` agree in **both** directions | `tests/test_api_docs.py` | 5 tests passed | — |
| one compiled kernel is correct at several row lengths from a symbolic export | `experiments/gpu_m18_dynamic_shapes.py` | 36 checks, exit 0 | `exp34_dynamic_shapes.txt` |
| fp32 / bf16 / fp16 take the input's dtype, accumulate in fp32, and stay inside a dtype-aware bound | `experiments/gpu_m19_dtypes.py` | 134 checks, exit 0 | `exp35_dtypes.txt` |
| end to end through AOTI, the generated kernel is in the package and the numbers are unchanged | `experiments/real_modules.py` | 1.19–1.57x on the in-frame rows | `exp33_real_modules_gpu.txt` |
| `log_softmax` in fp32 and fp16 fuses at 1.11–1.42x, and the excluded operators are byte-identical | `experiments/gpu_m21_i4_log_softmax.py` | 37 checks, exit 0 | `exp36_i4_log_softmax_gpu.txt` |
| the launch configuration is measured per shape, kept at the default unless it clears the noise margin, and baked into the package's op name | `experiments/gpu_m22_autotune.py` | 40 checks, exit 0; 2.74x end to end at `(1, 262144)` | `exp37_autotune_gpu.txt` |
| the README's limits table is true, and complete in both directions | `tests/test_readme_limits.py` | 3 tests passed (9 limit rows) | — |

## Integration placement

| claim | script | last result | log |
|---|---|---|---|
| rewriting the `ExportedProgram` reaches codegen, while every Inductor custom-pass hook does not | `gpu_diag_paths.py` (kept out of the repository) | 41 occurrences of the generated kernel vs 0 | — |
| the AOTI route is the documented one; the `torch.compile` backend does not reach codegen | `experiments/gpu_m13_aoti_codegen.py`, `experiments/gpu_m15_integration.py` | exit 0 | — |

## Decisions taken from measurement

| decision | evidence | where it is recorded |
|---|---|---|
| do **not** lower `softmax` (0.59x) or `layer_norm` (0.41x) | `experiments/gpu_m20_i4_ceiling.py` | `docs/roadmap.md` §I4, and pinned by `gpu_m21` §3 |
| do **not** build split reduction: its ceiling (0.0423 ms) does not beat the tuned single-program kernel (0.0411 ms), and Inductor's own is 0.0275 ms | `experiments/gpu_m22_autotune.py` §4 | `docs/roadmap.md` §I6 |
| attention-chain fusion and multi-GPU are architectural non-goals | architecture, not algebra | `docs/roadmap.md`, README |
