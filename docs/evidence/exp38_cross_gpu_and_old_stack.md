# Cross-GPU and old-stack verification

Two further machines, tested because the README's portability claim was "verified on one stack" and
the generated launch table says "regenerate on a different GPU". Both statements are now measured.

| machine | GPU | driver / CUDA | python | torch | triton |
|---|---|---|---|---|---|
| A | RTX 5090 D v2 | 12.8 | 3.12.11 | 2.11.0+cu128 | 3.6.0 |
| B | RTX 4090 D | 565.57.01 / 12.7 | 3.12.11 | 2.11.0+cu128 | 3.6.0 |
| C | Tesla V100-PCIE-32GB | 470.82.01 / 11.8 | 3.9.16 | **2.0.1+cu117** | **2.0.0** |

Both hosts are internal and reachable only through the gateway used for the earlier runs; the raw
logs are not committed (they carry host names and absolute paths), and the numbers below are the
runs' own summaries.

## Machine B (4090D): same stack, different GPU — `gpu_m22_autotune.py`

**40 checks, exit 0**, then after the robustness fix below **37 checks, exit 0** (fewer, because each
deviation adds a check and the fix removed three).

| | 5090 (A) | 4090D (B) |
|---|---|---|
| deviations from the default chosen | 4 | **1** |
| the deviations | `('single',262144)→(8192,8)`, `('many',262144)→(8192,8)`, `('many',1024)→(512,2)`, `('few',4096)→(512,2)` | `('single',262144)→(8192,8)` |
| a single row of 262144 | 3.04x | **3.22x** |
| end to end through AOTI at that shape | 2.74x (0.1315 → 0.0481 ms) | **2.57x** (0.1439 → 0.0560 ms) |
| Inductor's own lowering at that shape, relative to the tuned kernel | 1.00/0.67 | 1.00/0.80 |
| split reduction worth building? | no (0.0423 vs 0.0411 ms) | no (0.0626 vs 0.0468 ms) |

**What transfers and what does not.** The rule transfers exactly: one program for a long row wants
`BLOCK=8192, num_warps=8`, it is worth ~3x, and split reduction is not worth building on either GPU.
The TABLE does not transfer — four deviations on one card, one on the other — which is why
`launch_table.py` is a generated file carrying its machine in the header rather than a constant
somebody edits.

**A correction to the first report of this run.** I first wrote that a single timing stall
(`0.1358 ms` where the same configuration measured `0.0142 ms`) was reported as a 2.89x win, and
attributed it to `(1, 4096)`. That attribution was WRONG, and it came from reading grep output
line-by-line against the shape list: the grep had dropped one line, so every shape after the gap was
shifted by one. `0.1358 ms` is the *legitimate* cost of the default at `(1, 262144)`, where the
candidate that wins really is `(8192, 8)` and really is ~3x better. The correction is recorded here
rather than quietly edited out, because a mis-read measurement is exactly what this file exists to
catch.

**What the first run did show, and what was fixed.** With a single timing round, the *default itself*
was sometimes measured favourably and sometimes not, so candidates were compared against a noisy
baseline: that run chose four deviations on the 4090 D, including `(1, 256) -> (8192, 8)`, a shape
where the entire spread across candidates is ~2%. `autotune.search` now times every candidate in
three INTERLEAVED rounds and keeps the minimum, which removes the baseline's variance rather than the
candidate's luck. Re-measured, the same machine chose one deviation. The mechanism claimed is right;
the number I first attached to it was not.

**One thing is NOT verified.** `experiments/gpu_m22_autotune.py` was refactored to share
`autotune.measure_shapes` with `measure_launch_configs.py` (the two §1/§2 blocks). The shared code is
verified: it ran on both the 4090 D and the V100 through `measure_launch_configs.py`, and 62 CPU
tests pass. The refactored *experiment* run died after §1 on the 4090 D with no traceback
(process gone, log ends mid-section) and the cause is unidentified, so its 37-check result belongs to
the pre-refactor version. Recorded as a follow-up rather than assumed to work.

## Machine C (V100): a genuinely old stack — what works, measured

`torch 2.0.1` predates the whole integration surface, and the honest result is a split, not a pass or
a failure:

| | result on torch 2.0.1 |
|---|---|
| `import neptune_opt` | **works** (20 public names) |
| `tests/test_autotune.py` (launch configuration) | **11 passed, 0 failed** |
| `tests/test_neptune_opt.py` (gates, report) | **27 passed, 2 failed** |
| the 2 failures | `AttributeError: module 'torch' has no attribute 'export'` |
| `tests/test_recognize.py` | fails at import: `from torch.export import export` |
| `tests/test_lowering.py`, `tests/test_readme_limits.py` | need `torch.export`; fail at the first export |
| `torch.export` | **absent** (it became public in 2.1; 2.0 has `torch._export`) |
| `torch.library.triton_op` | **absent** |
| `torch._inductor.aoti_compile_and_package` | **absent** |

So the algebra, the recognition engine, the launch-configuration layer and the safety gates work on
a four-release-old stack, and the AOTI integration does not exist there at all — which is what the
README says, now with a number. Two concrete follow-ups are recorded in the roadmap backlog rather
than papered over: the test suite should SKIP a stack without `torch.export` instead of crashing at
import, and `test_recognize.py` should build its graphs with `make_fx` (available on 2.0.1) so the
recognition half is testable there.

## Adding a machine is now a script, and the V100 used it

Measuring the launch configuration no longer requires the export/AOTI stack, so the old machine could
contribute its own numbers:

```
    $ python experiments/measure_launch_configs.py      # on each host, detached, then poll
    4090 D : 1 deviation  ('single',262144): (8192,8)          -> registry: rtx-4090-d, rtx-5090-d-v2
    V100   : 2 deviations ('many',262144): (4096,4), ('single',262144): (8192,8)
                                                              -> registry: + v100-pcie-32gb
    both runs: "other machines' entries were left alone"
```

That both hosts merged into the registry they were shipped (rather than overwriting it) is the
property the merge was written for, and it is asserted on CPU by
`tests/test_autotune.py::test_merging_one_machine_keeps_the_others`.

The V100's two deviations are a useful cross-check on the physics: 262144 columns is the case where
the block size matters, and on every GPU measured it is the long single row (and, on the
bandwidth-poorer V100, the long *many*-row shape too) that wants a bigger block.
