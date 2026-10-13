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

**A measurement flaw this run exposed, and the fix.** On machine B one candidate came back at
`0.1358 ms` where the next run measured the same configuration at `0.0142 ms`; the search reports the
fastest candidate, so that single stall was reported as a **2.89x win** and became a table entry. The
next section of the same run re-measured it at 1.01x, i.e. the entry was harmless but wrong. Stalls
only ever ADD time on a shared staging GPU, so `autotune.search` now times each candidate `rounds`
(=3) times, INTERLEAVED, and keeps the minimum. Re-measured on the same machine:

```
    before:  rows=1 ncols=4096  default 0.1358 ms  ->  "2.89x better", deviation recorded
    after:   rows=1 ncols=4096  rounds 0.0145 0.0141 0.0144  -> 0.0141 ms, within noise, default kept
```

The whole re-run then chose exactly one deviation, the one that is real.

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
