"""
Choosing a generated kernel's launch configuration: `(BLOCK, num_warps)`.

Why a table rather than a formula
---------------------------------
The generated kernel is ONE PROGRAM PER ROW and loops over the row in `BLOCK`-wide chunks. Two
consequences, and neither resolves from first principles:

* `BLOCK` sets both how many iterations the loop takes and how much of the last one is wasted on a
  mask. A row of 1000 with `BLOCK=1024` leaves 24 of 1024 lanes idle for the whole loop; with
  `BLOCK=256` it runs four full iterations.
* `num_warps` sets the shape of the block-wide reduction. Lanes reduce first within a warp and then
  through shared memory, so too few warps serialises the tree while too many add synchronisation to
  a kernel that is otherwise bound by memory.

Which effect wins depends on the hardware and on the row length. So the answer is a MEASURED table
(`neptune_codegen/launch_table.py`, produced by `experiments/gpu_m22_autotune.py`, which prints the
search it ran), with `DEFAULT` -- the fixed value every measurement before I6 used -- as the
fallback for any shape that has not been measured.

The configuration is part of the OPERATOR'S IDENTITY, not a launch-time decision: an AOTI package
bakes the launch in, and the op is registered globally by name, so `op_handle` encodes the
configuration in that name. Two models with different row lengths then get two ops, each with the
configuration measured for its own shape, instead of whichever model registered first deciding for
both.
"""
from __future__ import annotations

import dataclasses
import math

# Every configuration the search may choose from. Kept small on purpose: the search costs one
# compile plus a timing run per entry, and it runs on shapes nobody has measured yet.
CANDIDATES: tuple = (
    (256, 1),
    (512, 2),
    (1024, 4),
    (2048, 4),
    (4096, 4),
    (2048, 8),
    (8192, 8),
)

# The value that shipped before I6 existed, and the fallback for an unmeasured shape. Every GPU
# result recorded in docs/evidence before `exp37` was produced with this.
DEFAULT: tuple = (1024, 4)

# `BLOCK` is bucketed to a power of two between these. Below the minimum the loop overhead dominates
# for every shape measured; above the maximum a single block exceeds the register budget of the
# kernel at 4 warps (measured: compile failure).
MIN_BUCKET = 256
MAX_BUCKET = 1 << 20

# Few rows means few programs, so occupancy has to come from inside the block; many rows means the
# grid provides it. Measured: a SINGLE row wants a different block from four rows at the same row
# length, so the single row gets its own class rather than sharing "few".
THIN_ROWS = 8

# What counts as a real difference. Measured on the target GPU: across the candidate set, timings
# for a short row spread by up to ~5% run to run, while the one configuration that clearly matters
# (a long row, one program) wins by 3x. So a candidate that beats the default by less than this is
# NOT a reason to change the default -- recording it would encode noise as a tuning decision.
NOISE = 0.05


def ncols_bucket(ncols) -> int:
    """The row length rounded UP to a power of two, clamped to the measured range."""
    n = max(1, int(ncols))
    return max(MIN_BUCKET, min(MAX_BUCKET, 1 << (n - 1).bit_length()))


def rows_class(rows) -> str:
    """"single", "few" or "many".

    One row is its own class because a single program has to find all of its parallelism inside the
    block, and the measurement says it wants a different block from four rows of the same length.
    A symbolic row count is treated as "many": a dynamic row count is a model whose batch varies,
    which is not the single-row case this class exists for, and guessing "single" would apply the
    wrong configuration to every real batch.
    """
    try:
        n = int(rows)
    except Exception:  # noqa: BLE001 -- a SymInt without a concrete value
        return "many"
    if n <= 1:
        return "single"
    return "few" if n <= THIN_ROWS else "many"


def shape_key(rows, ncols) -> tuple:
    return rows_class(rows), ncols_bucket(ncols)


def table() -> dict:
    """The measured table, imported lazily so a missing or broken data file cannot break import."""
    try:
        from neptune_codegen.launch_table import TABLE
    except Exception:  # noqa: BLE001
        return {}
    return {k: tuple(v) for k, v in TABLE.items() if tuple(v) in CANDIDATES}


def best_config(rows, ncols, dtype=None, device=None, verbose: bool = False) -> tuple:
    """`(BLOCK, num_warps)` for a row length, from the measured table or `DEFAULT`.

    `dtype` and `device` are accepted and currently unused: the search in `gpu_m22_autotune.py`
    measured both and they did not change the winner for any shape (see its log). They stay in the
    signature because that is the honest place for the answer to live if a future measurement says
    otherwise -- which is a measurement, not a guess.
    """
    key = shape_key(rows, ncols)
    cfg = table().get(key)
    if cfg is None:
        cfg = DEFAULT
        source = "default"
    else:
        source = "measured"
    if verbose:
        print(f"    launch config for rows={rows} ncols={ncols} -> {key}: "
              f"BLOCK={cfg[0]} num_warps={cfg[1]} ({source})")
    return cfg


def choose(results: list, default: tuple = DEFAULT, noise: float = NOISE) -> tuple:
    """What a measurement says the configuration should be: a config, and why.

    `results` are the MEASURED candidates. The rule is deliberately conservative:

        if the best candidate does not beat `default` by more than `noise`, keep `default`

    Two reasons, both from the measurements in `experiments/gpu_m22_autotune.py`. Most shapes show a
    spread of a few percent across ALL candidates, which is run-to-run noise -- picking its minimum
    would bake noise into a compiled artifact. And where the choice genuinely matters it does not
    matter marginally: a single row of 262144 wants `BLOCK=8192, num_warps=8` and is **3x** faster
    than the default, far outside the noise band.

    Returns `(config, gain_over_default, reason)`.
    """
    ranked = best_of(results)
    best = ranked[0]
    fix = next((r for r in results if (r.block, r.num_warps) == tuple(default)), None)
    if fix is None or not math.isfinite(best.ms) or best.ms <= 0:
        return tuple(default), float("nan"), "the default was not measurable: keeping it"
    gain = fix.ms / best.ms
    if gain <= 1.0 + noise:
        return tuple(default), gain, f"within noise ({gain:.2f}x): keeping the default"
    return (best.block, best.num_warps), gain, f"{gain:.2f}x better than the default"


def choose_many(per_shape: list, default: tuple = DEFAULT, noise: float = NOISE) -> tuple:
    """One choice for several shapes that share a bucket: the best WORST CASE.

    A bucket holds more than one row length, so a configuration has to be good for all of them, not
    for the one that happened to be timed first. Selecting per shape and keeping the last answer
    flapped at the noise boundary in `gpu_m22_autotune.py` -- one shape measured 1.05x and another
    1.0501x, and the bucket got two different answers. Here a candidate is scored by the WORST gain
    it achieves across the shapes, and the default is kept unless the winner clears the margin.

    `per_shape` is a list of measured candidate lists. Returns `(config, worst_case_gain, reason)`.
    """
    runs = []
    for results in per_shape:
        fix = next((r for r in results if (r.block, r.num_warps) == tuple(default)), None)
        if fix is not None and math.isfinite(fix.ms) and fix.ms > 0:
            runs.append({(r.block, r.num_warps): r.ms for r in results if math.isfinite(r.ms)})
    if not runs:
        return tuple(default), float("nan"), "the default was not measurable: keeping it"
    common = set(runs[0])
    for run in runs[1:]:
        common &= set(run)
    scored = []
    for cfg in common:
        default_ms = sum(r[tuple(default)] for r in runs) / len(runs)
        worst = min(r[tuple(default)] / r[cfg] for r in runs)
        scored.append((worst, cfg))
    if not scored:
        return tuple(default), float("nan"), "no configuration was measured for every shape"
    scored.sort(key=lambda t: (-t[0], t[1]))
    worst, cfg = scored[0]
    if cfg == tuple(default) or worst <= 1.0 + noise:
        return tuple(default), worst, f"within noise ({worst:.2f}x worst case): keeping the default"
    return cfg, worst, f"{worst:.2f}x better than the default in the worst case"


def op_key(lib: str, name: str, params: dict | None, keepdim: bool,
           block: int | None = None, num_warps: int | None = None) -> str:
    """The registered name of a generated op, including its parameters, rank and launch config.

    Pure Python, and shared with the op registration itself, so the claim "the configuration is
    part of the operator's identity" can be checked without a GPU.
    """
    tag = "_keep" if keepdim else ""
    for arg_name, val in sorted((params or {}).items()):
        tag += "_" + arg_name + f"{float(val):g}".replace(".", "p").replace("-", "m")
    if block is not None or num_warps is not None:
        tag += f"_b{block}w{num_warps}"
    return f"{lib}::{name}{tag}"


@dataclasses.dataclass
class Result:
    block: int
    num_warps: int
    ms: float

    def __str__(self) -> str:
        return f"BLOCK={self.block:5d} num_warps={self.num_warps} {self.ms:8.4f} ms"


def best_of(results: list) -> list:
    """The measured configurations, fastest first. Raises if NOTHING could be measured.

    Separate from `search` so the rule can be tested without a GPU: a run in which every candidate
    failed must not hand back a "winner". It did once -- a missing scalar parameter made all seven
    candidates fail to compile, and the sort simply returned the first one, `inf` milliseconds and
    all, which then became a table entry.
    """
    ok = [r for r in results if math.isfinite(r.ms)]
    if not ok:
        raise RuntimeError(
            "no candidate launch configuration could be measured "
            f"({len(results)} tried, all failed) -- refusing to report a winner from a run in "
            "which every measurement failed")
    return sorted(ok, key=lambda r: r.ms)


def search(kernel_name: str, rows: int, ncols: int, dtype=None, device: str = "cuda",
           reps: int = 100, candidates=CANDIDATES, params: dict | None = None,
           warmup: int = 5) -> list:
    """Time every candidate on a real tensor and return them fastest first.

    The input is made contiguous ONCE and the output allocated ONCE, so the comparison measures the
    launch configuration rather than the allocator. Launches are pipelined (`reps` in flight), which
    is the difference between measuring the kernel and measuring Python dispatch -- the lesson from
    `experiments/real_modules.py`.
    """
    import time

    import torch

    from neptune_codegen.skeleton import param_defaults
    from neptune_codegen.template import compile_skeleton

    if dtype is None:
        dtype = torch.float32
    params = {**param_defaults(kernel_name), **(params or {})}
    x = torch.randn(rows, ncols, device=device, dtype=dtype).abs()      # |x|: every skeleton's
    x = x.contiguous()                                                  # operand is reached by it
    out = torch.empty(rows, device=device, dtype=dtype)
    results = []
    for block, warps in candidates:
        try:
            kernel = compile_skeleton(kernel_name, params)
        except Exception as e:  # noqa: BLE001 -- a candidate that cannot compile is not a candidate
            results.append(Result(block, warps, float("inf")))
            print(f"      BLOCK={block:5d} num_warps={warps}  cannot compile: "
                  f"{type(e).__name__}: {str(e)[:40]}")
            continue

        def launch(kernel=kernel, block=block, warps=warps):
            kernel[(rows,)](x, out, rows, ncols, x.stride(0), BLOCK=block, num_warps=warps)

        try:
            for _ in range(warmup):
                launch()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                launch()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3 / reps
        except Exception as e:  # noqa: BLE001
            ms = float("inf")
            print(f"      BLOCK={block:5d} num_warps={warps}  cannot launch: "
                  f"{type(e).__name__}: {str(e)[:40]}")
        results.append(Result(block, warps, ms))
        print(f"      {results[-1]}")
    return best_of(results)
