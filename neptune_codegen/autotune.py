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

Which effect wins depends on the hardware and on the row length, so the answer is a MEASURED table,
per GPU. `neptune_codegen/launch_table.py` holds one entry per machine, `experiments/
measure_launch_configs.py` adds or refreshes one machine's entry without touching the others, and
`DEFAULT` -- the fixed value every measurement before I6 used -- is the fallback for a shape that has
not been measured.

The table does NOT transfer between GPUs. Measured: the same search on the same stack chose four
deviations from the default on an RTX 5090 D v2 and one on an RTX 4090 D, agreeing only on the case
that matters (a single long row). So a machine with no entry uses the DEFAULT rather than another
machine's numbers -- copying them would encode someone else's hardware as a decision.

The configuration is part of the OPERATOR'S IDENTITY, not a launch-time decision: an AOTI package
bakes the launch in, and the op is registered globally by name, so `op_handle` encodes the
configuration in that name. Two shapes then get two ops, each with the configuration measured for
its own shape, instead of whichever model registered first deciding for both.
"""
from __future__ import annotations

import dataclasses
import math
import os

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

# The value that shipped before I6 existed, and the fallback for an unmeasured shape OR an
# unmeasured machine. Every GPU result recorded before `exp37` was produced with this.
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

# What counts as a real difference. Measured on the target GPU: across the candidate set, timings for
# a short row spread by up to ~5% run to run, while the one configuration that clearly matters (a
# long row, one program) wins by 3x. So a candidate that beats the default by less than this is NOT a
# reason to change the default -- recording it would encode noise as a tuning decision.
NOISE = 0.05

# Where a machine's entry comes from when the generated file only has the legacy flat `TABLE`.
LEGACY_MACHINE = "legacy-flat-table"


# ---------------------------------------------------------------- machine identity
def machine_key(device_name: str | None = None) -> str:
    """A stable id for the GPU a table was measured on: 'rtx-5090-d-v2'.

    Normalised rather than raw, because the same card reports itself with different prefixes
    ("NVIDIA GeForce RTX 4090 D", "RTX 4090 D") across drivers and wrappers. With no name, the
    current CUDA device is asked -- that imports torch, so callers on a hot path should pass a name
    (or a device to `best_config`, which resolves it once).
    """
    name = device_name
    if name is None:
        try:                                        # pragma: no cover - needs a GPU
            import torch

            if not torch.cuda.is_available():
                return ""
            name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:  # noqa: BLE001
            return ""
    words = []
    for word in str(name).lower().replace("+", " ").split():
        if word in ("nvidia", "geforce", "tesla", "graphics", "corporation"):
            continue
        words.append("".join(ch if ch.isalnum() else "-" for ch in word).strip("-"))
    return "-".join(w for w in words if w)


def device_name_of(device) -> str | None:
    """The GPU name for a `torch.device`, or None when it cannot be resolved.

    Never imports torch: it is called from `best_config`, which must stay usable where torch is not
    importable at all (the CPU tests, and `neptune_recognize`'s pure paths). It only asks when torch
    is ALREADY imported by the caller, which is true on every path that reaches a real launch.
    """
    import sys

    if device is None or "torch" not in sys.modules:
        return None
    torch = sys.modules["torch"]
    try:
        index = getattr(device, "index", None)
        return torch.cuda.get_device_name(index if index is not None else 0)
    except Exception:  # noqa: BLE001 -- no CUDA, or a fake device: the default applies
        return None


# ---------------------------------------------------------------- shape bucketing
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


# ---------------------------------------------------------------- the measured tables
def _module_tables() -> dict:
    """`{machine: {'measured_on':..., 'table': {...}}}` from the generated file."""
    try:
        from neptune_codegen import launch_table
    except Exception:  # noqa: BLE001 -- a missing or broken data file cannot break import
        return {}
    raw = getattr(launch_table, "TABLES", None)
    if raw is None:
        flat = getattr(launch_table, "TABLE", None)
        if not flat:
            return {}
        raw = {LEGACY_MACHINE: {"measured_on": "flat TABLE (pre-registry format)",
                                "table": flat}}
    out = {}
    for machine, entry in raw.items():
        if isinstance(entry, dict) and "table" in entry:
            body = entry.get("table") or {}
            meta = {k: v for k, v in entry.items() if k != "table"}
        else:                                     # tolerate a bare {shape: config} mapping
            body, meta = entry or {}, {}
        out[machine] = {**meta, "table": _valid(body)}
    return out


def _valid(body) -> dict:
    """Drop anything that is not a configuration the search could have produced.

    A hand-edited table must not be able to introduce a block size nobody measured; the alternative
    is a silent compile failure or a wrong launch far from the edit.
    """
    return {k: tuple(v) for k, v in (body or {}).items() if tuple(v) in CANDIDATES}


def tables() -> dict:
    """Every measured machine, entries validated."""
    return _module_tables()


def machines() -> list:
    return sorted(tables())


def table(machine: str | None = None) -> dict:
    """The `{shape_key: config}` table for one machine (default: the current one).

    Deliberately no fallback to another machine: see the module docstring.
    """
    all_tables = tables()
    if machine is None:
        machine = machine_key()
    if machine in all_tables:
        return dict(all_tables[machine]["table"])
    if LEGACY_MACHINE in all_tables:
        # A file in the pre-registry format carries no machine, so it cannot be attributed to one;
        # applying it is the only reading that makes the numbers useful. This is a FORMAT fallback,
        # not a cross-machine copy -- the registry format never falls back to another GPU.
        return dict(all_tables[LEGACY_MACHINE]["table"])
    return {}


def measured_on(machine: str | None = None) -> str:
    all_tables = tables()
    if machine is None:
        machine = machine_key()
    return all_tables.get(machine, {}).get("measured_on", "")


def best_config(rows, ncols, dtype=None, device=None, device_name=None,
                verbose: bool = False) -> tuple:
    """`(BLOCK, num_warps)` for a row length on this machine: measured entry, else `DEFAULT`.

    `dtype` is accepted and unused: the search measured fp32/bf16/fp16 and the dtype did not change
    the winner for any shape. It stays in the signature because that is where the answer would live
    if a future measurement says otherwise -- which would be a measurement, not a guess.
    """
    if device_name is None:
        device_name = device_name_of(device)
    machine = machine_key(device_name) if device_name else ""
    key = shape_key(rows, ncols)
    # Always consult `table`, even with an unknown machine: that call owns the FORMAT fallback for
    # a pre-registry flat TABLE. Skipping it here made the legacy path dead code -- a bug this
    # test found, since the table() call it asserted on succeeded while best_config never looked.
    cfg = table(machine).get(key)
    source = "measured"
    if cfg is None:
        cfg, source = DEFAULT, ("default: machine not measured" if not machine
                                else f"default: {key} not measured here")
    if verbose:
        print(f"    launch config for rows={rows} ncols={ncols} on {machine or '<unknown gpu>'}"
              f" -> {key}: BLOCK={cfg[0]} num_warps={cfg[1]} ({source})")
    return cfg


# ---------------------------------------------------------------- the choice rule
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
        worst = min(r[tuple(default)] / r[cfg] for r in runs)
        scored.append((worst, cfg))
    if not scored:
        return tuple(default), float("nan"), "no configuration was measured for every shape"
    scored.sort(key=lambda t: (-t[0], t[1]))
    worst, cfg = scored[0]
    if cfg == tuple(default) or worst <= 1.0 + noise:
        return tuple(default), worst, f"within noise ({worst:.2f}x worst case): keeping the default"
    return cfg, worst, f"{worst:.2f}x better than the default in the worst case"


# ---------------------------------------------------------------- the generated registry
TABLE_HEADER = '''\
"""
MEASURED launch configurations -- GENERATED FILE, do not hand-edit a number.

Add or refresh ONE machine with:

    python experiments/measure_launch_configs.py

which prints every candidate timing it took and merges the result in, leaving the other machines
alone. Regenerate on the machine that will use the table: the table does NOT transfer between GPUs
(measured: four deviations from the default on an RTX 5090 D v2, one on an RTX 4090 D, agreeing only
on the single-long-row case). A machine with no entry here uses `autotune.DEFAULT`, which is what
every measurement before I6 used.

ONLY DEVIATIONS FROM THE DEFAULT ARE LISTED. Every other shape uses `autotune.DEFAULT`, because the
measurement did not justify anything else: candidates for a short row differ by a few percent run to
run, and a candidate has to beat the default by more than `autotune.NOISE` ({noise:.0%}) to be
recorded. Where the choice does matter it is not marginal -- a single row of 262144 columns is 3x
faster with the configuration recorded for it.

Keys inside `table` are `(rows class, ncols bucket)`; see `neptune_codegen/autotune.py` for what a
bucket is, why the class exists, and why a choice has to clear the noise margin.
"""
TABLES = {{
'''


def render_tables(all_tables: dict, noise: float = NOISE) -> str:
    """The generated file's text for a whole registry, deterministically ordered."""
    out = [TABLE_HEADER.format(noise=noise)]
    for machine in sorted(all_tables):
        entry = all_tables[machine]
        out.append(f"    {machine!r}: {{\n")
        out.append(f"        'measured_on': {entry.get('measured_on', '')!r},\n")
        out.append(f"        'noise': {float(entry.get('noise', noise))!r},\n")
        out.append("        'table': {\n")
        for key, cfg in sorted((entry.get("table") or {}).items()):
            out.append(f"            {tuple(key)!r}: {tuple(cfg)!r},\n")
        out.append("        },\n    },\n")
    out.append("}\n")
    return "".join(out)


def merge_table(all_tables: dict, machine: str, measured_on_text: str, body: dict,
                noise: float = NOISE) -> dict:
    """Add or replace ONE machine in a registry and return a new registry.

    Pure, so the merge can be tested without a GPU -- it is the part that decides whether a new
    machine's run erases the machine measured before it.
    """
    merged = {m: dict(e) for m, e in all_tables.items()}
    merged[machine] = {"measured_on": measured_on_text, "noise": noise, "table": _valid(body)}
    return merged


def write_tables(path: str, all_tables: dict, noise: float = NOISE) -> None:
    with open(path, "w") as f:
        f.write(render_tables(all_tables, noise))


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


def search(kernel_name: str, rows: int, ncols: int, dtype=None, device: str = "cuda",
           reps: int = 50, candidates=CANDIDATES, params: dict | None = None,
           warmup: int = 5, rounds: int = 3) -> list:
    """Time every candidate on a real tensor and return them fastest first.

    The input is made contiguous ONCE and the output allocated ONCE, so the comparison measures the
    launch configuration rather than the allocator. Launches are pipelined (`reps` in flight), which
    is the difference between measuring the kernel and measuring Python dispatch -- the lesson from
    `experiments/real_modules.py`.

    Each candidate is timed `rounds` times, INTERLEAVED, and the MINIMUM is kept. This is not
    decoration: on a shared staging GPU one candidate came back at 0.1358 ms where the same
    configuration measured 0.0142 ms on the next run, and a single such stall was reported as a
    2.89x win and would have become a table entry. Stalls only ever ADD time, so the minimum is the
    better estimate of a kernel's cost, and interleaving keeps a slow period from landing on one
    candidate only.
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
        except Exception as e:  # noqa: BLE001
            results.append(Result(block, warps, float("inf")))
            print(f"      BLOCK={block:5d} num_warps={warps}  cannot launch: "
                  f"{type(e).__name__}: {str(e)[:40]}")
            continue
        times = []
        for _round in range(rounds):
            t0 = time.perf_counter()
            for _ in range(reps):
                launch()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3 / reps)
        ms = min(times)
        results.append(Result(block, warps, ms))
        spread = f"  (rounds: {' '.join('%.4f' % t for t in times)})" if len(times) > 1 else ""
        print(f"      {results[-1]}{spread}")
    return best_of(results)


def measure_shapes(kernel_name: str, shapes, dtype=None, device: str = "cuda", reps: int = 50,
                   candidates=CANDIDATES, rounds: int = 3) -> tuple:
    """Measure a list of `(rows, ncols)` shapes and reduce each bucket to one choice.

    Returns `(table, log)` where `table` is `{shape_key: config}` holding ONLY deviations from the
    default, and `log` is the per-shape record the caller prints. Shared by
    `experiments/measure_launch_configs.py` (which needs no `torch.export`) and
    `experiments/gpu_m22_autotune.py` (which verifies the result end to end).
    """
    import torch

    if dtype is None:
        dtype = torch.float32
    per_bucket, log = {}, []
    for rows, ncols in shapes:
        key = shape_key(rows, ncols)
        print(f"    rows={rows:6d} ncols={ncols:7d}  (bucket {key[1]}, {key[0]})")
        results = search(kernel_name, rows, ncols, dtype=dtype, device=device, reps=reps,
                         candidates=candidates, rounds=rounds)
        picked, gain, why = choose(results)
        default = next((r for r in results if (r.block, r.num_warps) == tuple(DEFAULT)), None)
        print(f"      -> choose {picked}: {why}"
              f"{'' if default is None else f'  (default {default.ms:.4f} ms)'}")
        log.append({"shape": (rows, ncols), "key": key, "picked": picked, "gain": gain,
                    "why": why, "results": results})
        per_bucket.setdefault(key, []).append(results)
    table = {}
    for key, runs in sorted(per_bucket.items()):
        cfg, worst, why = choose_many(runs)
        print(f"      bucket {key} ({len(runs)} shape(s)) -> {cfg}: {why}")
        if cfg != DEFAULT:
            table[key] = cfg
    return table, log
