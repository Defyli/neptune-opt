"""
Measure this machine's launch configurations and merge them into the registry.

This is the entry point for ADDING A GPU. It needs only torch and Triton -- no `torch.export`, no
AOTInductor, no CUDA-graph machinery -- so a machine whose stack is too old for the integration can
still contribute its numbers. `experiments/gpu_m22_autotune.py` is the other half: it uses the same
measurement and then verifies the result end to end.

    python experiments/measure_launch_configs.py                 # merge into launch_table.py
    python experiments/measure_launch_configs.py --dry-run       # print, do not write
    python experiments/measure_launch_configs.py --shapes 1x262144,256x16384 --reps 100

The table is per GPU and is NOT copied between machines: the same search on the same stack chose four
deviations from the default on an RTX 5090 D v2 and one on an RTX 4090 D. Running this on a new GPU
adds its own entry and leaves every other entry untouched.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neptune_codegen as CG  # noqa: E402,F401  (imported for its side effect: path + registry)
from neptune_codegen import autotune  # noqa: E402

# The default shapes. Chosen so that buckets are shared by more than one shape (a bucket too coarse
# to hold its shapes then shows up as a disagreement) and every rows class is represented.
DEFAULT_SHAPES = (
    (1, 256), (1, 4096), (1, 262144),
    (4, 4096), (8, 1024),
    (256, 1024), (256, 4096), (256, 262144),
    (4096, 4096),
)


def _parse_shapes(text: str) -> tuple:
    shapes = []
    for part in text.split(","):
        rows, _, ncols = part.strip().partition("x")
        shapes.append((int(rows), int(ncols)))
    return tuple(shapes)


def _triton_version() -> str:
    try:
        import triton
        return triton.__version__
    except Exception:  # noqa: BLE001
        return "MISSING"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", default="logsumexp", help="skeleton to measure (default: logsumexp)")
    ap.add_argument("--shapes", default=None, help="rowsxncols,comma,separated (default: the standard set)")
    ap.add_argument("--reps", type=int, default=50, help="launches per round (default 50)")
    ap.add_argument("--rounds", type=int, default=3, help="interleaved rounds, minimum kept (default 3)")
    ap.add_argument("--dry-run", action="store_true", help="print the entry without writing it")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device: nothing to measure")
        return 2
    machine = autotune.machine_key()
    measured_on = (f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
                   f"triton {_triton_version()}")
    shapes = _parse_shapes(args.shapes) if args.shapes else DEFAULT_SHAPES

    print("=" * 100)
    print(f"MEASURING LAUNCH CONFIGURATIONS for {machine}")
    print(f"    {measured_on}")
    print(f"    candidates: {', '.join(f'({b},{w})' for b, w in autotune.CANDIDATES)}")
    print(f"    default: BLOCK={autotune.DEFAULT[0]} num_warps={autotune.DEFAULT[1]}"
          f"   noise margin: {autotune.NOISE:.0%}   kernel: {args.kernel}")
    print("=" * 100)
    try:
        body, _log = autotune.measure_shapes(args.kernel, shapes, reps=args.reps,
                                             rounds=args.rounds)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2

    print()
    print(f"    {len(body)} deviation(s) from the default on this machine:")
    for key, cfg in sorted(body.items()):
        print(f"      {key}: {cfg}")
    if not body:
        print("      (none: every shape measured is within the noise margin of the default)")

    path = os.path.join(os.path.dirname(_HERE), "neptune_codegen", "launch_table.py")
    merged = autotune.merge_table(autotune.tables(), machine, measured_on, body)
    if args.dry_run:
        print("\n    --dry-run: not writing launch_table.py. The entry would be:\n")
        print(autotune.render_tables({machine: merged[machine]}))
        return 0
    autotune.write_tables(path, merged)
    print(f"\n    wrote {len(merged)} machine(s) to {os.path.relpath(path, os.path.dirname(_HERE))}: "
          f"{', '.join(sorted(merged))}")
    print("    other machines' entries were left alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
