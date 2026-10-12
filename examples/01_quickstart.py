"""Quickstart: fuse one operator and compile it with AOTInductor.

    python examples/01_quickstart.py

Runs on CPU as far as it can and says so plainly; the AOTI step needs a CUDA device.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import neptune_opt
from examples.models import RMSNormMax, SoftmaxLogits, StableL2Norm


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(64, 4096, device=device)
    print(f"torch {torch.__version__}  device={device}")

    for model in (StableL2Norm(), SoftmaxLogits(), RMSNormMax()):
        name = type(model).__name__
        model = model.eval().to(device)

        # --- exactly what an existing AOTI flow already does -------------------------
        ep = torch.export.export(model, (x,))
        # --- the one line this library adds -----------------------------------------
        report = neptune_opt.apply(ep, verify_with=(model, (x,)))

        print(f"\n=== {name}")
        print(report.summary() or "  (nothing to do)")
        if not report.fusions:
            # Recognition needs no GPU, so a CPU-only machine is told which operator was
            # found even though no kernel could be generated.
            continue

        got = ep.module()(x)
        got = got[0] if isinstance(got, (tuple, list)) else got
        with torch.no_grad():
            ref = model(x)
        err = (got - ref).abs().max().item()
        print(f"  max|fused - eager| = {err:.3e}")

        if device != "cuda":
            print("  (skipping the AOTI package: it needs CUDA)")
            continue
        import tempfile
        pkg = os.path.join(tempfile.mkdtemp(), "model.pt2")
        torch._inductor.aoti_compile_and_package(ep, package_path=pkg)
        print(f"  AOTI package: {pkg} ({os.path.getsize(pkg) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
