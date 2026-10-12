"""How to wire this into a project that already compiles with AOTI.

The change is one line. Everything else -- how you export, which `dynamic_shapes` you pass,
how you ship the `.pt2` -- stays yours.

    BEFORE                                   AFTER
    -------------------------------------    ---------------------------------------------
    ep = torch.export.export(m, args)         ep = torch.export.export(m, args)
                                              neptune_opt.apply(ep)         # <- added
    aoti_compile_and_package(ep, ...)         aoti_compile_and_package(ep, ...)

Use `neptune_opt.export(m, args, **kw)` instead if you would rather not touch the export line;
it forwards every keyword straight to `torch.export.export`.

    python examples/02_existing_project.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import neptune_opt
from examples.models import TinyAttention


def build_and_ship(model: torch.nn.Module, example_inputs: tuple, out_dir: str) -> str:
    """A realistic export -> optimise -> package function."""
    # 1. export, keeping whatever semantics the project already relies on
    ep = torch.export.export(model, example_inputs, strict=False)

    # 2. let neptune_opt fuse what it recognises. It never raises: on any failure the graph is
    #    left untouched and the reason is in the report.
    report = neptune_opt.apply(ep, verify_with=(model, example_inputs))
    print(report.summary() or "  (nothing to fuse)")

    # 3. package. AOTI re-runs AOTAutograd on the program, which is exactly the step that
    #    turns the generated op into a Triton kernel -- so nothing else is needed here.
    pkg = os.path.join(out_dir, "model.pt2")
    torch._inductor.aoti_compile_and_package(ep, package_path=pkg)
    return pkg


def main() -> None:
    if not torch.cuda.is_available():
        print("This example needs CUDA (it packages a real AOTI artifact).")
        print("Run examples/01_quickstart.py and tests/ on CPU to try the rest.")
        return

    import tempfile

    ids = torch.arange(64)
    scores = torch.randn(64, 64)
    scores = scores.masked_fill(ids[None, :] > ids[:, None], float("-inf"))  # causal mask

    def causal_attention(weights: torch.Tensor) -> torch.Tensor:
        m = weights.amax(dim=-1, keepdim=True)
        e = torch.exp(weights - m)
        return e / e.sum(dim=-1, keepdim=True)

    model = torch.nn.Sequential()  # placeholder replaced below
    del model

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = scores

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return causal_attention(self.scores) @ x

    m = M().eval().cuda()
    x = torch.randn(64, 32, device="cuda")
    print("=== masked attention, exported and packaged")
    pkg = build_and_ship(m, (x,), tempfile.mkdtemp())
    print(f"  packaged: {pkg} ({os.path.getsize(pkg) / 1024:.0f} KiB)")

    print("\n=== a transformer-ish model")
    attn = TinyAttention(32).eval().cuda()
    pkg = build_and_ship(attn, (x,), tempfile.mkdtemp())
    print(f"  packaged: {pkg} ({os.path.getsize(pkg) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
