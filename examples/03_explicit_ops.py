"""The explicit operator library -- no recognition, no graph rewriting.

When recognition declines (see "Known limits" in the README) you can still use the generated
kernels directly by naming the operator. Same algebra, same code generation; you just choose
the operator instead of the compiler.

    python examples/03_explicit_ops.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import neptune_opt


def main() -> None:
    if not torch.cuda.is_available():
        print("This example builds GPU kernels, so it needs CUDA.")
        print("It still shows what each operator would need:")
    print()
    for name in ("stable_l2", "logsumexp", "rmsnorm_max", "softcap_lse", "linear_attn"):
        print(f"  neptune_opt.{name:12s} requires {neptune_opt.op_parameters(name)}")

    if not torch.cuda.is_available():
        print("\n(On a CUDA machine this next part runs.)")
        print("  neptune_opt.rmsnorm_max(x, eps=1e-6)")
        print("  neptune_opt.stable_l2(x)")
        print("  neptune_opt.softcap_lse(x, beta=0.5)")
        return

    x = torch.randn(64, 4096, device="cuda")

    print("\n=== the operator library")
    got = neptune_opt.rmsnorm_max(x, eps=1e-6)
    r = (x * x).mean(dim=-1, keepdim=True)
    ref = (x / torch.sqrt(r + 1e-6)).amax(dim=-1)
    print(f"  rmsnorm_max(x, eps=1e-6)   max|err| = {(got - ref).abs().max().item():.3e}")

    got = neptune_opt.stable_l2(x)
    m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    ref = (m * torch.sqrt(((x / m) ** 2).sum(dim=-1, keepdim=True))).squeeze(-1)
    print(f"  stable_l2(x)               max|err| = {(got - ref).abs().max().item():.3e}")

    print("\n=== these are ordinary torch ops, so they export and package like any other")
    class M(torch.nn.Module):
        def forward(self, t):
            return neptune_opt.rmsnorm_max(t, eps=1e-6)

    import tempfile
    ep = torch.export.export(M().eval(), (x,))
    pkg = os.path.join(tempfile.mkdtemp(), "m.pt2")
    torch._inductor.aoti_compile_and_package(ep, package_path=pkg)
    print(f"  packaged: {pkg} ({os.path.getsize(pkg) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
