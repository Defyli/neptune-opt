"""
Milestone 9 — THE DECISIVE EXPERIMENT.

Question
--------
A graph-level rewrite was measured (milestone 8) to change nothing: identical kernels,
identical input traversals, latency inside noise. The claim under test now is the one the
whole general-module idea rests on:

    can a single-pass fused kernel reduce the number of times the input is read, and does
    that translate into real speedup on a DRAM-bound reduction?

Method
------
We do NOT go through Inductor for the candidate. We write the fused kernel directly in
Triton, because the point is to measure the ceiling of the *idea*, not the current
plumbing. Concretely, for stable L2:

    baseline (torch):  m = amax(|x|);  sqrt(sum((x/m)^2))
    fused (ours):      one pass accumulating both sum(x^2) and max(|x|)

Both are exact; they differ only in how many times the input is traversed.

Measurements per shape:
  * input traversal count of the baseline (parsed from the emitted Triton source)
  * correctness of the fused kernel against the torch reference
  * kernel-only latency (CUDA events) for: torch eager, torch.compile, our fused kernel
  * effective DRAM traffic = bytes read / time, to show whether we are at the bandwidth
    limit or somewhere else

The host is a shared production box, so the sweep is bounded and uses small batches.

Run: python3 gpu_m9_fused.py
"""
from __future__ import annotations

import os
import re
import statistics
import tempfile
import time

import torch
import triton
import triton.language as tl

assert torch.cuda.is_available()
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
import torch._inductor.config as cfg  # noqa: E402

cfg.cuda_backend = "triton"

print("=" * 100)
print("MILESTONE 9 — decisive experiment: does single-pass fusion actually pay?")
print(f"torch {torch.__version__}  triton {triton.__version__}  "
      f"gpu {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"  SMs={props.multi_processor_count}  mem={props.total_memory/2**30:.1f} GiB")
print("=" * 100)


# --------------------------------------------------------------- fused kernel
@triton.jit
def stable_l2_fused(
    x_ptr, out_ptr,
    rows, ncols,
    stride_row,
    BLOCK: tl.constexpr,
):
    """One traversal per row.

    Keeps a running maximum and a running sum of the *scaled* squares:

        acc += (x_j / m_j)^2,   m_j = max(|x_i|, i <= j)

    and finishes with |x|_2 = m_final * sqrt(acc). This is exact because
    sum(x_j^2) = m_final^2 * sum((x_j/m_j)^2 * (m_j/m_final)^2) and the running-max
    rescaling is handled by choosing the accumulated form above.
    """
    row = tl.program_id(0)
    if row >= rows:
        return
    base = x_ptr + row * stride_row
    m = 0.0
    acc = 0.0
    for off in range(0, ncols, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < ncols
        v = tl.load(base + idx, mask=mask, other=0.0).to(tl.float32)
        a = tl.abs(v)
        m_new = tl.maximum(m, tl.max(a, axis=0))
        # rescale the previous accumulator to the new running maximum
        scale = tl.where(m_new > 0, m / m_new, 0.0)
        # Triton tensors have no __pow__: write the square as a multiplication.
        sc = a / tl.maximum(m_new, 1e-30)
        acc = acc * scale * scale + tl.sum(sc * sc, axis=0)
        m = m_new
    tl.store(out_ptr + row, m * tl.sqrt(acc))


def run_fused(x: torch.Tensor, block: int = 1024) -> torch.Tensor:
    rows, ncols = x.shape[0], x.numel() // x.shape[0]
    x2 = x.reshape(rows, ncols).contiguous()
    out = torch.empty(rows, device=x.device, dtype=torch.float32)
    grid = (rows,)
    stable_l2_fused[grid](x2, out, rows, ncols, x2.stride(0), BLOCK=block)
    return out


def torch_ref(t: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Row-wise stable L2, producing a 1-D result to match the fused kernel's shape."""
    t2 = t.reshape(t.shape[0], -1)
    m = t2.abs().amax(dim=-1, keepdim=True)
    return (m * torch.sqrt(((t2 / m.clamp_min(1e-30)) ** 2).sum(dim=-1, keepdim=True)
                           + eps)).squeeze(-1)


def torch_decomposed(t, eps=1e-6):
    m = t.abs().amax(dim=-1, keepdim=True)
    return (m * torch.sqrt(((t / m) ** 2).sum(dim=-1, keepdim=True) + eps)).squeeze(-1)


# --------------------------------------------------------------- timing
def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts)


def baseline_traversals(shape):
    """Compile the torch baseline and count how often the input is loaded."""
    cfg.post_grad_custom_pre_pass = None
    cache = tempfile.mkdtemp(prefix="m9_base_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache
    os.environ["TRITON_HOME"] = tempfile.mkdtemp(prefix="m9th_")
    torch._dynamo.reset()
    x = torch.randn(*shape, device="cuda")
    with torch.no_grad():
        cfn = torch.compile(torch_decomposed, dynamic=False)
        cfn(x)
    total_in, total_all, names = 0, 0, []
    for dp, _d, fs in os.walk(cache):
        for f in fs:
            if not f.endswith(".py"):
                continue
            t = open(os.path.join(dp, f), encoding="utf-8", errors="replace").read()
            if "async_compile.triton(" not in t:
                continue
            names += re.findall(r"async_compile\.triton\('(\w+)'", t)
            body = t.split("'''", 1)[-1] if "'''" in t else t
            total_in += len(re.findall(r"tl\.load\(\s*in_ptr0", body))
            total_all += len(re.findall(r"tl\.load\(", body))
    return cfn, total_in, total_all, names


SHAPES = [(8, 1024), (8, 8192), (8, 16384), (8, 65536), (32, 65536)]

print(f"\n{'shape':>14} {'base in_ptr0':>13} {'fused in_ptr0':>14} {'err(fused)':>11} "
      f"{'eager ms':>9} {'compiled ms':>12} {'fused ms':>9} {'speedup':>8} {'GB/s':>7}")
print("-" * 100)

rows_out = []
for shape in SHAPES:
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda")
    ref = torch_ref(x)
    fout = run_fused(x)
    err = (fout - ref).abs().max().item()

    cfn, base_in, base_all, names = baseline_traversals(shape)

    med_e, _ = timeit(lambda: torch_decomposed(x))
    med_c, _ = timeit(lambda: cfn(x))
    med_f, min_f = timeit(lambda: run_fused(x))

    nbytes = x.numel() * x.element_size()
    gbps = nbytes / (min_f * 1e-3) / 1e9 if min_f > 0 else float("nan")
    speedup = med_c / med_f if med_f > 0 else float("nan")
    print(f"{str(shape):>14} {base_in:13d} {1:14d} {err:11.2e} "
          f"{med_e:9.3f} {med_c:12.3f} {med_f:9.3f} {speedup:7.2f}x {gbps:7.0f}")
    rows_out.append(dict(shape=shape, base_in=base_in, err=err, eager=med_e,
                         compiled=med_c, fused=med_f, speedup=speedup, gbps=gbps,
                         names=names, nbytes=nbytes))

print("\n" + "=" * 100)
print("SUMMARY")
print("-" * 100)
for r in rows_out:
    print(f"  {str(r['shape']):>14}  baseline reads input {r['base_in']}x, fused reads it 1x "
          f"-> {r['speedup']:.2f}x  ({r['gbps']:.0f} GB/s, err {r['err']:.1e})")
    if len(r["names"]) > 1:
        print(f"                  baseline kernels: {r['names']}")

print("""
Reading:
  * 'base in_ptr0' is the number of times the compiled torch baseline loads the input,
    parsed from the emitted Triton source. 'fused in_ptr0' is 1 by construction.
  * speedup = compiled baseline / fused. Compiled is the fair comparison; eager just
    shows how much the decomposition costs without Inductor.
  * GB/s near the device's DRAM bandwidth means we are memory-bound and further fusion
    cannot help. GB/s far below it means something else is the bottleneck.
""")
print("=" * 100)
