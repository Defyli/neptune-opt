"""
I4 decision experiment: is there anything to GAIN by expanding softmax / log_softmax / layer_norm?

The roadmap's I4 says the biggest reach increase available is a lowering pass that expands
`aten._softmax` / `aten._log_softmax` / `aten.native_layer_norm` into decomposed subgraphs, so the
log-sum-exp (and layer norm's variance) becomes a node the region gate can replace.

Reach is not value. Those three are exactly the operators Inductor has DEDICATED lowerings for, and
`aten._softmax` in particular becomes a persistent reduction -- one kernel per row, the row held in
registers -- which is the best known shape for it. So before mutating graphs, measure the ceiling:
build what the lowering WOULD produce by hand, and race it against what Inductor already emits.

    baseline        torch.softmax / log_softmax / layer_norm, compiled
    with our LSE    exp(t - neptune.logsumexp(t))   <- what I4 would hand to Inductor
    decomposed      the max/exp/sum form written out, compiled, unfused

If the middle column does not win, I4 buys reach and costs performance, and that should be a
recorded decision rather than a surprise after the work.

Run on the GPU host:  python3 gpu_m20_i4_ceiling.py
"""
from __future__ import annotations

import os
import statistics
import sys
import time
import traceback

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neptune_codegen as CG  # noqa: E402
import neptune_opt as NO  # noqa: E402

# Large enough that the kernels, not the launch, dominate (see experiments/real_modules.py for the
# measurement bug that taught this).
X = torch.randn(256, 16384, device="cuda" if torch.cuda.is_available() else "cpu")


def _batch_ms(fn, x, reps: int = 20) -> float:
    with torch.no_grad():
        for _ in range(3):
            fn(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / reps


def _kernels_of(fn, x) -> tuple:
    """(cumulative triton kernel count, names) in the Inductor cache.

    CUMULATIVE, and therefore not a per-variant count: re-pointing TORCHINDUCTOR_CACHE_DIR after
    Inductor has loaded does not re-scope it (measured again here -- the numbers grow by one
    variant's worth each row: 5, 7, 8, 9, 10, 13, 17, 20). The DIFFERENCES between consecutive
    rows are the per-variant counts. Kept only because the differences are still informative, and
    labelled so nobody reads a single row's number as that variant's kernel count.
    """
    import glob
    import re

    d = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
    names = []
    for f in glob.glob(os.path.join(d, "**", "*.py"), recursive=True):
        t = open(f, encoding="utf-8", errors="replace").read()
        if "async_compile.triton(" in t:
            names += re.findall(r"async_compile\.triton\('(\w+)'", t)
    return len(set(names)), sorted(set(names))


def _lse(t):
    """Our generated log-sum-exp, used the way a lowering pass would use it."""
    return NO.logsumexp(t).unsqueeze(-1)


def test_softmax() -> None:
    print("\n--- softmax / log_softmax: is our LSE + elementwise faster than Inductor?")
    print(f"    {'variant':26s} {'ms':>9s} {'kernels':>8s}  kernels")
    # (label, variant, reference, baseline it is compared against). The reference is spelled out
    # per variant: an earlier version picked it by substring and `"softmax (" in "log_softmax
    # (torch)"` is True, so log_softmax would have been checked against softmax.
    sm = lambda t: torch.softmax(t, dim=-1)          # noqa: E731
    lsm = lambda t: torch.log_softmax(t, dim=-1)     # noqa: E731
    variants = [
        ("softmax (torch)", sm, sm, None),
        ("softmax via our LSE", lambda t: torch.exp(t - _lse(t)), sm, "softmax (torch)"),
        ("softmax decomposed", lambda t: (lambda m: torch.exp(t - m)
                                          / torch.exp(t - m).sum(-1, keepdim=True))(
            t.amax(dim=-1, keepdim=True)), sm, "softmax (torch)"),
        ("log_softmax (torch)", lsm, lsm, None),
        ("log_softmax via our LSE", lambda t: t - _lse(t), lsm, "log_softmax (torch)"),
    ]
    times = {}
    for label, fn, ref_fn, cmp_to in variants:
        try:
            with torch.no_grad():
                c = torch.compile(fn, dynamic=False)
                c(X)
            ms = _batch_ms(c, X)
            nk, names = _kernels_of(fn, X)
            with torch.no_grad():
                got = c(X)
                want = ref_fn(X)
            rel = ((got - want).abs() / want.abs().clamp_min(1e-3)).max().item()
            print(f"    {label:26s} {ms:9.4f} {nk:8d}  {', '.join(names[:2])}")
            times[label] = ms
            if cmp_to:
                print(f"      [{'PASS' if rel < 1e-3 else 'FAIL'}] numbers match; "
                      f"{times[cmp_to] / ms:5.2f}x vs torch's own")
            else:
                print(f"      [{'PASS' if rel < 1e-3 else 'FAIL'}] matches torch's operator")
        except Exception as e:  # noqa: BLE001
            print(f"    {label:26s} FAILED: {type(e).__name__}: {str(e)[:50]}")


def test_layer_norm() -> None:
    """Layer norm needs the mean AND the variance; the frame could only take the variance."""
    print("\n--- layer_norm: our frame can only take the VARIANCE half")
    print(f"    {'variant':30s} {'ms':>9s} {'kernels':>8s}")
    eps = 1e-5
    cols = X.shape[-1]
    variants = [
        ("layer_norm (torch)", lambda t: F.layer_norm(t, (t.shape[-1],))),
        ("mean + our variance", lambda t: (lambda m, v: (t - m) / torch.sqrt(v + eps))(
            t.mean(dim=-1, keepdim=True), NO.variance(t).unsqueeze(-1))),
        ("mean + torch var", lambda t: (lambda m, v: (t - m) / torch.sqrt(v + eps))(
            t.mean(dim=-1, keepdim=True), t.var(dim=-1, unbiased=False).unsqueeze(-1))),
    ]
    base = None
    for label, fn in variants:
        try:
            with torch.no_grad():
                c = torch.compile(fn, dynamic=False)
                c(X)
            ms = _batch_ms(c, X)
            nk, _names = _kernels_of(fn, X)
            print(f"    {label:30s} {ms:9.4f} {nk:8d}")
            if base is None:
                base = ms
            else:
                print(f"      {base / ms:5.2f}x vs torch's layer_norm")
        except Exception as e:  # noqa: BLE001
            print(f"    {label:30s} FAILED: {type(e).__name__}: {str(e)[:50]}")


def test_what_our_frame_can_actually_take() -> None:
    """For the record: of these three, which part is even inside the frame?"""
    print("\n--- what the frame could take, if the graph were expanded")
    import torch.nn as nn
    from torch.export import export

    # Real submodules, installed in __init__. Constructing `nn.LayerNorm(...)` inside a lambda
    # leaves it outside the traced module and torch.export refuses ("module is not installed as a
    # submodule") -- which torch 2.14 tolerates locally and 2.11 rejects, so the difference only
    # shows up on the GPU host.
    class One(nn.Module):
        """Exactly one operator, so an unused parameter cannot confound the verdict."""

        def __init__(self, kind, c):
            super().__init__()
            self.kind = kind
            if kind == "softmax":
                self.layer = nn.Softmax(dim=-1)
            elif kind == "log_softmax":
                self.layer = nn.LogSoftmax(dim=-1)
            elif kind == "layer_norm":
                self.layer = nn.LayerNorm(c)

        def forward(self, t):
            if self.kind == "logsumexp":
                return torch.logsumexp(t, dim=-1)
            return self.layer(t)

    x = X[:8, :1024]
    for name in ("softmax", "log_softmax", "layer_norm", "logsumexp"):
        # `.to(x.device)`: a parameterised module built on CPU cannot be exported against a CUDA
        # input ("Unhandled FakeTensor Device Propagation ... two different devices").
        ep = export(One(name, x.shape[-1]).to(x.device), (x,)).run_decompositions()
        m, why = __import__("neptune_recognize").recognize(ep.graph_module)
        got = m.skeleton if m else str(why)[:64]
        print(f"    {name:14s} -> {got}")
        ops = []
        for n in ep.graph_module.graph.nodes:
            if n.op == "call_function":
                ops.append(getattr(n.target, "__name__", str(n.target)))
        print(f"    {'':14s}    decomposed ops: {sorted(set(ops))[:6]}")


def main() -> int:
    if not torch.cuda.is_available():
        print("GPU required: this times compiled kernels.")
        return 1
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
    print("=" * 100)
    print("I4 DECISION EXPERIMENT -- what would expanding softmax/layer_norm actually buy?")
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  input {tuple(X.shape)}")
    print("=" * 100)
    # Each section guarded separately: one failure must not hide the others' results, which is
    # how a crash in the first section cost a whole GPU run. The failures are COUNTED -- guarding
    # without counting made the script exit 0 whatever happened, which is the silent-success
    # pattern this repository keeps having to fix.
    failed = []
    for fn in (test_what_our_frame_can_actually_take, test_softmax, test_layer_norm):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            print(f"\n{fn.__name__} FAILED: {type(e).__name__}: {str(e)[:90]}")
            traceback.print_exc()
    print("\n" + "=" * 100)
    print("READING: 'kernels' is the number Inductor emits for that variant. If torch's own")
    print("         operator is ONE kernel and the variant that uses our LSE is more, the frame")
    print("         cannot win there however good its reduction is -- which is what decides I4.")
    print("=" * 100)
    if failed:
        print(f"SECTIONS FAILED: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
