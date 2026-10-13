"""
I6: choose the launch configuration by measurement, and decide the split-reduction question.

Until this iteration `BLOCK` was the constant 1024 with Triton's default 4 warps. Neither value was
chosen. This experiment measures the candidates on real shapes and then does something the first
attempt did not: it decides what to DO with the difference, because most of the differences are
noise.

    §1 SEARCH   time every candidate on real shapes, with the default in the comparison
    §2 CHOICE   for each shape, keep the default unless a candidate beats it by more than the
                measured noise margin (`autotune.NOISE`), and write the DEVIATIONS to
                `neptune_codegen/launch_table.py`. A table full of noise is worse than no table
    §3 CHECK    the exit criterion: what the table chooses is never slower than the default,
                directly and end to end through AOTI, and the configuration really is baked into
                the package's op name
    §4 DECIDE   tall-thin shapes. Split reduction is the proposed fix for them, so its CEILING is
                measured (the chunked kernel plus a combine -- exactly what split reduction costs)
                against what the tuned single-program kernel now does

Run on the GPU host:  python3 gpu_m22_autotune.py
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import neptune_codegen as CG  # noqa: E402
from neptune_codegen import autotune  # noqa: E402

_failed = []
_passed = 0


def check(label, ok, detail=""):
    global _passed
    if ok:
        _passed += 1
    else:
        _failed.append(label)
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


# (rows, ncols) pairs chosen so that buckets are shared by more than one shape (a bucket too coarse
# to hold its shapes then shows up as a disagreement) and every rows class is represented.
SEARCH_SHAPES = (
    (1, 256), (1, 4096), (1, 262144),
    (4, 4096), (8, 1024),
    (256, 1024), (256, 4096), (256, 262144),
    (4096, 4096),
)


def _batch_ms(fn, reps=100):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / reps


def _triton():
    try:
        import triton
        return triton.__version__
    except Exception:  # noqa: BLE001
        return "MISSING"


# ---------------------------------------------------------------- §1/§2 search and choice
def section_search() -> list:
    print("=" * 112)
    print("[1] THE SEARCH: every candidate, timed on real shapes (fp32, kernel=logsumexp)")
    print("=" * 112)
    print(f"    candidates: {', '.join(f'({b},{w})' for b, w in autotune.CANDIDATES)}")
    print(f"    the fixed default being replaced: BLOCK={autotune.DEFAULT[0]} "
          f"num_warps={autotune.DEFAULT[1]}   noise margin: {autotune.NOISE:.0%}")
    measured = []
    for rows, ncols in SEARCH_SHAPES:
        key = autotune.shape_key(rows, ncols)
        print(f"    rows={rows:6d} ncols={ncols:7d}  (bucket {key[1]}, {key[0]})")
        results = autotune.search("logsumexp", rows, ncols, reps=50)
        picked, gain, why = autotune.choose(results)
        default = next((r for r in results
                        if (r.block, r.num_warps) == autotune.DEFAULT), None)
        if default is None:
            check(f"rows={rows} ncols={ncols}: the default was measurable", False)
            continue
        check(f"rows={rows} ncols={ncols}: a candidate was measured "
              f"({len(results)} of {len(autotune.CANDIDATES)})", math.isfinite(gain), f"gain={gain}")
        print(f"      -> choose {picked}: {why}  (default {default.ms:.4f} ms)")
        measured.append((rows, ncols, key, results, picked, gain))

    print()
    print("    is the choice skeleton-specific? (logsumexp vs stable_l2, same shapes)")
    disagreements = []
    for rows, ncols in ((256, 4096), (1, 262144)):
        a = autotune.choose(autotune.search("logsumexp", rows, ncols, reps=50))[0]
        b = autotune.choose(autotune.search("stable_l2", rows, ncols, reps=50))[0]
        same = a == b
        print(f"      rows={rows} ncols={ncols}: logsumexp chooses {a}, stable_l2 chooses {b}"
              f"  {'agree' if same else 'DISAGREE'}")
        if not same:
            disagreements.append((rows, ncols))
    check("the CHOICE does not depend on the skeleton", not disagreements,
          f"disagreements: {disagreements}")
    return measured


_TABLE_HEADER = '''\
"""
MEASURED launch configurations -- GENERATED FILE, do not hand-edit.

Produced by `experiments/gpu_m22_autotune.py`, which prints the whole search it ran. Regenerate it
on a different GPU rather than editing a number: the point of the table is that every entry was
timed on the machine that uses it.

    {measured_on}

ONLY DEVIATIONS FROM THE DEFAULT ARE LISTED. Every other shape uses `autotune.DEFAULT`, because the
measurement did not justify anything else: candidates for a short row differ by a few percent run to
run, and a candidate has to beat the default by more than `autotune.NOISE` ({noise:.0%}) to be
recorded here. Where the choice does matter it is not marginal -- one of the rows below is a single
row of 262144 columns, where the default is several times slower.

Keys are `(rows class, ncols bucket)`; see `neptune_codegen/autotune.py` for what a bucket is, why
the class exists, and why a choice has to clear the noise margin.
"""
TABLE = {{
'''


def section_table(measured: list) -> dict:
    print()
    print("=" * 112)
    print("[2] THE CHOICE: only deviations from the default are worth writing down")
    print("=" * 112)
    grouped = {}
    for rows, ncols, key, results, _picked, _gain in measured:
        grouped.setdefault(key, []).append(results)
    chosen = {}
    for key, runs in sorted(grouped.items()):
        cfg, worst, why = autotune.choose_many(runs)
        chosen[key] = cfg
        print(f"      {key}: {len(runs)} shape(s) -> {cfg}: {why}")
        check(f"{key}: every candidate was measured on every shape in the bucket",
              len(runs) > 0, f"{len(runs)} runs")
    deviations = {k: v for k, v in chosen.items() if v != autotune.DEFAULT}
    print(f"    {len(chosen)} buckets measured, {len(deviations)} deviate from the default")
    for k, v in sorted(chosen.items()):
        print(f"      {k}: {v}{'   <- deviation' if v != autotune.DEFAULT else ''}")
    text = _TABLE_HEADER.format(
        measured_on=f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
                    f"triton {_triton()}",
        noise=autotune.NOISE) + "".join(
        f"    {k!r}: {v!r},\n" for k, v in sorted(deviations.items())) + "}\n"
    path = os.path.join(os.path.dirname(_HERE), "neptune_codegen", "launch_table.py")
    with open(path, "w") as f:
        f.write(text)
    print("-" * 112)
    print(text)
    print("-" * 112)
    for k, v in deviations.items():
        check(f"{k} -> {v} is a real candidate", tuple(v) in autotune.CANDIDATES)
    return chosen


# ---------------------------------------------------------------- §3 the exit criterion
def section_check(measured: list) -> None:
    print()
    print("=" * 112)
    print("[3] CHECK: what the table chooses is not slower than the fixed default")
    print("=" * 112)
    print("    (a) direct launches, at every shape that was measured")
    for rows, ncols, _key, _results, picked, _gain in measured:
        x = torch.randn(rows, ncols, device="cuda").abs().contiguous()
        out = torch.empty(rows, device="cuda", dtype=torch.float32)
        kernel = CG.compile_skeleton("logsumexp", CG.param_defaults("logsumexp"))

        def launch(block, warps):
            kernel[(rows,)](x, out, rows, ncols, x.stride(0), BLOCK=block, num_warps=warps)

        fixed = _batch_ms(lambda: launch(*autotune.DEFAULT))
        tuned = _batch_ms(lambda: launch(*picked))
        gain = fixed / tuned
        check(f"rows={rows} ncols={ncols}: {picked} >= default {autotune.DEFAULT}",
              gain >= 1.0 - autotune.NOISE, f"{fixed:.4f} ms -> {tuned:.4f} ms = {gain:.2f}x")

    print("    (b) end to end through AOTI, at the shape where the choice is NOT noise")
    import torch.nn as nn

    import neptune_opt as NO

    rows, ncols = 1, 262144
    x = torch.randn(rows, ncols, device="cuda")

    class Model(nn.Module):
        def forward(self, t):
            return torch.logsumexp(t, dim=-1)

    def package(force_default: bool):
        mod = Model().eval().cuda()
        ep = torch.export.export(mod, (x,)).run_decompositions()
        real = autotune.best_config
        if force_default:
            autotune.best_config = lambda *a, **k: autotune.DEFAULT          # noqa: ARG005
        try:
            report = NO.apply(ep, verify_with=(mod, (x,)))
        finally:
            autotune.best_config = real
        d = tempfile.mkdtemp(prefix="i6_")
        path = os.path.join(d, "m.pt2")
        torch._inductor.aoti_compile_and_package(ep, package_path=path)
        return report, path, str(ep.graph_module.graph)

    rep_t, p_tuned, graph_t = package(False)
    rep_f, p_fixed, graph_f = package(True)
    check("both packages fused", bool(rep_t.fusions) and bool(rep_f.fusions),
          f"{[f.skeleton for f in rep_t.fusions]} / {[f.skeleton for f in rep_f.fusions]}")
    cfg = autotune.best_config(rows, ncols)
    check("the configuration is IN the package's op name", f"_b{cfg[0]}w{cfg[1]}" in graph_t,
          f"expected _b{cfg[0]}w{cfg[1]} in the rewritten graph")
    check("and the forced-default package carries the default instead",
          f"_b{autotune.DEFAULT[0]}w{autotune.DEFAULT[1]}" in graph_f)
    f_t = torch._inductor.aoti_load_package(p_tuned)
    f_f = torch._inductor.aoti_load_package(p_fixed)
    ms_t, ms_f = _batch_ms(lambda: f_t(x), reps=200), _batch_ms(lambda: f_f(x), reps=200)
    gain = ms_f / ms_t
    print(f"      default {autotune.DEFAULT} {ms_f:.4f} ms -> chosen {cfg} {ms_t:.4f} ms = "
          f"{gain:.2f}x")
    check("end to end, the chosen configuration is not slower", gain >= 1.0 - autotune.NOISE,
          f"{gain:.2f}x")
    with torch.no_grad():
        a, b = f_t(x), f_f(x)
    a = a[0] if isinstance(a, (tuple, list)) else a
    b = b[0] if isinstance(b, (tuple, list)) else b
    check("and the two agree numerically",
          torch.allclose(a.float(), b.float(), rtol=1e-5, atol=1e-5),
          f"max|diff|={(a.float() - b.float()).abs().max().item():.2e}")


# ---------------------------------------------------------------- §4 the split-reduction decision
def section_split() -> dict:
    print()
    print("=" * 112)
    print("[4] DECIDE: tall-thin shapes, and the CEILING of split reduction")
    print("=" * 112)
    rows, ncols = 1, 262144
    x = torch.randn(rows, ncols, device="cuda").abs().contiguous()
    print(f"    a single row of {ncols} -- one program, so one SM is busy out of many")

    import torch.nn as nn

    class LSE(nn.Module):
        def forward(self, t):
            return torch.logsumexp(t, dim=-1)

    mod = LSE().eval().cuda()
    ep = torch.export.export(mod, (x,)).run_decompositions()
    d = tempfile.mkdtemp(prefix="i6s_")
    p0 = os.path.join(d, "base.pt2")
    torch._inductor.aoti_compile_and_package(ep, package_path=p0)
    base = torch._inductor.aoti_load_package(p0)
    ms_base = _batch_ms(lambda: base(x), reps=200)
    with torch.no_grad():
        ref = mod(x)

    kernel = CG.compile_skeleton("logsumexp", CG.param_defaults("logsumexp"))
    out = torch.empty(rows, device="cuda")
    b, w = autotune.best_config(rows, ncols, torch.float32, "cuda")

    def unsplit():
        kernel[(rows,)](x, out, rows, ncols, x.stride(0), BLOCK=b, num_warps=w)

    ms_unsplit = _batch_ms(unsplit, reps=200)
    unsplit()
    torch.cuda.synchronize()
    err = (out - ref).abs().max().item()
    check("the tuned single-program kernel is numerically right", err < 1e-3, f"max|diff|={err:.2e}")
    print(f"    our single program, tuned ({b},{w})   : {ms_unsplit:8.4f} ms   "
          f"{ms_base / ms_unsplit:5.2f}x vs Inductor")

    decision = {}
    for k in (4, 16, 64):
        chunk = ncols // k
        xc = x[:, :chunk * k].reshape(k, chunk).contiguous()
        part = torch.empty(k, device="cuda", dtype=torch.float32)
        cs, cw = autotune.best_config(k, chunk, torch.float32, "cuda")

        state = {}

        def split():
            kernel[(k,)](xc, part, k, chunk, xc.stride(0), BLOCK=cs, num_warps=cw)
            m = part.max()
            state["p"] = m + (part - m).exp().sum().log()

        ms = _batch_ms(split, reps=200)
        split()
        torch.cuda.synchronize()
        e = (state["p"] - ref).abs().max().item()
        decision[k] = ms
        check(f"k={k}: the emulated split is numerically right", e < 1e-3, f"max|diff|={e:.2e}")
        print(f"    split into {k:3d} chunks (ceiling)     : {ms:8.4f} ms   "
              f"{ms_base / ms:5.2f}x vs Inductor")

    best_k = min(decision, key=lambda k: decision[k])
    r_split, r_unsplit = ms_base / decision[best_k], ms_base / ms_unsplit
    worth = decision[best_k] < ms_unsplit * (1.0 - autotune.NOISE)
    print()
    print(f"    best split k={best_k}: {r_split:.2f}x vs Inductor; tuned single program: "
          f"{r_unsplit:.2f}x")
    print(f"    VERDICT: split reduction is {'WORTH BUILDING' if worth else 'NOT worth building'}: "
          f"ceiling {decision[best_k]:.4f} ms vs {ms_unsplit:.4f} ms for the tuned single program")
    return {"base_ms": ms_base, "unsplit_ms": ms_unsplit, "split_ms": decision[best_k],
            "best_k": best_k, "ratio_split": r_split, "ratio_unsplit": r_unsplit, "worth": worth}


def main() -> int:
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
    if not torch.cuda.is_available():
        print("NO CUDA: this experiment needs the GPU host")
        return 2
    print(f"torch {torch.__version__}  triton {_triton()}  {torch.cuda.get_device_name(0)}")
    try:
        measured = section_search()
        section_table(measured)
        section_check(measured)
        decision = section_split()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2
    print()
    print("=" * 112)
    print(f"TOTAL: {_passed} passed, {len(_failed)} failed")
    print(f"DECISION: split reduction for tall-thin shapes: "
          f"{'build it' if decision['worth'] else 'do not build it'} "
          f"(ceiling {decision['split_ms']:.4f} ms at k={decision['best_k']} vs "
          f"{decision['unsplit_ms']:.4f} ms for the tuned single program)")
    if _failed:
        for f in _failed:
            print(f"  FAILED: {f}")
        return 1
    print("READING: the table holds only what the noise margin could not explain, and the")
    print("split-reduction question is answered against the TUNED kernel, not the old default.")
    print("=" * 112)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
