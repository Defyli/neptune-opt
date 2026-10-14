# API reference

The public surface of `neptune_opt`. `tests/test_api_docs.py` asserts that this list and
`neptune_opt.__all__` agree, so it cannot go stale.

## Recommended

### `apply(exported_program, values=None, verify_with=None, lower=True) -> Report`

Rewrite a recognised operator inside an `ExportedProgram`, **in place**. This is the recommended
entry point: one line in an existing AOTI flow.

```python
ep = torch.export.export(model, args)
report = neptune_opt.apply(ep)
torch._inductor.aoti_compile_and_package(ep, package_path="model.pt2")
```

* `values` — scalar values by argument name, e.g. `{"eps": 1e-6}`. Usually unnecessary: parameters
  are inferred from the graph, so a soft-cap `beta` or a decay offset is discovered rather than
  configured.
* `verify_with=(module, example_inputs)` — run the original module and the rewritten graph on the
  same inputs and compare. Sets `Report.verified`. Without it there is no numeric check, because
  `apply` has no access to the original module.

* `lower` — on by default. Expands a compound op the core decomposition table leaves whole
  (currently `log_softmax`, whose log-sum-exp is otherwise not a node at all) so the region gate has
  something to replace. It is **committed only when it enables a rewrite**, so a graph it cannot
  help is left byte-identical; `lower=False` is the control, and a test asserts the difference.

Never raises on a refusal: the graph is left alone and `Report.skipped` says why. When the graph
has not been decomposed yet and decomposing would have helped, it says that by name.

### `export(module, example_inputs, decompose=True, **export_kwargs) -> ExportedProgram`

`torch.export.export` followed by `apply`. A drop-in replacement that also fuses.

`decompose=True` (the default) runs the **core-aten decompositions** first, and for real models
that is not optional: `nn.Softmax`, `nn.LayerNorm` and `torch.logsumexp` each export as *one*
high-level op, which this frame cannot see. The decomposed form is what AOTAutograd hands
Inductor anyway. Pass `decompose=False` to fuse only what is already spelled out in the graph.

One operator the decompositions do **not** reach is `log_softmax`: `get_decompositions` has no
entry for the overload `torch.export` produces. That one is expanded by `apply`'s lowering pass
instead, which is why `nn.LogSoftmax` fuses in fp32 and fp16 with no extra step.

Other keyword arguments go to `torch.export.export` (`dynamic_shapes`, `strict`, ...).

### `Report`

`fusions` (list of `Fusion`), `skipped` (list of `(reason, detail)`), `verified` (bool).
`bool(report)` is `True` iff something was fused. `report.summary()` is never empty — silence is
not a result.

### `preview(exported_program, values=None) -> str`

What `apply` *would* do, without touching the graph.

### `op_parameters(name) -> tuple`

The scalar argument names an explicit operator needs. Empty for most.

## The explicit operator library

One function per registered skeleton, generated from the registry: `logsumexp`, `lse_biased`,
`lse_scaled`, `softcap_lse`, `stable_l2`, `sumsq_div`, `linear_attn`, `decay_renorm`,
`rmsnorm_max`, `variance`, `std`. Each handles `(rows, ncols)` and reduces the last dimension.

```python
neptune_opt.rmsnorm_max(x, eps=1e-6)
neptune_opt.variance(x)
neptune_opt.lse_scaled(x, T=8.0)
```

This is the zero-risk path: no recognition, no graph rewriting — you name the operator and the
kernel is generated for it. Useful when recognition declines, and as a check on what recognition
would have chosen.

## Performance fallback

### `compile_best(module, args, *, package_dir=None, margin=0.03, reps=20, rounds=3, verify=True, verbose=True, **export_kwargs) -> Choice`

Every other gate in this package protects **correctness**; this one protects **speed**. It compiles
the unmodified package AND the fused one, times both on `args`, and returns the faster — because a
numerically correct fusion can still be slower (measured: the two-pass plan for `var`/`std` is
**0.30x** at `(1, 262144)`).

```python
choice = neptune_opt.compile_best(model, (x,))
print(choice)                  # "shipped the fused package -- the fused package is 1.54x faster"
ship(choice.package_path)      # or: "shipped the baseline package -- 3.32x SLOWER, so it was dropped"
```

The price is deliberate and must be paid knowingly: **two AOTI compilations**, example inputs, a CUDA
device, and static shapes (an AOTI package is specialised to the shapes it was compiled for, and
those are also the shapes that get timed). It is a separate entry point rather than something
`apply()` does, because `apply()` runs before any compilation exists and so has nothing to time. When
the model has nothing to fuse, the second compilation is skipped — the common case pays nothing extra.

The choice rule: ship the fused package only if it beats the baseline by more than `margin` (3%), and
only after checking that its numbers still match. Slower, tied, numerically different, or failing to
compile all return the **unmodified baseline** with the measurement in `reason`.

### `Choice`

What the A/B measured: `package_path` (the one to ship), `used_fusion`, `baseline_ms`, `fused_ms`,
`speedup` (baseline / fused), `reason`, `report` (the `Report` from `apply`), `baseline_package` and
`fused_package` (both are kept, so the loser stays inspectable). It prints as one line.

## Legacy / gated

### `optimize(module, example_inputs, verbose=False)` — legacy

Returns an optimised module, which the caller then exports. Superseded by `apply`, which avoids the
extra export; emits a `DeprecationWarning`.

### `backend(gm, example_inputs, experimental=False)` — gated

`torch.compile(model, backend=neptune_opt.backend)`. **Does not work effectively**: the rewrite
happens and the results are correct, but the generated kernel does not reach codegen on this route,
so the op falls back to a Python launch. Refuses unless `experimental=True`.

## Also exported

`Fusion` (a fused operator: `skeleton`, `params`, `inferred`, rank), `LAST_REPORT` (the most recent
`Report`, for callers that cannot hold the return value), `__all__` (the exact list above).
