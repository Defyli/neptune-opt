"""Transactional local reduction fusion with algebra-derived CUDA execution plans.

Real-algebra matching is a legality check, not a universal IEEE floating-point or
performance proof. Numerical verification covers supplied cases; compile_best also
compares actual packages. Failed trials never publish their graph or signature.
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import math
import sys
import warnings
from typing import Any

import torch

import neptune_codegen as CG
from neptune_recognize import recognize_regions as _recognize_regions
from neptune_recognize.rewrite import region_contract, rewrite_match
from neptune_opt.ab import Choice, choose_faster as choose_faster, compile_best

__all__ = ["apply", "export", "optimize", "backend", "preview", "Report", "Fusion",
           "LAST_REPORT", "op_parameters", "compile_best", "Choice"]


@dataclasses.dataclass
class Fusion:
    skeleton: str
    params: dict
    input_name: str = ""
    output_name: str = ""
    plan: str = "serial"
    proof_kind: str = ""
    domain_assumptions: tuple = ()

    def __str__(self) -> str:
        if not self.params:
            return self.skeleton
        return f"{self.skeleton}({', '.join(f'{k}={v:g}' for k, v in sorted(self.params.items()))})"


@dataclasses.dataclass
class Report:
    fusions: list = dataclasses.field(default_factory=list)
    skipped: list = dataclasses.field(default_factory=list)
    verified: bool = False
    graph_changed: bool = False
    verification_cases: int = 0

    def __bool__(self) -> bool:
        return bool(self.fusions)

    def summary(self) -> str:
        lines = [f"  fused   : {fusion}" for fusion in self.fusions]
        lines.extend(f"  skipped : {reason}: {detail}" for reason, detail in self.skipped)
        if self.fusions:
            lines.append("  verified: fused output matches the supplied inputs" if self.verified
                         else "  unverified: real-algebra match; numerical verification not executed")
        return "\n".join(lines) or "  (nothing to do)"


LAST_REPORT: Report | None = None


def _outputs(x: Any) -> tuple:
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _clone_inputs(args):
    from torch.utils._pytree import tree_map

    return tree_map(lambda value: value.clone(memory_format=torch.preserve_format)
                    if isinstance(value, torch.Tensor) else value, args)


def _compare_trees(a, b, *, rtol=1e-3, atol=1e-4, max_ulps=None):
    from torch.utils._pytree import tree_flatten

    left, left_spec = tree_flatten(a)
    right, right_spec = tree_flatten(b)
    if left_spec != right_spec:
        return False, "output pytree structure differs"
    for index, (x, y) in enumerate(zip(left, right)):
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            if type(x) is not type(y) or x != y:
                return False, f"non-tensor output {index} differs"
            continue
        if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device:
            return False, f"output {index} shape/dtype/device differs"
        if x.is_floating_point() or x.is_complex():
            if not torch.equal(torch.isfinite(x), torch.isfinite(y)):
                return False, f"output {index} finite classification differs"
            if not torch.equal(torch.isnan(x), torch.isnan(y)):
                return False, f"output {index} NaN classification differs"
            if not x.is_complex() and (not torch.equal(torch.isposinf(x), torch.isposinf(y))
                                      or not torch.equal(torch.isneginf(x), torch.isneginf(y))):
                return False, f"output {index} infinity sign differs"
            try:
                torch.testing.assert_close(x, y, rtol=rtol, atol=atol, equal_nan=True)
            except AssertionError as error:
                if max_ulps is None or x.dtype not in (torch.float16, torch.bfloat16):
                    return False, f"output {index}: {error}"
                finite = torch.isfinite(x) & torch.isfinite(y)
                up = torch.nextafter(x, torch.full_like(x, float("inf")))
                down = torch.nextafter(x, torch.full_like(x, -float("inf")))
                bound = torch.maximum((up.float() - x.float()).abs(), (x.float() - down.float()).abs())
                if not ((x.float() - y.float()).abs()[finite] <= max_ulps * bound[finite]).all():
                    return False, f"output {index}: exceeds explicit {max_ulps}-ULP bound"
        elif not torch.equal(x, y):
            return False, f"output {index} integer/bool values differ"
    return True, ""


def _state_version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors intentionally have no version counter; compare their values.
        return None


def _snapshot_state(tensors):
    unique = {id(tensor): tensor for tensor in tensors if isinstance(tensor, torch.Tensor)}
    return [(tensor, _state_version(tensor), tensor.detach().clone()) for tensor in unique.values()]


def _state_mutated(snapshot):
    return any(_state_version(tensor) != version or not _compare_trees(tensor, saved, rtol=0, atol=0)[0]
               for tensor, version, saved in snapshot)


def _restore_state(snapshot):
    with torch.no_grad():
        for tensor, version, saved in snapshot:
            if _state_version(tensor) != version or not _compare_trees(tensor, saved, rtol=0, atol=0)[0]:
                tensor.copy_(saved)


def _verify(module: torch.nn.Module, candidate: torch.nn.Module, args: tuple,
            *, rtol=1e-3, atol=1e-4, max_ulps=None) -> tuple[bool, str]:
    """Differential execution with read-only state; invoke serially on exclusive modules."""
    state = _snapshot_state([value for mod in (module, candidate)
                             for value in (*mod.parameters(), *mod.buffers())])
    try:
        with torch.no_grad():
            left, right = _clone_inputs(args), _clone_inputs(args)
            pristine_left, pristine_right = _clone_inputs(left), _clone_inputs(right)
            a = module(*left)
            if _state_mutated(state):
                return False, "reference mutated parameters or buffers"
            b = candidate(*right)
            if _state_mutated(state):
                return False, "candidate mutated parameters or buffers"
            for pristine, current in ((pristine_left, left), (pristine_right, right)):
                ok, detail = _compare_trees(pristine, current, rtol=0, atol=0)
                if not ok:
                    return False, f"inputs mutated: {detail}"
            return _compare_trees(a, b, rtol=rtol, atol=atol, max_ulps=max_ulps)
    except Exception as error:  # noqa: BLE001 -- numerical execution is a refusal
        return False, f"{type(error).__name__}: {error}"
    finally:
        _restore_state(state)


def _trace(module, args):
    from torch.fx.experimental.proxy_tensor import make_fx

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            return make_fx(module)(*args)
    finally:
        module.train(was_training)


def _graph_trial(gm):
    # Share potentially huge state tensors, copying only functional graph structure.
    trial = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
    trial.meta = copy.copy(gm.meta)
    if "neptune_replaced_roots" in trial.meta:
        trial.meta["neptune_replaced_roots"] = copy.deepcopy(trial.meta["neptune_replaced_roots"])
    return trial


def _rewrite_regions(gm, values, *, plan="serial", max_regions=None, lower=False,
                     experimental_log_softmax=False):
    report = Report()
    limit = len(list(gm.graph.nodes)) if max_regions is None else max_regions
    for _ in range(limit):
        matches, reason = _recognize_regions(gm, values)
        eligible = []
        for match in matches:
            _, refusal = region_contract(match)
            if refusal is None:
                eligible.append(match)
            elif not any(detail == refusal for _, detail in report.skipped):
                report.skipped.append(("refused", refusal))
        if not eligible and lower:
            from neptune_recognize.lowering import _is_compound, lower_compound, _commit
            for node in list(gm.graph.nodes):
                if not _is_compound(node):
                    continue
                if "log_softmax" in str(node.target) and not experimental_log_softmax:
                    continue
                tentative = _graph_trial(gm)
                if not lower_compound(tentative, nodes=node.name):
                    continue
                lowered_matches, _ = _recognize_regions(tentative, values)
                chosen = next((match for match in lowered_matches
                               if match.region_root.meta.get("neptune_lowering") == node.name
                               and region_contract(match)[1] is None), None)
                if chosen is None:
                    continue
                fusion = Fusion(chosen.skeleton, dict(chosen.params), chosen.input_node.name,
                                chosen.region_root.name, plan, chosen.proof_kind, chosen.domain_assumptions)
                call, refusal = rewrite_match(tentative, chosen, plan=plan)
                if call is None:
                    raise RuntimeError(refusal)
                # The lowering API commits only graph structure. Track its output
                # separately, including squeeze/cast outputs outside the fused root.
                replacements = copy.deepcopy(gm.meta.get("neptune_replaced_roots", {}))
                replacements.update(tentative.meta.get("neptune_replaced_roots", {}))
                old_users = {user.name for user in node.users}
                for user in tentative.graph.nodes:
                    if user.name in old_users:
                        old_user = next(original for original in gm.graph.nodes
                                        if original.name == user.name)
                        old_leaves, _ = torch.utils._pytree.tree_flatten((old_user.args, old_user.kwargs))
                        new_leaves, _ = torch.utils._pytree.tree_flatten((user.args, user.kwargs))
                        for old_leaf, new_leaf in zip(old_leaves, new_leaves):
                            if old_leaf is node and isinstance(new_leaf, torch.fx.Node):
                                replacements[node.name] = new_leaf.name
                # Lowering and fusion of this marked region are one transaction. An
                # existing region cannot consume its budget and leave it expanded.
                _commit(gm, tentative)
                gm.meta["neptune_replaced_roots"] = replacements
                report.fusions.append(fusion)
                break
            else:
                if not report.fusions and not report.skipped:
                    report.skipped.append(("no-fusion", reason))
                break
            continue
        if not eligible:
            if not report.fusions and not report.skipped:
                report.skipped.append(("no-fusion", reason))
            break
        match = eligible[0]
        fusion = Fusion(match.skeleton, dict(match.params), match.input_node.name,
                        match.region_root.name, plan, match.proof_kind, match.domain_assumptions)
        call, refusal = rewrite_match(gm, match, plan=plan)
        if call is None:
            raise RuntimeError(refusal)
        report.fusions.append(fusion)
        # Re-discover after each mutation; matches may overlap or refer to deleted nodes.
    report.graph_changed = bool(report.fusions)
    return report


def _rewrite_graph(gm, values=None, *, plan="serial", max_regions=None):
    """Return a separate candidate on success, the original graph on any failure."""
    try:
        trial = _graph_trial(gm)
        report = _rewrite_regions(trial, values, plan=plan, max_regions=max_regions)
        return (trial if report.fusions else gm), report
    except Exception as error:  # noqa: BLE001
        return gm, Report(skipped=[("rewrite-failed", f"{type(error).__name__}: {error}")])


def _candidate_program(exported, trial):
    """Prepare all contracts before the public constructor's first validation."""
    from torch.export.graph_signature import ConstantArgument
    from torch.utils._pytree import tree_flatten

    signature = copy.deepcopy(exported.graph_signature)
    calls = copy.deepcopy(exported.module_call_graph)
    nodes = {node.name: node for node in trial.graph.nodes}
    replacements = copy.deepcopy(trial.meta.get("neptune_replaced_roots", {}))

    def resolve(name):
        # A surviving value has not been replaced; stale mappings from earlier
        # transactions must never rename graph/user inputs or live intermediates.
        if name in nodes:
            return name
        seen = set()
        while name not in nodes:
            if name in seen:
                raise RuntimeError("cyclic rewritten root mapping")
            seen.add(name)
            if name not in replacements:
                raise RuntimeError(f"module signature references removed intermediate {name}")
            name = replacements[name]
        return name

    renamed = {name: resolve(name) for name in replacements if name not in nodes}
    output = next(node for node in trial.graph.nodes if node.op == "output")
    leaves, _ = tree_flatten(output.args[0])
    if len(leaves) != len(signature.output_specs):
        raise RuntimeError("rewrite changed exported output arity")
    for index, (spec, leaf) in enumerate(zip(signature.output_specs, leaves)):
        if isinstance(spec.arg, ConstantArgument):
            if isinstance(leaf, torch.fx.Node) or type(leaf) is not type(spec.arg.value) or leaf != spec.arg.value:
                raise RuntimeError("rewrite changed a constant graph output")
        elif hasattr(spec.arg, "name"):
            if not isinstance(leaf, torch.fx.Node):
                raise RuntimeError("rewrite changed a graph output into a constant")
            if spec.arg.name != leaf.name:
                if spec.arg.name in nodes or renamed.get(spec.arg.name) != leaf.name:
                    raise RuntimeError("rewrite changed output without a root replacement")
            signature.output_specs[index] = dataclasses.replace(
                spec, arg=dataclasses.replace(spec.arg, name=leaf.name))
    # Graph inputs (including user argument names) are never renamed. Module
    # inputs may reference an internal root, unlike root graph input_specs.
    for spec in signature.input_specs:
        if hasattr(spec.arg, "name") and spec.arg.name not in nodes:
            raise RuntimeError("rewrite removed an exported graph input")
    for entry in calls:
        if entry.signature is None:
            continue
        for field in ("inputs", "outputs"):
            arguments = getattr(entry.signature, field)
            setattr(entry.signature, field, [dataclasses.replace(arg, name=resolve(arg.name))
                    if hasattr(arg, "name") and not isinstance(arg, ConstantArgument)
                    else arg for arg in arguments])
    kwargs = dict(root=trial, graph=trial.graph, graph_signature=signature,
                  state_dict=exported.state_dict,
                  range_constraints=copy.deepcopy(exported.range_constraints),
                  module_call_graph=calls, example_inputs=exported.example_inputs,
                  constants=exported.constants)
    parameters = inspect.signature(torch.export.ExportedProgram).parameters
    if "verifiers" in parameters:
        kwargs["verifiers"] = list(exported.verifiers)
    elif "verifier" in parameters:
        kwargs["verifier"] = exported.verifier
    return torch.export.ExportedProgram(**kwargs)


def apply(exported_program, values: dict | None = None, verify_with: tuple | None = None,
          lower: bool = True, *, plan="serial", max_regions=None,
          verification_inputs=None, rtol=1e-3, atol=1e-4,
          experimental_log_softmax=False, max_ulps=None) -> Report:
    """Transactionally fuse local regions in an existing ExportedProgram.

    Unknown operations, casts and other inputs are preserved boundaries. Optional
    compound lowering defaults to eligible LSE only; experimental_log_softmax enables
    the cancellation-prone x-LSE readout with supplied-case verification, not a full-domain
    guarantee. Lowering never decomposes SDPA, softmax,
    LayerNorm or the global decomposition table. Any generation, signature or numerical
    failure leaves the original graph and parameters untouched.

    Without verify_with the result is explicitly numerically UNVERIFIED. Real-algebra
    matching is not an IEEE proof. plan='split' is explicit and experimental; serial is
    the default. Verification covers supplied cases, not the whole dynamic range.
    """
    from neptune_codegen.template import validate_plan

    global LAST_REPORT
    report = Report()
    try:
        validate_plan(plan)
        if max_regions is not None and (type(max_regions) is not int or max_regions <= 0):
            raise ValueError("max_regions must be a positive integer or None")
        if any(not math.isfinite(tolerance) or tolerance < 0 for tolerance in (rtol, atol)):
            raise ValueError("verification tolerances must be finite and nonnegative")
        if max_ulps is not None and (type(max_ulps) is not int or max_ulps < 0):
            raise ValueError("max_ulps must be a nonnegative integer or None")
        signature = exported_program.graph_signature
        if signature.buffers_to_mutate or signature.user_inputs_to_mutate:
            raise ValueError("state/input mutation is outside inference reduction fusion")
        trial = _graph_trial(exported_program.graph_module)
        report = _rewrite_regions(trial, values, plan=plan, max_regions=max_regions, lower=lower,
                                  experimental_log_softmax=experimental_log_softmax)
        if not report.fusions:
            LAST_REPORT = report
            return report
        candidate = _candidate_program(exported_program, trial)
        if verify_with is not None:
            module, args = verify_with
            cases = [tuple(args)]
            if verification_inputs is not None:
                cases.extend(tuple(case) for case in verification_inputs)
            executable = candidate.module()
            for index, case in enumerate(cases):
                # A user reference can capture original state in a Python closure rather
                # than register it as a submodule. Protect the EP's tensors too.
                state = _snapshot_state(exported_program.state_dict.values())
                try:
                    if rtol == 1e-3 and atol == 1e-4 and max_ulps is None:
                        ok, detail = _verify(module, executable, case)
                    else:
                        ok, detail = _verify(module, executable, case, rtol=rtol, atol=atol,
                                             max_ulps=max_ulps)
                    if _state_mutated(state):
                        ok, detail = False, "verification mutated original exported state"
                finally:
                    _restore_state(state)
                if not ok:
                    report = Report(skipped=[("verify-failed", f"case {index}: {detail}")],
                                    verification_cases=index + 1)
                    LAST_REPORT = report
                    return report
            report.verified, report.verification_cases = True, len(cases)
        # All fallible graph work is finished. Publishing object references does not
        # run generated code or expose a partially rewritten original graph.
        exported_program._graph_module = candidate.graph_module
        exported_program._graph_signature = candidate.graph_signature
        exported_program._module_call_graph = candidate.module_call_graph
        report.graph_changed = True
    except Exception as error:  # noqa: BLE001 -- optional optimization fails closed
        reason = "triton-missing" if isinstance(error, ModuleNotFoundError) and "triton" in str(error) else "rewrite-failed"
        report = Report(skipped=[(reason, f"{type(error).__name__}: {error}")])
    LAST_REPORT = report
    return report


def preview(exported_program, values: dict | None = None, *, lower=True,
            experimental_log_softmax=False):
    """Analyze apply's local regions/targeted lowering without generating kernels."""
    from neptune_recognize.lowering import _is_compound, lower_compound, _commit

    trial = _graph_trial(exported_program.graph_module)
    if lower:
        for node in list(trial.graph.nodes):
            if not _is_compound(node) or ("log_softmax" in str(node.target) and not experimental_log_softmax):
                continue
            tentative = _graph_trial(trial)
            if not lower_compound(tentative, nodes=node.name):
                continue
            candidates, _ = _recognize_regions(tentative, values)
            if any(match.region_root.meta.get("neptune_lowering") == node.name
                   and region_contract(match)[1] is None for match in candidates):
                _commit(trial, tentative)
    matches, reason = _recognize_regions(trial, values)
    eligible = [match for match in matches if region_contract(match)[1] is None]
    if matches and not eligible:
        reason = region_contract(matches[0])[1]
    return eligible, reason


def export(module, example_inputs, decompose=True, **export_kwargs):
    """Export then apply; core decomposition is explicit and is not an integration hook."""
    ep = torch.export.export(module, tuple(example_inputs), **export_kwargs)
    if decompose:
        ep = ep.run_decompositions()
    apply(ep, verify_with=(module, tuple(example_inputs)))
    return ep


def op_parameters(name: str) -> tuple:
    sk = CG.SKELETONS[name]
    declared = {sym for sym, _ in sk.defaults}
    return tuple(arg for sym, arg in sk.params if sym not in declared)


def _make_op(name):
    def op(x, *, plan="serial", **params):
        missing = [parameter for parameter in op_parameters(name) if parameter not in params]
        if missing:
            raise TypeError(f"neptune_opt.{name}() needs {', '.join(missing)}")
        values = {**CG.param_defaults(name), **params}
        return CG.op_handle(name, params=values or None, plan=plan)(x)

    op.__name__ = name
    op.__qualname__ = name
    op.__doc__ = f"Algebra-derived {name}; explicit serial/split CUDA plan."
    return op


for _op_name in CG.supported():
    globals()[_op_name] = _make_op(_op_name)
__all__ += [name for name in CG.supported()]


def optimize(module, example_inputs, *, verbose=False):
    """Legacy module-returning form; a failed candidate returns the original module."""
    warnings.warn("neptune_opt.optimize() is superseded by apply()/export()",
                  DeprecationWarning, stacklevel=2)
    try:
        ep = torch.export.export(module, tuple(example_inputs))
    except Exception as error:  # noqa: BLE001
        return module, Report(skipped=[("export-failed", f"{type(error).__name__}: {error}")])
    report = apply(ep, verify_with=(module, tuple(example_inputs)))
    if verbose:
        print(report.summary(), file=sys.stderr)
    return (ep.module() if report.fusions else module), report


def backend(gm, example_inputs, experimental=False):
    """torch.compile route remains gated until its actual generated kernels are verified."""
    if not experimental:
        raise RuntimeError("neptune_opt.backend is experimental; use export/AOTI, or pass experimental=True")
    import torch._inductor

    global LAST_REPORT
    candidate, LAST_REPORT = _rewrite_graph(gm)
    return torch._inductor.compile(candidate, list(example_inputs))
