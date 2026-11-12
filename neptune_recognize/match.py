"""
Which registered skeleton does this graph implement? Decided by comparing expressions.
"""
from __future__ import annotations

import dataclasses

import sympy as sp

import neptune_codegen as CG
from neptune_codegen import c_sym, vanishes
from neptune_recognize.interp import (S_RESULT, _INTRINSIC_SYMBOLS, _IGNORED,
                                      interpret, region_inputs, _tensor_value, _flatten_nodes, _op)
from neptune_recognize.params import _solve_for


def _intrinsic_substitution(sk):
    """Replace a skeleton's INTRINSIC parameters with the canonical graph symbol.

    An intrinsic parameter is one the kernel reads at run time (`N` is `ncols`). There is no
    value to solve for, so instead of trying to produce a number the matcher rewrites the
    pattern to use the same symbol the graph uses -- which is what makes the operator work
    when the row length is symbolic and no number exists.
    """
    out = {}
    for sym, name in getattr(sk, "intrinsic_params", ()):
        canon = _INTRINSIC_SYMBOLS.get(name)
        if canon is not None:
            out[sym] = canon
    return out


def _producer_pattern(sk):
    """The skeleton's phi: the expression its max-like reduction reduces over."""
    derived = CG.derived_producer_expr(sk)
    if derived is not None:
        return derived
    return sp.Abs(c_sym) if sk.producer != "signed" else c_sym


def _solve_params(sk, target, expr, solved, supplied):
    """Solve the skeleton's not-yet-known scalars so that `target == expr`.

    Returns the extended dict, or None if some parameter cannot be pinned down. Parameters
    are solved ONE AT A TIME for the same reason as elsewhere: handing `sp.solve` an equation
    with more unknowns than it can determine routes it into `_solve_undetermined`.
    """
    unknowns = [sym for sym, arg in sk.params
                if arg not in (supplied or {}) and sym not in solved]
    local = dict(solved)
    if not unknowns:
        return local
    d = sp.simplify(target.subs(local) - expr)
    for u in list(unknowns):
        if not d.has(u):
            continue
        val = _solve_for(d, u)
        if val is None:
            return None
        local[u] = val
        d = sp.simplify(d.subs({u: val}))
    return local


def _find_region_root(gm, env, sk, cand, solved, supplied, *, exact=False):
    """Locate the node computing the skeleton's `finalize`, i.e. the operator's edge.

    This is what makes the pass usable inside a real model, and it is also a correctness fix:
    the previous guard only required the tail to be free of the data symbol, so `l2 * 2.0`
    reported the OUTER multiply as the region root and a rewrite would have silently dropped
    the ×2. Requiring the tail to equal the declared `finalize` pins the boundary exactly --
    the inner multiply matches, the outer one does not, and the ×2 is left alone.

    The last matching node in topological order is taken, so the region is as large as the
    operator's own algebra allows and no larger.
    """
    from neptune_codegen import _FINAL_ACC, _FINAL_M

    # NB: the graph's OWN producer symbol, not the canonical S_RESULT used for matching
    # phi/psi. Using S_RESULT here silently excluded every node mentioning the producer.
    target = (sk.finalize.xreplace({_FINAL_ACC: cand.consumer_sym,
                                    _FINAL_M: cand.producer_sym})
              .subs(_intrinsic_substitution(sk)).subs(solved))
    # Supplied parameters are part of the finalize too, not just phi/psi.
    target = target.subs({sym: sp.Rational(str(float(supplied[arg])))
                         for sym, arg in sk.params if arg in (supplied or {})})
    best = None
    for node in gm.graph.nodes:
        expr = env.get(node)
        if expr is None or not expr.has(cand.consumer_sym):
            continue
        local = _solve_params(sk, target, expr, solved, supplied)
        if local is None:
            continue
        residual = target.subs(local) - expr
        if sp.simplify(residual) == 0 if exact else vanishes(residual):
            best = (node, local)
    return best if best else (None, None)


@dataclasses.dataclass
class Match:
    """A recognised operator TOGETHER with the parameter values it implies.

    Carrying the values matters: recognition infers `beta = 0.5` from the graph, and the
    generated kernel has to be built with that 0.5. Returning only a name would leave the
    caller to guess, which is exactly the silent-wrong-constant failure this project keeps
    finding. `inferred` lists the arguments that were discovered rather than supplied.
    """

    input_node: object
    skeleton: str
    params: dict
    inferred: tuple = ()
    # the graph node that computes the consumer reduction, and the symbol standing for its
    # result. `rewrite` needs both: the node to bound the region, the symbol to tell which
    # downstream expressions still belong to the operator.
    consumer_node: object = None
    consumer_sym: object = None
    producer_sym: object = None
    # the node computing the operator's FINAL value (the `finalize` tail). Replacing this,
    # rather than the graph output, keeps whatever the surrounding model does with the
    # result.
    region_root: object = None
    region_nodes: tuple = ()
    proof_kind: str = "legacy-symbolic-or-numeric-filter"
    domain_assumptions: tuple = ()

    def __repr__(self) -> str:
        extra = f" params={self.params}" + (f" (inferred: {', '.join(self.inferred)})"
                                            if self.inferred else "")
        return f"<{self.skeleton}{extra}>"


def _match_candidate(gm, input_node, cand, values, env, *, exact=False):
    phi, psi = cand.phi, cand.psi
    prod_kind, cons_kind = cand.prod_kind, cand.cons_kind
    consumer_node = cand.consumer_node

    equivalent = (lambda expression: sp.simplify(expression) == 0) if exact else vanishes
    tried = []
    matches = []
    for name, sk in sorted(CG.SKELETONS.items(), key=lambda entry: (len(entry[1].params), entry[0])):
        if name in CG.NEGATIVE_CONTROLS:
            continue
        if sk.producer_reducer != prod_kind or sk.consumer_reducer != cons_kind:
            tried.append(f"{name}(reducer {prod_kind}/{cons_kind})")
            continue
        vals = (values or {}).get(name, {})
        intrinsic = _intrinsic_substitution(sk)
        phi_pat = _producer_pattern(sk).subs(intrinsic)
        psi_pat = CG.term_expr(sk).xreplace({CG.s2_sym: S_RESULT}).subs(intrinsic)

        # (a) substitute whatever the caller supplied
        known_args = {arg: float(vals[arg]) for sym, arg in sk.params if arg in vals}
        known = {sym: sp.Rational(str(float(vals[arg]))) for sym, arg in sk.params if arg in vals}
        unknown = [sym for sym, arg in sk.params
                   if arg not in vals and sym not in intrinsic]
        phi_pat = phi_pat.subs(known)
        psi_pat = psi_pat.subs(known)

        # (b) infer the rest from the graph, then re-check BOTH expressions.
        # Solve INCREMENTALLY, one side at a time. The equations must come from whichever
        # side actually mentions the parameter (`N` in g = c/(N+s) never appears in the
        # producer), and handing sp.solve the whole coupled system makes it return [] even
        # when the first equation alone pins the parameter down.
        # Initialized OUTSIDE the branch. It used to be bound only inside `if unknown:`, so a
        # skeleton with no non-intrinsic scalar parameters -- `std` and `variance` are exactly
        # that -- reached `_find_region_root(..., solved, ...)` with the name unbound and raised
        # UnboundLocalError. The bug was dormant only because no param-free sum/sum skeleton
        # matched a graph yet; it fires the moment one does. Declaring it here also removes a
        # second hazard: on the old code path a dict solved for an EARLIER skeleton could
        # survive into the next loop iteration, so the tail check could see another operator's
        # parameter values.
        solved: dict = {}
        if unknown:
            remaining = list(unknown)
            for target, graph_side in ((phi_pat, phi), (psi_pat, psi)):
                d = sp.simplify(target.subs(solved) - graph_side)
                progress = True
                while remaining and progress:
                    progress = False
                    for u in list(remaining):
                        if not d.has(u):
                            continue
                        # one parameter at a time, symbolic then numeric. Handing
                        # sp.solve a coupled system returns [] and can crash on
                        # `_solve_undetermined`.
                        val = _solve_for(d, u)
                        if val is None:
                            continue
                        solved[u] = val
                        remaining.remove(u)
                        d = sp.simplify(d.subs({u: val}))
                        progress = True
                        break
            if remaining:
                # A parameter that appears ONLY in the finalize cannot be solved from
                # phi/psi -- the log-family's additive constant is exactly that case. Leave
                # it to `_find_region_root`, which compares against the operator's tail.
                d_phi = sp.simplify(phi_pat.subs(solved) - phi)
                d_psi = sp.simplify(psi_pat.subs(solved) - psi)
                relevant = [u for u in remaining if d_phi.has(u) or d_psi.has(u)]
                if relevant:
                    tried.append(f"{name}(param-unsolved {relevant})")
                    continue
            phi_pat = phi_pat.subs(solved)
            psi_pat = psi_pat.subs(solved)

        if equivalent(phi_pat - phi) and equivalent(psi_pat - psi):
            # A zero-singular power kernel uses the documented guarded coordinate. Do
            # not introduce that guard into an unguarded division or erase another clamp.
            if CG.coordinate_needs_floor(sk) != cand.producer_guarded:
                tried.append(f"{name}(producer-coordinate-contract-differs)")
                continue
            # The tail must equal the declared `finalize`; otherwise this is not the
            # operator. This gate runs BEFORE the match is accepted.
            root, tail_solved = _find_region_root(gm, env, sk, cand, solved, known_args,
                                                  exact=exact)
            if root is None:
                tried.append(f"{name}(no-finalize)")
                continue
            all_solved = {**solved, **tail_solved}
            merged = {**known_args, **{arg: float(val) for sym, arg in sk.params
                                       for key, val in all_solved.items() if key == sym}}
            inferred = tuple(sorted(arg for sym, arg in sk.params if sym in all_solved))
            if exact:
                # The emitted scalar is a Python float literal. Certify THAT binding,
                # not a higher-precision root which is subsequently rounded away.
                bindings = {sym: sp.Rational(str(merged[arg])) for sym, arg in sk.params
                            if arg in merged}
                if len(bindings) != len(sk.params):
                    continue
                if not equivalent(_producer_pattern(sk).subs(intrinsic).subs(bindings) - phi):
                    continue
                if not equivalent(CG.term_expr(sk).xreplace({CG.s2_sym: S_RESULT}).subs(intrinsic).subs(bindings) - psi):
                    continue
                actual_tail = sk.finalize.xreplace({CG._FINAL_ACC: cand.consumer_sym,
                                                    CG._FINAL_M: cand.producer_sym})
                if not equivalent(actual_tail.subs(intrinsic).subs(bindings) - env[root]):
                    continue
                # Any exact legal skeleton implements this same region. Prefer the
                # first minimal-parameter one instead of solving all unrelated families.
                return Match(input_node, name, merged, inferred, consumer_node,
                             cand.consumer_sym, cand.producer_sym, root), "matched"
            matches.append((len(unknown), name, merged, inferred, root))

    if matches:
        # An exact match (no inference needed) beats one that had to invent parameter
        # values; among equals, the alphabetical order is stable and predictable.
        matches.sort(key=lambda m: (m[0], m[1]))
        _n, name, merged, inferred, root = matches[0]
        return Match(input_node, name, merged, inferred, consumer_node,
                     cand.consumer_sym, cand.producer_sym, root), "matched"

    return None, "no registered skeleton matches (tried: " + ", ".join(tried) + ")"


def _region_nodes(root, anchor):
    """Collect the full dependency closure up to one tensor boundary.

    Nothing outside the closure is erased, and other tensor/parameter inputs cannot
    masquerade as scalar constants. Opaque computation before anchor remains intact.
    """
    region, frontier = set(), [root]
    while frontier:
        node = frontier.pop()
        if node is anchor or node in region:
            continue
        if node.op in ("placeholder", "get_attr", "output"):
            return None
        region.add(node)
        frontier.extend(node.all_input_nodes)
    return tuple(region)


def recognize_regions(gm, values=None):
    """Discover exact single-row regions inside a multi-input/opaque surrounding DAG.

    All primitive semantics are shared with strict interpretation. Unknown nodes are
    boundaries, never identity operations. A candidate remains one data symbol, even
    when its preserved producer is a GEMM or combines several model inputs.
    """
    values = values or {}
    by_root = {}
    reasons = []
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    anchors = region_inputs(gm)
    pair_count = 0
    for anchor in anchors:
        why = []
        info = interpret(gm, why=why, input_node=anchor, local=True)
        if info is None:
            reasons.extend(why)
            continue
        _, candidates, env = info
        pair_count += len(candidates)
        for candidate in candidates:
            if not candidate.phi.has(c_sym) or not candidate.psi.has(c_sym):
                continue
            match, reason = _match_candidate(gm, anchor, candidate, values, env, exact=True)
            if match is None:
                reasons.append(reason)
                continue
            root = match.region_root
            closure = _region_nodes(root, anchor)
            if closure is None:
                reasons.append("a matched region has more than one data boundary")
                continue
            if candidate.producer_node not in closure or candidate.consumer_node not in closure:
                continue
            # Removing low-precision intermediate rounding needs separate, explicit
            # certification. Casts are boundaries; genuine fp32 regions are discoverable
            # inside a bf16 graph without silently deleting those casts.
            source_value = _tensor_value(anchor)
            region_values = [_tensor_value(node) for node in closure]
            if source_value is not None:
                if any(value is not None and hasattr(value, "dtype") and value.is_floating_point()
                       and value.dtype != source_value.dtype for value in region_values):
                    reasons.append("mixed-dtype arithmetic remains a region boundary")
                    continue
            params = {**CG.param_defaults(match.skeleton), **match.params}
            try:
                if not CG.SKELETONS[match.skeleton].two_pass:
                    certificate = CG.derive_merge_certificate(CG.SKELETONS[match.skeleton], params)
                    assumptions = certificate.assumptions
                else:
                    assumptions = ("finite real inputs and finite reachable arithmetic", "ncols > 0")
            except (NotImplementedError, KeyError, ValueError) as error:
                reasons.append(str(error))
                continue
            match.region_nodes = tuple(sorted(closure, key=order.__getitem__))
            match.proof_kind = "exact-symbolic-real-algebra"
            match.domain_assumptions = assumptions
            previous = by_root.get(root)
            if previous is None or len(match.region_nodes) > len(previous.region_nodes):
                by_root[root] = match
    matches = sorted(by_root.values(), key=lambda match: order[match.region_root])
    if matches:
        return matches, f"matched {len(matches)} local reduction region(s)"
    detail = reasons[-1] if reasons else "no dependent reduction pair"
    compounds = sorted({str(node.target) for node in gm.graph.nodes if node.op == "call_function"
                        and str(node.target).startswith(("aten.logsumexp.", "aten.log_softmax.",
                                                         "aten._log_softmax."))})
    if compounds:
        detail += "; compound operations require targeted lowering: " + ", ".join(compounds)
    return [], (f"no certified local region: analyzed {len(anchors)} row-data boundaries, "
                f"{pair_count} dependent pair(s); {detail}")


def recognize(gm, values=None):
    """Which registered skeleton does this graph implement?

    Returns (Match | None, reason). `values` optionally supplies scalar parameter values by
    argument name; parameters NOT supplied are inferred from the graph, so an ALiBi bias or
    a soft-cap beta is discovered rather than configured.
    """
    values = values or {}
    why: list = []
    info = interpret(gm, why=why)
    if info is None:
        return None, why[0] if why else "graph is not a producer/consumer reduction pair"
    input_node, candidates, env = info

    for cand in candidates:
        m, why = _match_candidate(gm, input_node, cand, values, env)
        if m is not None:
            return m, "matched"
    return None, "no registered skeleton matches (" + str(len(candidates)) + " candidate pair(s))"


def region_root(gm, match, env):
    """The graph node that computes the operator's FINAL value, or None if unsafe.

    The recognised pair (producer, consumer) only says how the reduction is built. The
    operator's tail -- the `finalize` that turns (accumulator, producer) into the result --
    is a pointwise chain after the consumer reduction, and a rewrite must replace THAT
    value, never the whole graph output.

    The tail is identified by a property of the algebra rather than by a pattern list: after
    the consumer reduction the value no longer depends on the data symbol `c`, only on the
    two reduction results. Walking forward from the consumer over nodes whose expression
    mentions nothing else therefore stops exactly at the operator's boundary -- and if the
    result is then combined with real data (a residual add, a matmul) that node mentions `c`
    and is excluded, so those graphs are refused instead of silently having the rest of the
    model deleted.

    That last point is the reason this function exists: the previous rewrite replaced the
    graph OUTPUT with the fused call, which for `add(fused_result, residual)` would have
    discarded the residual.
    """
    # The graph's OWN symbols, not the canonical S_RESULT used for pattern matching:
    # `env` holds `$r0`/`$r1`, and using S_RESULT here rejected every node that mentioned
    # the producer (measured: `mul` came out as `$r0*sqrt($r1)`, so the whole tail was
    # excluded and the gate refused a perfectly good whole-graph operator).
    if getattr(match, "region_root", None) is not None:
        return match.region_root
    allowed = {match.producer_sym, match.consumer_sym}
    region = {match.consumer_node}
    frontier = [match.consumer_node]
    while frontier:
        node = frontier.pop()
        for user in node.users:
            if user in region:
                continue
            expr = env.get(user)
            if expr is None or not expr.free_symbols <= allowed:
                continue
            region.add(user)
            frontier.append(user)

    out = next((n for n in reversed(list(gm.graph.nodes)) if n.op == "output"), None)
    if out is None or not out.args:
        return None
    # The output node's arguments are a nested structure, and its shape differs between
    # tracing paths: make_fx puts the tensor node directly in args[0], a Dynamo graph puts a
    # TUPLE there. Assuming the former meant the walk never started and every torch.compile
    # fusion was refused as "embedded" while the identical make_fx graph was accepted.
    leaves = []
    _flatten_nodes(out.args, leaves)
    if len(leaves) != 1:
        return None                      # several outputs: not a single operator
    val = leaves[0]
    while (hasattr(val, "op") and val.op in ("call_function", "call_method")
           and _op(val) in _IGNORED):
        nxt = [a for a in val.args if hasattr(a, "op")]
        if len(nxt) != 1:
            break
        val = nxt[0]
    return val if val in region else None
