"""
The API reference must match the surface it documents.

A stale limits section already bit this repository once: the README still described variance as
unreachable after the two-pass plan made it work, and mis-attributed why LayerNorm is out of frame.
Documentation that contradicts the code is worse than no documentation, so the public surface is
asserted here rather than maintained by hand.
"""
from __future__ import annotations

import ast
import inspect
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neptune_opt as NO  # noqa: E402

_DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "docs", "api.md")
_README = Path(__file__).resolve().parents[1] / "README.md"


def _prefixed() -> set:
    """Names written as `neptune_opt.<name>` -- the doc saying "this is the API"."""
    text = open(_DOC, encoding="utf-8").read()
    return {n for n in re.findall(r"neptune_opt\.([A-Za-z_][A-Za-z0-9_]*)", text)
            if not n.startswith("__")}


def _documented() -> set:
    """Names the doc mentions at all: the prefixed form, plus anything in backticks.

    Deliberately lenient in this direction. The strict direction is `_prefixed()`: a phantom
    `neptune_opt.thing` is a broken promise, while a bare `` `Report` `` in prose is just prose,
    and requiring the prefix everywhere would make the document unreadable.
    """
    text = open(_DOC, encoding="utf-8").read()
    ticked = {n for n in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)", text)
              if not n.startswith("__")}
    return _prefixed() | ticked


def test_the_api_doc_exists_and_is_not_empty():
    assert os.path.exists(_DOC), _DOC
    assert len(open(_DOC, encoding="utf-8").read()) > 500


def test_every_public_name_is_documented():
    """`__all__` is the contract; the doc has to cover all of it."""
    missing = sorted(set(NO.__all__) - _documented())
    assert not missing, (
        f"public names missing from docs/api.md: {missing}. Either document them or take them "
        f"out of __all__."
    )


def test_the_doc_does_not_promise_names_that_do_not_exist():
    """The other direction: the doc must not mention a `neptune_opt.x` that is not exported.

    The operator names are the interesting case -- the doc lists them by hand, so a skeleton
    renamed in the registry would leave the doc advertising a function nobody can call.
    """
    import neptune_codegen as CG

    # `neptune_opt.<op>` functions are generated from the registry, so a name is only real if it
    # is a supported skeleton.
    known = set(NO.__all__) | set(CG.supported())
    unknown = sorted(n for n in _prefixed() if n not in known)
    assert not unknown, f"docs/api.md promises non-existent neptune_opt.{unknown}"


def test_the_explicit_op_library_covers_every_supported_skeleton():
    """The doc claims one function per registered skeleton; that is generated, so assert it."""
    import neptune_codegen as CG

    for name in CG.supported():
        assert hasattr(NO, name), f"neptune_opt.{name} is missing but {name} is supported"
        assert name in NO.__all__, f"neptune_opt.{name} is not exported"
    for name in CG.NEGATIVE_CONTROLS:
        assert name not in NO.__all__, f"{name} is a negative control and must not be exported"


def test_documented_entry_points_are_callable():
    for name in ("apply", "export", "preview", "op_parameters", "optimize", "backend"):
        assert callable(getattr(NO, name)), name


def test_readme_names_resolve_to_real_package_attributes():
    import neptune_codegen as CG

    text = _README.read_text(encoding="utf-8")
    modules = {"neptune_opt": NO, "neptune_codegen": CG}
    for package, name in re.findall(r"\b(neptune_opt|neptune_codegen)\.([A-Za-z_]\w*)", text):
        assert hasattr(modules[package], name), f"README promises missing {package}.{name}"


def test_readme_operator_tables_match_both_registries():
    import neptune_codegen as CG

    text = _README.read_text(encoding="utf-8")
    section = text.split("## What it fuses\n", 1)[1].split("\n## ", 1)[0]
    scalar, structured = section.split("### Structured specifications\n", 1)
    scalar_names = re.findall(r"^\|\s*`([A-Za-z_]\w*)`\s*\|", scalar, re.M)
    assert len(scalar_names) == len(set(scalar_names)), "duplicate README scalar row"
    assert set(scalar_names) == set(CG.supported())
    rows = re.findall(r"^\|\s*`([A-Za-z_]\w*)`\s*\|\s*`([A-Za-z_]\w*)`\s*\|",
                      structured, re.M)
    assert len(rows) == len(dict(rows)), "duplicate README structured row"
    assert dict(rows) == {name: spec.operation for name, spec in CG.STATE_REDUCTIONS.items()}


def test_readme_python_examples_compile_and_use_current_signatures():
    import neptune_codegen as CG

    text = _README.read_text(encoding="utf-8")
    examples = re.findall(r"^```python\n(.*?)^```$", text, re.M | re.S)
    assert examples, "README has no Python examples"
    modules = {"neptune_opt": NO, "neptune_codegen": CG}
    for index, source in enumerate(examples):
        tree = ast.parse(source, filename=f"README example {index}")
        compile(tree, f"README example {index}", "exec")  # Syntax only; no GPU execution.
        for node in ast.walk(tree):
            if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute)
                    or not isinstance(node.func.value, ast.Name)
                    or node.func.value.id not in modules):
                continue
            function = getattr(modules[node.func.value.id], node.func.attr)
            assert callable(function), f"README calls non-callable {node.func.attr}"
            # Validate argument names / required parameters without evaluating tensors.
            assert all(keyword.arg is not None for keyword in node.keywords)
            inspect.signature(function).bind(
                *[None for _ in node.args],
                **{keyword.arg: None for keyword in node.keywords})


def test_readme_repository_links_have_existing_targets():
    text = _README.read_text(encoding="utf-8")
    links = re.findall(r"\]\(\s*([^\s)]+)\s*\)", text)
    local = [link.split("#", 1)[0] for link in links
             if not link.startswith(("https://", "http://", "#", "mailto:"))]
    assert local, "README has no repository links"
    root = _README.parent
    for link in local:
        target = (root / link).resolve()
        assert target.is_relative_to(root), f"README link leaves repository: {link}"
        assert target.exists(), f"README has a broken repository link: {link}"
