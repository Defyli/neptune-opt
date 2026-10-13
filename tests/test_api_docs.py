"""
The API reference must match the surface it documents.

A stale limits section already bit this repository once: the README still described variance as
unreachable after the two-pass plan made it work, and mis-attributed why LayerNorm is out of frame.
Documentation that contradicts the code is worse than no documentation, so the public surface is
asserted here rather than maintained by hand.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neptune_opt as NO  # noqa: E402

_DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "docs", "api.md")


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
