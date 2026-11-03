"""Offline checks on unreal/*.py: they parse and never build a positional Rotator."""

import ast
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parent.parent / "unreal").glob("*.py"))


def _positional_rotators(tree):
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == "Rotator":
                bad.append(node.lineno)
    return bad


def test_scripts_found():
    assert SCRIPTS


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_parses(path):
    ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_positional_rotator(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    bad = _positional_rotators(tree)
    assert not bad, (
        "%s: unreal.Rotator needs pitch=/yaw=/roll= keywords (lines %s)"
        % (path.name, bad)
    )
