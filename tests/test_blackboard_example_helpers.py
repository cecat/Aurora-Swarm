"""Unit tests for the Blackboard example's board_view_for_prompt helper."""

import importlib.util
import sys
from pathlib import Path

import pytest

# Load board_view_for_prompt from examples/blackboard_example.py without running main
_examples_dir = Path(__file__).resolve().parent.parent / "examples"
_spec = importlib.util.spec_from_file_location(
    "blackboard_example",
    _examples_dir / "blackboard_example.py",
    submodule_search_locations=[str(_examples_dir)],
)
assert _spec is not None and _spec.loader is not None
_example_module = importlib.util.module_from_spec(_spec)
# Avoid running main when loading (script only runs on __name__ == "__main__")
sys.modules["blackboard_example"] = _example_module
_spec.loader.exec_module(_example_module)
board_view_for_prompt = _example_module.board_view_for_prompt


def test_board_view_empty():
    """Empty board returns placeholder."""
    out = board_view_for_prompt({}, "full", 0)
    assert out == "(Board is empty.)"


def test_board_view_full():
    """full strategy includes all entries."""
    board = {
        "hypotheses": ["H1", "H2"],
        "critiques": ["C1"],
    }
    out = board_view_for_prompt(board, "full", 0)
    assert "**hypotheses** (2 entries):" in out
    assert "  1. H1" in out
    assert "  2. H2" in out
    assert "**critiques** (1 entries):" in out
    assert "  1. C1" in out


def test_board_view_last_n():
    """last_n limits entries per section to the last N."""
    board = {
        "hypotheses": ["H1", "H2", "H3", "H4", "H5"],
        "critiques": ["C1", "C2"],
    }
    out = board_view_for_prompt(board, "last_n", 2)
    assert "H4" in out
    assert "H5" in out
    assert "H1" not in out
    assert "H2" not in out
    assert "H3" not in out
    assert "C1" in out
    assert "C2" in out
    assert "**hypotheses** (2 entries):" in out
    assert "**critiques** (2 entries):" in out


def test_board_view_last_n_zero():
    """last_n with value 0 yields no entries per section."""
    board = {"hypotheses": ["H1"], "critiques": ["C1"]}
    out = board_view_for_prompt(board, "last_n", 0)
    assert "H1" not in out
    assert "C1" not in out
    assert "0 entries" in out


def test_board_view_max_chars_truncates_at_line():
    """max_chars truncates at a line boundary."""
    board = {
        "hypotheses": ["short", "another"],
        "critiques": ["critique"],
    }
    out = board_view_for_prompt(board, "max_chars", 50)
    assert len(out) <= 50 or out.endswith("\n")
    # Should include at least the first section header and maybe part of content
    assert "**hypotheses**" in out or "hypotheses" in out.lower()


def test_board_view_max_chars_large_value_unchanged():
    """max_chars with value larger than content leaves content unchanged."""
    board = {"hypotheses": ["H1"], "critiques": ["C1"]}
    full = board_view_for_prompt(board, "full", 0)
    truncated = board_view_for_prompt(board, "max_chars", 10000)
    assert truncated == full
