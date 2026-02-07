"""Tests for the MCP server tools (logic only, no actual MCP transport)."""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def sample_py_files(tmp_path):
    """Create sample Python files for testing."""
    (tmp_path / "main.py").write_text(
        "def main():\n    print('hello')\n\ndef helper():\n    return 42\n"
    )
    (tmp_path / "utils.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"
    )
    (tmp_path / "readme.txt").write_text("This is not a Python file.\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.py").write_text("class Nested:\n    pass\n")
    return tmp_path


class TestCompactTextLogic:
    """Test the compact_text tool's core logic."""

    def test_auto_mode_detects_code(self):
        from semcompress.chunking.code import detect_code

        code = "import os\nimport sys\n\ndef main():\n    pass\n\nclass Foo:\n    pass\n"
        assert detect_code(code) is True

    def test_auto_mode_rejects_prose(self):
        from semcompress.chunking.code import detect_code

        prose = "This is a normal sentence. Here is another one. And a third."
        assert detect_code(prose) is False


class TestCompactFilesLogic:
    """Test file reading and concatenation logic used by compact_files."""

    def test_reads_multiple_files(self, sample_py_files):
        paths = [
            str(sample_py_files / "main.py"),
            str(sample_py_files / "utils.py"),
        ]
        parts = []
        for path in paths:
            with open(path) as f:
                parts.append(f"# FILE: {path}\n{f.read()}")
        full = "\n\n".join(parts)
        assert "def main" in full
        assert "def add" in full

    def test_skips_missing_files(self, sample_py_files):
        paths = [
            str(sample_py_files / "main.py"),
            str(sample_py_files / "nonexistent.py"),
        ]
        parts = []
        for path in paths:
            if os.path.isfile(path):
                with open(path) as f:
                    parts.append(f.read())
        assert len(parts) == 1

    def test_empty_paths_list(self):
        parts = []
        for path in []:
            if os.path.isfile(path):
                with open(path) as f:
                    parts.append(f.read())
        assert len(parts) == 0


class TestCompactDirectoryLogic:
    """Test directory walking logic used by compact_directory."""

    def test_finds_py_files(self, sample_py_files):
        extensions = [".py"]
        found = []
        skip_dirs = {"__pycache__", ".git", "node_modules", ".venv"}
        for root, dirs, fnames in os.walk(str(sample_py_files)):
            dirs[:] = [d for d in sorted(dirs) if d not in skip_dirs]
            for f in sorted(fnames):
                if any(f.endswith(ext) for ext in extensions):
                    found.append(os.path.join(root, f))
        assert len(found) == 3  # main.py, utils.py, sub/nested.py

    def test_filters_by_extension(self, sample_py_files):
        extensions = [".txt"]
        found = []
        for root, dirs, fnames in os.walk(str(sample_py_files)):
            for f in sorted(fnames):
                if any(f.endswith(ext) for ext in extensions):
                    found.append(f)
        assert len(found) == 1
        assert found[0] == "readme.txt"

    def test_respects_token_budget(self, sample_py_files):
        from semcompress.token_counting import count_tokens

        extensions = [".py"]
        max_tokens = 10  # Very small budget
        parts = []
        total_tokens = 0
        budget_exhausted = False

        for root, dirs, fnames in os.walk(str(sample_py_files)):
            if budget_exhausted:
                break
            dirs[:] = sorted(dirs)
            for f in sorted(fnames):
                if not any(f.endswith(ext) for ext in extensions):
                    continue
                path = os.path.join(root, f)
                with open(path) as fh:
                    content = fh.read()
                entry_tokens = count_tokens(content, "whitespace")
                if total_tokens + entry_tokens > max_tokens and parts:
                    budget_exhausted = True
                    break
                parts.append(content)
                total_tokens += entry_tokens

        # Should have loaded at least 1 file but not all 3
        assert 1 <= len(parts) < 3

    def test_skips_pycache(self, sample_py_files):
        cache_dir = sample_py_files / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "cached.pyc").write_text("bytecode")

        skip_dirs = {"__pycache__", ".git", "node_modules", ".venv"}
        found = []
        for root, dirs, fnames in os.walk(str(sample_py_files)):
            dirs[:] = [d for d in sorted(dirs) if d not in skip_dirs]
            for f in fnames:
                found.append(f)

        assert "cached.pyc" not in found
