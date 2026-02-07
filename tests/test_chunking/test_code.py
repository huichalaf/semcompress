"""Tests for code-aware chunking."""

from __future__ import annotations

from semcompress.chunking.code import detect_code, split_code

PYTHON_CODE = """
import os
import sys

def hello(name):
    print(f"Hello, {name}")

class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}"

async def fetch_data(url):
    async with aiohttp.get(url) as resp:
        return await resp.json()

@decorator
def decorated():
    pass
""".strip()

PROSE_TEXT = """
The quick brown fox jumps over the lazy dog. This is a simple sentence
about animals and their behaviors. In the forest, many creatures live
together in harmony. The birds sing in the morning and the wolves howl
at night. Nature is beautiful and complex.
""".strip()

JS_CODE = """
const express = require('express');
const app = express();

function handleRequest(req, res) {
    res.json({ status: 'ok' });
}

app.listen(3000);
""".strip()

GO_CODE = """
package main

import "fmt"

func main() {
    fmt.Println("hello")
}

func add(a, b int) int {
    return a + b
}
""".strip()

RUST_CODE = """
pub struct Config {
    name: String,
}

impl Config {
    pub fn new(name: &str) -> Self {
        Config { name: name.to_string() }
    }
}

fn main() {
    let c = Config::new("test");
}
""".strip()


class TestDetectCode:
    def test_detects_python(self):
        assert detect_code(PYTHON_CODE) is True

    def test_detects_javascript(self):
        assert detect_code(JS_CODE) is True

    def test_detects_go(self):
        assert detect_code(GO_CODE) is True

    def test_detects_rust(self):
        assert detect_code(RUST_CODE) is True

    def test_rejects_prose(self):
        assert detect_code(PROSE_TEXT) is False

    def test_empty_string(self):
        assert detect_code("") is False

    def test_short_code_snippet(self):
        # Only 2 lines, 1 code signal = 50% > 15% threshold
        assert detect_code("import os\nprint('hi')") is True

    def test_mixed_content_mostly_prose(self):
        # 1 code line out of 10 = 10% < 15% threshold
        text = "This is prose.\n" * 9 + "import os\n"
        assert detect_code(text) is False


class TestSplitCode:
    def test_splits_python_functions(self):
        blocks = split_code(PYTHON_CODE)
        assert len(blocks) >= 4  # imports, hello, Greeter, fetch_data, decorated

    def test_splits_on_class(self):
        blocks = split_code(PYTHON_CODE)
        class_blocks = [b for b in blocks if "class Greeter" in b]
        assert len(class_blocks) == 1

    def test_splits_on_async(self):
        blocks = split_code(PYTHON_CODE)
        async_blocks = [b for b in blocks if "async def" in b]
        assert len(async_blocks) == 1

    def test_splits_on_decorator(self):
        blocks = split_code(PYTHON_CODE)
        decorated = [b for b in blocks if "@decorator" in b]
        assert len(decorated) == 1

    def test_preserves_all_content(self):
        blocks = split_code(PYTHON_CODE)
        joined = "\n".join(blocks)
        keywords = [
            "import os", "def hello", "class Greeter",
            "async def fetch_data", "@decorator",
        ]
        for keyword in keywords:
            assert keyword in joined

    def test_empty_string(self):
        assert split_code("") == []

    def test_single_function(self):
        code = "def foo():\n    return 42"
        blocks = split_code(code)
        assert len(blocks) >= 1
        assert "def foo" in blocks[0]

    def test_fallback_to_blank_line_split(self):
        # No def/class boundaries, should fall back to blank-line splitting
        text = "x = 1\ny = 2\n\nz = 3\nw = 4"
        blocks = split_code(text)
        assert len(blocks) == 2

    def test_indented_methods(self):
        code = """
class Foo:
    def method_a(self):
        pass

    def method_b(self):
        pass
""".strip()
        blocks = split_code(code)
        assert len(blocks) >= 2
