import pytest

from semcompress import compact
from semcompress.compactor import Compactor
from semcompress.models import CompactorConfig

# These tests use the real sentence-transformers model.
# They test the full public API end-to-end.

pytestmark = pytest.mark.slow


class TestCompactFunction:
    def test_basic_compaction(self, sample_text):
        result = compact(sample_text, ratio=0.5)

        assert result.text
        assert result.compacted_tokens < result.original_tokens
        assert result.ratio < 1.0
        assert result.chunks_removed > 0
        assert result.iterations >= 1

    def test_token_budget(self, sample_text):
        result = compact(sample_text, tokens=30)

        assert result.compacted_tokens <= result.original_tokens

    def test_different_methods(self, sample_text):
        result_proj = compact(sample_text, ratio=0.5, method="projection")
        result_cos = compact(sample_text, ratio=0.5, method="cosine")

        # Both should produce valid results
        assert result_proj.text
        assert result_cos.text

    def test_high_ratio_returns_more_text(self, sample_text):
        result_low = compact(sample_text, ratio=0.3)
        result_high = compact(sample_text, ratio=0.7)

        assert result_high.compacted_tokens >= result_low.compacted_tokens


class TestCompactorClass:
    def test_compact(self, sample_text):
        config = CompactorConfig(target_ratio=0.5)
        c = Compactor(config)
        result = c.compact(sample_text)

        assert result.text
        assert result.ratio <= 1.0

    def test_compact_batch(self, sample_text):
        config = CompactorConfig(target_ratio=0.5)
        c = Compactor(config)
        results = c.compact_batch([sample_text, sample_text])

        assert len(results) == 2
        for r in results:
            assert r.text

    def test_empty_text(self):
        config = CompactorConfig()
        c = Compactor(config)
        result = c.compact("")

        assert result.text == ""
        assert result.original_tokens == 0
        assert result.ratio == 1.0

    def test_short_text_below_target(self):
        config = CompactorConfig(target_tokens=1000)
        c = Compactor(config)
        result = c.compact("Short text.")

        assert result.text == "Short text."
        assert result.ratio == 1.0
        assert result.chunks_removed == 0

    def test_invalid_method_raises(self):
        config = CompactorConfig(scoring_method="invalid")
        with pytest.raises(ValueError, match="Unknown scoring method"):
            Compactor(config)
