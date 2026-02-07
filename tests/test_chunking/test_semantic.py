from semcompress.chunking.semantic import SemanticChunker
from semcompress.models import CompactorConfig


class TestSemanticChunker:
    def test_creates_groups(self, mock_embedder):
        config = CompactorConfig(similarity_threshold=0.3)
        chunker = SemanticChunker(mock_embedder, config)

        text = (
            "Machine learning is great. Deep learning uses neural networks. "
            "The weather is sunny today. It might rain tomorrow."
        )
        groups = chunker.create_groups(text)

        assert len(groups) >= 1
        # All original sentences should be present across groups
        all_sentences = [s for g in groups for s in g]
        assert len(all_sentences) >= 2

    def test_single_sentence(self, mock_embedder):
        config = CompactorConfig()
        chunker = SemanticChunker(mock_embedder, config)

        groups = chunker.create_groups("Just one sentence.")
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_empty_text(self, mock_embedder):
        config = CompactorConfig()
        chunker = SemanticChunker(mock_embedder, config)

        groups = chunker.create_groups("")
        assert groups == []

    def test_high_threshold_splits_more(self, mock_embedder):
        config_high = CompactorConfig(similarity_threshold=0.99)
        config_low = CompactorConfig(similarity_threshold=0.0)

        chunker_high = SemanticChunker(mock_embedder, config_high)
        chunker_low = SemanticChunker(mock_embedder, config_low)

        text = (
            "Machine learning is a field of AI. It uses data to learn patterns. "
            "Cooking requires fresh ingredients. A good recipe needs time."
        )

        groups_high = chunker_high.create_groups(text)
        groups_low = chunker_low.create_groups(text)

        # Higher threshold should create more groups (more splits)
        assert len(groups_high) >= len(groups_low)
