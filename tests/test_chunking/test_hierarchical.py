from semcompress.chunking.hierarchical import HierarchicalChunker
from semcompress.models import CompactorConfig


class TestHierarchicalChunker:
    def test_produces_chunk_nodes(self, mock_embedder, sample_text):
        config = CompactorConfig()
        chunker = HierarchicalChunker(mock_embedder, config)

        nodes = chunker.chunk(sample_text)

        assert len(nodes) >= 1
        for node in nodes:
            assert len(node.children) >= 1
            assert node.token_count > 0
            for child in node.children:
                assert child.text
                assert child.token_count > 0
                assert child.index >= 0

    def test_children_indices_are_unique(self, mock_embedder, sample_text):
        config = CompactorConfig()
        chunker = HierarchicalChunker(mock_embedder, config)

        nodes = chunker.chunk(sample_text)

        all_indices = [c.index for n in nodes for c in n.children]
        assert len(all_indices) == len(set(all_indices))

    def test_all_text_preserved(self, mock_embedder, sample_text):
        config = CompactorConfig()
        chunker = HierarchicalChunker(mock_embedder, config)

        nodes = chunker.chunk(sample_text)

        all_child_texts = " ".join(c.text for n in nodes for c in n.children)
        # Every word in original should appear in chunked output
        for word in sample_text.split()[:10]:
            assert word in all_child_texts or word.rstrip(".,!?") in all_child_texts

    def test_empty_text(self, mock_embedder):
        config = CompactorConfig()
        chunker = HierarchicalChunker(mock_embedder, config)

        nodes = chunker.chunk("")
        assert nodes == []
