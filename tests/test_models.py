
from semcompress.models import Chunk, ChunkNode, CompactionResult, CompactorConfig


class TestChunk:
    def test_creation(self):
        chunk = Chunk(text="Hello world", index=0, token_count=2)
        assert chunk.text == "Hello world"
        assert chunk.index == 0
        assert chunk.token_count == 2
        assert chunk.is_removed is False
        assert chunk.embedding is None
        assert chunk.importance_score == 0.0

    def test_char_count(self):
        chunk = Chunk(text="Hello", index=0)
        assert chunk.char_count == 5


class TestChunkNode:
    def test_active_children(self):
        children = [
            Chunk(text="First", index=0, token_count=1),
            Chunk(text="Second", index=1, token_count=1, is_removed=True),
            Chunk(text="Third", index=2, token_count=1),
        ]
        node = ChunkNode(text="First Second Third", index=0, children=children)

        assert len(node.active_children) == 2
        assert node.active_children[0].text == "First"
        assert node.active_children[1].text == "Third"

    def test_active_text(self):
        children = [
            Chunk(text="Hello", index=0, token_count=1),
            Chunk(text="removed", index=1, token_count=1, is_removed=True),
            Chunk(text="world", index=2, token_count=1),
        ]
        node = ChunkNode(text="Hello removed world", index=0, children=children)
        assert node.active_text == "Hello world"

    def test_active_token_count(self):
        children = [
            Chunk(text="Hello", index=0, token_count=5),
            Chunk(text="removed", index=1, token_count=3, is_removed=True),
            Chunk(text="world", index=2, token_count=4),
        ]
        node = ChunkNode(text="test", index=0, children=children)
        assert node.active_token_count == 9


class TestCompactorConfig:
    def test_defaults(self):
        config = CompactorConfig()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.device == "cpu"
        assert config.target_ratio == 0.5
        assert config.target_tokens is None
        assert config.scoring_method == "projection"
        assert config.token_counter == "whitespace"

    def test_custom_values(self):
        config = CompactorConfig(
            model_name="custom-model",
            target_ratio=0.3,
            scoring_method="cosine",
        )
        assert config.model_name == "custom-model"
        assert config.target_ratio == 0.3
        assert config.scoring_method == "cosine"


class TestCompactionResult:
    def test_creation(self):
        result = CompactionResult(
            text="compacted",
            original_tokens=100,
            compacted_tokens=50,
            ratio=0.5,
            iterations=3,
            chunks_removed=5,
            chunks_kept=5,
        )
        assert result.text == "compacted"
        assert result.ratio == 0.5
        assert result.iterations == 3
