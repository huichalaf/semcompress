import numpy as np
import pytest

from semcompress.models import Chunk, ChunkNode
from semcompress.scoring.cosine import CosineScorer


class TestCosineScorer:
    def test_identical_vectors_score_one(self):
        scorer = CosineScorer()

        parent = ChunkNode(text="parent", index=0)
        parent.embedding = np.array([1.0, 2.0, 3.0])

        child = Chunk(text="child", index=0)
        child.embedding = np.array([1.0, 2.0, 3.0])

        scores = scorer.score(parent, [child])
        assert scores[0] == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_score_zero(self):
        scorer = CosineScorer()

        parent = ChunkNode(text="parent", index=0)
        parent.embedding = np.array([1.0, 0.0])

        child = Chunk(text="child", index=0)
        child.embedding = np.array([0.0, 1.0])

        scores = scorer.score(parent, [child])
        assert scores[0] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_score_negative(self):
        scorer = CosineScorer()

        parent = ChunkNode(text="parent", index=0)
        parent.embedding = np.array([1.0, 0.0])

        child = Chunk(text="child", index=0)
        child.embedding = np.array([-1.0, 0.0])

        scores = scorer.score(parent, [child])
        assert scores[0] == pytest.approx(-1.0, abs=1e-6)

    def test_magnitude_invariant(self):
        """Cosine similarity should not depend on vector magnitudes."""
        scorer = CosineScorer()

        parent = ChunkNode(text="parent", index=0)
        parent.embedding = np.array([1.0, 1.0])

        child_small = Chunk(text="small", index=0)
        child_small.embedding = np.array([0.1, 0.1])

        child_large = Chunk(text="large", index=1)
        child_large.embedding = np.array([100.0, 100.0])

        scores = scorer.score(parent, [child_small, child_large])
        assert scores[0] == pytest.approx(scores[1], abs=1e-6)

    def test_no_parent_embedding_raises(self):
        scorer = CosineScorer()
        parent = ChunkNode(text="parent", index=0)
        child = Chunk(text="child", index=0)
        child.embedding = np.array([1.0])

        with pytest.raises(ValueError, match="no embedding"):
            scorer.score(parent, [child])

    def test_zero_child_norm(self):
        scorer = CosineScorer()

        parent = ChunkNode(text="parent", index=0)
        parent.embedding = np.array([1.0, 0.0])

        child = Chunk(text="child", index=0)
        child.embedding = np.array([0.0, 0.0])

        scores = scorer.score(parent, [child])
        assert scores[0] == 0.0
