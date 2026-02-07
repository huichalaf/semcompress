import numpy as np

from semcompress.models import Chunk, ChunkNode
from semcompress.scoring.projection import ProjectionScorer


class TestProjectionScorer:
    def test_aligned_vectors_score_high(self):
        scorer = ProjectionScorer()

        parent = ChunkNode(text="parent", index=0)
        parent.embedding = np.array([1.0, 0.0, 0.0])

        children = [
            Chunk(text="aligned", index=0),
            Chunk(text="orthogonal", index=1),
            Chunk(text="opposite", index=2),
        ]
        children[0].embedding = np.array([1.0, 0.0, 0.0])   # Perfectly aligned
        children[1].embedding = np.array([0.0, 1.0, 0.0])   # Orthogonal
        children[2].embedding = np.array([-1.0, 0.0, 0.0])  # Opposite

        scores = scorer.score(parent, children)

        assert scores[0] > scores[1]   # Aligned > orthogonal
        assert scores[1] > scores[2]   # Orthogonal > opposite
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)
        assert scores[2] == pytest.approx(-1.0)

    def test_partially_aligned(self):
        scorer = ProjectionScorer()

        parent = ChunkNode(text="parent", index=0)
        parent.embedding = np.array([1.0, 0.0])

        child = Chunk(text="child", index=0)
        child.embedding = np.array([0.7, 0.7])

        scores = scorer.score(parent, [child])
        assert scores[0] == pytest.approx(0.7, abs=1e-6)

    def test_no_parent_embedding_raises(self):
        scorer = ProjectionScorer()
        parent = ChunkNode(text="parent", index=0)
        child = Chunk(text="child", index=0)
        child.embedding = np.array([1.0, 0.0])

        with pytest.raises(ValueError, match="no embedding"):
            scorer.score(parent, [child])

    def test_child_without_embedding_scores_zero(self):
        scorer = ProjectionScorer()

        parent = ChunkNode(text="parent", index=0)
        parent.embedding = np.array([1.0, 0.0])

        child = Chunk(text="child", index=0)  # No embedding

        scores = scorer.score(parent, [child])
        assert scores[0] == 0.0

    def test_zero_parent_norm(self):
        scorer = ProjectionScorer()

        parent = ChunkNode(text="parent", index=0)
        parent.embedding = np.array([0.0, 0.0, 0.0])

        child = Chunk(text="child", index=0)
        child.embedding = np.array([1.0, 0.0, 0.0])

        scores = scorer.score(parent, [child])
        assert scores[0] == 0.0


import pytest  # noqa: E402
