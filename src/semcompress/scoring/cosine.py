from __future__ import annotations

import numpy as np

from semcompress.models import Chunk, ChunkNode
from semcompress.scoring.base import BaseScorer


class CosineScorer(BaseScorer):
    """Score importance by cosine similarity between child and parent embeddings.

    Measures pure directional alignment, ignoring vector magnitude:

        score_i = dot(child_i, parent) / (||child_i|| * ||parent||)

    When embeddings are normalized (default), this produces identical results
    to ProjectionScorer.
    """

    def score(self, parent: ChunkNode, children: list[Chunk]) -> list[float]:
        if parent.embedding is None:
            raise ValueError("Parent chunk has no embedding computed")

        parent_norm = np.linalg.norm(parent.embedding)
        if parent_norm < 1e-10:
            return [0.0] * len(children)

        scores = []
        for child in children:
            if child.embedding is None:
                scores.append(0.0)
                continue
            child_norm = np.linalg.norm(child.embedding)
            if child_norm < 1e-10:
                scores.append(0.0)
                continue
            cos_sim = float(np.dot(child.embedding, parent.embedding) / (child_norm * parent_norm))
            scores.append(cos_sim)

        return scores
