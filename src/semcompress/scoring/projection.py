from __future__ import annotations

import numpy as np

from semcompress.models import Chunk, ChunkNode
from semcompress.scoring.base import BaseScorer


class ProjectionScorer(BaseScorer):
    """Score importance by projecting child embeddings onto the parent's direction.

    The scalar projection measures how much each child's embedding vector
    contributes to the parent's semantic direction:

        score_i = dot(child_i, parent) / ||parent||

    Higher scores mean the child is more aligned with — and contributes more to —
    the overall meaning of the parent section.
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
            else:
                proj = float(np.dot(child.embedding, parent.embedding) / parent_norm)
                scores.append(proj)

        return scores
