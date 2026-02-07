from __future__ import annotations

from semcompress.models import ChunkNode, CompactionResult, CompactorConfig
from semcompress.scoring.base import BaseScorer


class CompactionEngine:
    """Core compaction loop: iteratively removes least-important sub-chunks.

    Algorithm:
        1. Compute embeddings for all sub-chunks and parent chunks.
        2. Score each sub-chunk's importance (projection onto parent direction).
        3. Iteratively remove the lowest-scoring chunks in batches.
        4. After each batch removal, recompute affected parent embeddings and re-score.
        5. Stop when target token count is reached or no more chunks can be removed.
    """

    def __init__(self, embedder, scorer: BaseScorer, config: CompactorConfig):
        self._embedder = embedder
        self._scorer = scorer
        self._config = config

    def compact(self, nodes: list[ChunkNode]) -> CompactionResult:
        """Run the iterative compaction loop.

        Args:
            nodes: List of ChunkNodes with children (from HierarchicalChunker).

        Returns:
            CompactionResult with compacted text and metadata.
        """
        # Phase 1: Compute all embeddings
        self._compute_all_embeddings(nodes)

        # Phase 2: Initial scoring
        self._score_all(nodes)

        # Phase 3: Determine target
        original_tokens = sum(n.token_count for n in nodes)

        if self._config.target_tokens is not None:
            target_tokens = self._config.target_tokens
        elif self._config.target_ratio is not None:
            target_tokens = int(original_tokens * self._config.target_ratio)
        else:
            target_tokens = int(original_tokens * 0.5)

        target_tokens = max(0, target_tokens)
        current_tokens = original_tokens

        # Phase 4: Iterative removal
        iterations = 0
        total_removed = 0

        while current_tokens > target_tokens and iterations < self._config.max_iterations:
            # Collect removable candidates
            candidates = []
            for node in nodes:
                active = node.active_children
                if len(active) <= self._config.min_chunks_to_keep:
                    continue
                for child in active:
                    if child.importance_score <= self._config.min_importance_threshold:
                        continue
                    candidates.append((child, node))

            if not candidates:
                break

            # Sort by importance ascending (least important first)
            candidates.sort(key=lambda x: x[0].importance_score)

            # Determine how many to remove this iteration
            n_active = sum(len(n.active_children) for n in nodes)
            n_to_remove = max(1, int(n_active * self._config.batch_removal_fraction))

            # Remove chunks, tracking affected parents
            affected_parent_ids = set()
            removed_this_round = 0

            for chunk, parent_node in candidates[:n_to_remove]:
                # Re-check parent still has enough children
                if len(parent_node.active_children) <= self._config.min_chunks_to_keep:
                    continue

                chunk.is_removed = True
                current_tokens -= chunk.token_count
                affected_parent_ids.add(id(parent_node))
                removed_this_round += 1
                total_removed += 1

                if current_tokens <= target_tokens:
                    break

            if removed_this_round == 0:
                break

            # Recompute embeddings and scores for affected parents
            for node in nodes:
                if id(node) in affected_parent_ids:
                    active_text = node.active_text
                    if active_text.strip():
                        node.embedding = self._embedder.embed_single(active_text)
                        active = node.active_children
                        if active:
                            scores = self._scorer.score(node, active)
                            for child, score in zip(active, scores):
                                child.importance_score = score

            iterations += 1

        # Phase 5: Reconstruct text
        text_parts = []
        kept_chunks = []
        removed_chunks = []

        for node in nodes:
            active_texts = []
            for child in node.children:
                if child.is_removed:
                    removed_chunks.append(child)
                else:
                    kept_chunks.append(child)
                    active_texts.append(child.text)
            if active_texts:
                text_parts.append(self._config.chunk_separator.join(active_texts))

        compacted_text = self._config.parent_separator.join(text_parts)

        return CompactionResult(
            text=compacted_text,
            original_tokens=original_tokens,
            compacted_tokens=current_tokens,
            ratio=current_tokens / original_tokens if original_tokens > 0 else 1.0,
            iterations=iterations,
            chunks_removed=total_removed,
            chunks_kept=len(kept_chunks),
            removed_chunks=removed_chunks,
            kept_chunks=kept_chunks,
        )

    def _compute_all_embeddings(self, nodes: list[ChunkNode]) -> None:
        """Batch-compute embeddings for all children and parents."""
        # Collect all child texts for batch embedding
        all_children = []
        for node in nodes:
            for child in node.children:
                all_children.append(child)

        if all_children:
            child_texts = [c.text for c in all_children]
            child_embeddings = self._embedder.embed(child_texts)
            for child, emb in zip(all_children, child_embeddings):
                child.embedding = emb

        # Compute parent embeddings from full parent text
        parent_texts = [node.text for node in nodes if node.text.strip()]
        if parent_texts:
            parent_embeddings = self._embedder.embed(parent_texts)
            text_idx = 0
            for node in nodes:
                if node.text.strip():
                    node.embedding = parent_embeddings[text_idx]
                    text_idx += 1

    def _score_all(self, nodes: list[ChunkNode]) -> None:
        """Score all active children against their parent."""
        for node in nodes:
            active = node.active_children
            if not active or node.embedding is None:
                continue
            scores = self._scorer.score(node, active)
            for child, score in zip(active, scores):
                child.importance_score = score
