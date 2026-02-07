from __future__ import annotations

import numpy as np
import pytest

from semcompress.embeddings.base import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """Deterministic embedder for fast tests. No model download required.

    Generates reproducible vectors from text hashes so that:
    - Same text always produces the same embedding
    - Different texts produce different (but deterministic) embeddings
    - Semantically similar texts (sharing words) will have somewhat similar vectors
    """

    def __init__(self, dim: int = 384):
        self._dim = dim

    def embed_single(self, text: str) -> np.ndarray:
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        vec = rng.randn(self._dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([]).reshape(0, self._dim)
        return np.array([self.embed_single(t) for t in texts])

    @property
    def dimension(self) -> int:
        return self._dim


@pytest.fixture
def mock_embedder():
    return MockEmbedder(dim=384)


@pytest.fixture
def small_mock_embedder():
    """Small embedder for readable test vectors."""
    return MockEmbedder(dim=8)


@pytest.fixture
def sample_text():
    return (
        "Machine learning is a subset of artificial intelligence. "
        "It enables computers to learn from data without being explicitly programmed. "
        "Deep learning is a specialized form of machine learning using neural networks. "
        "Natural language processing allows machines to understand human language. "
        "Computer vision enables machines to interpret visual information from the world. "
        "Reinforcement learning trains agents through trial and error in an environment. "
        "Transfer learning allows models to apply knowledge from one task to another. "
        "The field of AI has seen rapid advancement in recent years. "
        "Large language models have transformed how we interact with technology. "
        "Ethical considerations in AI development are becoming increasingly important."
    )


@pytest.fixture
def long_sample_text():
    return (
        "The history of artificial intelligence dates back to the 1950s. "
        "Alan Turing proposed the concept of machine intelligence in his seminal paper. "
        "The Dartmouth Conference in 1956 is considered the birth of AI as a field. "
        "Early AI research focused on symbolic reasoning and problem solving. "
        "Expert systems became popular in the 1980s for domain-specific tasks. "
        "The AI winter periods saw reduced funding and interest in the field. "
        "Machine learning emerged as a dominant paradigm in the 1990s. "
        "Support vector machines and random forests became widely used algorithms. "
        "The advent of big data in the 2000s fueled new breakthroughs. "
        "Deep learning revolutionized the field starting around 2012. "
        "Convolutional neural networks achieved superhuman performance in image recognition. "
        "Recurrent neural networks enabled advances in sequence modeling. "
        "The transformer architecture introduced in 2017 changed everything. "
        "Attention mechanisms allowed models to process long-range dependencies. "
        "BERT and GPT models demonstrated the power of pre-training. "
        "Large language models scaled to billions of parameters. "
        "Generative AI created new possibilities for content creation. "
        "Multimodal models can now process text, images, and audio together. "
        "AI agents are becoming more autonomous and capable. "
        "The future of AI promises even more transformative applications."
    )
