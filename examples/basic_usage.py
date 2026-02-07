"""Basic usage of semcompress — compress text while preserving meaning."""

from semcompress import compact

# Sample text (a long passage about AI)
text = (
    "Machine learning is a subset of artificial intelligence that focuses on building "
    "systems that learn from data. It enables computers to identify patterns and make "
    "decisions without being explicitly programmed for each scenario. Deep learning is "
    "a specialized form of machine learning that uses neural networks with multiple layers. "
    "Natural language processing allows machines to understand, interpret, and generate "
    "human language in meaningful ways. Computer vision enables machines to interpret and "
    "understand visual information from the world around them. Reinforcement learning "
    "trains agents to make sequential decisions through trial and error in an environment. "
    "Transfer learning allows models to apply knowledge gained from one task to improve "
    "performance on a different but related task. The field of artificial intelligence has "
    "seen rapid advancement in recent years, driven by increases in computing power and "
    "data availability. Large language models have fundamentally transformed how we "
    "interact with technology and access information. Ethical considerations in AI "
    "development are becoming increasingly important as these systems affect more areas "
    "of daily life."
)

# Compress to 50% of original size
result = compact(text, ratio=0.5)

print("=== Original ===")
print(f"Tokens: {result.original_tokens}")
print()
print("=== Compacted ===")
print(f"Tokens: {result.compacted_tokens}")
print(f"Ratio: {result.ratio:.0%}")
print(f"Iterations: {result.iterations}")
print(f"Chunks removed: {result.chunks_removed}")
print(f"Chunks kept: {result.chunks_kept}")
print()
print(result.text)
