"""Advanced usage with custom configuration."""

from semcompress import Compactor, CompactorConfig

text = (
    "The history of artificial intelligence dates back to the 1950s when Alan Turing "
    "proposed the concept of machine intelligence. The Dartmouth Conference in 1956 is "
    "widely considered the birth of AI as an academic field. Early AI research focused "
    "primarily on symbolic reasoning and logical problem solving. Expert systems became "
    "popular in the 1980s for handling domain-specific tasks in business and medicine. "
    "The AI winter periods saw dramatically reduced funding and diminished interest in "
    "the field. Machine learning emerged as the dominant paradigm in the 1990s with the "
    "development of statistical methods. Support vector machines and random forests "
    "became widely used classification algorithms. The advent of big data in the 2000s "
    "provided the fuel for new computational breakthroughs. Deep learning revolutionized "
    "the field starting around 2012 with AlexNet's ImageNet victory. The transformer "
    "architecture introduced in 2017 fundamentally changed natural language processing. "
    "Large language models scaled to billions of parameters demonstrated emergent "
    "capabilities. Generative AI opened new possibilities for automated content creation. "
    "Multimodal models now seamlessly process text, images, and audio together. The "
    "future of AI promises even more transformative applications across every industry."
)

# Configure for aggressive compression with precise control
config = CompactorConfig(
    target_ratio=0.3,              # Keep only 30%
    scoring_method="projection",   # Use projection-based scoring
    batch_removal_fraction=0.05,   # Remove 5% per iteration (more precise)
    similarity_threshold=0.4,      # Topic shift sensitivity
    min_chunks_to_keep=1,          # Always keep at least 1 sentence per section
)

compactor = Compactor(config)
result = compactor.compact(text)

print(f"Original:  {result.original_tokens} tokens")
print(f"Compacted: {result.compacted_tokens} tokens ({result.ratio:.0%})")
print(f"Iterations: {result.iterations}")
print()
print("--- Kept sentences ---")
for chunk in result.kept_chunks:
    print(f"  [{chunk.importance_score:.3f}] {chunk.text[:80]}...")
print()
print("--- Removed sentences ---")
for chunk in result.removed_chunks:
    print(f"  [{chunk.importance_score:.3f}] {chunk.text[:80]}...")
