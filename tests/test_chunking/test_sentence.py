from semcompress.chunking.sentence import split_sentences


class TestSplitSentences:
    def test_basic_split(self):
        text = "First sentence. Second sentence. Third sentence."
        sentences = split_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence."
        assert sentences[2] == "Third sentence."

    def test_question_marks(self):
        text = "Is this a question? Yes it is. Really?"
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_exclamation_marks(self):
        text = "Wow! That is amazing. Incredible!"
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_abbreviations_preserved(self):
        text = "Dr. Smith went to the store. He bought milk."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert "Dr. Smith" in sentences[0]

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   ") == []

    def test_single_sentence(self):
        text = "Just one sentence here"
        sentences = split_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == text

    def test_paragraph_fallback(self):
        text = "first paragraph content here\n\nsecond paragraph content here"
        sentences = split_sentences(text)
        assert len(sentences) == 2

    def test_preserves_content(self):
        text = "Hello world. Goodbye world."
        sentences = split_sentences(text)
        reconstructed = " ".join(sentences)
        assert "Hello world" in reconstructed
        assert "Goodbye world" in reconstructed
