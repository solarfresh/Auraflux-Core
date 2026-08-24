from auraflux_core.rag.chunkers.utils import split_paragraphs_bilingual


def test_split_paragraphs_pure_english_hotpotqa_style():
    """
    Reproduces the over-fragmentation issue with HotpotQA English text.
    Text containing single newlines after periods (e.g., 'Sentence.\n') should not be broken
    into separate paragraphs; single newlines inside English text should be replaced with spaces.
    """
    english_text = (
        "Scott Derrickson is an American director.\n"
        "He directed Doctor Strange in 2016.\n"
        "The movie was a critical success."
    )

    paragraphs = split_paragraphs_bilingual(english_text)

    # Expected: Single newlines in English text are joined into a single coherent paragraph.
    assert len(paragraphs) == 1
    assert paragraphs[0] == "Scott Derrickson is an American director. He directed Doctor Strange in 2016. The movie was a critical success."


def test_split_paragraphs_english_with_question_mark_newline():
    """
    Tests that single newlines following English punctuation like '?\n' or '!\n'
    do not erroneously trigger double newline splits for English text.
    """
    english_text = "Who is Scott Derrickson?\nHe is an American film director."

    paragraphs = split_paragraphs_bilingual(english_text)

    # Expected: Question mark followed by single newline stays in a single paragraph.
    assert len(paragraphs) == 1
    assert paragraphs[0] == "Who is Scott Derrickson? He is an American film director."


def test_split_paragraphs_pure_english_double_newline():
    """
    Tests standard double newlines (\\n\\n) in English text to ensure they are preserved
    as separate paragraphs.
    """
    english_text = "First paragraph content.\n\nSecond paragraph content."

    paragraphs = split_paragraphs_bilingual(english_text)

    assert len(paragraphs) == 2
    assert paragraphs[0] == "First paragraph content."
    assert paragraphs[1] == "Second paragraph content."


def test_split_paragraphs_cjk_sentence_end_with_newline():
    """
    Tests CJK (Chinese) text with sentence-ending punctuation followed by newlines ('。\\n').
    CJK text should process sentence-ending newlines as structural paragraph breaks.
    """
    cjk_text = "這是第一段內容。\n這是第一段的延伸。\n\n這是第二段內容。"

    paragraphs = split_paragraphs_bilingual(cjk_text)

    assert len(paragraphs) == 3
    assert paragraphs[0] == "這是第一段內容。"
    assert paragraphs[1] == "這是第一段的延伸。"
    assert paragraphs[2] == "這是第二段內容。"
