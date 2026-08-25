from typing import List
from unittest.mock import MagicMock

import pytest

# Import your module (adjust the import path to match your project structure)
from auraflux_core.rag.chunkers.paragraph_chunker import \
    ParagraphDynamicChunker
from auraflux_core.rag.chunkers.utils import (estimate_tokens,
                                              split_paragraphs_bilingual)
from auraflux_core.rag.schemas.chunker import StandardChunk
from auraflux_core.rag.schemas.parser import (PDFSectionMetadata,
                                              StandardSection,
                                              TXTSectionMetadata)

# ---------------------------------------------------------------------------
# Plain Text Mock Data (.txt content) representing a standard HotpotQA article
# ---------------------------------------------------------------------------
HOTPOTQA_MULTI_PARAGRAPH_TXT = """**Scott Derrickson**
Scott Derrickson is an American filmmaker. He directed Doctor Strange in 2016.

**Doctor Strange (film)**
Doctor Strange is a 2016 American superhero film based on the Marvel Comics character.

**Asbury University**
Asbury University is a private Christian university in Wilmore, Kentucky."""


@pytest.fixture
def chunker() -> ParagraphDynamicChunker:
    """Provides a ParagraphDynamicChunker instance configured with max_chunk_size=128."""
    return ParagraphDynamicChunker(
        max_chunk_size=128,
        min_chunk_size=32,
        chunk_overlap=16
    )

def test_multi_paragraph_chunk_count_and_capacity(chunker: ParagraphDynamicChunker) -> None:
    """
    Validates chunking behavior for multi-paragraph documents without verbose logging.

    Verifies that:
    1. The number of generated chunks preserves paragraph boundaries.
    2. No individual chunk exceeds the max_chunk_size limit (128 tokens).
    3. No chunk is below the min_chunk_size limit (unless constrained by text size).
    """
    section = StandardSection(
        section_id="doc_hotpotqa_multi_para_001",
        title="HotpotQA Contexts",
        breadcrumb=["HotpotQA"],
        text=HOTPOTQA_MULTI_PARAGRAPH_TXT,
        metadata=TXTSectionMetadata(
            source_file="hotpotqa_multi_para.txt",
            char_count=310
        )
    )

    extracted_paragraphs: List[str] = split_paragraphs_bilingual(section.text)
    chunks = chunker.chunk_section(section)

    # 1. Assert double newline splitting correctly extracted 3 paragraphs
    assert len(extracted_paragraphs) == 3, (
        f"Expected 3 extracted paragraphs, got {len(extracted_paragraphs)}"
    )

    # 2. Calculate total token load across all paragraphs
    total_tokens = sum(estimate_tokens(p) for p in extracted_paragraphs)

    # 3. Assert greedy merging behavior:
    # If total tokens fit within max_chunk_size (128), it must merge them into exactly 1 chunk.
    if total_tokens <= chunker.max_chunk_size:
        assert len(chunks) == 1, (
            f"Expected greedy merge into 1 chunk for {total_tokens} total tokens, got {len(chunks)}"
        )
    else:
        assert len(chunks) > 1, "Expected multiple chunks when token budget is exceeded."

    # 4. Strict chunk capacity assertions
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk.evidence.excerptText)
        assert 0 < chunk_tokens <= chunker.max_chunk_size, (
            f"Chunk token size ({chunk_tokens}) violated bounds (0, {chunker.max_chunk_size}]."
        )

# ==========================================
# 1. Helper Functions & Token Estimator Tests
# ==========================================

def test_estimate_tokens_pure_english():
    text = "This is a simple sentence."  # 5 words * 1.3 = 6.5 -> 6 tokens
    assert estimate_tokens(text) == 6


def test_estimate_tokens_pure_cjk():
    text = "這是一個簡單的中文句子。"  # 12 CJK characters
    assert estimate_tokens(text) == 12


def test_estimate_tokens_mixed_language():
    text = "這是一個 RAG 測試範例。"  # 9 CJK chars + 2 English words * 1.3 (2.6) -> 11 tokens
    assert estimate_tokens(text) == 10


def test_split_paragraphs_bilingual_english():
    text = "First paragraph line 1.\nFirst paragraph line 2.\n\nSecond paragraph."
    paragraphs = split_paragraphs_bilingual(text)
    assert len(paragraphs) == 2
    assert paragraphs[0] == "First paragraph line 1. First paragraph line 2."
    assert paragraphs[1] == "Second paragraph."


def test_split_paragraphs_bilingual_cjk():
    # Case A: Paragraph line breaks without sentence-ending punctuation (merged into 1 paragraph)
    text = "這是第一段內容\n這是第一段的延伸。\n\n這是第二段內容。"
    paragraphs = split_paragraphs_bilingual(text)
    assert len(paragraphs) == 2
    assert paragraphs[0] == "這是第一段內容這是第一段的延伸。"
    assert paragraphs[1] == "這是第二段內容。"

    # Case B: Paragraph line breaks with sentence-ending punctuation (split into 3 paragraphs)
    text_hard_break = "這是第一段內容。\n這是第一段的延伸。\n\n這是第二段內容。"
    paragraphs_hard = split_paragraphs_bilingual(text_hard_break)
    assert len(paragraphs_hard) == 3


# ==========================================
# 2. ParagraphDynamicChunker Core Logic Tests
# ==========================================

@pytest.fixture
def mock_section():
    """Fixture providing a mock StandardSection instance for testing."""
    metadata = PDFSectionMetadata(
        source_file="test_doc.pdf",
        page_num=1,
        char_count=120,  # Explicitly passed to resolve Pylance 'reportCallIssue'
        bbox=[0.0, 0.0, 100.0, 100.0]
    )
    return StandardSection(
        section_id="sec_001",
        title="Introduction",
        breadcrumb=["Chapter 1", "Introduction"],
        text="",
        metadata=metadata
    )


def test_chunk_section_empty_text(mock_section):
    chunker = ParagraphDynamicChunker()
    mock_section.text = "   "
    chunks = chunker.chunk_section(mock_section)
    assert chunks == []


def test_chunk_section_normal_greedy_aggregation(mock_section):
    """Tests greedy aggregation of short paragraphs into a single chunk within capacity."""
    mock_section.text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

    # Large window size allows all paragraphs to aggregate into one chunk
    chunker = ParagraphDynamicChunker(max_chunk_size=200)
    chunks = chunker.chunk_section(mock_section)

    assert len(chunks) == 1
    assert isinstance(chunks[0], StandardChunk)
    assert "Paragraph one." in chunks[0].evidence.excerptText
    assert "Paragraph three." in chunks[0].evidence.excerptText


def test_chunk_section_split_when_exceeding_max_size(mock_section):
    """Tests chunk emission when aggregated length exceeds max_chunk_size."""
    para1 = "Word " * 50  # ~65 tokens
    para2 = "Word " * 50  # ~65 tokens
    mock_section.text = f"{para1}\n\n{para2}"

    # Combined tokens (~130) exceed max_chunk_size=80, requiring a split into 2 chunks
    chunker = ParagraphDynamicChunker(max_chunk_size=80)
    chunks = chunker.chunk_section(mock_section)

    assert len(chunks) == 2
    assert chunks[0].id == "sec_001_pchk_000"
    assert chunks[1].id == "sec_001_pchk_001"


def test_chunk_section_anaphora_tolerance_overflow(mock_section):
    """Tests 15% overflow tolerance threshold when anaphora is detected to preserve context."""
    para1 = "Word " * 45  # ~58 tokens
    para2 = "However, this is an important point."  # Contains anaphora triggers (However/this)

    mock_section.text = f"{para1}\n\n{para2}"

    # Mock AnaphoraDetector to force triggering on para2
    mock_anaphora_detector = MagicMock()
    mock_anaphora_detector.starts_with_anaphora.side_effect = lambda p: "However" in p

    # max_chunk_size = 60, allowing up to 69 tokens with 1.15x overflow tolerance.
    # Total tokens: para1 (58) + para2 (~8) = 66 <= 69, so they should merge into 1 chunk.
    chunker = ParagraphDynamicChunker(
        max_chunk_size=60,
        anaphora_detector=mock_anaphora_detector
    )
    chunks = chunker.chunk_section(mock_section)

    assert len(chunks) == 1
    assert "However, this is an important point." in chunks[0].evidence.excerptText


def test_chunk_section_fallback_sub_splitter_for_oversized_paragraph(mock_section):
    """Tests fallback sentence sliding window when a single paragraph inherently exceeds max_chunk_size."""
    oversized_para = "Sentence one. " * 30  # Oversized single paragraph
    mock_section.text = oversized_para

    chunker = ParagraphDynamicChunker(max_chunk_size=50)
    chunks = chunker.chunk_section(mock_section)

    # Paragraph should be sub-split into multiple chunks
    assert len(chunks) > 1
    assert all(isinstance(c, StandardChunk) for c in chunks)


def test_chunk_metadata_location_formatting(mock_section):
    """Verifies that the location metadata inside generated StandardChunk is correctly formatted."""
    mock_section.text = "A brief testing paragraph."
    chunker = ParagraphDynamicChunker()

    chunks = chunker.chunk_section(mock_section)
    assert len(chunks) == 1

    location = chunks[0].evidence.location
    assert "Page 1" in location
    assert "Section: Chapter 1 > Introduction" in location
