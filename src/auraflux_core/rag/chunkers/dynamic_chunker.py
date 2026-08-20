import re
from typing import List, Dict, Any, Optional

from auraflux_core.rag.chunkers.base import BaseChunker
from auraflux_core.rag.schemas.parser import StandardSection
from auraflux_core.rag.schemas.chunker import StandardChunk, ChunkEvidence
from auraflux_core.rag.config.regex_patterns import DEFAULT_SENTENCE_PATTERNS, SentencePatternCollection


class DynamicChunker(BaseChunker):
    """
    Universal Sentence-Aware Dynamic Overlapping Chunker.
    Splits sections based on CJK and English sentence boundaries with target size and overlap.
    Outputs the unified pipeline entity `StandardChunk` directly.
    """

    def __init__(
        self,
        max_chunk_size: int = 500,
        chunk_overlap: int = 100,
        patterns: Optional[SentencePatternCollection] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

        # Inject custom sentence pattern collection or fallback to system default singleton
        self.patterns = patterns or DEFAULT_SENTENCE_PATTERNS
        self.sentence_splitter = self.patterns.default_sentence_splitter

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits raw text into discrete sentences while preserving punctuation."""
        raw_sentences = self.sentence_splitter.findall(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text]

    def _join_sentences(self, sentences: List[str]) -> str:
        """
        Smart sentence joiner to prevent unnatural spacing between CJK characters
        while preserving word spacing for Latin/English text.
        """
        if not sentences:
            return ""

        result = sentences[0]
        for next_sent in sentences[1:]:
            if result and next_sent:
                last_char = result[-1]
                first_char = next_sent[0]
                if not re.search(r'[\u4e00-\u9fff]', last_char) and not re.search(r'[\u4e00-\u9fff]', first_char):
                    result += " " + next_sent
                else:
                    result += next_sent
            else:
                result += next_sent
        return result.strip()

    def _build_location_string(self, section: StandardSection) -> str:
        """Constructs a readable location string based on section metadata and breadcrumbs."""
        breadcrumb_path = " > ".join(section.breadcrumb) if section.breadcrumb else section.title

        page_num = getattr(section.metadata, "page_num", None)
        if page_num is not None:
            return f"Page {page_num}, Section: {breadcrumb_path}"
        return f"Section: {breadcrumb_path}"

    def _create_chunk_data(
        self,
        section: StandardSection,
        sentences: List[str],
        chunk_idx: int
    ) -> StandardChunk:
        """Helper method to construct a StandardChunk model with Layer 4 Evidence initialized."""
        chunk_text = self._join_sentences(sentences)
        location_str = self._build_location_string(section)
        source_file_id = getattr(section.metadata, "source_file", "unknown_file")

        # Layer 4: Fact & Evidence Initialization
        evidence = ChunkEvidence(
            excerptText=chunk_text,
            location=location_str
        )

        return StandardChunk(
            id=f"{section.section_id}_chk_{chunk_idx:03d}",
            fileId=source_file_id,
            evidence=evidence,
            keywords=None,     # Will be populated at Step 4 (LLM Extraction)
            concept=None,      # Will be populated at Step 6 (LLM Reasoning)
            alignment=None,    # Will be populated at Step 6 (LLM Reasoning)
            vectors=None       # Will be calculated and attached at Step 7
        )

    def chunk_section(self, section: StandardSection) -> List[StandardChunk]:
        """
        Splits a single StandardSection into sentence-boundary-aligned StandardChunk objects.
        """
        text = section.text.strip()
        if not text:
            return []

        if len(text) <= self.max_chunk_size:
            return [self._create_chunk_data(section, [text], chunk_idx=0)]

        sentences = self._split_into_sentences(text)
        chunks: List[StandardChunk] = []

        current_sentences: List[str] = []
        current_len = 0
        chunk_idx = 0

        for sentence in sentences:
            sent_len = len(sentence)

            if current_len + sent_len > self.max_chunk_size and current_sentences:
                chunks.append(self._create_chunk_data(section, current_sentences, chunk_idx))
                chunk_idx += 1

                overlap_sentences: List[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break

                current_sentences = overlap_sentences
                current_len = overlap_len

            current_sentences.append(sentence)
            current_len += sent_len

        if current_sentences:
            chunks.append(self._create_chunk_data(section, current_sentences, chunk_idx))

        return chunks
