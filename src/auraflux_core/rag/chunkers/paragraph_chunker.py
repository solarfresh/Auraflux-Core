import re
from typing import Any, Dict, List, Optional, Tuple

from auraflux_core.rag.chunkers.base import BaseChunker
from auraflux_core.rag.config.regex_patterns import (DEFAULT_ANAPHORA_PATTERNS,
                                                     DEFAULT_SENTENCE_PATTERNS,
                                                     AnaphoraPatternCollection,
                                                     SentencePatternCollection)
from auraflux_core.rag.schemas.chunker import ChunkEvidence, StandardChunk
from auraflux_core.rag.schemas.parser import StandardSection


class AnaphoraDetector:
    """
    Responsibility: Evaluates whether a given text snippet begins with
    anaphoric or demonstrative expressions requiring preceding context.
    """

    def __init__(self, patterns: Optional[AnaphoraPatternCollection] = None):
        self.patterns = patterns or DEFAULT_ANAPHORA_PATTERNS
        self.anaphora_pattern = self.patterns.anaphora_reference_pattern

    def starts_with_anaphora(self, text: str) -> bool:
        """Returns True if the text segment relies heavily on immediate preceding context."""
        return bool(self.anaphora_pattern.search(text))


class SentenceSlidingWindow:
    """
    Responsibility: Executes sentence-level sliding window segmentation
    on oversized or merged text blocks.
    """

    def __init__(
        self,
        max_chunk_size: int,
        chunk_overlap: int,
        patterns: Optional[SentencePatternCollection] = None
    ):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.patterns = patterns or DEFAULT_SENTENCE_PATTERNS
        self.sentence_splitter = self.patterns.default_sentence_splitter

    def _split_into_sentences(self, text: str) -> List[str]:
        raw_sentences = self.sentence_splitter.findall(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text]

    def split(self, text_block: str) -> Tuple[List[str], str]:
        """
        Splits a text block into chunks within max_chunk_size and maintains trailing sentence overlap.

        Returns:
            Tuple[List[str], str]: (List of slice results, trailing sentence overlap buffer)
        """
        sentences = self._split_into_sentences(text_block)
        chunks: List[str] = []
        current_sentences: List[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)
            if current_len + sent_len > self.max_chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences).strip())

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

        tail_buffer = " ".join(current_sentences).strip()
        return chunks, tail_buffer


class ParagraphSubSplitter:
    """Responsibility: Fallback sentence-level sliding window splitting for oversized paragraphs."""

    def __init__(self, max_chunk_size: int, chunk_overlap: int, sentence_splitter: Any):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_splitter = sentence_splitter

    def _split_into_sentences(self, text: str) -> List[str]:
        raw_sentences = self.sentence_splitter.findall(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text]

    def split_paragraph(self, para_text: str) -> List[str]:
        sentences = self._split_into_sentences(para_text)
        chunk_texts: List[str] = []
        current_sentences: List[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)
            if current_len + sent_len > self.max_chunk_size and current_sentences:
                chunk_texts.append(" ".join(current_sentences).strip())

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
            chunk_texts.append(" ".join(current_sentences).strip())

        return chunk_texts


class ParagraphDynamicChunker(BaseChunker):
    """
    Responsibility: Orchestrates paragraph-level dynamic chunking.
    Aggregates paragraphs greedily within size limits and delegates anaphora detection
    and sentence-level re-segmentation to specialized components when boundaries break context.
    """

    def __init__(
        self,
        max_chunk_size: int = 800,
        min_chunk_size: int = 150,
        chunk_overlap: int = 100,
        anaphora_detector: Optional[AnaphoraDetector] = None,
        sentence_window: Optional[SentenceSlidingWindow] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap

        # Inject components or fallback to default instances
        self.anaphora_detector = anaphora_detector or AnaphoraDetector()
        self.sentence_window = sentence_window or SentenceSlidingWindow(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap
        )

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Splits raw section text into structural paragraph units."""
        raw_paragraphs = re.split(r'\n\s*\n|\n', text)
        return [p.strip() for p in raw_paragraphs if p.strip()]

    def _join_texts(self, text_list: List[str]) -> str:
        """Joins text segments with appropriate line breaks based on language context."""
        if not text_list:
            return ""
        result = text_list[0]
        for next_text in text_list[1:]:
            if result and next_text:
                last_char = result[-1]
                first_char = next_text[0]
                # Insert double newline for Western non-CJK character boundaries
                if not re.search(r'[\u4e00-\u9fff]', last_char) and not re.search(r'[\u4e00-\u9fff]', first_char):
                    result += "\n\n" + next_text
                else:
                    result += "\n" + next_text
            else:
                result += next_text
        return result.strip()

    def _build_location_string(self, section: StandardSection) -> str:
        """Constructs human-readable metadata location path."""
        breadcrumb_path = " > ".join(section.breadcrumb) if section.breadcrumb else section.title
        page_num = getattr(section.metadata, "page_num", None)
        if page_num is not None:
            return f"Page {page_num}, Section: {breadcrumb_path}"
        return f"Section: {breadcrumb_path}"

    def _create_chunk_data(self, section: StandardSection, text_content: str, chunk_idx: int) -> StandardChunk:
        """Encapsulates string content into the standardized StandardChunk schema."""
        return StandardChunk(
            id=f"{section.section_id}_pchk_{chunk_idx:03d}",
            fileId=getattr(section.metadata, "source_file", "unknown_file"),
            evidence=ChunkEvidence(excerptText=text_content, location=self._build_location_string(section)),
            keywords=None,
            concept=None,
            alignment=None,
            vectors=None
        )

    def chunk_section(self, section: StandardSection) -> List[StandardChunk]:
        """
        Executes section chunking workflow.

        Algorithm Flow:
        1. Greedily aggregates paragraphs until max_chunk_size is reached.
        2. Upon capacity overflow, inspects the next paragraph via AnaphoraDetector.
        3. If anaphora is detected, rejects paragraph boundary, merges buffer + paragraph,
           and delegates re-segmentation to SentenceSlidingWindow.
        4. Otherwise, safely emits current buffer and resets for the next chunk.
        """
        text = section.text.strip()
        if not text:
            return []

        raw_paragraphs = self._split_into_paragraphs(text)
        if not raw_paragraphs:
            return []

        chunk_data_list: List[StandardChunk] = []
        chunk_idx = 0
        current_buffer: List[str] = []
        current_buffer_len = 0

        i = 0
        n = len(raw_paragraphs)

        while i < n:
            para = raw_paragraphs[i]
            para_len = len(para)

            # Case 1: Paragraph fits within capacity -> Accumulate greedily
            if current_buffer_len + para_len <= self.max_chunk_size:
                current_buffer.append(para)
                current_buffer_len += para_len
                i += 1
                continue

            # Case 2: Capacity reached -> Perform anaphora inspection
            if self.anaphora_detector.starts_with_anaphora(para) and current_buffer:
                # Merge current buffer with incoming anaphoric paragraph
                combined_block = self._join_texts(current_buffer + [para])

                # Delegate sentence-level sliding window segmentation
                sub_chunks, tail_buffer = self.sentence_window.split(combined_block)

                for t in sub_chunks:
                    chunk_data_list.append(self._create_chunk_data(section, t, chunk_idx))
                    chunk_idx += 1

                # Re-initialize buffer with sentence window overlap tail
                current_buffer = [tail_buffer] if tail_buffer else []
                current_buffer_len = len(tail_buffer) if tail_buffer else 0
                i += 1
            else:
                # Case 3: Safe boundary -> Emit accumulated buffer as chunk
                chunk_str = self._join_texts(current_buffer)
                chunk_data_list.append(self._create_chunk_data(section, chunk_str, chunk_idx))
                chunk_idx += 1

                current_buffer = []
                current_buffer_len = 0

        # Flush remaining buffer content
        if current_buffer:
            chunk_str = self._join_texts(current_buffer)
            # Final safety check for single oversized remaining buffer
            if len(chunk_str) > self.max_chunk_size:
                sub_chunks, _ = self.sentence_window.split(chunk_str)
                for t in sub_chunks:
                    chunk_data_list.append(self._create_chunk_data(section, t, chunk_idx))
                    chunk_idx += 1
            else:
                chunk_data_list.append(self._create_chunk_data(section, chunk_str, chunk_idx))

        return chunk_data_list