from typing import Optional
from auraflux_core.rag.schemas.chunker import ChunkKeywords, ChunkEvidence, StandardChunk
from auraflux_core.rag.filters.base import BaseFilter


# ----------------------------------------------------------------------
# Grounding Gate
# ----------------------------------------------------------------------
class GroundedInEvidenceFilter(BaseFilter[StandardChunk]):
    def passes(self, chunk: StandardChunk) -> bool:
        return self._is_grounded_in_evidence(chunk.keywords, chunk.evidence)

    def _is_grounded_in_evidence(
        self,
        keywords: Optional[ChunkKeywords],
        evidence: ChunkEvidence
    ) -> bool:
        """
        Validates that extracted keywords (triples/tags) are actually grounded
        in the raw excerpt text to prevent hallucinated metadata.
        """
        if keywords is None:
            return False

        if not evidence.excerpt_text:
            return False

        text_lower = evidence.excerpt_text.lower()

        # Check if at least one triple subject/object exists in the raw text
        has_grounded_triple = any(
            triple.subject.lower() in text_lower or triple.object.lower() in text_lower
            for triple in keywords.triples
        )

        # Check if at least one tag exists in the raw text
        has_grounded_tag = any(
            tag.lower() in text_lower
            for tag in keywords.tags
        )

        return has_grounded_triple or has_grounded_tag


# ----------------------------------------------------------------------
# Semantic Quality & Density Gate
# ----------------------------------------------------------------------
class SemanticQualityFilter(BaseFilter[StandardChunk]):
    def passes(self, chunk: StandardChunk) -> bool:
        return self._is_valid_semantic_quality(chunk.keywords)

    def _is_valid_semantic_quality(self, keywords: Optional[ChunkKeywords]) -> bool:
        """
        Blocks low-density chunks containing no triples and no tags,
        preventing them from triggering expensive Stage 2 operations.
        """
        if keywords is None:
            return False

        # Early-exit condition: Completely void of strongly bound triples and tags
        if len(keywords.triples) == 0 and len(keywords.tags) == 0:
            return False

        return True
