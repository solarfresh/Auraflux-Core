import re

from auraflux_core.rag.config.regex_patterns import (DEFAULT_NOISE_PATTERNS,
                                                     NoisePatternCollection)
from auraflux_core.rag.filters.base import BaseFilter
from auraflux_core.rag.schemas.chunker import ChunkEvidence, StandardChunk


# ----------------------------------------------------------------------
# Step 2: Filter 1 - Static Rule-Based Gate
# ----------------------------------------------------------------------
class StaticRuleFilter(BaseFilter[StandardChunk]):
    def __init__(self, patterns: NoisePatternCollection = DEFAULT_NOISE_PATTERNS):
        self.patterns = patterns

    def passes(self, chunk: StandardChunk) -> bool:
        return self._is_valid_static_rule(chunk.evidence)

    def _is_valid_static_rule(self,
        evidence: ChunkEvidence,
    ) -> bool:
        """
        Filters out chunks that are too short, contain common layout noise,
        or have high gibberish/symbol ratios (0 Token Cost).
        """
        text = evidence.excerpt_text.strip()

        # 1. Length check (Discard if fewer than 50 characters)
        if len(text) < 50:
            return False

        # 2. Header, footer, and TOC noise pattern matching
        if (
            self.patterns.page_number.search(text)
            or self.patterns.copyright_notice.search(text)
            or self.patterns.table_of_contents.search(text)
        ):
            return False

        # 3. Gibberish and special character ratio check
        # Discard if non-word and non-CJK characters (punctuation/symbols/noise)
        # exceed 40%
        valid_chars = re.findall(r'[\w\u4e00-\u9fff]', text)
        valid_ratio = len(valid_chars) / len(text) if len(text) > 0 else 0
        if (1 - valid_ratio) > 0.4:
            return False

        return True
