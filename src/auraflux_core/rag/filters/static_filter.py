import re
from auraflux_core.rag.schemas.chunker import ChunkEvidence
from auraflux_core.rag.config.regex_patterns import NoisePatternCollection, DEFAULT_NOISE_PATTERNS

# ----------------------------------------------------------------------
# Step 2: Filter 1 - Static Rule-Based Gate
# ----------------------------------------------------------------------
def is_valid_static_rule(
    evidence: ChunkEvidence,
    patterns: NoisePatternCollection = DEFAULT_NOISE_PATTERNS
) -> bool:
    """
    Filters out chunks that are too short, contain common layout noise,
    or have high gibberish/symbol ratios (0 Token Cost).
    """
    text = evidence.excerptText.strip()

    # 1. Length check (Discard if fewer than 50 characters)
    if len(text) < 50:
        return False

    # 2. Header, footer, and TOC noise pattern matching
    if (
        patterns.page_number.search(text)
        or patterns.copyright_notice.search(text)
        or patterns.table_of_contents.search(text)
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
