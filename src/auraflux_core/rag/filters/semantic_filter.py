from typing import Optional
from auraflux_core.rag.schemas.chunker import ChunkKeywords

# ----------------------------------------------------------------------
# Step 5: Filter 2 - Semantic Quality & Density Gate
# ----------------------------------------------------------------------
def is_valid_semantic_quality(keywords: Optional[ChunkKeywords]) -> bool:
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