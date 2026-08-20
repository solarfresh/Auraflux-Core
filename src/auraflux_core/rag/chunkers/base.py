from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from auraflux_core.rag.schemas.parser import StandardSection
from auraflux_core.rag.schemas.chunker import StandardChunk


class BaseChunker(ABC):
    """
    Abstract Base Class for all chunking strategy implementations in Stage 2.
    Enforces a standard chunking interface across different chunker types
    (e.g., DynamicChunker, SemanticChunker, CodeChunker).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the chunker with optional execution configurations.
        :param config: Dictionary containing chunking parameters (e.g., max_size, overlap)
        """
        self.config = config or {}

    @abstractmethod
    def chunk_section(self, section: StandardSection) -> List[StandardChunk]:
        """
        Abstract method to process and split a single StandardSection into a list of StandardChunk models.

        :param section: Input StandardSection model from Stage 1
        :return: List of validated StandardChunk Pydantic models
        """
        pass

    def chunk_sections(self, sections: List[StandardSection]) -> List[StandardChunk]:
        """
        Batch processing method to iterate over a list of StandardSections
        and return a flattened list of StandardChunks.

        :param sections: List of StandardSection models from Stage 1
        :return: Flattened list of StandardChunk Pydantic models
        """
        all_chunks: List[StandardChunk] = []
        for section in sections:
            chunks = self.chunk_section(section)
            all_chunks.extend(chunks)
        return all_chunks
