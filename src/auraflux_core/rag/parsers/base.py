import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Union

from auraflux_core.rag.schemas.parser import StandardSection

logger = logging.getLogger(__name__)

FileInput = Union[str, Path, bytes, IO[bytes]]


class BaseParser(ABC):
    """
    Abstract Base Class for all document parsers (PDFParser, DOCXParser, TXTParser).
    Enforces a unified interface supporting File Path, Bytes, or IO Streams.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def parse(
        self,
        file_input: FileInput,
        filename: Optional[str] = None
    ) -> List[StandardSection]:
        """
        Abstract method to execute format-specific document parsing.

        :param file_input: Target document (Path, raw bytes, or IO stream)
        :param filename: Optional source filename (used for metadata/breadcrumbs)
        :return: List of validated StandardSection Pydantic models
        """
        pass

    def safe_parse(
        self,
        file_input: FileInput,
        filename: Optional[str] = None
    ) -> List[StandardSection]:
        """
        High-level wrapper method with built-in error handling and schema validation.
        """
        source_name = filename or self._extract_source_name(file_input)

        try:
            sections = self.parse(file_input, filename=source_name)
            if not sections:
                logger.warning(f"Parsing result for {source_name} is empty.")
            return sections
        except Exception as e:
            logger.error(f"Error parsing document {source_name}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to parse document '{source_name}': {str(e)}") from e

    def _extract_source_name(self, file_input: FileInput) -> str:
        """
        Extracts a human-readable file/source name from various input types.

        This helper method safely inspects the provided file input (Path, string,
        or file-like object) to determine a suitable filename for logging, metadata,
        and breadcrumb generation.

        Args:
            file_input (FileInput): The input payload, which can be a file path,
                raw bytes, or a file-like stream object.

        Returns:
            str: Extracted filename or 'unnamed_document' if name resolution fails.
        """
        # Case 1: Input is a string or Path object
        if isinstance(file_input, (str, Path)):
            return Path(file_input).name

        # Case 2: Input is a file-like stream with a 'name' attribute (e.g., Django File or Open File)
        if hasattr(file_input, "name") and isinstance(getattr(file_input, "name"), str):
            return Path(getattr(file_input, "name")).name

        # Case 3: Fallback for raw bytes or unnamed streams without a 'name' attribute
        return "unnamed_document"
