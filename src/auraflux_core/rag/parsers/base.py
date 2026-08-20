from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from auraflux_core.rag.schemas.parser import StandardSection


class BaseParser(ABC):
    """
    Abstract Base Class for all document parsers (PDFParser, DOCXParser, TXTParser).
    Enforces a unified interface and error handling mechanism across format parsers.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the parser with optional configuration parameters.
        :param config: Dictionary containing parser settings (e.g., encoding, regex rules, OCR toggles)
        """
        self.config = config or {}

    @abstractmethod
    def parse(self, file_path: str | Path) -> List[StandardSection]:
        """
        Abstract method to execute format-specific document parsing.

        :param file_path: Target document file path
        :return: List of validated StandardSection Pydantic models
        """
        pass

    def safe_parse(self, file_path: str | Path) -> List[StandardSection]:
        """
        High-level wrapper method with built-in error handling and schema validation.
        Prevents corrupt files or parsing exceptions from crashing the pipeline.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"[Parser Error] File not found: {file_path}")

        try:
            sections = self.parse(path)
            if not sections:
                print(f"⚠️ [Parser Warning] Parsing result for {path.name} is empty.")
            return sections
        except Exception as e:
            print(f"❌ [Parser Exception] Error parsing document {path.name}: {str(e)}")
            raise RuntimeError(f"Failed to parse document '{path.name}': {str(e)}") from e