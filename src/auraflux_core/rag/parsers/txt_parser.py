from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chardet

from auraflux_core.rag.config.regex_patterns import (DEFAULT_HEADER_PATTERNS,
                                                     HeaderPatternCollection)
from auraflux_core.rag.parsers.base import BaseParser, FileInput
from auraflux_core.rag.schemas.parser import (StandardSection,
                                              TXTSectionMetadata)


class TXTParser(BaseParser):
    """
    Parser implementation for plain text (.txt) files.
    Supports file paths, raw bytes, and file streams directly.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        patterns: Optional[HeaderPatternCollection] = None
    ):
        super().__init__(config)
        self.patterns = patterns or DEFAULT_HEADER_PATTERNS
        self._compiled_patterns = self.patterns.get_all_patterns()
        self.terminal_punctuations = ('。', '！', '？', '.', '!', '?', ':', '：', ';', '；')

    def _detect_and_decode_bytes(self, raw_bytes: bytes) -> Tuple[str, str]:
        """
        Detects encoding automatically using chardet with fallback mechanisms.
        """
        detection = chardet.detect(raw_bytes)
        detected_encoding = detection.get("encoding") or "utf-8"

        encodings_to_try = [detected_encoding, "utf-8", "utf-8-sig", "big5", "gbk", "cp950"]

        for enc in encodings_to_try:
            try:
                content = raw_bytes.decode(enc)
                return content, enc
            except (UnicodeDecodeError, TypeError):
                continue

        return raw_bytes.decode("utf-8", errors="ignore"), "utf-8-lossy"

    def _resolve_to_bytes(self, file_input: FileInput) -> bytes:
        """
        Resolves various file input types into a unified raw binary bytes object.

        This helper method normalizes input payloads—whether provided as file paths,
        raw bytes, or active file streams—into a consistent `bytes` format required for
        subsequent encoding detection and content parsing.

        Args:
            file_input (FileInput): The target document input (Path, string, bytes,
                or file-like stream object).

        Returns:
            bytes: Raw binary content of the document.

        Raises:
            FileNotFoundError: If the provided path string or Path object does not exist.
            TypeError: If the input type is unsupported.
        """
        # Case 1: Input is a string path or Path instance
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_input}")
            return path.read_bytes()

        # Case 2: Input is already raw binary bytes
        elif isinstance(file_input, bytes):
            return file_input

        # Case 3: Input is a readable stream (e.g., BytesIO, Django File, or Open File)
        elif hasattr(file_input, "read"):
            content = file_input.read()
            return content if isinstance(content, bytes) else content.encode("utf-8")

        # Case 4: Fallback for invalid/unsupported input types
        else:
            raise TypeError(f"Unsupported input type: {type(file_input)}")

    def _is_implicit_heading(self, line: str) -> bool:
        line_str = line.strip()
        if not line_str:
            return False

        for pattern in self._compiled_patterns:
            if pattern.match(line_str):
                return True

        if len(line_str) < 40 and not line_str.endswith(self.terminal_punctuations):
            return True

        return False

    def parse(
        self,
        file_input: FileInput,
        filename: Optional[str] = None
    ) -> List[StandardSection]:
        """
        Parses a .txt source into a list of StandardSection Pydantic models.
        """
        source_name = filename or self._extract_source_name(file_input)
        raw_bytes = self._resolve_to_bytes(file_input)
        content, used_encoding = self._detect_and_decode_bytes(raw_bytes)

        lines = content.replace('\r\n', '\n').split('\n')

        sections: List[StandardSection] = []
        current_breadcrumb: List[str] = [source_name]
        current_title: str = "Preamble"
        current_lines: List[str] = []
        section_counter: int = 1

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            if self._is_implicit_heading(line_str):
                if current_lines:
                    full_text = "\n".join(current_lines).strip()
                    if full_text:
                        sections.append(
                            StandardSection(
                                section_id=f"sec_txt_{section_counter:03d}",
                                breadcrumb=list(current_breadcrumb),
                                title=current_title,
                                text=full_text,
                                metadata=TXTSectionMetadata(
                                    source_file=source_name,
                                    file_type="txt",
                                    char_count=len(full_text),
                                    encoding=used_encoding
                                )
                            )
                        )
                        section_counter += 1
                        current_lines = []

                current_title = line_str
                current_breadcrumb = [source_name, line_str]
                continue

            current_lines.append(line_str)

        if current_lines:
            full_text = "\n".join(current_lines).strip()
            if full_text:
                sections.append(
                    StandardSection(
                        section_id=f"sec_txt_{section_counter:03d}",
                        breadcrumb=list(current_breadcrumb),
                        title=current_title,
                        text=full_text,
                        metadata=TXTSectionMetadata(
                            source_file=source_name,
                            file_type="txt",
                            char_count=len(full_text),
                            encoding=used_encoding
                        )
                    )
                )

        return sections
