from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chardet
from parsers.base import BaseParser
from schemas.parser import StandardSection, TXTSectionMetadata

from auraflux_core.rag.config.regex_patterns import (DEFAULT_HEADER_PATTERNS,
                                                     HeaderPatternCollection)


class TXTParser(BaseParser):
    """
    Parser implementation for plain text (.txt) files.
    Features:
    1. Automatic encoding detection with safe fallbacks (chardet -> UTF-8 -> Big5/GBK).
    2. Multi-language (English & Chinese) implicit structure mining via HeaderPatternCollection.
    3. Structural section output mapped to StandardSection with typed TXTSectionMetadata.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        patterns: Optional[HeaderPatternCollection] = None
    ):
        super().__init__(config)
        # Inject custom multi-language pattern collection or fallback to system defaults
        self.patterns = patterns or DEFAULT_HEADER_PATTERNS
        self._compiled_patterns = self.patterns.get_all_patterns()

        # Terminal punctuations for Chinese & English to avoid misidentifying sentence ends as headers
        self.terminal_punctuations = ('。', '！', '？', '.', '!', '?', ':', '：', ';', '；')

    def _detect_and_read_encoding(self, file_path: Path) -> Tuple[str, str]:
        """
        Detects file encoding automatically using chardet with fallback mechanisms.
        :return: Tuple of (decoded_content_string, used_encoding_name)
        """
        raw_bytes = file_path.read_bytes()

        # Detect encoding
        detection = chardet.detect(raw_bytes)
        detected_encoding = detection.get("encoding") or "utf-8"

        # Sequential fallbacks for Traditional/Simplified Chinese and general UTF formats
        encodings_to_try = [detected_encoding, "utf-8", "utf-8-sig", "big5", "gbk", "cp950"]

        for enc in encodings_to_try:
            try:
                content = raw_bytes.decode(enc)
                return content, enc
            except (UnicodeDecodeError, TypeError):
                continue

        # Last resort: decode as utf-8 and ignore corrupted byte sequences
        return raw_bytes.decode("utf-8", errors="ignore"), "utf-8-lossy"

    def _is_implicit_heading(self, line: str) -> bool:
        """
        Determines whether a line qualifies as an implicit heading using multi-language regex library
        or short visual isolated line heuristics.
        """
        line_str = line.strip()
        if not line_str:
            return False

        # Rule A: Match against loaded Chinese & English Regex pattern library
        for pattern in self._compiled_patterns:
            if pattern.match(line_str):
                return True

        # Rule B: Short isolated visual line without Chinese or English terminal punctuation marks
        if len(line_str) < 40 and not line_str.endswith(self.terminal_punctuations):
            return True

        return False

    def parse(self, file_path: str | Path) -> List[StandardSection]:
        """
        Parses a .txt file into a list of StandardSection Pydantic models.

        :param file_path: Target .txt file path
        :return: List of validated StandardSection instances
        """
        path = Path(file_path)
        content, used_encoding = self._detect_and_read_encoding(path)

        # Normalize Windows/Unix line breaks
        lines = content.replace('\r\n', '\n').split('\n')

        sections: List[StandardSection] = []
        current_breadcrumb: List[str] = [path.name]
        current_title: str = "Preamble"
        current_lines: List[str] = []
        section_counter: int = 1

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if current line triggers a new section heading
            if self._is_implicit_heading(line_str):
                # Flush previous accumulated text into a StandardSection if content exists
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
                                    source_file=path.name,
                                    file_type="txt",
                                    char_count=len(full_text),
                                    encoding=used_encoding
                                )
                            )
                        )
                        section_counter += 1
                        current_lines = []

                # Update heading title and breadcrumb trail
                current_title = line_str
                current_breadcrumb = [path.name, line_str]
                continue

            current_lines.append(line_str)

        # Flush final remaining buffer lines
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
                            source_file=path.name,
                            file_type="txt",
                            char_count=len(full_text),
                            encoding=used_encoding
                        )
                    )
                )

        return sections
