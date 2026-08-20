import re
from typing import List, Pattern
from pydantic import BaseModel, Field


class HeaderPatternCollection(BaseModel):
    """
    Strongly-typed collection of compiled Regular Expressions for implicit header detection.
    Supports both English and Chinese document layout conventions.
    """
    # 1. Markdown syntax (# Title, ## Subtitle)
    markdown_headers: Pattern = Field(
        default=re.compile(r'^#{1,6}\s+.*$'),
        description="Matches Markdown heading syntaxes (# Title, ## Subtitle)"
    )

    # 2. Chinese Chapters & Legal Terms (第一章, 第 2 條, 附錄一)
    chinese_chapters: Pattern = Field(
        default=re.compile(r'^(?:第[0-9一二三四五六七八九十百千]+\s*[章節條項編]|附錄[一二三四五六七八九十0-9]*)\s*.*$'),
        description="Matches Chinese legal/chapter/appendix prefixes (第一章, 第 2 條, 附錄 A)"
    )

    # 3. English Chapters, Sections & Appendixes (Chapter 1, Section 2.1, Appendix A)
    english_chapters: Pattern = Field(
        default=re.compile(
            r'^(?i:'  # Case-insensitive flag
            r'chapter\s+[0-9a-z]+|'
            r'section\s+[0-9a-z\.]+|'
            r'part\s+[0-9a-z]+|'
            r'appendix\s+[a-z0-9]+|'
            r'article\s+[0-9a-z]+'
            r')(?:\s*[:\.-]\s*.*|\s+.*)?$'
        ),
        description="Matches English structural headers (Chapter 1, Section 2.1, Appendix A, Article IV)"
    )

    # 4. Numeric Sections (1., 1.1, 1.1.2)
    numeric_sections: Pattern = Field(
        default=re.compile(r'^[0-9]+(?:\.[0-9]+)*\s+.*$'),
        description="Matches hierarchical numeric section headers (1., 1.1, 1.1.2)"
    )

    # 5. Enumerate Listings - Chinese & English (一、, A., (a), 1))
    listings: Pattern = Field(
        default=re.compile(r'^(?:[一二三四五六七八九十]+[、,]|[A-Za-z0-9]+[\.\)])\s+.*$'),
        description="Matches Chinese and English enumeration listings (一、, A., 1), a.)"
    )

    # 6. Bracketed Section Headers ([摘要], (二), [Note], (1))
    bracket_sections: Pattern = Field(
        default=re.compile(r'^(?:\[[^\]]+\]|\（[0-9一二三四五六七八九十]+\）|\([0-9a-zA-Z]+\))\s*.*$'),
        description="Matches bracketed section tags ([Summary], （一）, (a), [Note])"
    )

    # 7. Common English Document Structural Markers (Overview, Executive Summary)
    english_keywords: Pattern = Field(
        default=re.compile(
            r'^(?i:'
            r'abstract|introduction|overview|background|executive summary|'
            r'table of contents|scope|prerequisites|architecture|conclusion|'
            r'references|acknowledgments|faq|changelog'
            r')(?:\s*[:\.-]\s*.*)?$'
        ),
        description="Matches standard English standalone structural section words"
    )

    class Config:
        arbitrary_types_allowed = True

    def get_all_patterns(self) -> List[Pattern]:
        """Returns all compiled regular expressions as a list."""
        return [
            self.markdown_headers,
            self.chinese_chapters,
            self.english_chapters,
            self.numeric_sections,
            self.listings,
            self.bracket_sections,
            self.english_keywords
        ]


# Default singleton instance for general usage
DEFAULT_HEADER_PATTERNS = HeaderPatternCollection()
