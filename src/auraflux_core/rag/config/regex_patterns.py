import re
from typing import List, Pattern
from pydantic import BaseModel, Field


import re
from typing import Pattern
from pydantic import BaseModel, Field


class AnaphoraPatternCollection(BaseModel):
    """
    Strongly-typed collection of compiled Regular Expressions for detecting
    anaphoric pronouns, demonstratives, and discourse connectors across English and CJK text.
    """
    # Detects leading closing quotes/brackets followed by strong demonstratives or discourse markers
    anaphora_reference_pattern: Pattern = Field(
        default=re.compile(
            r'^\s*(?:[」』”’"\'\）\]]*\s*)?'  # Matches optional leading closing quotes or brackets
            r'(?:'
            # CJK Demonstratives & References (中文強指代詞與脈絡錨點)
            r'這[句項種個本篇案點話]|此[項個點類言]|上述|前述|該[項個條]|這些|那些|對此|'
            r'因此|結果|總結來說|換句話說|如前所述|'
            # English Demonstratives & Discourse Anchors
            r'this\s+|these\s+|those\s+|such\s+|therefore|as\s+a\s+result|however|'
            r'in\s+other\s+words|for\s+instance|in\s+addition|furthermore'
            r')'
            r'[\s\n\u4e00-\u9fff,，:\.\-\(（]',
            re.IGNORECASE
        ),
        description="Matches text starting with strong anaphoric pronouns or contextually dependent connectors."
    )

    class Config:
        arbitrary_types_allowed = True


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


class SentencePatternCollection(BaseModel):
    """
    Strongly-typed collection of compiled Regular Expressions for sentence segmentation.
    Supports CJK (Chinese, Japanese, Korean) and Western sentence terminators and line breaks.
    """
    # Standard sentence terminators including CJK and Latin punctuation, plus inline newlines
    default_sentence_splitter: Pattern = Field(
        default=re.compile(r'([^。！？.!?\n]+[。！？.!?\n]*(?:[」』”’"\'）\]]*))'),
        description="Splits text into discrete sentences using CJK/Latin terminators and newlines."
    )

    class Config:
        arbitrary_types_allowed = True


# Default singleton instance for general usage
DEFAULT_ANAPHORA_PATTERNS = AnaphoraPatternCollection()
DEFAULT_HEADER_PATTERNS = HeaderPatternCollection()
DEFAULT_SENTENCE_PATTERNS = SentencePatternCollection()
