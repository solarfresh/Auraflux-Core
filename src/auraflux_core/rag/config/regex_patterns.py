import re
from typing import List, Pattern

from pydantic import BaseModel, ConfigDict, Field


class AnaphoraPatternCollection(BaseModel):
    """
    Strongly-typed collection of compiled Regular Expressions for detecting
    anaphoric pronouns, demonstratives, and discourse connectors across English and CJK text.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    anaphora_reference_pattern: Pattern = Field(
        default=re.compile(
            r'^\s*(?:[」』”’"\'\）\]]*\s*)?'  # 可選的前置引號/括號與空白
            r'(?:'
            # 1. CJK 指代詞與脈絡錨點
            r'這[句項種個本篇案點話]|此[項個點類言]|上述|前述|該[項個條]|這些|那些|對此|'
            r'因此|結果|總結來說|換句話說|如前所述|'
            # 2. English Demonstratives & Discourse Anchors
            r'\bthis\b|\bthese\b|\bthose\b|\bsuch\b|\btherefore\b|\bas\s+a\s+result\b|\bhowever\b|'
            r'\bin\s+other\s+words\b|\bfor\s+instance\b|\bin\s+addition\b|\bfurthermore\b'  # 修復: 補上缺少的運算子並加上 \b
            r')'
            r'[\s\n\u4e00-\u9fff,，:\.\-\(（]',
            re.IGNORECASE
        ),
        description="Matches text starting with strong anaphoric pronouns or contextually dependent connectors."
    )


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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


class KeywordPatternCollection(BaseModel):
    """
    Strongly-typed collection of compiled Regular Expressions for keyword and triple processing.
    Handles enumeration delimiters (CJK and Western) via primary and secondary regex passes.
    """
    # Pass 1: Primary pattern for explicit punctuation, thousands separators, and English connectors
    explicit_delimiter_splitter: re.Pattern = Field(
        default=re.compile(
            # 1. Punctuation followed by optional spaces and single-char Chinese connectors
            # Strictly requires preceding punctuation (e.g., "、並", "、及", ",與", "；或")
            r'[,，、;；]\s*(?:及|與|並|跟|或)|'

            # 2. English conjunctions (consumes optional preceding comma and surrounding whitespace)
            # Prevents duplicate splitting between a comma and 'and' which leaves empty string tokens
            r'(?:[,，]\s*)?\s+(?:and|or|plus|along with|together with|as well as)\s+|'

            # 3. Multi-character Chinese connectors
            r'以及|暨|'

            # 4. Standalone commas not acting as thousands separators
            # Skips "1,000" while properly splitting commas in lists like "5, 10" and "A, 5,"
            r'(?<!\d),(?!\d)|(?<=\d),(?!\d{3}(?!\d))|'

            # 5. Standalone Chinese punctuation marks
            r'[，、;；]',
            re.IGNORECASE
        ),
        description="First-pass pattern splitting explicit CJK/Western punctuation and primary conjunctions."
    )

    # Pass 2: Secondary pattern for inline CJK verb/noun parallel phrase connectors (e.g., "成長並擴張")
    contextual_enumeration_splitter: re.Pattern = Field(
        default=re.compile(
            # Matches "並" / "與" / "及" when positioned between two 2-char CJK terms
            r"(?:(?<=[\u4e00-\u9fa5\w”」』】）\)])\s*(?:並|與|及|暨|跟)\s*(?=[\u4e00-\u9fa5\w“「『【（\(]))"
        ),
        description="Second-pass pattern targeting inline CJK connectors in parallel verb/noun phrases."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ----------------------------------------------------------------------
# Compiled Noise Patterns Collection
# ----------------------------------------------------------------------
class NoisePatternCollection(BaseModel):
    """
    Strongly-typed collection of compiled Regular Expressions for noise detection.
    Matches page numbers, headers, footers, copyrights, and table of contents entries.
    """
    page_number: Pattern = Field(
        default=re.compile(r'Page\s+\d+\s+of\s+\d+|第\s*\d+\s*頁/共\s*\d+\s*頁', re.IGNORECASE),
        description="Matches standard page numbering formats (e.g., 'Page 1 of 5', '第 1 頁/共 5 頁')."
    )
    copyright_notice: Pattern = Field(
        default=re.compile(r'All\s+Rights\s+Reserved|Copyright\s+©|版權所有', re.IGNORECASE),
        description="Matches legal copyright and rights reservation statements."
    )
    table_of_contents: Pattern = Field(
        default=re.compile(r'^\s*(Table\s+of\s+Contents|目\s*錄)\s*$', re.IGNORECASE),
        description="Matches Table of Contents header lines."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


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

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TripleCheckerPatternCollection(BaseModel):
    """
    Strongly-typed collection of compiled Regular Expressions for TripleRuleCheckerTool.
    Supports CJK (Chinese, Japanese) and English pronoun, noise, and structural flaw detection.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pronoun_pattern: Pattern = Field(
        default=re.compile(
            r"(?:"
            # English Personal & Possessive Pronouns
            r"\b(?:your|yours|you|my|mine|our|ours|us|their|theirs)\b|"
            # CJK Personal & Possessive Pronouns
            r"你[的們]?|我[的們]?|您[的]?|他[的們]?|她[的們]?|它[的們]?"
            r")",
            re.IGNORECASE
        ),
        description="Matches English and CJK personal and possessive pronouns inside predicates."
    )

    weak_predicate_pattern: Pattern = Field(
        default=re.compile(
            r"(?:"
            # English Weak Verb Phrases
            r"\b(?:gives?\s+\w+|provides?\s+\w+\s+with|helps?\s+\w+|allows?\s+\w+\s+to)\b|"
            # CJK Weak Verb Phrases
            r"提供[你您我他給]?|給予[你您我他給]?|讓[你您我他]|幫[你您我他]|為[你您我他]"
            r")",
            re.IGNORECASE
        ),
        description="Matches predicates containing weak verbs or embedded personal object fillers."
    )

    @staticmethod
    def build_spo_patterns(subject: str, predicate: str, obj: str) -> List[Pattern]:
        """
        Dynamically constructs compiled regex patterns for Subject-Predicate-Object (S-P-O)
        and Subject-Object-Predicate (S-O-P) order verification without distance constraints.
        """
        s_esc = re.escape(subject)
        p_esc = re.escape(predicate)
        o_esc = re.escape(obj)

        # Standard SPO: Subject -> Predicate -> Object
        spo_regex = re.compile(rf"{s_esc}.*?{p_esc}.*?{o_esc}", re.IGNORECASE | re.DOTALL)
        # Inverted / Passive SOP: Subject -> Object -> Predicate
        sop_regex = re.compile(rf"{s_esc}.*?{o_esc}.*?{p_esc}", re.IGNORECASE | re.DOTALL)

        return [spo_regex, sop_regex]


# Default singleton instance for general usage
DEFAULT_ANAPHORA_PATTERNS = AnaphoraPatternCollection()
DEFAULT_KEYWORD_PATTERNS = KeywordPatternCollection()
DEFAULT_HEADER_PATTERNS = HeaderPatternCollection()
DEFAULT_NOISE_PATTERNS = NoisePatternCollection()
DEFAULT_SENTENCE_PATTERNS = SentencePatternCollection()
DEFAULT_TRIPLE_CHECKER_PATTERNS = TripleCheckerPatternCollection()
