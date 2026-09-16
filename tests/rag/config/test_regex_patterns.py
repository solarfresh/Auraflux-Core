import pytest
from auraflux_core.rag.config.regex_patterns import DEFAULT_KEYWORD_PATTERNS

patterns = DEFAULT_KEYWORD_PATTERNS
explicit_delimiter_splitter = patterns.explicit_delimiter_splitter
contextual_enumeration_splitter = patterns.contextual_enumeration_splitter


class TestExplicitDelimiterThousandsSeparator:

    def test_simple_thousands(self):
        assert explicit_delimiter_splitter.split("10,000") == ["10,000"]

    def test_multiple_groups(self):
        assert explicit_delimiter_splitter.split("2,536,227") == ["2,536,227"]

    def test_number_with_trailing_text(self):
        result = explicit_delimiter_splitter.split("59,026 Express.js \"hello world\" HTTP requests per second")
        assert result == ["59,026 Express.js \"hello world\" HTTP requests per second"]

    def test_number_embedded_in_sentence(self):
        result = explicit_delimiter_splitter.split("Bun bundles 10,000 React components in 269ms")
        assert result == ["Bun bundles 10,000 React components in 269ms"]


class TestExplicitDelimiterCommaAndEnglishConnectors:

    def test_basic_comma_list(self):
        raw = explicit_delimiter_splitter.split("Python, Java, Go")
        assert [x.strip() for x in raw if x.strip()] == ["Python", "Java", "Go"]

    def test_comma_with_space_after_number(self):
        raw = explicit_delimiter_splitter.split("the values are A, 5, and C")
        assert [x.strip() for x in raw if x.strip()] == ["the values are A", "5", "C"]

    def test_mixed_number_and_list(self):
        raw = explicit_delimiter_splitter.split("5, 10, and 15")
        assert [x.strip() for x in raw if x.strip()] == ["5", "10", "15"]

    def test_english_connectors(self):
        assert [x.strip() for x in explicit_delimiter_splitter.split("A and B") if x.strip()] == ["A", "B"]
        assert [x.strip() for x in explicit_delimiter_splitter.split("A or B") if x.strip()] == ["A", "B"]
        assert [x.strip() for x in explicit_delimiter_splitter.split("A as well as B") if x.strip()] == ["A", "B"]

    def test_or_does_not_break_words(self):
        raw = explicit_delimiter_splitter.split("before the deadline")
        assert [x.strip() for x in raw if x.strip()] == ["before the deadline"]


class TestExplicitDelimiterCJKPunctuationAndMultiChar:

    def test_dunhao_and_comma_split(self):
        assert [x.strip() for x in explicit_delimiter_splitter.split("蘋果、香蕉、橘子") if x.strip()] == ["蘋果", "香蕉", "橘子"]
        assert [x.strip() for x in explicit_delimiter_splitter.split("設計，製造，測試") if x.strip()] == ["設計", "製造", "測試"]

    def test_multi_char_connectors(self):
        assert [x.strip() for x in explicit_delimiter_splitter.split("設計以及製造") if x.strip()] == ["設計", "製造"]
        assert [x.strip() for x in explicit_delimiter_splitter.split("產學暨研究中心") if x.strip()] == ["產學", "研究中心"]

    def test_punctuation_preceded_connectors(self):
        assert [x.strip() for x in explicit_delimiter_splitter.split("A、B、及C") if x.strip()] == ["A", "B", "C"]
        assert [x.strip() for x in explicit_delimiter_splitter.split("設計,與製造") if x.strip()] == ["設計", "製造"]


class TestContextualInlineCJKConnectors:

    def test_inline_parallel_verbs_split(self):
        assert contextual_enumeration_splitter.split("成長並擴張") == ["成長", "擴張"]
        assert contextual_enumeration_splitter.split("整理與分析") == ["整理", "分析"]
