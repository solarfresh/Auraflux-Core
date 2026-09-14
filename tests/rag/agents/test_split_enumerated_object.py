from unittest.mock import MagicMock

import pytest

from auraflux_core.rag.agents.keywords_extractor import ExtractKeywordsAgent
from auraflux_core.rag.config.regex_patterns import DEFAULT_KEYWORD_PATTERNS


@pytest.fixture
def agent():
    mock_config = MagicMock()
    mock_config.name = "TestKeywordsExtractor"
    mock_config.pipeline_name = "plan_and_execute"
    mock_config.language = "zh"
    mock_config.llm.provider = "openai"
    mock_config.llm.model = "gpt-4o"
    mock_config.llm.temperature = 0.0
    mock_config.llm.max_tokens = 2000

    mock_client_manager = MagicMock()

    return ExtractKeywordsAgent(
        config=mock_config, client_manager=mock_client_manager
    )


class TestSplitEnumeratedObject:

    def test_preserve_thousands_separator_and_embedded_numbers(self, agent):
        patterns = DEFAULT_KEYWORD_PATTERNS

        assert agent._split_enumerated_object("10,000", patterns) == ["10,000"]
        assert agent._split_enumerated_object("2,536,227", patterns) == ["2,536,227"]

        long_text = "59,026 Express.js \"hello world\" HTTP requests per second"
        assert agent._split_enumerated_object(long_text, patterns) == [long_text]

        benchmark_text = "Bun bundles 10,000 React components in 269ms"
        assert agent._split_enumerated_object(benchmark_text, patterns) == [benchmark_text]

    def test_explicit_delimiters_and_english_connectors(self, agent):
        patterns = DEFAULT_KEYWORD_PATTERNS

        assert agent._split_enumerated_object("Python, Java, Go", patterns) == ["Python", "Java", "Go"]
        assert agent._split_enumerated_object("the values are A, 5, and C", patterns) == ["the values are A", "5", "C"]
        assert agent._split_enumerated_object("5, 10, and 15", patterns) == ["5", "10", "15"]
        assert agent._split_enumerated_object("A as well as B", patterns) == ["A", "B"]
        assert agent._split_enumerated_object("before the deadline", patterns) == ["before the deadline"]

    def test_cjk_punctuation_and_multi_char_connectors(self, agent):
        patterns = DEFAULT_KEYWORD_PATTERNS

        assert agent._split_enumerated_object("蘋果、香蕉、橘子", patterns) == ["蘋果", "香蕉", "橘子"]
        assert agent._split_enumerated_object("設計，製造，測試", patterns) == ["設計", "製造", "測試"]
        assert agent._split_enumerated_object("設計以及製造", patterns) == ["設計", "製造"]
        assert agent._split_enumerated_object("產學暨研究中心", patterns) == ["產學", "研究中心"]
        assert agent._split_enumerated_object("A、B、及C", patterns) == ["A", "B", "C"]
        assert agent._split_enumerated_object("設計,與製造", patterns) == ["設計", "製造"]

    # ----------------------------------------------------------------------
    # 4. 上下文感知的兩階段列舉拆解 (Context-Aware Two-Stage Split)
    # ----------------------------------------------------------------------
    def test_two_stage_contextual_enumeration_splitting(self, agent):
        patterns = DEFAULT_KEYWORD_PATTERNS

        assert agent._split_enumerated_object("獲利、成長並擴張", patterns) == ["獲利", "成長", "擴張"]
        assert agent._split_enumerated_object("收集、整理並分析", patterns) == ["收集", "整理", "分析"]
        assert agent._split_enumerated_object("蘋果、香蕉跟橘子", patterns) == ["蘋果", "香蕉", "橘子"]

        assert agent._split_enumerated_object("設計並製造", patterns) == ["設計並製造"]
        assert agent._split_enumerated_object("本季營收並未達成目標", patterns) == ["本季營收並未達成目標"]
        assert agent._split_enumerated_object("結果並非如此", patterns) == ["結果並非如此"]
        assert agent._split_enumerated_object("此政策涉及個資保護規範", patterns) == ["此政策涉及個資保護規範"]
        assert agent._split_enumerated_object("該公司積極參與國際標準制定", patterns) == ["該公司積極參與國際標準制定"]

    def test_text_stripping_and_clean_up_integration(self, agent):
        patterns = DEFAULT_KEYWORD_PATTERNS

        assert agent._split_enumerated_object("《蘋果》, 『香蕉』, 「橘子」", patterns) == ["蘋果", "香蕉", "橘子"]
        assert agent._split_enumerated_object("「成長」並『擴張』、獲利", patterns) == ["成長", "擴張", "獲利"]
