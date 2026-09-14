import pytest

from auraflux_core.rag.config.regex_patterns import DEFAULT_KEYWORD_PATTERNS
from auraflux_core.rag.tools.triple_processor_tool import TripleProcessorTool


@pytest.fixture
def tool():
    """Fixture providing a fresh TripleProcessorTool instance without LLM mocks."""
    return TripleProcessorTool()


class TestTripleProcessorTool:

    def test_preserve_thousands_separator_and_embedded_numbers(self, tool):
        patterns = DEFAULT_KEYWORD_PATTERNS

        assert tool._split_enumerated_object("10,000", patterns) == ["10,000"]
        assert tool._split_enumerated_object("2,536,227", patterns) == ["2,536,227"]

        long_text = "59,026 Express.js \"hello world\" HTTP requests per second"
        assert tool._split_enumerated_object(long_text, patterns) == [long_text]

        benchmark_text = "Bun bundles 10,000 React components in 269ms"
        assert tool._split_enumerated_object(benchmark_text, patterns) == [benchmark_text]

    def test_explicit_delimiters_and_english_connectors(self, tool):
        patterns = DEFAULT_KEYWORD_PATTERNS

        assert tool._split_enumerated_object("Python, Java, Go", patterns) == ["Python", "Java", "Go"]
        assert tool._split_enumerated_object("the values are A, 5, and C", patterns) == ["the values are A", "5", "C"]
        assert tool._split_enumerated_object("5, 10, and 15", patterns) == ["5", "10", "15"]
        assert tool._split_enumerated_object("A as well as B", patterns) == ["A", "B"]
        assert tool._split_enumerated_object("before the deadline", patterns) == ["before the deadline"]

    def test_cjk_punctuation_and_multi_char_connectors(self, tool):
        patterns = DEFAULT_KEYWORD_PATTERNS

        assert tool._split_enumerated_object("蘋果、香蕉、橘子", patterns) == ["蘋果", "香蕉", "橘子"]
        assert tool._split_enumerated_object("設計，製造，測試", patterns) == ["設計", "製造", "測試"]
        assert tool._split_enumerated_object("設計以及製造", patterns) == ["設計", "製造"]
        assert tool._split_enumerated_object("產學暨研究中心", patterns) == ["產學", "研究中心"]
        assert tool._split_enumerated_object("A、B、及C", patterns) == ["A", "B", "C"]
        assert tool._split_enumerated_object("設計,與製造", patterns) == ["設計", "製造"]

    def test_two_stage_contextual_enumeration_splitting(self, tool):
        patterns = DEFAULT_KEYWORD_PATTERNS

        assert tool._split_enumerated_object("獲利、成長並擴張", patterns) == ["獲利", "成長", "擴張"]
        assert tool._split_enumerated_object("收集、整理並分析", patterns) == ["收集", "整理", "分析"]
        assert tool._split_enumerated_object("蘋果、香蕉跟橘子", patterns) == ["蘋果", "香蕉", "橘子"]

        assert tool._split_enumerated_object("設計並製造", patterns) == ["設計並製造"]
        assert tool._split_enumerated_object("本季營收並未達成目標", patterns) == ["本季營收並未達成目標"]
        assert tool._split_enumerated_object("結果並非如此", patterns) == ["結果並非如此"]
        assert tool._split_enumerated_object("此政策涉及個資保護規範", patterns) == ["此政策涉及個資保護規範"]
        assert tool._split_enumerated_object("該公司積極參與國際標準制定", patterns) == ["該公司積極參與國際標準制定"]

    def test_text_stripping_and_clean_up_integration(self, tool):
        patterns = DEFAULT_KEYWORD_PATTERNS

        assert tool._split_enumerated_object("《蘋果》, 『香蕉』, 「橘子」", patterns) == ["蘋果", "香蕉", "橘子"]
        assert tool._split_enumerated_object("「成長」並『擴張』、獲利", patterns) == ["成長", "擴張", "獲利"]

    # ----------------------------------------------------------------------
    # Tool 端到端 process_triples 與 run 方法測試
    # ----------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_process_triples_e2e(self, tool):
        raw_triples = [
            {
                "subject": "《Bun》",
                "predicate": "「bundles」",
                "object": "10,000 React components、269ms",
                "metric_name": "count",
                "normalized_value": "10000",
                "unit": "count",
                "operator": "=="
            }
        ]

        processed = tool.process_triples(raw_triples)

        assert len(processed) == 2
        assert processed[0]["subject"] == "Bun"
        assert processed[0]["predicate"] == "bundles"
        assert processed[0]["object"] == "10,000 React components"
        assert processed[0]["normalized_value"] == 10000.0

        assert processed[1]["object"] == "269ms"

    @pytest.mark.asyncio
    async def test_run_async_interface(self, tool):
        raw_triples = [
            {
                "subject": "Python",
                "predicate": "supports",
                "object": "Linux, macOS, and Windows"
            }
        ]

        results = await tool.run(triples=raw_triples)
        assert len(results) == 3
        assert [r["object"] for r in results] == ["Linux", "macOS", "Windows"]
