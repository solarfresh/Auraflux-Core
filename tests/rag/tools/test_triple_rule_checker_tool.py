from typing import Any, Dict

import pytest

from auraflux_core.core.schemas.tools import ToolConfig
from auraflux_core.rag.tools.triple_processor_tool import TripleRuleCheckerTool


@pytest.fixture
def checker_tool():
    """Fixture to initialize TripleRuleCheckerTool with default configurations."""
    return TripleRuleCheckerTool(config=ToolConfig())

def create_triple(subject: str, predicate: str, obj: str) -> Dict[str, Any]:
    """Helper function to create a minimal valid triple dictionary."""
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "subject_resolved": None,
        "data_target": None,
        "metric_name": None,
        "normalized_value": None,
        "unit": None,
        "operator": None,
    }

# ==============================================================================
# 1. Inspect Single Triple Test Cases
# ==============================================================================

def test_clean_triple_passes(checker_tool):
    """Verifies that a valid and compliant triple passes inspection without error reasons."""
    valid_triple = {
        "subject": "Bun v1.0",
        "subject_resolved": None,
        "predicate": "supports",
        "object": "TypeScript SDK",
        "data_target": None,
        "metric_name": None,
        "normalized_value": None,
        "unit": None,
        "operator": None,
    }
    reasons = checker_tool.inspect_single_triple(valid_triple)
    assert len(reasons) == 0


def test_english_pronoun_in_predicate(checker_tool):
    """Detects English personal/possessive pronouns inside the predicate (e.g., 'gives your agents')."""
    flawed_triple = {
        "subject": "Cosmic",
        "predicate": "gives your agents",
        "object": "TypeScript SDK",
    }
    reasons = checker_tool.inspect_single_triple(flawed_triple)
    assert any("personal/possessive pronoun" in r for r in reasons)


def test_chinese_pronoun_in_predicate(checker_tool):
    """Detects CJK personal/possessive pronouns inside the predicate (e.g., '提供你的代理人')."""
    flawed_triple = {
        "subject": "Cosmic",
        "predicate": "提供你的代理人",
        "object": "TypeScript SDK",
    }
    reasons = checker_tool.inspect_single_triple(flawed_triple)
    assert any("personal/possessive pronoun" in r for r in reasons)


def test_weak_verb_phrase_in_predicate(checker_tool):
    """Detects weak verb phrases with embedded target fillers (e.g., 'provides user with')."""
    flawed_triple = {
        "subject": "System",
        "predicate": "provides user with",
        "object": "API Key",
    }
    reasons = checker_tool.inspect_single_triple(flawed_triple)
    assert any("weak verb phrase" in r for r in reasons)


def test_excessive_predicate_length(checker_tool):
    """Detects predicates exceeding the maximum word count threshold (>6 words)."""
    flawed_triple = {
        "subject": "Model",
        "predicate": "is capable of performing advanced deep learning reasoning on",
        "object": "Data",
    }
    reasons = checker_tool.inspect_single_triple(flawed_triple)
    assert any("Predicate is too long" in r for r in reasons)


def test_unresolved_anaphora_in_subject(checker_tool):
    """Detects unresolved demonstrative pronouns in subject without subject_resolved field."""
    flawed_triple = {
        "subject": "這項技術",
        "subject_resolved": None,  # Coreference not resolved
        "predicate": "降低",
        "object": "延遲",
    }
    reasons = checker_tool.inspect_single_triple(flawed_triple)
    assert any("unresolved anaphoric pronoun" in r for r in reasons)


def test_resolved_anaphora_passes(checker_tool):
    """Verifies that anaphoric pronouns pass inspection when subject_resolved is populated."""
    valid_triple = {
        "subject": "這項技術",
        "subject_resolved": "AuraFlux Engine",  # Properly resolved
        "predicate": "降低",
        "object": "延遲",
    }
    reasons = checker_tool.inspect_single_triple(valid_triple)
    assert len(reasons) == 0


def test_empty_nodes_detection(checker_tool):
    """Detects empty or whitespace-only subject/object node fields."""
    flawed_triple = {
        "subject": "  ",
        "predicate": "connects to",
        "object": "Database",
    }
    reasons = checker_tool.inspect_single_triple(flawed_triple)
    assert any("Subject or Object is empty" in r for r in reasons)


# ==============================================================================
# 2. Filter & Workflow Batch Processing Test Cases
# ==============================================================================

def test_filter_triples_splitting(checker_tool):
    """Verifies that batch filtering correctly routes triples into clean and flagged sets."""
    raw_triples = [
        {
            "subject": "Bun",
            "predicate": "executes",
            "object": "JavaScript",
        },
        {
            "subject": "Cosmic",
            "predicate": "gives your agents",
            "object": "SDK",
        },
    ]

    clean, flagged = checker_tool.filter_triples(raw_triples)

    assert len(clean) == 1
    assert clean[0].subject == "Bun"

    assert len(flagged) == 1
    assert flagged[0].subject == "Cosmic"
    assert hasattr(flagged[0], "flag_reasons")
    assert len(flagged[0].flag_reasons) > 0


# ==============================================================================
# 3. Async Tool Run Execution Test Cases
# ==============================================================================

@pytest.mark.asyncio
async def test_tool_async_run_interface(checker_tool):
    """Verifies the async BaseTool.run() execution interface and dictionary response structure."""
    raw_triples = [
        {"subject": "A", "predicate": "helps user to", "object": "B"},
        {"subject": "X", "predicate": "binds", "object": "Y"},
    ]

    result = await checker_tool.run(triples=raw_triples)

    assert result["has_flagged"] is True
    assert result["total_count"] == 2
    assert result["flagged_count"] == 1
    assert len(result["clean_triples"]) == 1
    assert len(result["flagged_triples"]) == 1

def test_anomaly_invalid_operator(checker_tool):
    """Detects anomaly where operator contains unauthorized text (e.g., 'decrease') instead of valid enum symbols."""
    flawed_triple = {
        "subject": "memory use",
        "predicate": "fell between",
        "object": "13 and 48 percent",
        "operator": "decrease",  # Invalid
    }
    reasons = checker_tool.inspect_single_triple(flawed_triple)
    # Must fail if the rule checker doesn't catch it
    assert len(reasons) > 0, "Expected rule checker to flag invalid operator, but none found."
    assert any("operator" in r.lower() for r in reasons)


def test_anomaly_missing_normalized_value_with_unit(checker_tool):
    """Detects anomaly where a quantitative unit is present but normalized_value is improperly left null."""
    flawed_triple = {
        "subject": "memory use",
        "predicate": "fell between",
        "object": "13 and 48 percent",
        "metric_name": "memory_usage",
        "normalized_value": None,  # Invalid: unit is percent
        "unit": "percent",
    }
    reasons = checker_tool.inspect_single_triple(flawed_triple)
    # Must fail if the rule checker doesn't catch it
    assert len(reasons) > 0, "Expected rule checker to flag missing normalized_value, but none found."
    assert any("normalized_value" in r.lower() or "metric" in r.lower() for r in reasons)


def test_anomaly_unnecessary_subject_resolved(checker_tool):
    """Detects anomaly where subject_resolved is improperly populated for a standard non-anaphoric subject noun."""
    flawed_triple = {
        "subject": "memory use",
        "subject_resolved": "memory_usage",  # Invalid: not an anaphoric pronoun
        "predicate": "fell between",
        "object": "13 and 48 percent",
    }
    reasons = checker_tool.inspect_single_triple(flawed_triple)
    assert len(reasons) > 0, "Expected rule checker to flag unnecessary subject_resolved, but none found."
    assert any("subject_resolved" in r.lower() for r in reasons)

def test_clean_metric_all_or_none_passes(checker_tool):
    """Verifies that both fields being present (or both being null) passes the consistency check cleanly."""
    # Case A: Both present (valid metric)
    valid_metric_triple = {
        "subject": "CPU usage",
        "predicate": "is",
        "object": "85 percent",
        "metric_name": "cpu_utilization",
        "normalized_value": 85.0,
        "unit": "percent",
        "operator": "==",
    }
    assert len(checker_tool.inspect_single_triple(valid_metric_triple)) == 0

    # Case B: Both null (valid non-metric triple)
    valid_non_metric_triple = {
        "subject": "Bun",
        "predicate": "supports",
        "object": "TypeScript",
        "metric_name": None,
        "normalized_value": None,
        "unit": None,
        "operator": None,
    }
    assert len(checker_tool.inspect_single_triple(valid_non_metric_triple)) == 0

# ==============================================================================
# 5. Metric Consistency Mixed/Partial None Test Cases (New)
# ==============================================================================

def test_clean_metric_partial_none_optional_fields(checker_tool):
    """Verifies that unit being None while core fields and operator are present passes cleanly."""

    # Case 1: unit is None, but core metric and operator are present -> Pass
    triple_missing_unit = {
        "subject": "count",
        "predicate": "equals",
        "object": "50",
        "metric_name": "item_count",
        "normalized_value": 50.0,
        "unit": None,      # unit is optional (can be None)
        "operator": "==",  # operator is mandatory when metric is present
    }
    assert len(checker_tool.inspect_single_triple(triple_missing_unit)) == 0

def test_anomaly_missing_operator_when_metric_present(checker_tool):
    """Detects anomaly where metric data is present but operator is improperly left null/None."""
    flawed_triple = {
        "subject": "count",
        "predicate": "equals",
        "object": "50",
        "metric_name": "item_count",
        "normalized_value": 50.0,
        "unit": None,
        "operator": None,  # Invalid: operator is now mandatory
    }
    reasons = checker_tool.inspect_single_triple(flawed_triple)
    assert len(reasons) > 0, "Expected rule checker to flag missing operator, but none found."
    assert any("operator" in r.lower() for r in reasons)

def test_anomaly_orphan_unit_without_core_metric(checker_tool):
    """Detects anomaly where a unit or operator is provided, but core metric fields (metric_name & normalized_value) are null."""
    flawed_triple = {
        "subject": "response time",
        "predicate": "takes",
        "object": "fast",
        "metric_name": None,       # Missing core
        "normalized_value": None,  # Missing core
        "unit": "ms",              # Orphan unit!
        "operator": None,
    }
    reasons = checker_tool.inspect_single_triple(flawed_triple)
    assert len(reasons) > 0, "Expected rule checker to flag orphan unit/operator without core metric, but none found."
    assert any("Inconsistent metric data" in r for r in reasons)

# =============================================================================
# Bilingual Unit Tests for SPO Grounding & Sequence Rules
# =============================================================================

class TestTripleSPORules:

    def test_valid_spo_sequence_english(self, checker_tool):
        """Tests standard Subject-Predicate-Object order in English."""
        excerpt = "Apple Inc. announced the new iPhone 16 at the annual keynote event."
        triple = create_triple(
            subject="Apple Inc.",
            predicate="announced",
            obj="iPhone 16"
        )

        result = checker_tool.inspect_triples(triples=[triple], excerpt_text=excerpt)

        assert len(result.clean_triples) == 1
        assert len(result.flagged_triples) == 0
        assert not result.has_flagged

    def test_valid_spo_sequence_chinese(self, checker_tool):
        """Tests standard Subject-Predicate-Object order in Traditional Chinese."""
        excerpt = "台積電在南部科學園區興建了最先進的二奈米晶圓廠。"
        triple = create_triple(
            subject="台積電",
            predicate="興建了",
            obj="二奈米晶圓廠"
        )

        result = checker_tool.inspect_triples(triples=[triple], excerpt_text=excerpt)

        assert len(result.clean_triples) == 1
        assert len(result.flagged_triples) == 0

    def test_valid_sop_passive_sequence_english(self, checker_tool):
        """Tests valid passive/inverted Subject-Object-Predicate order in English."""
        excerpt = "The fiscal budget for 2025 was approved by the board of directors."
        triple = create_triple(
            subject="The fiscal budget",
            predicate="approved",
            obj="board of directors"
        )

        result = checker_tool.inspect_triples(triples=[triple], excerpt_text=excerpt)

        assert len(result.clean_triples) == 1
        assert len(result.flagged_triples) == 0

    def test_valid_sop_passive_sequence_chinese(self, checker_tool):
        """Tests valid passive/inverted Subject-Object-Predicate order in Traditional Chinese."""
        excerpt = "這項全新的AI服務由專案小組於上個月成功研發。"
        triple = {
            "subject": "這項全新的AI服務",
            "subject_resolved": "全新的AI服務",
            "predicate": "研發",
            "object": "專案小組",
            "data_target": None,
            "metric_name": None,
            "normalized_value": None,
            "unit": None,
            "operator": None,
        }

        result = checker_tool.inspect_triples(triples=[triple], excerpt_text=excerpt)

        assert len(result.clean_triples) == 1
        assert len(result.flagged_triples) == 0

    def test_invalid_out_of_order_sequence_english(self, checker_tool):
        """
        Tests flagging when components appear in wrong order (e.g. Predicate before Subject).
        In this excerpt, 'launched' appears BEFORE 'Google'.
        """
        excerpt = "Recently launched products include Pixel 9, which Google engineered with AI."
        triple = create_triple(
            subject="Google",
            predicate="launched",
            obj="Pixel 9"
        )

        result = checker_tool.inspect_triples(triples=[triple], excerpt_text=excerpt)

        assert len(result.clean_triples) == 0
        assert len(result.flagged_triples) == 1

        flagged_reasons = result.flagged_triples[0].flag_reasons
        assert any("do not follow a valid sequence order" in reason for reason in flagged_reasons)

    def test_invalid_out_of_order_sequence_chinese(self, checker_tool):
        """
        Tests flagging when components appear in inverted order not matching SPO or SOP.
        Here predicate '出版' appears before subject '聯經出版'.
        """
        excerpt = "隆重出版這本歷史巨著的是著名的聯經出版公司。"
        triple = create_triple(
            subject="聯經出版公司",
            predicate="出版",
            obj="歷史巨著"
        )

        result = checker_tool.inspect_triples(triples=[triple], excerpt_text=excerpt)

        assert len(result.clean_triples) == 0
        assert len(result.flagged_triples) == 1

        flagged_reasons = result.flagged_triples[0].flag_reasons
        assert any("do not follow a valid sequence order" in reason for reason in flagged_reasons)

    def test_missing_component_in_excerpt_english(self, checker_tool):
        """Tests flagging when a triple entity is completely absent from the excerpt."""
        excerpt = "Microsoft acquired GitHub to empower developers globally."
        triple = create_triple(
            subject="Microsoft",
            predicate="acquired",
            obj="GitLab"  # Absent entity
        )

        result = checker_tool.inspect_triples(triples=[triple], excerpt_text=excerpt)

        assert len(result.clean_triples) == 0
        assert len(result.flagged_triples) == 1

        flagged_reasons = result.flagged_triples[0].flag_reasons
        assert any("Object 'GitLab' is not found" in reason for reason in flagged_reasons)

    def test_missing_component_in_excerpt_chinese(self, checker_tool):
        """Tests flagging when a Chinese predicate is completely absent from the excerpt."""
        excerpt = "華碩電腦在台北發表了最新的雙螢幕筆記型電腦。"
        triple = create_triple(
            subject="華碩電腦",
            predicate="收購",  # Absent predicate (text has '發表了')
            obj="筆記型電腦"
        )

        result = checker_tool.inspect_triples(triples=[triple], excerpt_text=excerpt)

        assert len(result.clean_triples) == 0
        assert len(result.flagged_triples) == 1

        flagged_reasons = result.flagged_triples[0].flag_reasons
        assert any("Predicate '收購' is not found" in reason for reason in flagged_reasons)

    def test_multiple_triples_mixed_results(self, checker_tool):
        """Tests processing multiple triples simultaneously where some pass and some fail."""
        excerpt = "NVIDIA developed CUDA technology to accelerate high performance computing."

        valid_triple = create_triple(
            subject="NVIDIA",
            predicate="developed",
            obj="CUDA technology"
        )
        invalid_triple = create_triple(
            subject="CUDA technology",
            predicate="developed",
            obj="NVIDIA"  # Out of order (Object before Subject in text)
        )

        result = checker_tool.inspect_triples(
            triples=[valid_triple, invalid_triple],
            excerpt_text=excerpt
        )

        assert result.total_count == 2
        assert len(result.clean_triples) == 1
        assert len(result.flagged_triples) == 1
        assert result.has_flagged is True
