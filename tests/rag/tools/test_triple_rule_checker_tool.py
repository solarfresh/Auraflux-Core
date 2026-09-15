import pytest

from auraflux_core.core.schemas.tools import ToolConfig
from auraflux_core.rag.tools.triple_processor_tool import TripleRuleCheckerTool


@pytest.fixture
def checker_tool():
    """Fixture to initialize TripleRuleCheckerTool with default configurations."""
    return TripleRuleCheckerTool(config=ToolConfig())


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
