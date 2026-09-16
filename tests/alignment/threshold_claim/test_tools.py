import pytest
from auraflux_core.alignment.threshold_claim.tools import ThresholdCalculatorTool


@pytest.fixture
def calculator_tool():
    return ThresholdCalculatorTool()


@pytest.mark.asyncio
async def test_calculator_status_verified(calculator_tool):
    """Test Case 1: All metrics match baseline and satisfy boundary conditions (VERIFIED)."""
    claim_metrics = [
        {
            "raw_text": "500萬TWD",
            "metric_name": "budget",
            "normalized_value": 5000000.0,
            "unit": "TWD",
            "operator": "<="
        }
    ]
    baseline_metrics = [
        {
            "raw_text": "600萬TWD",
            "metric_name": "budget",
            "normalized_value": 6000000.0,
            "unit": "TWD",
            "operator": "<="
        }
    ]

    result = await calculator_tool.run(claim_metrics, baseline_metrics)

    assert result["status"] == "VERIFIED"
    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0]["is_satisfied"] is True
    assert result["comparisons"][0]["overflow_value"] == 0.0
    assert len(result["unprocessed_warnings"]) == 0


@pytest.mark.asyncio
async def test_calculator_status_violated(calculator_tool):
    """Test Case 2: Claimed metric exceeds baseline boundary limit (VIOLATED)."""
    claim_metrics = [
        {
            "raw_text": "700萬TWD",
            "metric_name": "budget",
            "normalized_value": 7000000.0,
            "unit": "TWD",
            "operator": "<="
        }
    ]
    baseline_metrics = [
        {
            "raw_text": "600萬TWD",
            "metric_name": "budget",
            "normalized_value": 6000000.0,
            "unit": "TWD",
            "operator": "<="
        }
    ]

    result = await calculator_tool.run(claim_metrics, baseline_metrics)

    assert result["status"] == "VIOLATED"
    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0]["is_satisfied"] is False
    assert result["comparisons"][0]["overflow_value"] == 1000000.0


@pytest.mark.asyncio
async def test_calculator_status_partially_verified_unit_mismatch(calculator_tool):
    """Test Case 3: Unit mismatch or unresolved boundary ambiguity (PARTIALLY_VERIFIED)."""
    claim_metrics = [
        {
            "raw_text": "500萬USD",
            "metric_name": "budget",
            "normalized_value": 5000000.0,
            "unit": "USD",
            "operator": "<="
        }
    ]
    baseline_metrics = [
        {
            "raw_text": "600萬TWD",
            "metric_name": "budget",
            "normalized_value": 6000000.0,
            "unit": "TWD",
            "operator": "<="
        }
    ]

    result = await calculator_tool.run(claim_metrics, baseline_metrics)

    assert result["status"] == "PARTIALLY_VERIFIED"
    assert result["comparisons"][0]["is_satisfied"] is None
    assert len(result["unprocessed_warnings"]) == 1
    assert "Unit mismatch" in result["unprocessed_warnings"][0]


@pytest.mark.asyncio
async def test_calculator_status_unsupported(calculator_tool):
    """Test Case 4: Missing corresponding baseline metric data (UNSUPPORTED)."""
    claim_metrics = [
        {
            "raw_text": "500萬TWD",
            "metric_name": "budget",
            "normalized_value": 5000000.0,
            "unit": "TWD",
            "operator": "<="
        }
    ]
    baseline_metrics = []  # No baseline metrics retrieved from RAG context

    result = await calculator_tool.run(claim_metrics, baseline_metrics)

    assert result["status"] == "UNSUPPORTED"
    assert len(result["comparisons"]) == 1
    assert result["comparisons"][0]["is_satisfied"] is None
    assert len(result["unprocessed_warnings"]) == 1
    assert "Baseline reference metric missing" in result["unprocessed_warnings"][0]
