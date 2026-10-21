import pytest
from pydantic import ValidationError
from auraflux_core.rag.schemas.agents import ExtractKeywordsAgentConfig


def test_extract_keywords_agent_config_defaults(base_agent_params):
    """
    Test Case: Verify ExtractKeywordsAgentConfig uses expected default values.
    """
    config = ExtractKeywordsAgentConfig(**base_agent_params)
    assert config.temperature == 0.0
    assert config.max_tokens == 32768
    assert config.pipeline_name == 'plan_and_execute'

def test_extract_keywords_agent_config_invalid_max_tokens(base_agent_params):
    """
    Test Case: Verify ValidationError is raised when max_tokens is invalid (e.g., -1).
    """
    with pytest.raises(ValidationError):
        ExtractKeywordsAgentConfig(**base_agent_params, max_tokens=-1)

def test_extract_keywords_agent_config_invalid_temperature(base_agent_params):
    """
    Test Case: Verify ValidationError is raised when temperature is out of range (e.g., -0.1 or 3.0).
    """
    with pytest.raises(ValidationError):
        ExtractKeywordsAgentConfig(**base_agent_params, temperature=-0.1)

    with pytest.raises(ValidationError):
        ExtractKeywordsAgentConfig(**base_agent_params, temperature=3.0)
