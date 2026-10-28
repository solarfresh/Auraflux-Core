import pytest
from pydantic import ValidationError
from auraflux_core.core.schemas.orchestrators import OrchestratorConfig


def test_orchestrator_config_defaults():
    config = OrchestratorConfig()
    assert config.timeout == 30.0
    assert config.max_retries == 3
    assert config.strategy is None
    assert config.tools is None
    assert config.agents is None


def test_orchestrator_config_valid_timeout():
    config = OrchestratorConfig(timeout=10.5)
    assert config.timeout == 10.5


def test_orchestrator_config_invalid_timeout():
    with pytest.raises(ValidationError) as exc_info:
        OrchestratorConfig(timeout=0)
    assert "timeout must be greater than 0" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        OrchestratorConfig(timeout=-5.0)
    assert "timeout must be greater than 0" in str(exc_info.value)


def test_orchestrator_config_extra_forbidden():
    with pytest.raises(ValidationError) as exc_info:
        OrchestratorConfig.model_validate({"unknown_field": "not_allowed"})
    assert "Extra inputs are not permitted" in str(exc_info.value)
