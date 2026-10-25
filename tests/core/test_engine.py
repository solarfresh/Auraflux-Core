from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auraflux_core.core.schemas.clients import ClientConfig
from auraflux_core.engine import AurafluxEngine


@pytest.mark.asyncio
class TestAurafluxEngine:
    async def test_engine_initialization_success(self):
        """Happy Path: Successful engine initialization."""
        engine = AurafluxEngine(config={"debug": True})
        assert engine.is_initialized is False

        await engine.initialize()

        assert engine.is_initialized is True

    async def test_engine_double_initialization_raises_error(self):
        """Edge Case: Double initialization raises RuntimeError."""
        engine = AurafluxEngine()
        await engine.initialize()

        with pytest.raises(RuntimeError, match="AurafluxEngine has already been initialized."):
            await engine.initialize()

    async def test_engine_uninitialized_access_raises_error(self):
        """Edge Case: Calling protected methods before initialization raises RuntimeError."""
        engine = AurafluxEngine()

        with pytest.raises(RuntimeError, match="AurafluxEngine is not initialized."):
            engine._ensure_initialized()

    async def test_engine_initialization_failure_raises_error(self):
        """Edge Case: Initialization failure (e.g. dependency error) is propagated."""
        # Create a valid ClientConfig object for the mock
        mock_config = ClientConfig(providers=[])

        engine = AurafluxEngine(config={"client_config": mock_config})

        with patch("auraflux_core.core.clients.client_manager.ClientManager.initialize", side_effect=RuntimeError("Connection failed")):
            with pytest.raises(RuntimeError, match="Connection failed"):
                await engine.initialize()

        assert engine.is_initialized is False

    async def test_execute_agent_uninitialized_raises_error(self):
        """Edge Case: Calling execute_agent before initialization raises RuntimeError."""
        engine = AurafluxEngine()

        with pytest.raises(RuntimeError, match="AurafluxEngine is not initialized."):
            await engine.execute_agent(agent_type="test", prompt="hello")

    async def test_execute_agent_unknown_type_raises_error(self):
        """Edge Case: Calling execute_agent with unknown type raises ValueError."""
        engine = AurafluxEngine()
        await engine.initialize()

        with pytest.raises(ValueError, match="is not registered in AGENT_REGISTRY"):
            await engine.execute_agent(agent_type="unknown", prompt="hello")

    async def test_execute_agent_invalid_temperature_raises_error(self):
        """Edge Case: Calling execute_agent with out-of-bounds temperature raises ValueError."""
        engine = AurafluxEngine()
        await engine.initialize()

        mock_agent_instance = MagicMock()
        mock_agent_class = MagicMock(return_value=mock_agent_instance)
        mock_config_class = MagicMock()

        mock_impl = MagicMock()
        mock_impl.agent_class = mock_agent_class
        mock_impl.config_class = mock_config_class

        with patch.dict(engine.agent_registry, {"mock_agent": mock_impl}, clear=False):
            with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
                await engine.execute_agent(
                    agent_type="mock_agent",
                    prompt="test-prompt",
                    temperature=2.5,
                )

    async def test_execute_agent_success(self):
        """Happy Path: Successful execution of an agent via execute_agent."""
        engine = AurafluxEngine()
        await engine.initialize()

        mock_agent_instance = MagicMock()
        mock_agent_instance.execute = AsyncMock(return_value="agent-response")

        mock_agent_class = MagicMock(return_value=mock_agent_instance)
        mock_config_class = MagicMock()

        mock_impl = MagicMock()
        mock_impl.agent_class = mock_agent_class
        mock_impl.config_class = mock_config_class

        with patch.dict(engine.agent_registry, {"mock_agent": mock_impl}, clear=False):
            result = await engine.execute_agent(
                agent_type="mock_agent",
                prompt="test-prompt",
                provider="openai",
                model_id="gpt-4o",
                temperature=0.5,
            )

            assert result == "agent-response"
            mock_agent_class.assert_called_once()
            mock_agent_instance.execute.assert_awaited_once_with(prompt="test-prompt")

    async def test_execute_orchestrator_success(self):
        """Happy Path: Successful execution of an orchestrator via execute_orchestrator."""
        engine = AurafluxEngine()
        await engine.initialize()

        with patch("auraflux_core.core.orchestrators.base.BaseOrchestrator.execute") as mock_exec:
            mock_exec.return_value = MagicMock(status=MagicMock(value="completed"), metadata={})

            result = await engine.execute_orchestrator(
                orchestrator_type="default",
                initial_state={"messages": []},
                agents_map={"dummy": MagicMock()}
            )

            assert isinstance(result, dict)
            assert result["status"] == "success"
            assert "final_state" in result
            assert "metadata" in result
            mock_exec.assert_awaited_once()

    async def test_execute_orchestrator_uninitialized(self):
        """Edge Case: Calling execute_orchestrator before initialization raises RuntimeError."""
        engine = AurafluxEngine()
        with pytest.raises(RuntimeError, match="AurafluxEngine is not initialized."):
            await engine.execute_orchestrator(orchestrator_type="default", initial_state={})

    async def test_execute_orchestrator_invalid_type(self):
        """Edge Case: Calling execute_orchestrator with invalid type raises ValueError."""
        engine = AurafluxEngine()
        await engine.initialize()
        with pytest.raises(ValueError, match="is not supported"):
            await engine.execute_orchestrator(orchestrator_type="invalid_type", initial_state={})

    async def test_execute_orchestrator_invalid_state(self):
        """Edge Case: Calling execute_orchestrator with invalid initial_state raises ValueError."""
        engine = AurafluxEngine()
        await engine.initialize()
        with pytest.raises(ValueError, match="initial_state must be a dictionary"):
            await engine.execute_orchestrator(orchestrator_type="default", initial_state=cast(dict, "invalid_state"))

