from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auraflux_core.core.schemas.clients import ClientConfig
from auraflux_core.core.schemas.engine import EngineConfig
from auraflux_core.core.schemas.orchestrators import OrchestratorConfig
from auraflux_core.engine import AurafluxEngine


@pytest.fixture
def mock_engine_config() -> EngineConfig:
    return EngineConfig(client_config=ClientConfig(providers=[]))

@pytest.mark.asyncio
class TestAurafluxEngine:
    async def test_engine_initialization_success(self, mock_engine_config: EngineConfig):
        """Happy Path: Successful engine initialization."""
        engine = AurafluxEngine(config=mock_engine_config)
        assert engine.is_initialized is False

        await engine.initialize()

        assert engine.is_initialized is True

    async def test_engine_double_initialization_raises_error(self, mock_engine_config: EngineConfig):
        """Edge Case: Double initialization raises RuntimeError."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()

        with pytest.raises(RuntimeError, match="AurafluxEngine has already been initialized."):
            await engine.initialize()

    async def test_engine_uninitialized_access_raises_error(self, mock_engine_config: EngineConfig):
        """Edge Case: Calling protected methods before initialization raises RuntimeError."""
        engine = AurafluxEngine(config=mock_engine_config)

        with pytest.raises(RuntimeError, match="AurafluxEngine is not initialized."):
            engine._ensure_initialized()

    async def test_engine_initialization_failure_raises_error(self, mock_engine_config: EngineConfig):
        """Edge Case: Initialization failure (e.g. dependency error) is propagated."""
        engine = AurafluxEngine(config=mock_engine_config)

        with patch("auraflux_core.core.clients.client_manager.ClientManager.initialize", side_effect=RuntimeError("Connection failed")):
            with pytest.raises(RuntimeError, match="Connection failed"):
                await engine.initialize()

        assert engine.is_initialized is False

    async def test_execute_agent_uninitialized_raises_error(self, mock_engine_config: EngineConfig):
        """Edge Case: Calling execute_agent before initialization raises RuntimeError."""
        engine = AurafluxEngine(config=mock_engine_config)

        with pytest.raises(RuntimeError, match="AurafluxEngine is not initialized."):
            await engine.execute_agent(agent_type="test", payload={"input": "hello"}, config=MagicMock())

    async def test_execute_agent_unknown_type_raises_error(self, mock_engine_config: EngineConfig):
        """Edge Case: Calling execute_agent with unknown type raises ValueError."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()

        with pytest.raises(ValueError, match="is not registered in AGENT_REGISTRY"):
            await engine.execute_agent(agent_type="unknown", payload={"input": "hello"}, config=MagicMock())

    async def test_execute_agent_structured_success(self, mock_engine_config: EngineConfig):
        """Happy Path: execute_agent returns structured success dictionary."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()

        mock_agent_instance = MagicMock()
        mock_agent_instance.execute = AsyncMock(return_value={"output": "success"})

        mock_agent = MagicMock()
        mock_agent.build_agent = MagicMock(return_value=mock_agent_instance)
        mock_agent.config_class = MagicMock

        with patch.dict(engine.agent_registry, {"mock_agent": mock_agent}, clear=False):
            res = await engine.execute_agent(
                agent_type="mock_agent",
                payload={"input": "test"},
                config=MagicMock()
            )

            assert isinstance(res, dict)
            assert res["status"] == "success"
            assert res["result"] == {"output": "success"}
            assert res["metadata"] == {"agent_type": "mock_agent"}

    async def test_execute_agent_structured_error(self, mock_engine_config: EngineConfig):
        """Edge Case: execute_agent catches exception and returns structured error dictionary."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()

        mock_agent_instance = MagicMock()
        mock_agent_instance.execute = AsyncMock(side_effect=RuntimeError("Agent internal error"))

        mock_agent = MagicMock()
        mock_agent.build_agent = MagicMock(return_value=mock_agent_instance)
        mock_agent.config_class = MagicMock

        with patch.dict(engine.agent_registry, {"mock_agent": mock_agent}, clear=False):
            res = await engine.execute_agent(
                agent_type="mock_agent",
                payload={"input": "test"},
                config=MagicMock()
            )

            assert isinstance(res, dict)
            assert res["status"] == "failed"
            assert "Agent internal error" in res["error"]
            assert res["metadata"] == {"agent_type": "mock_agent"}

    async def test_execute_orchestrator_success(self, mock_engine_config: EngineConfig):
        """Happy Path: Successful execution of an orchestrator via execute_orchestrator."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()

        with patch("auraflux_core.core.orchestrators.base.BaseOrchestrator.execute") as mock_exec:
            mock_state = MagicMock()
            mock_state.status = MagicMock(value="completed")
            mock_state.metadata = {}
            mock_exec.return_value = mock_state

            result = await engine.execute_orchestrator(
                orchestrator_type="default",
                payload={"messages": []},
                config=OrchestratorConfig(agents={"dummy": MagicMock()})
            )

            assert isinstance(result, dict)
            assert result["status"] == "success"
            assert "final_state" in result
            assert "metadata" in result
            mock_exec.assert_awaited_once()

    async def test_execute_orchestrator_uninitialized(self, mock_engine_config: EngineConfig):
        """Edge Case: Calling execute_orchestrator before initialization raises RuntimeError."""
        engine = AurafluxEngine(config=mock_engine_config)
        with pytest.raises(RuntimeError, match="AurafluxEngine is not initialized."):
            await engine.execute_orchestrator(orchestrator_type="default", payload={})

    async def test_execute_orchestrator_invalid_type(self, mock_engine_config: EngineConfig):
        """Edge Case: Calling execute_orchestrator with invalid type raises ValueError."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()
        with pytest.raises(ValueError, match="is not supported"):
            await engine.execute_orchestrator(orchestrator_type="invalid_type", payload={})

    async def test_execute_orchestrator_invalid_state(self, mock_engine_config: EngineConfig):
        """Edge Case: Calling execute_orchestrator with invalid payload raises ValueError."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()
        with pytest.raises(ValueError, match="payload must be a dictionary"):
            await engine.execute_orchestrator(orchestrator_type="default", payload=cast(dict, "invalid_state"))

    async def test_execute_orchestrator_invalid_config_type(self, mock_engine_config: EngineConfig):
        """Edge Case: Calling execute_orchestrator with invalid config type raises ValueError."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()
        with pytest.raises(ValueError, match="config must be an instance of OrchestratorConfig"):
            await engine.execute_orchestrator(
                orchestrator_type="default",
                payload={},
                config="invalid_config" # type: ignore
            )

    async def test_execute_embedding_success_single(self, mock_engine_config: EngineConfig):
        """Happy Path: execute_embedding with single text."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()

        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        mock_embedding = MagicMock()
        mock_embedding.embedding_class = MagicMock(return_value=mock_embedding_instance)

        with patch.dict("auraflux_core.engine.EMBEDDING_REGISTRY", {"default": mock_embedding}, clear=False):
            res = await engine.execute_embedding(
                embedding_type="default",
                payload={"text": "hello world"},
                config=MagicMock()
            )

            assert isinstance(res, dict)
            assert res["status"] == "success"
            assert res["result"] == [0.1, 0.2, 0.3]
            assert res["metadata"] == {"embedding_type": "default"}

    async def test_execute_embedding_success_batch(self, mock_engine_config: EngineConfig):
        """Happy Path: execute_embedding with batch texts."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()

        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed_documents = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

        mock_embedding = MagicMock()
        mock_embedding.embedding_class = MagicMock(return_value=mock_embedding_instance)

        with patch.dict("auraflux_core.engine.EMBEDDING_REGISTRY", {"default": mock_embedding}, clear=False):
            res = await engine.execute_embedding(
                embedding_type="default",
                payload={"texts": ["hello", "world"]},
                config=MagicMock()
            )

            assert isinstance(res, dict)
            assert res["status"] == "success"
            assert res["result"] == [[0.1, 0.2], [0.3, 0.4]]
            assert res["metadata"] == {"embedding_type": "default"}

    async def test_execute_embedding_uninitialized(self, mock_engine_config: EngineConfig):
        """Edge Case: Calling execute_embedding before initialization raises RuntimeError."""
        engine = AurafluxEngine(config=mock_engine_config)
        with pytest.raises(RuntimeError, match="AurafluxEngine is not initialized."):
            await engine.execute_embedding(embedding_type="default", payload={"text": "test"}, config=MagicMock())

    async def test_execute_embedding_invalid_type(self, mock_engine_config: EngineConfig):
        """Edge Case: Calling execute_embedding with unknown type raises ValueError."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()
        with pytest.raises(ValueError, match="is not registered in EMBEDDING_REGISTRY"):
            await engine.execute_embedding(embedding_type="unknown", payload={"text": "test"}, config=MagicMock())

    async def test_execute_embedding_missing_payload_fields(self, mock_engine_config: EngineConfig):
        """Edge Case: Calling execute_embedding with missing payload fields raises ValueError."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()
        with pytest.raises(ValueError, match="payload must contain either 'texts'"):
            await engine.execute_embedding(embedding_type="default", payload={}, config=MagicMock())

    async def test_execute_embedding_failure_handling(self, mock_engine_config: EngineConfig):
        """Edge Case: execute_embedding catches exception and returns structured failure dictionary."""
        engine = AurafluxEngine(config=mock_engine_config)
        await engine.initialize()

        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed_query = AsyncMock(side_effect=RuntimeError("Embedding failure"))

        mock_embedding = MagicMock()
        mock_embedding.embedding_class = MagicMock(return_value=mock_embedding_instance)

        with patch.dict("auraflux_core.engine.EMBEDDING_REGISTRY", {"default": mock_embedding}, clear=False):
            res = await engine.execute_embedding(
                embedding_type="default",
                payload={"text": "fail test"},
                config=MagicMock()
            )

            assert isinstance(res, dict)
            assert res["status"] == "failed"
            assert "Embedding failure" in res["error"]
            assert res["metadata"] == {"embedding_type": "default"}


