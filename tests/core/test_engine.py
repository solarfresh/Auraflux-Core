from unittest.mock import patch

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
