from typing import Any, Dict, Optional

from auraflux_core.agents import AGENT_REGISTRY
from auraflux_core.core.clients.client_manager import ClientManager
from auraflux_core.core.schemas.agents import AgentConfig
from auraflux_core.core.schemas.embeddings import EmbeddingConfig
from auraflux_core.core.schemas.engine import EngineConfig
from auraflux_core.core.schemas.orchestrators import OrchestratorConfig
from auraflux_core.embeddings import EMBEDDING_REGISTRY
from auraflux_core.orchestrators import ORCHESTRATOR_REGISTRY


class AurafluxEngine:
    def __init__(self, config: EngineConfig):
        """
        Initialize the AurafluxEngine.

        :param config: EngineConfig configuration.
        """
        self.config = config
        self._is_initialized = False
        self.client_manager: Optional[ClientManager] = None
        self.agent_registry = AGENT_REGISTRY

    @property
    def is_initialized(self) -> bool:
        """
        Check if the engine is initialized.
        """
        return self._is_initialized

    async def initialize(self) -> None:
        """
        Initialize the engine, ClientManager, and AgentRegistry.

        :raises RuntimeError: If the engine is already initialized.
        :raises ValueError: If client_config is missing in the engine configuration.
        """
        if self._is_initialized:
            raise RuntimeError("AurafluxEngine has already been initialized.")

        try:
            # 1. Initialize ClientManager if configuration is provided
            client_config = self.config.client_config if self.config else None
            if not client_config:
                raise ValueError("client_config is required in EngineConfig for AurafluxEngine initialization.")

            self.client_manager = ClientManager(config=client_config)
            await self.client_manager.initialize()

            # 2. AGENT_REGISTRY is already imported and statically available

            self._is_initialized = True
        except Exception as e:
            # Explicitly clear state on initialization failure
            self._is_initialized = False
            raise RuntimeError(f"Failed to initialize AurafluxEngine: {e}") from e

    def _ensure_initialized(self) -> None:
        """
        Ensure the engine is initialized.
        This is a defensive helper method for other engine components that
        must not execute before initialization.

        If a method requiring initialization is called on an uninitialized engine,
        it will be blocked by this defensive check.

        :raises RuntimeError: If the engine is not initialized.
        """
        if not self._is_initialized:
            raise RuntimeError("AurafluxEngine is not initialized.")

    # Example of a core method protected by _ensure_initialized
    def get_agent_class(self, agent_name: str):
        self._ensure_initialized()
        impl = self.agent_registry.get(agent_name)
        return impl.agent_class if impl else None

    async def execute_agent(
        self,
        agent_type: str,
        payload: Dict[str, Any],
        config: AgentConfig,
    ) -> Dict[str, Any]:
        """
        Execute an agent by type with the given payload and optional configuration.

        :param agent_type: The registered agent type string.
        :param payload: The input payload dictionary for the agent execution.
        :param config: Optional AgentConfig instance.
        :return: A dictionary containing the status, result/error, and metadata.
        :raises RuntimeError: If the engine is not initialized.
        :raises ValueError: If the agent_type is not registered or config is invalid.
        """
        self._ensure_initialized()
        if self.client_manager is None:
            raise RuntimeError("ClientManager must not be None after initialization.")

        agent_impl = self.agent_registry.get(agent_type)
        if not agent_impl:
            raise ValueError(f"Agent type '{agent_type}' is not registered in AGENT_REGISTRY.")

        if config is None:
            raise ValueError(f"config must be an instance of {config_class.__name__}")

        try:
            agent = agent_impl.build_agent(config=config, client_manager=self.client_manager)
            result = await agent.execute(payload=payload)

            return {
                "status": "success",
                "result": result,
                "metadata": {"agent_type": agent_type},
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "metadata": {"agent_type": agent_type},
            }

    async def execute_orchestrator(
        self,
        orchestrator_type: str,
        payload: dict,
        config: Optional[OrchestratorConfig] = None,
    ) -> dict:
        """
        Execute an orchestrator by type with the given initial state.

        :param orchestrator_type: The orchestrator type string.
        :param payload: The initial state dictionary.
        :param config: Optional execution configuration.
        :return: A dictionary containing the final state and metadata.
        :raises RuntimeError: If the engine is not initialized.
        :raises ValueError: If the orchestrator_type is not supported or payload is invalid.
        """
        self._ensure_initialized()

        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary.")

        if config is not None:
            if not isinstance(config, OrchestratorConfig):
                raise ValueError("config must be an instance of OrchestratorConfig.")

        impl = ORCHESTRATOR_REGISTRY.get(orchestrator_type)
        if not impl:
            raise ValueError(f"Orchestrator type '{orchestrator_type}' is not supported.")

        try:
            orchestrator = impl.build_orchestrator(config=config)

            final_state = await orchestrator.execute(payload=payload)

            return {
                "status": "success" if final_state.status.value == "completed" else "failed",
                "final_state": final_state.metadata,
                "metadata": {"orchestrator_type": orchestrator_type},
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "metadata": {"orchestrator_type": orchestrator_type},
            }

    async def execute_embedding(
        self,
        embedding_type: str,
        payload: Dict[str, Any],
        config: EmbeddingConfig,
    ) -> Dict[str, Any]:
        """
        Execute embedding generation by type with the given payload and configuration.

        :param embedding_type: The registered embedding type string.
        :param payload: The input payload dictionary containing 'texts' or 'text'.
        :param config: EmbeddingConfig instance.
        :return: A dictionary containing the status, result/error, and metadata.
        :raises RuntimeError: If the engine is not initialized or ClientManager is None.
        :raises ValueError: If embedding_type is not registered, config is invalid, or payload is missing texts/text.
        """
        self._ensure_initialized()
        if self.client_manager is None:
            raise RuntimeError("ClientManager must not be None after initialization.")

        embedding_impl = EMBEDDING_REGISTRY.get(embedding_type)
        if not embedding_impl:
            raise ValueError(f"Embedding type '{embedding_type}' is not registered in EMBEDDING_REGISTRY.")

        if config is None:
            raise ValueError("config must be provided as an instance of EmbeddingConfig.")

        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary.")

        has_texts = "texts" in payload and isinstance(payload["texts"], list)
        has_text = "text" in payload and isinstance(payload["text"], str)

        if not has_texts and not has_text:
            raise ValueError("payload must contain either 'texts' (List[str]) or 'text' (str).")

        try:
            embedding_instance = embedding_impl.embedding_class(config=config, client_manager=self.client_manager)

            if has_texts:
                result = await embedding_instance.embed_documents(texts=payload["texts"])
            else:
                result = await embedding_instance.embed_query(text=payload["text"])

            return {
                "status": "success",
                "result": result,
                "metadata": {"embedding_type": embedding_type},
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "metadata": {"embedding_type": embedding_type},
            }
