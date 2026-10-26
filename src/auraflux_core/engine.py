from typing import Any, Dict, Optional

from auraflux_core.agents import AGENT_REGISTRY
from auraflux_core.core.clients.client_manager import ClientManager
from auraflux_core.core.schemas.engine import EngineConfig
from auraflux_core.core.schemas.orchestrators import OrchestratorConfig
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
        prompt: str,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        prompt_config: Optional[Any] = None,
        temperature: Optional[float] = None,
    ) -> Any:
        """
        Execute an agent by type with the given prompt and optional configuration overrides.

        :param agent_type: The registered agent type string.
        :param prompt: The input prompt for the agent.
        :param provider: Optional LLM provider override.
        :param model_id: Optional model ID override.
        :param prompt_config: Optional prompt configuration override.
        :param temperature: Optional temperature override.
        :return: The execution result from the agent.
        :raises RuntimeError: If the engine is not initialized.
        :raises ValueError: If the agent_type is not registered.
        """
        self._ensure_initialized()

        agent_impl = self.agent_registry.get(agent_type)
        if not agent_impl:
            raise ValueError(f"Agent type '{agent_type}' is not registered in AGENT_REGISTRY.")

        agent_class = agent_impl.agent_class
        config_class = agent_impl.config_class

        # Build config kwargs based on config_class or default AgentConfig
        config_kwargs: Dict[str, Any] = {
            "name": agent_type,
            "provider": provider or "openai",
            "model": model_id or "gpt-4o",
        }
        if prompt_config:
            config_kwargs["prompt_config"] = prompt_config
        if temperature is not None:
            if not (0.0 <= temperature <= 2.0):
                raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temperature}")
            config_kwargs["temperature"] = temperature

        agent_config = config_class(**config_kwargs)

        # Instantiate agent
        agent = agent_class(config=agent_config, client_manager=self.client_manager)

        return await agent.execute(prompt=prompt)

    async def execute_orchestrator(
        self,
        orchestrator_type: str,
        initial_state: dict,
        agents_map: Optional[Dict[str, Any]] = None,
        config: Optional[OrchestratorConfig] = None,
    ) -> dict:
        """
        Execute an orchestrator by type with the given initial state.

        :param orchestrator_type: The orchestrator type string.
        :param initial_state: The initial state dictionary.
        :param agents_map: Optional agents mapping.
        :param config: Optional execution configuration.
        :return: A dictionary containing the final state and metadata.
        :raises RuntimeError: If the engine is not initialized.
        :raises ValueError: If the orchestrator_type is not supported or initial_state is invalid.
        """
        self._ensure_initialized()

        if not isinstance(initial_state, dict):
            raise ValueError("initial_state must be a dictionary.")

        if config is not None:
            if not isinstance(config, OrchestratorConfig):
                raise ValueError("config must be an instance of OrchestratorConfig.")

        impl = ORCHESTRATOR_REGISTRY.get(orchestrator_type)
        if not impl:
            raise ValueError(f"Orchestrator type '{orchestrator_type}' is not supported.")

        try:
            strategy = impl.strategy_class(router=impl.router_class())
            orchestrator = impl.orchestrator_class(
                strategy=strategy,
                tools=[],
                agents=agents_map or {},
                config=config
            )

            final_state = await orchestrator.execute(input_data=initial_state)

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
