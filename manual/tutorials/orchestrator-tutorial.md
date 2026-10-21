# Orchestrator Basic Tutorial (`orchestrator-tutorial.md`)

This tutorial introduces the core design principles and basic usage of Orchestrators within the **Auraflux-Core** framework. Orchestrators are responsible for coordinating multiple agents, tools, complex data flows, and state transitions.

---

## 1. Core Concepts

Within the Auraflux framework, Orchestrators inherit from [`BaseOrchestrator`](src/auraflux_core/core/orchestrators/base.py:16) with core responsibilities including:
- **Workflow Encapsulation**: Encapsulating multi-agent execution sequences (e.g., Extraction -> Validation -> Refinement) into a single high-level interface.
- **State Management**: Tracking execution stages and context via [`OrchestratorState`](src/auraflux_core/core/orchestrators/state.py) and strategy patterns (e.g., [`DualLayerStrategy`](src/auraflux_core/core/orchestrators/strategies/dual_layer_strategy.py)).
- **Interface Consistency**: Providing a unified `.run(input_data)` method externally.

---

## 2. Basic Implementation Example

Below is a basic example illustrating how to implement a custom orchestrator to manage agent tasks:

```python
import asyncio
from typing import Any, Dict
from auraflux_core.core.orchestrators.base import BaseOrchestrator
from auraflux_core.core.clients.client_manager import ClientManager
from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.schemas.agents import AgentConfig

class SimpleTaskOrchestrator(BaseOrchestrator):
    """
    Basic orchestrator demonstration: sequentially invokes an agent and aggregates results.
    """
    def __init__(self, agent: BaseAgent, client_manager: ClientManager):
        self.agent = agent
        self.client_manager = client_manager

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the orchestrator workflow.
        """
        # Invoke the agent's standard run interface
        agent_output = await self.agent.run(payload=input_data)

        return {
            "status": "success",
            "input": input_data,
            "output": agent_output
        }
```

---

## 3. Best Practices

1. **Dependency Injection**: Inject `ClientManager` and `Agent` instances via `__init__` to ensure easy mocking during testing.
2. **Single Responsibility**: Each orchestrator should focus on a specific business workflow (e.g., document preprocessing, multi-path retrieval, or alignment verification).
