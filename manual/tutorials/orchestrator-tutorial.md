# Orchestrator Basic Tutorial (`orchestrator-tutorial.md`)

This tutorial introduces the core design principles and basic usage of Orchestrators within the **Auraflux-Core** framework. Orchestrators are responsible for coordinating multiple agents, tools, complex data flows, and state transitions.

---

## 1. Core Concepts

Within the Auraflux framework, Orchestrators inherit from [`BaseOrchestrator`](src/auraflux_core/core/orchestrators/base.py:16) with core responsibilities including:
- **Workflow Encapsulation**: Encapsulating multi-agent execution sequences (e.g., Extraction -> Validation -> Refinement) into a single high-level interface.
- **State Management**: Tracking execution stages and context via [`OrchestratorState`](src/auraflux_core/core/orchestrators/state.py) and strategy patterns (e.g., [`DualLayerStrategy`](src/auraflux_core/core/orchestrators/strategies/dual_layer_strategy.py)).
- **Interface Consistency**: Providing a unified `.execute(input_data)` method externally.

---

## 2. Basic Implementation Example (Hexagonal & Strategy Pattern)

Below is an example demonstrating how to implement a custom orchestrator leveraging Hexagonal Architecture and Strategy patterns, ensuring properly typed agent management (`Dict[str, Any]`):

```python
import asyncio
from typing import Any, Dict, List
from auraflux_core.core.orchestrators.base import BaseOrchestrator
from auraflux_core.core.orchestrators.strategies.base import OrchestrationStrategy
from auraflux_core.core.tools.base_tool import BaseTool

class MyCustomStrategy(OrchestrationStrategy):
    async def execute(self, input_data: Any, tools: Dict[str, Any], agents: Dict[str, Any], state: Any) -> Any:
        # Strategy logic here
        return state

class SimpleTaskOrchestrator(BaseOrchestrator):
    """
    Orchestrator demonstration: utilizes Strategy pattern and typed agents dictionary.
    """
    def __init__(self, strategy: OrchestrationStrategy, tools: List[BaseTool], agents: Dict[str, Any]):
        super().__init__(strategy=strategy, tools=tools, agents=agents)

# Usage Example:
# strategy = MyCustomStrategy()
# agents = {"assistant": my_agent_instance}
# orchestrator = SimpleTaskOrchestrator(strategy=strategy, tools=[], agents=agents)
# result = await orchestrator.execute(input_data={"task": "do something"})
```

---

## 3. Best Practices

1. **Dependency Injection**: Inject `Strategy`, `Tools`, and `Agent` instances via `__init__` to ensure easy mocking during testing.
2. **Type Strictness**: Always pass `agents` as `Dict[str, Any]` to the `BaseOrchestrator` constructor.
3. **Single Responsibility**: Each orchestrator should focus on a specific business workflow.
