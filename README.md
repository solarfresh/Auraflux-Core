# Auraflux Core

`auraflux_core` serves as the primary engine for the Auraflux system, designed to facilitate a highly modular and scalable **Multi-Agent System (MAS)**. It provides a model-agnostic framework that streamlines LLM orchestration, complex tool integration, and high-performance asynchronous request dispatching.

## 🌟 Key Features

* **Unified LLM Orchestration**: Manage multiple providers (e.g., Google Gemini, OpenAI) through a centralized `ClientManager`.
* **Dual-Protocol**: Highlighting that the system can handle both modern APIs and legacy/open-source Prompt-based methods is a major selling point.
* **Reflective vs. Direct**: Replaced the technical `AND_PROCESS` with the more descriptive `REFLECTIVE`, which is a standard term in Agentic AI.
* **Production-Ready Dispatching**: A queue-based request system designed for high-concurrency environments like Celery workers, ensuring non-blocking operations.
* **Extensible Base Architecture**: Robust abstract base classes (`BaseAgent`, `BaseTool`, `BaseHandler`) allow for seamless customization of system components.
* **Cross-Model Prompt Alignment**: Dynamic system message mapping via `_message_mapper` to align prompts across different languages and model families.

## 🏗 System Architecture

### 1. Agent Logic (`core/agents`)
Agents are defined by an `AgentConfig`, which governs their identity, model parameters (temperature, max tokens), and conversation constraints like `turn_limit`.
* **BaseAgent**: Implements the core lifecycle, including message history management and tool-call post-processing.

### 2. Client Management & Dispatching (`core/clients`)
The `ClientManager` acts as the single point of entry for all LLM requests.
* **Request Queue**: Handles thread-safe request submission.
* **Dispatcher**: A background task that routes requests to specific handlers (e.g., `GeminiHandler`) and manages timeouts and retries.

### 3. Tool Framework (`core/tools`)
All external capabilities must inherit from `BaseTool`.
* Tools provide structured metadata (`get_parameters`) required for accurate LLM function calling.
* Asynchronous execution via `await tool.run()`.

## 🛠 Setup and Configuration

### Requirements
* Python 3.12+
* Pydantic V2
* Google GenAI SDK (for Gemini support)

### Environment Configuration
```bash
# Set API keys based on the handlers you intend to use
GEMINI_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key
```

## 📖 Development Guide

### Creating a Custom Tool
```python
from auraflux_core.core.tools.base_tool import BaseTool

class SearchTool(BaseTool):
    async def run(self, query: str):
        # Implementation of the search logic
        return f"Results for {query}"

    def get_name(self): return "web_search"
    def get_description(self): return "Search the web for real-time info"
    def get_parameters(self):
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}}
        }
```

### Initializing an Agent
```python
from auraflux_core.core.schemas.agents import AgentConfig

# Define the agent's behavior and model constraints
config = AgentConfig(
    name="Researcher",
    provider="GOOGLE",
    model="gemini-1.5-pro",
    temperature=0.7
)
```

## 📂 Module Index

| Module | Description |
| :--- | :--- |
| `core.agents` | Core agent logic and reasoning cycle. |
| `core.clients` | LLM client management and provider handlers. |
| `core.schemas` | Pydantic data models for configurations and messaging. |
| `core.tools` | Standard interface for external tool integration. |
| `canvases` | Scenario-specific workspace implementations. |
