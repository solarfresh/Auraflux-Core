# Getting Started with Auraflux-Core

This guide helps you set up your development environment, install dependencies, and run your first validation steps for **Auraflux-Core**.

## 1. Prerequisites

- Python 3.10+
- Conda or virtual environment tool (as required by project environment governance rules)

## 2. Installation

Clone the repository and install dependencies within your designated virtual environment:

```bash
# Example using conda or venv
pip install -r requirements.txt
```

## 3. Basic Configuration

Ensure environment variables or configuration files under [`src/auraflux_core/core/configs/`](src/auraflux_core/core/configs/settings.py) are properly set for your LLM providers (e.g., Gemini, OpenAI, vLLM).

## 4. Running Tests

Verify your installation by running tests through pytest in your isolated environment:

```bash
pytest
```

---

## 5. Recommended Learning Path & Next Steps

To ensure a smooth learning curve, follow this recommended progression path when building with **Auraflux-Core**:

1. **Phase 1: Foundations & Individual Agents**
   - **[Basic Agent Tutorial (`basic-agent-tutorial.md`)](./basic-agent-tutorial.md)**: Learn how to initialize client managers, configure prompts, and execute individual agents.
2. **Phase 2: Workflow Orchestration**
   - **[Orchestrator Tutorial (`orchestrator-tutorial.md`)](./orchestrator-tutorial.md)**: Master how to wrap and coordinate agents and tools into structured orchestrators using `BaseOrchestrator`.
3. **Phase 3: High-Level Engine Entry Point**
   - **[Auraflux Engine Tutorial (`engine-tutorial.md`)](./engine-tutorial.md)**: Explore [`AurafluxEngine`](src/auraflux_core/engine.py:11) as a high-level entry point integrating `ClientManager`, `AGENT_REGISTRY`, and `ORCHESTRATOR_REGISTRY`.
4. **Phase 4: Advanced Modules & Integration**
   - **[Core Module Usage (`usage.md`)](./usage.md)**: Dive deeper into core components, client management, alignment strategies, and RAG pipelines.
   - **[Backend Integration Guide (`backend-integration.md`)](./backend-integration.md)**: Learn how to share and invoke Agents and Orchestrators cleanly inside web backend services (such as FastAPI).
