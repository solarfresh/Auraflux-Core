# Core Module Usage Guide

This guide details how to utilize the primary modules of **Auraflux-Core**.

## 1. Core Module (`core`)

The `core` module handles client management and execution pipelines.

- **Client Manager**: Manage multi-provider LLM clients (`GeminiHandler`, `OpenAIHandler`, `VLLMHandler`).
- **Pipelines**: Execute agent tasks using `DirectPipeline` or `PlanAndExecutePipeline`.

```python
from auraflux_core.core.clients.client_manager import ClientManager
from auraflux_core.core.agents.generic_agent import GenericAgent

# Initialize client and agent example usage
```

## 2. Alignment Module (`alignment`)

The `alignment` module provides checks and strategies to verify outputs against objective and threshold claims, along with mental model evaluation.

- **Mental Model**: Evaluate reasoning coherence.
- **Objective/Threshold Claims**: Verify constraint compliance and numerical thresholds.

## 3. RAG Module (`rag`)

The `rag` module supports document ingestion, chunking, filtering, and hybrid retrieval.

- **Chunkers**: Use `DynamicChunker` or `ParagraphChunker`.
- **Retrievers**: Utilize `OpenSearchRetriever` and `HybridRetriever`.
- **Examples**: See [`rag-orchestrator-example.md`](./examples/rag/rag-orchestrator-example.md) for a complete RAG orchestrator guide.
