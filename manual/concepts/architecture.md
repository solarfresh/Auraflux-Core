# System Architecture Overview

**Auraflux-Core** is structured into distinct, modular subsystems designed for agentic workflows, alignment verification, and retrieval-augmented generation (RAG).

## High-Level Architecture Diagram

```mermaid
graph TD
    A[Client Request] --> B[Core Router / Orchestrator]
    B --> C[Core Agents & Pipelines]
    B --> D[Alignment Subsystem]
    B --> E[RAG Subsystem]

    subgraph Core Subsystem
        C --> C1[Base Agent / Generic Agent]
        C --> C2[Pipelines Direct & Plan-Execute]
    end

    subgraph Alignment Subsystem
        D --> D1[Mental Model]
        D --> D2[Objective Claim]
        D --> D3[Threshold Claim]
    end

    subgraph RAG Subsystem
        E --> E1[Parsers & Chunkers]
        E --> E2[Retrievers & Filters]
        E --> E3[Hybrid Retriever & Processors]
    end
```

## Subsystem Breakdown

1. **`src/auraflux_core/core/`**: Provides foundational agent classes, client management handlers (Gemini, OpenAI, vLLM), orchestration routers, adapters, and messaging formatters.
2. **`src/auraflux_core/alignment/`**: Implements specialized alignment strategies and verification pipelines including mental models, objective claims, and threshold claims.
3. **`src/auraflux_core/rag/`**: Contains document parsers, dynamic chunkers, semantic/static filters, and hybrid retrievers for robust knowledge retrieval.
