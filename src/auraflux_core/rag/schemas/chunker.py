from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

# --- Primitive Types and Enums ---
ImpactLevel = Literal['strategic', 'tactical', 'operational']


# --- Layer 1: Alignment & Scope Layer ---
class ChunkScope(BaseModel):
    """
    Defines scope and boundaries for strategy alignment.
    """
    domain: str = Field(
        default="",
        description="Target business or technical domain (e.g., 'IT Architecture & Compliance')"
    )
    impactLevel: ImpactLevel = Field(
        default="operational",
        description="Level of impact on decision-making"
    )
    boundaries: List[str] = Field(
        default_factory=list,
        description="Non-negotiable rules or hard constraints (e.g., ['No public cloud routing'])"
    )


class ChunkAlignment(BaseModel):
    """
    Contextual questions and non-negotiable boundaries for driving discussions.
    """
    targetQuestion: str = Field(
        default="",
        description="Core decision dilemma or question triggered by this chunk"
    )
    scope: ChunkScope = Field(
        default_factory=ChunkScope,
        description="Boundary parameters and impact domain"
    )


# --- Layer 2: Abstraction Layer ---
class ChunkConcept(BaseModel):
    """
    High-level concepts and structural propositions (does NOT duplicate raw text details).
    """
    title: str = Field(
        default="",
        description="High-level proposition or rule title (e.g., 'Data Sovereignty vs. Architectural Agility')"
    )
    description: str = Field(
        default="",
        description="Contextual description explaining real-world impact and constraint mechanisms"
    )


# --- Layer 3: Token & Entity-Relation Layer ---
class TripleItem(BaseModel):
    """
    Expresses a bound semantic triple: Subject -> Predicate -> Object.
    Provides a closed-world statement ensuring entities, logical conditions, and quantities remain coupled.
    """
    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Relation / Predicate / Operator")
    object: str = Field(..., description="Metric / Constraint / Object entity")


class ChunkKeywords(BaseModel):
    """
    Captures bound entity-metric pairs and general domain tags to prevent association mismatch.
    """
    triples: List[TripleItem] = Field(
        default_factory=list,
        description="List of bound semantic triples ensuring entity-limit associations"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="General high-level domain or thematic tags (e.g., ['finance', 'compliance'])"
    )


# --- Layer 4: Fact & Evidence Layer ---
class ChunkEvidence(BaseModel):
    """
    Raw text snippets and location pointers for grounding and auditability.
    """
    excerpt_text: str = Field(..., description="Exact verbatim excerpt from the document (100–300 words)")
    location: str = Field(..., description="Location pointer within source document (e.g., 'Page 5, Section 3.2')")


# --- Multi-Vector Embedding Layer ---
class ChunkVectors(BaseModel):
    """
    Individual vector representations corresponding to dense vector fields in OpenSearch/Vector DB.
    """
    questionVector: Optional[List[float]] = Field(None, description="Embedding vector for `alignment.targetQuestion`")
    conceptVector: Optional[List[float]] = Field(None, description="Embedding vector for `concept.title` & `concept.description`")
    evidenceVector: Optional[List[float]] = Field(None, description="Embedding vector for `evidence.excerpt_text`")


# --- Unified Repository Chunk Entity ---
class StandardChunk(BaseModel):
    """
    Unified Repository Chunk Entity (1:1 mirror of the TypeScript ChunkData interface).
    Serves as the single data model throughout the entire processing pipeline.

    Pipeline Ingestion Lifecycle:
    - Stage 2 (Chunking): Instantiates `ChunkData` with `id`, `fileId`, and Layer 4 (`evidence`).
    - Step 4 (LLM Extraction): Populates Layer 3 (`keywords`).
    - Step 6 (LLM Reasoning): Populates Layer 2 (`concept`) and Layer 1 (`alignment`).
    - Step 7 (Storage): Calculates and attaches `vectors`, followed by DB persistence.
    """
    id: str = Field(..., description="Unique identifier for the chunk")
    fileId: str = Field(..., description="Unique identifier of the parent document")

    # Layer 4: Fact & Evidence (Populated at Stage 2 creation)
    evidence: ChunkEvidence = Field(..., description="Layer 4: Raw Fact & Evidence")

    # Layer 3: Keyword Tokens (Populated at Step 4 after LLM extraction)
    keywords: Optional[ChunkKeywords] = Field(
        default_factory=ChunkKeywords,
        description="Layer 3: Bound Semantic Triples & Keywords"
    )

    # Layer 2 & Layer 1: Abstract Concepts & Alignment (Populated at Step 6 after LLM reasoning)
    concept: Optional[ChunkConcept] = Field(None, description="Layer 2: Abstract Concept")
    alignment: Optional[ChunkAlignment] = Field(None, description="Layer 1: Alignment & Scope")

    # Multi-Vector Embeddings (Calculated and attached at Step 7)
    vectors: Optional[ChunkVectors] = Field(None, description="Multi-Vector Embeddings")

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )