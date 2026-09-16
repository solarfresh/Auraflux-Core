import re
from typing import List, Literal, Optional

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

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
    Expresses a bound semantic triple: Subject -> Predicate -> Object,
    optionally enriched with normalized quantitative metric metadata and coreference resolution.
    Provides a closed-world statement ensuring entities, logical conditions, and quantities remain coupled.
    """

    subject: str = Field(..., description="Subject entity as verbatim from text")
    subject_resolved: Optional[str] = Field(
        None,
        description="Resolved actual entity name(s) if subject is a pronoun, relative pronoun, or generic term (e.g., 'C++ / Rust')"
    )
    predicate: str = Field(..., description="Relation / Predicate / Operator as verbatim from text")
    object: str = Field(..., description="Metric / Constraint / Object entity as verbatim from text")

    # Target field binding for metric (automatically resolved via post-processing)
    data_target: Optional[str] = Field(
        None,
        description="Field that contains the raw quantitative text span: strictly 'subject' or 'object'"
    )

    # Embedded Quantitative Metric Normalization Properties
    metric_name: Optional[str] = Field(
        None,
        description="Standardized metric identifier (e.g., 'budget', 'duration', 'sla_time', 'penalty')"
    )
    normalized_value: Optional[float] = Field(
        None,
        description="Pure numeric value converted strictly to the standard base unit"
    )
    unit: Optional[str] = Field(
        None,
        description="Standard unit symbol (e.g., 'TWD', 'USD', 'day', 'hour', 'm2', 'kg', 'GB', '%')"
    )
    operator: Optional[str] = Field(
        None,
        description="Boundary condition operator strictly chosen from ['<=', '>=', '==', '<', '>']"
    )

    @field_validator("normalized_value", mode="before")
    @classmethod
    def parse_empty_float(cls, v):
        """Coerces empty strings, null literals, or None values into None for float type safety."""
        if v == "" or v is None or v == "null":
            return None
        return float(v)

    @field_validator("subject_resolved", "data_target", "metric_name", "unit", "operator", mode="before")
    @classmethod
    def parse_empty_string(cls, v):
        """Coerces empty strings or null literals into None for string properties."""
        if v == "" or v == "null":
            return None
        return v

    @model_validator(mode="after")
    def auto_assign_data_target_fallback(self) -> "TripleItem":
        """
        Fallback Logic: If 'normalized_value' exists but LLM omitted 'data_target',
        programmatically assign 'data_target' via deterministic string matching.
        """
        # Primary check: Only execute fallback if LLM omitted data_target
        if self.normalized_value is not None and self.data_target is None:
            # Format numeric string for substring matching (e.g., 10000.0 -> "10000")
            val_str = str(self.normalized_value)
            val_str_clean = val_str[:-2] if val_str.endswith(".0") else val_str

            # Check literal occurrence of raw number or unit symbol within fields
            in_object = val_str_clean in self.object or (
                self.unit is not None and self.unit in self.object
            )
            in_subject = val_str_clean in self.subject or (
                self.unit is not None and self.unit in self.subject
            )

            if in_object and not in_subject:
                self.data_target = "object"
            elif in_subject and not in_object:
                self.data_target = "subject"
            elif in_object and in_subject:
                # Priority goes to object if numerical string exists in both
                self.data_target = "object"
            else:
                # Fallback heuristic: Check for digits via Regex if metric units were transformed
                if re.search(r"\d+", self.object):
                    self.data_target = "object"
                elif re.search(r"\d+", self.subject):
                    self.data_target = "subject"
                else:
                    self.data_target = "object"

        return self


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
    excerptText: str = Field(..., description="Exact verbatim excerpt from the document (100–300 words)")
    location: str = Field(..., description="Location pointer within source document (e.g., 'Page 5, Section 3.2')")


# --- Multi-Vector Embedding Layer ---
class ChunkVectors(BaseModel):
    """
    Individual vector representations corresponding to dense vector fields in OpenSearch/Vector DB.
    """
    questionVector: Optional[List[float]] = Field(None, description="Embedding vector for `alignment.targetQuestion`")
    conceptVector: Optional[List[float]] = Field(None, description="Embedding vector for `concept.title` & `concept.description`")
    evidenceVector: Optional[List[float]] = Field(None, description="Embedding vector for `evidence.excerptText`")


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