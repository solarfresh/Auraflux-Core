import re
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

PremiseT = TypeVar("PremiseT")
QuantT = TypeVar("QuantT")

ClaimStatus = Literal[
    "VERIFIED",
    "PARTIALLY_VERIFIED",
    "UNSUPPORTED",
    "VIOLATED",
    "FAIL",
    "BORDERLINE"
]


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


class ClaimVerdict(BaseModel):
    """Shared verdict envelope produced by every diagnostic agent
    (mental_model / objective / threshold). Downstream cross-checking
    logic (Step 3) consumes this common shape uniformly, regardless
    of which agent produced it.

    Subclasses MUST add their own `diagnostics` field (structure differs
    per agent's main-battlefield gate) and MAY add agent-specific fields
    (e.g. threshold's `preset_options`).
    """

    proposition_id: str = Field(..., description="Unique identifier for the atomic claim.")
    claim_text: str = Field(..., description="The original atomic claim statement.")
    triples: List[TripleItem] = Field(
        default_factory=list,
        description="Bound semantic triples representing the structured facts of the claim."
    )
    status: ClaimStatus = Field(..., description="Final verification state shared across all agents.")
    verification_proofs: List[str] = Field(
        default_factory=list,
        description="Extracted proof points, citations, or references supporting the verdict."
    )
    compliance_gap: Optional[str] = Field(
        default=None,
        description="Gap narrative when status indicates a violation, block, or unsupported claim."
    )


class DiagnosticAnalysisBase(BaseModel, Generic[PremiseT, QuantT]):
    """Shared 3-gate diagnostic contract. Concrete agents only parametrize
    the two fields that may be upgraded into a strongly-typed 'main
    battlefield' structure; boundary_conflicts stays free-form across
    all agents since none of them have formalized its shape yet.
    """

    implicit_premises: PremiseT
    quantification_requirements: QuantT
    boundary_conflicts: Dict[str, Any] = Field(default_factory=dict)
