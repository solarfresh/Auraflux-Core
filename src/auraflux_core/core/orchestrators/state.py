import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class OrchestratorStatus(str, Enum):
    IDLE = "IDLE"
    INGESTING = "INGESTING"
    PROCESSING = "PROCESSING"
    VALIDATING = "VALIDATING"
    REFINING = "REFINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ExecutionStep(BaseModel):
    """Represents a single atomic action within the orchestration loop."""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    actor_name: str
    input_params: Dict[str, Any]
    output_data: Optional[Any] = None
    error: Optional[str] = None
    token_usage: Optional[int] = 0
    duration_ms: Optional[float] = 0.0


class OrchestratorState(BaseModel):
    """
    Maintains the global state of an orchestration session.
    Refactored to be domain-agnostic.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex[:8]}")
    status: OrchestratorStatus = OrchestratorStatus.IDLE

    # Context & Data
    source_identifier: Optional[str] = None  # Renamed from source_file for generality
    current_unit_id: Optional[str] = None    # Renamed from chunk_id for generality

    # --- Domain Agnostic Result Container ---
    # Use this to store nodes/edges for Graph tasks,
    # or text/blocks for other types of orchestration.
    output: Dict[str, Any] = Field(default_factory=dict)

    # Audit & History
    history: List[ExecutionStep] = []
    retry_count: int = 0
    total_token_count: int = 0

    # Experimental Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_step(self, step: ExecutionStep):
        self.history.append(step)
        if step.token_usage:
            self.total_token_count += step.token_usage

    def update_status(self, new_status: OrchestratorStatus):
        self.status = new_status
