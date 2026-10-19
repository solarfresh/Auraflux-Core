import json
import uuid
from datetime import datetime, timezone
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


class ExecutionCursor(BaseModel):
    """Structured cursor to track exact progress for session resilience and resume fidelity."""
    agent_name: str = Field(default="")
    macro_iteration: int = Field(default=0, ge=0)
    completed_steps_count: int = Field(default=0, ge=0)


class ExecutionStep(BaseModel):
    """Represents a single atomic action within the orchestration loop."""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    actor_name: str
    method_name: str
    input_params: Dict[str, Any]
    output_data: Optional[Any] = None
    error: Optional[str] = None
    token_usage: Optional[int] = 0
    duration_ms: Optional[float] = 0.0


class OrchestratorState(BaseModel):
    """
    Maintains the global state of an orchestration session[cite: 4].
    Refactored to be domain-agnostic and supports checkpoint serialization[cite: 4].
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Schema version control for backward compatibility during resume[cite: 4]
    schema_version: str = Field(default="1.0.0")

    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex[:8]}")
    status: OrchestratorStatus = OrchestratorStatus.IDLE

    # Context & Data
    source_identifier: Optional[str] = None
    current_unit_id: Optional[str] = None

    # --- Domain Agnostic Result Container ---
    output: Dict[str, Any] = Field(default_factory=dict)

    # Audit & History
    history: List[ExecutionStep] = Field(default_factory=list)
    retry_count: int = 0
    attempt_count: int = Field(default=1, ge=1)  # Formalized typed attempt tracking for circuit breaker resilience
    total_token_count: int = 0
    last_error: Optional[str] = Field(default=None)
    cursor: ExecutionCursor = Field(default_factory=ExecutionCursor)

    # Experimental Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_step(self, step: ExecutionStep):
        self.history.append(step)
        if step.token_usage:
            self.total_token_count += step.token_usage

    def update_status(self, new_status: OrchestratorStatus, error_message: Optional[str] = None):
        self.status = new_status
        if error_message:
            self.last_error = error_message

    def bump_attempt(self) -> int:
        """Increments and returns the attempt count safely for resume persistence."""
        self.attempt_count += 1
        return self.attempt_count

    def set_last_result(self, result: Any):
        """Safely sets the last result in metadata with state-level serialization sanitization."""
        try:
            json.dumps(result)
            self.metadata["last_result"] = result
        except Exception:
            self.metadata["last_result"] = str(result)

    def dump_checkpoint(self) -> str:
        """
        Serializes the current orchestrator state into a JSON string
        for checkpoint persistence (Session Resilience)[cite: 4].
        """
        return self.model_dump_json(indent=2)

    @classmethod
    def load_checkpoint(cls, json_data: str) -> "OrchestratorState":
        """
        Deserializes a JSON string back into an OrchestratorState instance,
        enabling seamless session recovery[cite: 4].
        """
        data = json.loads(json_data)

        # Optional: Add schema version validation hook here if schema evolves
        if data.get("schema_version") != "1.0.0":
            # Handle migration logic or raise compatibility warning
            pass

        return cls.model_validate(data)
