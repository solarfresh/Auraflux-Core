class OrchestrationError(Exception):
    """Base exception for all orchestrator-related errors."""
    pass


class OrchestrationLimitExceededError(OrchestrationError):
    """Raised when orchestration exceeds the maximum allowed iterations (infinite loop defense)."""
    pass


class MaxResumeAttemptsExceededError(OrchestrationError):
    """Raised when session resumption exceeds the maximum allowed retry attempts."""
    pass


class CheckpointError(OrchestrationError):
    """Raised when saving or loading checkpoints fails."""
    pass


class CheckpointCorruptedError(CheckpointError):
    """Raised when a loaded checkpoint is invalid, corrupted, or has signature/version mismatches."""
    pass


class AgentPipelineError(OrchestrationError):
    """Raised when an agent's pipeline definition is missing, invalid, or malformed."""
    pass


class InvalidOrchestrationStateError(OrchestrationError):
    """Raised when an illegal state transition or resume operation is attempted."""
    pass


class ConcurrencyConflictError(OrchestrationError):
    """Raised when a concurrent execution collision or race condition is detected for a session."""
    pass
