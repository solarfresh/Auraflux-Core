import logging
import sys
from typing import Any, Optional

import structlog
from structlog.types import Processor

from .settings import settings


def setup_logging(
    name: str = "auraflux",
    json_format: Optional[bool] = None,
) -> Any:
    """
    Sets up structured logging for the application using structlog and standard logging.

    Args:
        name (str): The logger name identifier. Defaults to 'auraflux'.
        json_format (Optional[bool]): Explicitly enable JSON rendering.
            If None, falls back to `settings.json_logs` or non-tty detection.

    Returns:
        structlog.stdlib.BoundLogger: Configured structured logger instance.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Determine whether to render structured JSON or human-readable console logs
    use_json = (
        json_format
        if json_format is not None
        else getattr(settings, "json_logs", not sys.stdout.isatty())
    )

    # Shared processors applied before final rendering
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if use_json:
        # JSON pipeline for Production & Async Data Pipelines
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        # Development / Jupyter-friendly console pipeline
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure root standard library handler
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Disable propagation on named loggers to prevent duplicate output
    logger = logging.getLogger(name)
    logger.propagate = False

    return structlog.get_logger(name)


def get_logger(name: str = "auraflux") -> structlog.stdlib.BoundLogger:
    """
    Retrieves a bound structlog logger instance.

    Args:
        name (str): Name of the logger category.

    Returns:
        structlog.stdlib.BoundLogger: Bound logger instance.
    """
    return structlog.get_logger(name)
