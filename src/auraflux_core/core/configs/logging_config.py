import logging
import sys
from .settings import settings

def setup_logging(name='twinrad'):
    """
    Sets up the application's logging configuration.

    The log level is set based on the `settings.log_level` environment variable.
    If not specified, it defaults to 'INFO'.

    @param name: The name of the logger. Defaults to 'twinrad'.
    @return: Configured logger instance.
    """
    # Create the logger instance
    logger = logging.getLogger(name)

    # Get the log level from settings, default to INFO if not set
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Create a console handler to print logs to the standard output
    handler = logging.StreamHandler(sys.stdout)

    # Define a log message format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    # Prevent duplicate log messages in some environments
    logger.propagate = False

    return logger