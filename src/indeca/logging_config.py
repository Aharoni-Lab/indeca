import logging
import logging.handlers
from pathlib import Path
from typing import Union

"""
Internal logging configuration for minian_bin package.

This module configures logging for the minian_bin package internals.
By default, the package creates a 'logs' directory and writes logs to 'minian_bin.log'.
Users can control the logging level using set_package_log_level().

Example:
    # In your application:
    from minian_bin import set_package_log_level
    set_package_log_level(logging.DEBUG)
"""

# Set up package logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create the package's root logger
logger = logging.getLogger("minian_bin")

# Create handler for package logging
file_handler = logging.handlers.RotatingFileHandler(
    log_dir / "minian_bin.log", maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Configure handler
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)  # Capture all levels in file

# Add handler to logger
logger.addHandler(file_handler)
logger.addHandler(
    logging.NullHandler()
)  # Prevent "No handlers could be found" warnings
logger.setLevel(logging.WARNING)  # Set default level to WARNING


def set_package_log_level(level: Union[int, str]) -> None:
    """
    Set the logging level for the minian_bin package.
    The default level is WARNING if this function is not called.

    Parameters
    ----------
    level : Union[int, str]
        Logging level. Can be either a string ('DEBUG', 'INFO', 'WARNING',
        'ERROR', 'CRITICAL') or an integer (logging.DEBUG, logging.INFO, etc.)
    """
    if isinstance(level, str):
        level = level.upper()
        if not hasattr(logging, level):
            raise ValueError(f"Invalid logging level: {level}")
        level = getattr(logging, level)

    logger.setLevel(level)
    # Add a test message to verify logging is working
    logger.info(
        f"minian_bin logging initialized with level: "
        f"{level if isinstance(level, str) else logging.getLevelName(level)}"
    )
    logger.debug(
        "This is a test DEBUG message - you should only see this if level is DEBUG"
    )


# Get a logger for a specific module within the package
def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module within minian_bin.
    For internal package use only.
    Parameters
    ----------
    module_name : str
        Name of the module (e.g., 'pipeline', 'deconv')
    Returns
    -------
    logging.Logger
        Logger instance for the module
    """
    return logging.getLogger(f"minian_bin.{module_name}")
