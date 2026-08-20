"""Centralised logging configuration for the application.

Creates a `logs/` directory if it doesn't exist and configures the root logger
to write timestamped, level-tagged records into a per-day log file. Other
modules should import `get_logger` rather than calling `logging.basicConfig`
again, so every module shares one consistent format and destination.
"""

import logging                  # stdlib logging framework (loggers, handlers, formatters)
import os                       # filesystem helpers: directory creation + path joining
from datetime import datetime   # used to stamp the log filename with the current date

# str: directory (relative to the process's working directory) where log files live.
LOGS_DIR = "logs"

# Ensure the directory exists before logging starts.
# exist_ok=True makes this a no-op on subsequent runs instead of raising FileExistsError.
os.makedirs(LOGS_DIR, exist_ok=True)

# str: full path to today's log file, e.g. "logs/log_2026-08-16.log".
# NOTE: the date is evaluated once at import time, so a process that runs past
# midnight keeps writing to the file it opened on startup.
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}.log")

# Configure the ROOT logger once, at import time. Every logger created later
# propagates its records up to this handler.
#   filename -> append records to LOG_FILE instead of stderr
#   format   -> "2026-08-16 10:31:04,512 - INFO - message"
#   level    -> drop anything below INFO (i.e. DEBUG) at the root
logging.basicConfig(
    filename=LOG_FILE,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger wired into the application-wide configuration.

    Args:
        name (str): Identifier for the logger, conventionally the caller's
            `__name__`. Dotted names form a hierarchy — "app.db" is a child
            of "app" — which is what lets records propagate to the root
            handler configured above.

    Returns:
        logging.Logger: A logger set to INFO level. Calls with the same `name`
        return the same underlying object, so this is safe to call repeatedly.
    """
    logger: logging.Logger = logging.getLogger(name)  # cached per-name by the logging module
    logger.setLevel(logging.INFO)                     # suppress DEBUG on this logger specifically
    return logger
