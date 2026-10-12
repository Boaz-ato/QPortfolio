"""Logging utilities for qportfolio.

Libraries should not configure the root logger. We attach a NullHandler so
that log records are silently discarded unless the application configures
a handler. Users can enable logging with:

    import logging
    logging.getLogger("qportfolio").setLevel(logging.DEBUG)
    logging.getLogger("qportfolio").addHandler(logging.StreamHandler())
"""

from __future__ import annotations

import logging

LOGGER_NAME = "qportfolio"

logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the 'qportfolio' hierarchy."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
