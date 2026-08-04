"""Structured JSON-line logging used across the backend.

Logs are written both to stdout (for local dev) and to
backend/storage/logs/guidely.log so they can be inspected after the
process exits.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "storage" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "guidely.log"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured with JSON output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_JsonFormatter())
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(_LOG_FILE)
    file_handler.setFormatter(_JsonFormatter())
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, message: str, **fields) -> None:
    """Log a structured event with arbitrary extra key/value fields."""
    logger.info(message, extra={"extra_fields": fields})
