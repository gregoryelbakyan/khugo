"""Small, structured logging helpers shared by commands."""

from __future__ import annotations

import json
import logging
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as newline-delimited JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        if hasattr(record, "details"):
            payload["details"] = record.details
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once for machine-readable command output."""
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
