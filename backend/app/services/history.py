"""
History service — persist predictions to a JSON file.
"""
from __future__ import annotations

import json

from app.config import HISTORY_FILE, PREVIEW_LEN


def _ensure_dir() -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)


def load() -> list[dict]:
    _ensure_dir()
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save(history: list[dict]) -> None:
    _ensure_dir()
    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append(
    history: list[dict],
    *,
    prediction_id: str,
    text: str,
    prediction: str,
    confidence: float,
    timestamp: str,
) -> list[dict]:
    """Prepend a new entry (most-recent-first) and return the updated list."""
    entry = {
        "id":           prediction_id,
        "text_preview": text[:PREVIEW_LEN] + ("…" if len(text) > PREVIEW_LEN else ""),
        "prediction":   prediction,
        "confidence":   confidence,
        "timestamp":    timestamp,
    }
    return [entry] + history


def delete_one(history: list[dict], item_id: str) -> tuple[list[dict], bool]:
    """Return (updated_list, was_found)."""
    new = [h for h in history if h["id"] != item_id]
    return new, len(new) < len(history)
