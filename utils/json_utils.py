from __future__ import annotations

from typing import Any


def stringify_dict_keys(value: Any) -> Any:
    """Recursively convert dictionary keys to strings for UI-safe JSON rendering."""
    if isinstance(value, dict):
        return {str(key): stringify_dict_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stringify_dict_keys(item) for item in value]
    return value


def model_or_value_to_json(value: Any) -> dict[str, Any]:
    """Convert a pydantic model or plain value into a JSON-safe dictionary."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return stringify_dict_keys(value)  # type: ignore[return-value]

