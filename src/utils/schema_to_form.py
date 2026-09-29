from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_schema(schema_source: Any) -> dict[str, Any]:
    """Normalize a Pandera schema object, YAML path, or dict to a Python dict."""
    if isinstance(schema_source, dict):
        return schema_source

    if hasattr(schema_source, "to_yaml"):
        return yaml.safe_load(schema_source.to_yaml()) or {}

    if isinstance(schema_source, (str, Path)):
        path = Path(schema_source)
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    raise TypeError(f"Unsupported schema source: {type(schema_source)!r}")


def build_form_fields(schema_source: Any) -> list[dict[str, Any]]:
    """Translate a schema YAML dict into HTML-friendly field metadata."""
    schema = load_schema(schema_source)
    columns = schema.get("columns", {})
    fields: list[dict[str, Any]] = []

    for name, definition in columns.items():
        if not isinstance(definition, dict):
            continue

        dtype = str(definition.get("dtype", "")).lower()
        options = definition.get("isin")
        minimum = definition.get("greater_than_or_equal_to")
        maximum = definition.get("less_than_or_equal_to")

        if isinstance(options, list) and options:
            field_type = "select"
            default = options[0]
            input_type = "select"
        elif any(token in dtype for token in ("float", "double")):
            field_type = "number"
            input_type = "number"
            default = float(minimum) if minimum is not None else 0.0
        elif any(token in dtype for token in ("int", "bool")):
            field_type = "number"
            input_type = "number"
            default = int(minimum) if minimum is not None else 0
        else:
            field_type = "text"
            input_type = "text"
            default = ""

        field = {
            "name": name,
            "label": name.replace("_", " ").title(),
            "input_type": input_type,
            "field_type": field_type,
            "default": default,
            "options": options,
            "step": "0.01" if "float" in dtype or "double" in dtype else "1",
            "min": minimum,
            "max": maximum,
            "required": True,
        }
        fields.append(field)

    return fields


def render_form_html(schema_source: Any) -> str:
    """Return a simple HTML fragment for a generated inference form."""
    form_fields = build_form_fields(schema_source)
    html = []
    for field in form_fields:
        label = field["label"]
        name = field["name"]
        options = field.get("options")
        if options:
            html.append(f'<label>{label}: <select id="{name}">')
            for option in options:
                selected = " selected" if option == field["default"] else ""
                html.append(f'<option value="{option}"{selected}>{option}</option>')
            html.append("</select></label>")
        else:
            attrs = [f'id="{name}"', 'type="number"' if field["input_type"] == "number" else 'type="text"']
            if field.get("step"):
                attrs.append(f'step="{field["step"]}"')
            if field.get("min") is not None:
                attrs.append(f'min="{field["min"]}"')
            if field.get("max") is not None:
                attrs.append(f'max="{field["max"]}"')
            if field.get("default") is not None:
                attrs.append(f'value="{field["default"]}"')
            html.append(f'<label>{label}: <input {" ".join(attrs)} /></label>')
    return "\n".join(html)
