"""Extract color labels from retained object descriptions."""

from __future__ import annotations

import re
from typing import Any


COLOR_ALIASES = (
    ("light blue", "light blue"),
    ("dark brown", "dark brown"),
    ("olive green", "olive green"),
    ("turquoise", "turquoise"),
    ("silver", "silver"),
    ("orange", "orange"),
    ("purple", "purple"),
    ("yellow", "yellow"),
    ("green", "green"),
    ("brown", "brown"),
    ("white", "white"),
    ("black", "black"),
    ("pink", "pink"),
    ("cyan", "cyan"),
    ("teal", "teal"),
    ("grey", "gray"),
    ("gray", "gray"),
    ("blue", "blue"),
    ("red", "red"),
)


def infer_color(obj: dict[str, Any]) -> str:
    explicit = obj.get("color")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower().replace("grey", "gray")
    texts = [
        obj.get("target_id", ""),
        *(obj.get("short_noun_phrases") or []),
        obj.get("discriminative_description", ""),
        obj.get("description", ""),
    ]
    for raw in texts:
        text = re.sub(r"[_-]+", " ", str(raw).lower())
        matches = []
        for phrase, normalized in COLOR_ALIASES:
            match = re.search(rf"\b{re.escape(phrase)}\b", text)
            if match:
                matches.append((match.start(), -len(phrase), normalized))
        if matches:
            return min(matches)[2]
    raise ValueError(f"No stage-one color label is available for {obj.get('target_id')!r}")
