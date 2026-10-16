"""Mask-aware VLM selection of one canonical frame per deformable object."""


from __future__ import annotations


from collections.abc import Mapping

from dataclasses import dataclass

from typing import Any

EXPECTED_MODEL = "google/gemini-3-flash-preview"




@dataclass(frozen=True, slots=True)
class ReferenceFrameSelection:
    target_id: str
    reference_frame: int
    confidence: float
    reason: str
    alternative_frames: tuple[int, ...]
    active_frame_count: int
    contact_sheets: tuple[str, ...]
    agent_response: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "reference_frame": self.reference_frame,
            "confidence": self.confidence,
            "reason": self.reason,
            "alternative_frames": list(self.alternative_frames),
            "active_frame_count": self.active_frame_count,
            "contact_sheets": list(self.contact_sheets),
            "agent_response": dict(self.agent_response),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReferenceFrameSelection":
        return cls(
            target_id=str(payload["target_id"]),
            reference_frame=int(payload["reference_frame"]),
            confidence=float(payload["confidence"]),
            reason=str(payload["reason"]),
            alternative_frames=tuple(
                int(frame) for frame in payload.get("alternative_frames", [])
            ),
            active_frame_count=int(payload["active_frame_count"]),
            contact_sheets=tuple(
                str(path) for path in payload.get("contact_sheets", [])
            ),
            agent_response=dict(payload.get("agent_response", {})),
        )
