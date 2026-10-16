"""Evidence validation and native-resolution frames for conditional 2D routing."""


from __future__ import annotations


from typing import Any, Mapping



def gravity_artifact(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a categorical image-plane direction, not metric gravity calibration."""
    direction = raw.get("gravity")
    vectors = {"screen_down": [0, 1, 0], "screen_inward": [0, 0, 1]}
    if not isinstance(direction, str) or direction not in vectors:
        raise ValueError('Choose "gravity": "screen_down" or "screen_inward"; return one string.')
    return dict(
        direction=direction, reason=f"VLM selected {direction} from the two image-plane gravity directions.",
        coordinate_frame="image plane; x right, y down, z away from viewer",
        direction_unit_camera=vectors[direction], magnitude=None,
        parameterization="Canonical category, not metric calibration",
    )
