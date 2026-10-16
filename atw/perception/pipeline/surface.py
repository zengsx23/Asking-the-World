"""Scene support and wall constraints from segmented background geometry."""

from typing import Any, Mapping

import numpy as np

from atw.perception.rigid_tracking import PointCloudSupportConstraint


def support_constraint_from_gravity_summary(
    summary: Mapping[str, Any]
) -> PointCloudSupportConstraint | None:
    """A wall's collision normal is independent of its in-plane gravity."""
    if summary.get("status") != "completed":
        return None
    selected = summary["selected"]
    surface_type = summary.get("scene_surface_type")
    if surface_type in {"wall_plane", "support_plane"}:
        plane = selected[surface_type]
        normal, offset = plane["normal_world"], plane["offset_world"]
    else:
        normal = selected["up_direction_world"]
        offset = selected["full_mesh"]["support_plane_offset_world"]
    return PointCloudSupportConstraint(
        up_world=np.asarray(normal, dtype=np.float64), plane_offset_world=float(offset)
    )


def wall_box_reference_priors(surface_summary, motion_decision, rigid_shapes):
    """Activate side contact only for static boxes on the classified wall."""
    if (surface_summary.get('status') != 'completed'
            or surface_summary.get('scene_surface_type') != 'wall_plane'):
        return {}
    plane = surface_summary['selected']['wall_plane']
    return {name: dict(surface_type='wall_plane', tracking_branch='static',
                       normal_world=plane['normal_world'], offset_world=plane['offset_world'])
            for name, shape in rigid_shapes.items()
            if shape == 'box' and motion_decision['objects'][name]['tracking_branch'] == 'static'}
