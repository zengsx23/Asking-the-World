"""Reference mesh proportions, learned pose refinement, then metric alignment."""

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Any

import numpy as np

from atw.perception.rigid_reconstruction import ProjectionAlignment, optimize_reference_scale_and_depth
from atw.perception.wall_box_alignment import WallBoxConstraint


@dataclass(slots=True)
class ReferenceMeshAlignment:
    mesh: Any
    pose_camera: np.ndarray
    alignment: ProjectionAlignment
    rendered_mask: np.ndarray
    visible_mask: np.ndarray
    diagnostics: dict[str, Any]
    timings: dict[str, float]


def align_reference_mesh(*, foundationpose, mesh, seed_pose_camera, rgb, mask,
                         camera_points, intrinsics, all_object_mask_union,
                         rigid_shape: str, reference_surface_prior=None,
                         camera_to_world=None) -> ReferenceMeshAlignment:
    """Stretch boxes/irregular meshes and refine reference poses with FoundationPose.

    Candidate poses are continued before applying scale/depth corrections. Mesh
    handles are retained only for this reference, avoiding cross-object caches.
    Spheres/cylinders retain the single five-iteration reference alignment.
    Static wall boxes constrain every learned update and the final metric fit.
    """
    started = perf_counter()
    wall = WallBoxConstraint.from_prior(reference_surface_prior, rigid_shape=rigid_shape,
                                       camera_to_world=camera_to_world)
    axes = (0, 1, 2) if wall is not None else (None,)
    signs = {axis: (1 if seed_pose_camera[:3, axis] @ wall.normal >= 0 else -1)
             for axis in axes if axis is not None}
    session = foundationpose.prepare_reference(
        rgb=rgb, depth=camera_points[..., 2], mask=mask, intrinsic=intrinsics)
    vertices = np.asarray(mesh.vertices)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    meshes = {}
    coarse, fine = {}, {}
    history, stages = [], []
    alignment_seconds = 0.

    def evaluate(logs, seed, iterations, axis=None):
        nonlocal alignment_seconds
        key = tuple(np.round(logs, 7))
        if key not in meshes:
            candidate_mesh = mesh.copy()
            candidate_mesh.vertices = center + (vertices - center) * np.exp(logs)
            meshes[key] = (candidate_mesh, session.prepare_mesh(candidate_mesh))
        candidate_mesh, prepared = meshes[key]
        constraint_args = {}
        if wall is not None:
            constraint_args['pose_constraint'] = lambda pose: wall.project_pose(
                pose, np.asarray(candidate_mesh.vertices), axis=axis, sign=signs[axis])
        pose, diagnostic = session.refine(
            prepared=prepared, seed_pose_4x4=seed, iterations=iterations, **constraint_args)
        t = perf_counter()
        output_pose = pose
        if wall is not None:
            fitted = wall.fit_scale_and_depth(vertices=np.asarray(candidate_mesh.vertices),
                faces=np.asarray(candidate_mesh.faces), pose=pose, axis=axis, sign=signs[axis],
                intrinsics=intrinsics, target_mask=mask,
                all_object_mask_union=all_object_mask_union, target_camera_points=camera_points)
            if fitted is None:
                alignment_seconds += perf_counter() - t
                history.append(dict(logs=np.asarray(logs).tolist(), face_axis=axis,
                                    iterations=iterations, valid=False, reason='no_visible_wall_projection'))
                return None
            alignment, output_pose, rendered, visible = fitted
        else:
            alignment, rendered, visible = optimize_reference_scale_and_depth(
                vertices_camera=np.asarray(candidate_mesh.vertices) @ pose[:3, :3].T + pose[:3, 3],
                faces=np.asarray(candidate_mesh.faces), object_origin_camera=pose[:3, 3],
                intrinsics=intrinsics, target_mask=mask,
                all_object_mask_union=all_object_mask_union, target_camera_points=camera_points)
        alignment_seconds += perf_counter() - t
        area = abs(alignment.final_area_ratio - 1)
        depth = alignment.final_relative_depth_error
        violation = max(area - .01, 0) + 4 * max(depth - .005, 0)
        valid = bool(np.isfinite(area + depth) and area <= .01 and depth <= .005)
        score = alignment.final_visible_iou - .002 * float(np.sum(logs ** 2)) - 10 * violation
        history.append(dict(logs=np.asarray(logs).tolist(), iterations=iterations, face_axis=axis,
                            pose_camera=pose.tolist(), output_pose_camera=output_pose.tolist(), iou=alignment.final_visible_iou,
                            valid=valid, score=score, alignment=alignment.to_metadata()))
        return dict(mesh=candidate_mesh, pose=pose, output_pose=output_pose, axis=axis,
                    alignment=alignment, rendered=rendered,
                    visible=visible, logs=np.asarray(logs), score=score, valid=valid,
                    diagnostic=diagnostic)

    def best_complete():
        if not fine:
            raise ValueError('no reference candidate has a visible projection under wall contact')
        valid = [c for c in fine.values() if c['valid']]
        return max(valid or list(fine.values()), key=lambda c: c['score'])

    def finish(count):
        ranked = sorted(((k, c) for k, c in coarse.items() if k not in fine),
                        key=lambda item: item[1]['score'], reverse=True)
        for key, c in ranked[:count]:
            result = evaluate(c['logs'], c['pose'], 3, c['axis'])
            if result is not None:
                fine[key] = result
        return best_complete()

    def add(logs, axis=None):
        key = (axis, *np.round(logs, 7))
        if key not in coarse and np.max(np.abs(logs)) <= np.log(1.25):
            result = evaluate(logs, seed_pose_camera, 2, axis)
            if result is not None:
                coarse[key] = result

    def snapshot(stage, best):
        stages.append(dict(stage=stage, coarse_count=len(coarse), fine_count=len(fine),
                           reference_iou=best['alignment'].final_visible_iou,
                           elapsed_seconds=perf_counter() - started))

    stretch_enabled = rigid_shape in {'box', 'irregular'}
    axis_screening = None
    if stretch_enabled:
        directions = np.array([[1., -1., 0.], [1., 0., -1.], [0., 1., -1.]])
        if wall is not None:
            for axis in axes:
                add(np.zeros(3), axis)
            screen = []
            for axis in axes:
                candidate = coarse.get((axis, 0., 0., 0.))
                screen.append(dict(axis=axis, iou=candidate['alignment'].final_visible_iou
                                   if candidate is not None else None))
            ranked = sorted(screen, key=lambda row: row['iou'] if row['iou'] is not None else -1.,
                            reverse=True)
            axes = tuple(sorted(row['axis'] for row in ranked[:2] if row['iou'] is not None))
            axis_screening = dict(candidates=screen, retained_axes=list(axes),
                                  rejected_axes=[row['axis'] for row in screen if row['axis'] not in axes])
            for key in list(coarse):
                if key[0] not in axes:
                    del coarse[key]
        for axis in axes:
            add(np.zeros(3), axis)
            for step in (.1, .2):
                for direction in directions:
                    for sign in (-1, 1):
                        add(step * sign * direction, axis)
        best = finish(6)
        snapshot('global_top6', best)
        if not best['valid'] or best['alignment'].final_visible_iou < .94:
            parents = sorted((c for c in fine.values() if c['valid']),
                             key=lambda c: c['score'], reverse=True)[:2]
            for parent in parents or [best]:
                for direction in directions:
                    for sign in (-1, 1):
                        add(parent['logs'] + .05 * sign * direction, parent['axis'])
            best = finish(4)
            snapshot('local_top4', best)
        if not best['valid'] or best['alignment'].final_visible_iou < .94:
            for direction in directions:
                for sign in (-1, 1):
                    add(best['logs'] + .025 * sign * direction, best['axis'])
            best = finish(3)
            snapshot('fine_top3', best)
    else:
        best = evaluate(np.zeros(3), seed_pose_camera, 5)

    warnings = []
    if not best['valid']:
        warnings.append('reference_alignment_area_or_depth_tolerance_not_met')
        logging.warning('Reference alignment did not meet area/depth tolerances')
    timings = {**session.timings, 'area_depth_alignment_seconds': alignment_seconds,
               'total_seconds': perf_counter() - started}
    diagnostics = {**best['diagnostic'], 'timings': timings,
                   'dimension_ratio_search': {
                       'enabled': stretch_enabled, 'ratio_bounds': [.8, 1.25],
                       'ratios': np.exp(best['logs']).tolist(),
                       'logs': best['logs'].tolist(),
                       'candidate_count': sum(c['iterations'] == 2 for c in history) if wall is not None else len(meshes),
                       'refinement_call_count': len(history), 'stages': stages,
                       'constraints_satisfied': best['valid'], 'warnings': warnings,
                       'candidates': history,
                   }}
    if wall is not None:
        points = (np.asarray(best['mesh'].vertices) * best['alignment'].uniform_scale) @ best['output_pose'][:3, :3].T + best['output_pose'][:3, 3]
        diagnostics['wall_contact'] = dict(enabled=True, face_axis=best['axis'],
            axis_screening=axis_screening,
            axis_sign=signs[best['axis']], normal_camera=wall.normal.tolist(), offset_camera=wall.offset,
            minimum_clearance=float(np.min(points @ wall.normal)-wall.offset),
            normal_alignment_cosine=float(signs[best['axis']] * best['output_pose'][:3, best['axis']] @ wall.normal),
            uniform_scale_depth_method='wall_tangent_translation_and_scale',
            reference_surface_prior=reference_surface_prior)
    return ReferenceMeshAlignment(best['mesh'], best['output_pose'], best['alignment'],
                                  best['rendered'], best['visible'], diagnostics, timings)
