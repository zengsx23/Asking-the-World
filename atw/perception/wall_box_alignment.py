"""Reference-frame box alignment on a classified wall contact manifold."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from atw.perception.mesh_projection import render_mask_occluded_mesh
from atw.perception.rigid_reconstruction import ProjectionAlignment, _mask_iou


@dataclass(frozen=True)
class WallBoxConstraint:
    normal: np.ndarray
    offset: float

    @classmethod
    def from_prior(cls, prior, *, rigid_shape, camera_to_world):
        if (prior is None or rigid_shape != 'box'
                or prior['surface_type'] != 'wall_plane'
                or prior['tracking_branch'] != 'static'):
            return None
        normal_world = np.asarray(prior['normal_world'], dtype=float)
        camera = np.asarray(camera_to_world, dtype=float)
        normal = camera[:3, :3].T @ normal_world
        length = np.linalg.norm(normal)
        if not np.isfinite(length) or length <= 0:
            raise ValueError('wall normal must be finite and nonzero')
        offset = (float(prior['offset_world']) - normal_world @ camera[:3, 3]) / length
        if not np.isfinite(offset):
            raise ValueError('wall offset must be finite')
        return cls(normal / length, float(offset))

    def project_pose(self, pose, vertices, *, axis, sign, scale=1.):
        """Keep in-wall yaw and translation; align one axis and close the gap."""
        pose = np.array(pose, dtype=float, copy=True)
        source = pose[:3, axis]
        target = sign * self.normal
        cross = np.cross(source, target)
        sine = np.linalg.norm(cross)
        cosine = np.clip(source @ target, -1., 1.)
        if sine > 1e-10:
            correction = Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix()
        elif cosine < 0:
            tangent = pose[:3, (axis + 1) % 3]
            correction = Rotation.from_rotvec(np.pi * tangent).as_matrix()
        else:
            correction = np.eye(3)
        pose[:3, :3] = correction @ pose[:3, :3]
        world = (np.asarray(vertices) * scale) @ pose[:3, :3].T + pose[:3, 3]
        pose[:3, 3] += (self.offset - np.min(world @ self.normal)) * self.normal
        return pose

    def fit_scale_and_depth(self, *, vertices, faces, pose, axis, sign,
                            intrinsics, target_mask, all_object_mask_union,
                            target_camera_points):
        """Fit scale and wall-tangent translation; never translate off the wall.

        Rasterized area/depth and silhouette centroid form a small three-variable
        least-squares problem. Finite pixel-sized differences handle rasterization;
        every trial is projected onto contact before measurement.
        """
        mask = np.asarray(target_mask, bool)
        valid = mask & np.isfinite(target_camera_points).all(axis=-1) & (target_camera_points[..., 2] > 0)
        if not valid.any():
            raise ValueError('target mask has no valid depth pixels')
        target_depth = float(np.median(target_camera_points[..., 2][valid]))
        target_area = int(mask.sum())
        yy, xx = np.nonzero(mask); centroid = np.array([xx.mean(), yy.mean()])
        basis = pose[:3, [i for i in range(3) if i != axis]]
        # Variables are log scale and two translations measured in object lengths.
        length = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        pixel_world = target_depth / max(float(intrinsics[0, 0]), float(intrinsics[1, 1]))
        steps = np.array([.01, max(pixel_world / length, .002), max(pixel_world / length, .002)])
        def evaluate(x):
            scale = float(np.exp(x[0]))
            current = np.array(pose, copy=True)
            current[:3, 3] += basis @ (x[1:] * length)
            current = self.project_pose(current, vertices, axis=axis, sign=sign, scale=scale)
            camera_vertices = (vertices * scale) @ current[:3, :3].T + current[:3, 3]
            r = render_mask_occluded_mesh(vertices_camera=camera_vertices, faces=faces,
                intrinsic=intrinsics, image_shape=mask.shape,
                occluder_mask=all_object_mask_union, self_mask=mask)
            raw, visible = r['rendered_mask'], r['visible_mask']
            if not visible.any():
                return None
            y, xpixel = np.nonzero(visible)
            depth = float(np.median(r['rendered_depth'][visible]))
            area_ratio = float(visible.sum() / target_area)
            residual = np.r_[(area_ratio - 1) / .01, (depth - target_depth) / (target_depth * .005),
                             (np.array([xpixel.mean(), y.mean()]) - centroid) / 2.]
            return dict(pose=current, scale=scale, raw=raw, visible=visible, depth=depth,
                        area_ratio=area_ratio, residual=residual, cost=float(residual @ residual))
        x = np.zeros(3); first = current = evaluate(x)
        if current is None:
            # A coarse face hypothesis can project entirely outside the image.
            return None
        trace = []
        for iteration in range(12):
            trace.append(dict(iteration=iteration, uniform_scale=current['scale'],
                area_ratio=current['area_ratio'], relative_depth_error=abs(current['depth']-target_depth)/target_depth,
                residual_cost=current['cost']))
            if (abs(current['area_ratio']-1) <= .01 and abs(current['depth']-target_depth)/target_depth <= .005
                    and np.linalg.norm(current['residual'][2:]) <= .5):
                break
            columns = []
            for j in range(3):
                delta = np.zeros(3); delta[j] = steps[j]
                sample = evaluate(x + delta)
                columns.append((sample['residual']-current['residual'])/steps[j] if sample is not None else np.zeros(4))
            jac = np.column_stack(columns)
            delta = np.linalg.lstsq(np.vstack((jac, np.eye(3)*.1)),
                                    np.r_[-current['residual'], np.zeros(3)], rcond=None)[0]
            delta /= max(1., np.max(np.abs(delta)) / .2)
            improved = False
            for factor in (1., .5, .25):
                trial_x = x + factor * delta; trial = evaluate(trial_x)
                if trial is not None and trial['cost'] < current['cost'] - 1e-8:
                    x, current = trial_x, trial; improved = True; break
            if not improved:
                break
        final = current
        relative = abs(final['depth']-target_depth)/target_depth
        alignment = ProjectionAlignment(uniform_scale=final['scale'], camera_depth_translation=0.,
            target_median_depth=target_depth, initial_visible_iou=_mask_iou(first['visible'],mask),
            final_visible_iou=_mask_iou(final['visible'],mask), initial_raw_iou=_mask_iou(first['raw'],mask),
            final_raw_iou=_mask_iou(final['raw'],mask), final_area_ratio=final['area_ratio'],
            final_depth_error=final['depth']-target_depth, final_relative_depth_error=relative,
            final_occluded_rendered_pixels=int((final['raw'] & ~final['visible']).sum()),
            iteration_count=len(trace), stop_reason='wall_contact_area_and_depth_tolerance'
            if abs(final['area_ratio']-1)<=.01 and relative<=.005 else 'wall_contact_metric_constraints_unmet',
            alignment_trace=tuple(trace), method='wall_constrained_foundationpose_then_contact_scale_depth_alignment')
        return alignment, final['pose'], final['raw'], final['visible']
