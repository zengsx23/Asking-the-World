"""Fixed-parameter native scene loading and forward experiments; no upstream writes."""
from dataclasses import replace
import json

import numpy as np
from atw.polyworld.contphy_experiments import ContPhyExperiments, extend, rows


class NativeExperiments(ContPhyExperiments):
    # Internal adaptive mass selection; the public tool does not accept factors.
    mass_factor_bounds = (.001, 1000.)

    def _load(self):
        if self.loaded is not None:
            return self.loaded
        if self.kind == 'cloth':
            return super()._load()
        if self.kind == 'soft_body':
            from atw.polyworld.soft_body_swr import load_softbody_scene
            from atw.world_modeling.softbody_observed_radius import MODEL
            if self.fit['model'] != MODEL:
                raise ValueError('Unsupported native softbody recipe')
            scene, _ = load_softbody_scene(self.paths['tracking'], reconstruction_dir=self.paths['reconstruction'])
            with np.load(self.paths['swr']/self.fit['prediction_npz']) as archive:
                self.theta = archive['theta'].copy()
                self.loaded = replace(scene, initial_centers=archive['initial_centers'].copy())
        elif self.kind == 'fluid':
            from atw.polyworld.fluid_swr import FluidScene, FluidEmitter
            if self.fit['model'] != 'agentic-fluid-route-v1':
                raise ValueError('Unsupported native fluid recipe')
            metadata = json.loads((self.paths['swr']/self.fit['observations']).read_text())
            with np.load(self.paths['swr']/'observations.npz') as archive:
                self.loaded = FluidScene(self.paths['swr'], metadata['video_path'], metadata['fps'],
                    *metadata['image_wh'], tuple(metadata['fluid_ids']), archive['target_masks'].copy(),
                    archive['obstacle_mask'].copy(), tuple(FluidEmitter(e['object_id'],e['fluid_id'],
                    tuple(e['xy']),tuple(e['active_frames'])) for e in metadata['emitters']),
                    np.empty((0,2),np.float32),np.empty(0,np.int32),np.empty(0,np.int32),1,
                    particle_emitter=np.empty(0,np.int32),collision_obstacle_mask=archive['collision_obstacle_mask'].copy())
            self.fluid_metadata = metadata
            # Explicit internal adapter fields for shared property measurements.
            self.fit = dict(self.fit, image_wh=metadata['image_wh'], fps=metadata['fps'],
                emitters=metadata['emitters'], smoothing_length_pixels=self.fit['physical_support_pixels'])
            self.backend.fit = self.fit
        else:
            self.loaded = self._load_rope()
        return self.loaded

    def _load_rope(self):
        from atw.polyworld.rope_swr import prepare_rope_scenes
        from atw.polyworld.rope_topology import available_pulleys
        from atw.perception.object_reconstruction import read_packed_mask_archive
        fits = {r['rope_id']: r for r in self.fit['ropes']}
        source = self.fit['source']
        scenes, source = prepare_rope_scenes(source, mask_archive=self.backend.mask_archive_path,
            mask_index=self.backend.mask_index_path, node_count=next(iter(fits.values()))['node_count'],
            roles=source['rope_roles'])
        tracks = read_packed_mask_archive(self.backend.mask_archive_path, self.backend.mask_index_path)
        selected = []
        for scene in scenes:
            if scene.rope_id not in fits:
                continue
            fit = fits[scene.rope_id]
            catalog = {b.object_id: b for b in (*scene.bodies, *available_pulleys(scene, tracks, source))}
            bodies = tuple(replace(catalog[b['object_id']], role=b['role'], dynamic=b['dynamic'],
                radius=b['radius_normalized'], parent_id=b.get('parent_id'),
                initial_center=tuple(b.get('initial_center_normalized', catalog[b['object_id']].initial_center)),
                endpoint_offset=tuple(b.get('endpoint_offset_normalized', catalog[b['object_id']].endpoint_offset)),
                attachment_source=b.get('attachment_source', catalog[b['object_id']].attachment_source),
                default_relative_mass=b.get('default_relative_mass',1.)) for b in fit['bodies'])
            saved = {b.object_id:b for b in bodies}
            bodies += tuple(b for b in scene.bodies if b.object_id not in saved and b.role=='attachment'
                and not b.dynamic and b.parent_id in saved and not saved[b.parent_id].dynamic)
            selected.append(replace(scene, bodies=bodies, initial_positions=np.asarray(fit['initial_centerline_normalized']),
                mass_model=fit['optimization'].get('mass_model','particle'),
                route_visits=tuple(tuple(v) for v in fit['routing'].get('visits',())),
                endpoint_pinned=tuple(fit['endpoint_pinned']),
                event_frame_offset=int(fit['event_frame'])-scene.frames[0] if fit.get('event_frame') is not None else 0))
        return tuple(selected)

    def _fluid(self, original, extension, removed, masses, emission, audit, emission_fluid_id=None):
        # The shared forward solver uses the same PBF configuration and emission
        # convention. Native fits supply one phase per E source, never color merges.
        fit = self.fit
        sources={r['static_object_id'] for r in self.fluid_metadata['source_records']}
        if set(removed)&sources:
            raise ValueError('Static source removal is not an obstacle-only intervention')
        if emission_fluid_id is not None:
            records = {r['source_id']:r for r in self.fluid_metadata['source_records']}
            target = records[emission_fluid_id]
            candidates = [r for r in records.values() if r['fluid_object_id']==target['fluid_object_id']
                and r['first_visible_frame']<target['first_visible_frame']]
            if candidates:
                earlier = min(candidates,key=lambda r:r['first_visible_frame'])['source_id']
                theta = np.asarray(fit['parameter_vector'],np.float32).copy()
                count = len(original.fluid_ids)
                first, later = original.fluid_ids.index(earlier), original.fluid_ids.index(emission_fluid_id)
                for offset in (3,3+count):theta[offset+later]=theta[offset+first]
                self.fit = dict(fit,parameter_vector=theta.tolist())
        try:
            result = super()._fluid(original,extension,removed,masses,emission,audit,emission_fluid_id)
        finally:
            self.fit = fit
        if emission_fluid_id is not None and candidates:
            result['assumptions'].append(
                'The selected later source uses fitted viscosity and relative density from the earlier '
                'same-color source; emitter parameters are unchanged.')
        return result

    def _soft_body(self, original, extension, removed, masses, emission, audit):
        from atw.polyworld.soft_body_swr import make_parameter_layout, _contact_events, _numpy_deformation
        allowed = set(original.object_ids) | {o.object_id for o in original.obstacles}
        if removed - allowed:
            raise ValueError(f"Objects are absent from the fitted soft-body world: {sorted(removed-allowed)}")
        selected = [i for i,n in enumerate(original.object_ids) if n not in removed]
        if not selected:
            raise ValueError("At least one soft body must remain")
        count = original.frame_count + round(extension * original.fps)
        changes = dict(object_ids=tuple(original.object_ids[i] for i in selected),
            frames=tuple(range(original.frames[0], original.frames[0]+count)),
            obstacles=tuple(o for o in original.obstacles if o.object_id not in removed))
        for key in ("target_centers", "target_subset_centers", "target_covariances", "target_deformations",
                    "reference_subset_offsets", "reference_subset_covariances",
                    "shape_observed_mask", "shape_contact_recovered_mask", "observed_mask"):
            changes[key] = extend(getattr(original,key)[:,selected], count, fill=0)
        for key in ("initial_centers", "radii", "wall_depths"):
            changes[key] = getattr(original,key)[selected].copy()
        for key in ("first_offsets", "trajectory_paths", "valid_frame_counts", "point_counts"):
            changes[key] = tuple(getattr(original,key)[i] for i in selected)
        scene = replace(original, **changes)
        if not audit:
            # No observed pair normals or trajectory-derived penetration allowances in edited/future motion.
            scene = replace(scene, observed_mask=np.zeros_like(scene.observed_mask), target_centers=np.zeros_like(scene.target_centers))
        from atw.world_modeling.softbody_observed_radius import make_engine
        from atw.world_modeling.softbody_ballistic_utils import FIELDS
        engine = make_engine(scene, device=self.device, substeps=self.fit["substeps_per_frame"])
        layout = make_parameter_layout(scene.object_count)
        original_layout = make_parameter_layout(original.object_count)
        theta = np.zeros(layout.size,np.float32)
        for i,j in enumerate(selected):
            theta[2*i:2*i+2] = self.theta[2*j:2*j+2]
            for field,_,_ in FIELDS:
                theta[getattr(layout,field).start+i] = self.theta[getattr(original_layout,field).start+j]
        if not audit:
            active = np.arange(count)[:,None] >= np.asarray(scene.first_offsets)[None]
            engine.scene = replace(engine.scene, observed_mask=active.astype(float))
        predicted, contacts = engine.rollout_numpy(theta)
        trajectories, geometry = {}, {}
        for i,name in enumerate(scene.object_ids):
            world = (predicted[:,i,:1]*scene.wall_lateral_world + predicted[:,i,1:2]*scene.gravity_down_world
                     + scene.wall_depths[i]*scene.wall_normal_world)
            trajectories[name] = rows(scene.frames[scene.first_offsets[i]:], world[scene.first_offsets[i]:], "center_world")
            for row, value in zip(trajectories[name], predicted[scene.first_offsets[i]:,i]):
                row["plane_lateral_down_m"] = value[:2].tolist()
            with np.load(scene.trajectory_paths[i]) as archive:
                reference_frame = int(np.asarray(archive["reference_frame"]).item())
                reference_world = np.asarray(archive["model_xyz"][reference_frame], dtype=np.float64)
            valid = np.isfinite(reference_world).all(axis=1)
            reference_world = reference_world[valid]
            if len(reference_world) > 2048:
                reference_world = reference_world[np.linspace(0, len(reference_world)-1, 2048).astype(int)]
            reference_plane = np.stack((reference_world @ scene.wall_lateral_world,
                                        reference_world @ scene.gravity_down_world), axis=-1)
            plane_offsets = reference_plane - reference_plane.mean(axis=0)
            depth_offsets = reference_world @ scene.wall_normal_world - scene.wall_depths[i]
            deformation = _numpy_deformation(predicted[:, i])
            deformed_plane = predicted[:, i, None, :2] + np.einsum(
                "tij,pj->tpi", deformation, plane_offsets)
            geometry[name] = (deformed_plane[..., :1] * scene.wall_lateral_world
                + deformed_plane[..., 1:2] * scene.gravity_down_world
                + (scene.wall_depths[i] + depth_offsets)[None, :, None] * scene.wall_normal_world)
        return dict(trajectories=trajectories, geometry=geometry, coordinate_system="Retained 3D world coordinates",
            events=[dict(event,time_s=event["frame_index"]/scene.fps) for event in _contact_events(scene,contacts)],
            obstacles=[dict(object_id=o.object_id, center_plane=o.center, half_extents=o.half_extents) for o in scene.obstacles],
            assumptions=["Fixed effective soft-body parameters and static walls; surviving balls retain their modeled entry states.",
                         "New experiments use physical contact separation without observed pair-normal guidance; the native fixed-radius contact recipe and output projection are retained."])
