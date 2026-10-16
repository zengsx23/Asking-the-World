"""Load physical backends from explicit asset manifests.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from atw.polyworld.scene_evidence import SceneEvidence
from atw.polyworld.physics_experiments import PhysicsExperiments, encode_saved_parameters, mask_visibility
from atw.world_probing.evidence import write


class NativeRigid(PhysicsExperiments):
    def __init__(self, backend, tracking, reconstruction, device):
        from atw.polyworld.swr_sysid import load_rotational_scene
        self.scene, _ = load_rotational_scene(tracking, reconstruction_dir=reconstruction)
        self.fit = backend.fit
        self.theta = encode_saved_parameters(self.scene, self.fit)
        self.output, self.device = backend.output, device
        self.rollouts = backend.rollouts
        self.root = tracking.parent

    def assets(self):
        return dict(limitations=['Planar fixed-height rigid proxies; fitted parameters, not physical ground truth.',
            'Bodies appear at their first usable 3D state and persist beyond their last tracked state.',
            'Finite event coverage does not establish infinite-time absence.'])

    def simulate(self, extension_seconds, remove_objects=None, option_id=None):
        from atw.polyworld.physics_experiments import experiment_scene
        from atw.polyworld.swr_sysid import _predicted_poses_world
        removed = sorted(set(remove_objects or []))
        scene = experiment_scene(self.scene, extension_seconds, removed)
        predicted, events = self._engine(scene).rollout_numpy(self.theta)
        centers, poses = _predicted_poses_world(scene, predicted)
        rid = f'rollout_{len(self.rollouts)+1:04d}'
        value = dict(rollout_id=rid, source_kind='simulation', object_ids=list(scene.object_ids),
            remove_objects=removed, extension_seconds=extension_seconds,
            end_seconds=(scene.frame_count-1)/scene.fps, observed_end_seconds=(self.scene.frame_count-1)/scene.fps,
            event_capacity_reached=len(events)>=128,
            events=[dict(event_time_s=e.time_s, frame_float=e.time_s*scene.fps,
                object_ids=[scene.object_ids[e.object_i],scene.object_ids[e.object_j]],
                normal_impulse=e.normal_impulse_magnitude) for e in events],
            predicted_trajectories={name:[] if name in removed else [dict(frame_index=f,time_s=f/scene.fps,
                plane_xy=predicted[f,i,:2].tolist(),yaw_delta_rad=float(predicted[f,i,2]),
                point_cloud_center_world=centers[f,i].tolist(),pose_4x4_world=poses[f,i].tolist())
                for f in range(scene.first_offsets[i],scene.frame_count)] for i,name in enumerate(scene.object_ids)})
        self.rollouts[rid] = value
        write(self.output/'rollouts'/f'{rid}.json', value)
        return {k:v for k,v in value.items() if k!='predicted_trajectories'}


class NativeEvidence(SceneEvidence):
    def change_mass(self, object_id, direction, option_id=None):
        from atw.world_probing.engines.rope_mass import select_mass_factor
        if self.engine != 'rope':
            raise ValueError('Mass interventions require the rope engine')
        self._ids([object_id])
        diagnostics = {}
        index = getattr(self, '_mass_selection_index', 0) + 1
        self._mass_selection_index = index
        try:
            factor, binding = select_mass_factor(self._contphy._load(), self.fit['ropes'],
                object_id, direction, self._contphy.device, diagnostics)
        finally:
            # Retained locally, excluded from canonical tool evidence and prompts.
            write(self.output/'mass_selection'/f'{index:04d}.json', diagnostics)
        result = self.simulate(0., mass_scales={binding['object_id']:factor},
                               option_id=option_id, rope_initial_state='rest')
        return result | dict(object_id=object_id, direction=direction, factor=factor,
                             effective_mass_factor=factor, mass_binding=binding)

    def __init__(self, inputs, tools, output, *, tracking_only=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.video = Path(inputs['video_path'])
        self.selected_scene = inputs['stage_one_result']['data']['scene']
        self.sources = self.selected_scene['sources']
        registry = {a['artifact_id']:a for a in inputs['stage_one_result']['artifacts']}
        def path(identifier):
            if identifier not in registry:
                raise ValueError(f'Unknown explicit scene artifact: {identifier}')
            target = tools.run_dir/registry[identifier]['path']
            if not target.is_file():
                raise ValueError('Referenced scene artifact is missing')
            return target
        self.artifact_path = path
        if tracking_only:
            self.manifest = dict(engine=self.selected_scene['engine'])
            self.fit, self.paths = {}, {}
        else:
            manifest_path = path(self.sources['swr_asset_id'])
            self.manifest = json.loads(manifest_path.read_text())
            if self.manifest['availability']!='fitted':
                raise ValueError('The selected scene has no fitted SWR')
            if Path(self.manifest['video_path']).resolve()!=self.video.resolve():
                raise ValueError('SWR and question videos differ')
            self.fit_path = path(self.manifest['result_asset_id'])
            self.fit = json.loads(self.fit_path.read_text())
            self.paths = dict(swr=self.fit_path.parent)
        self.engine = {'softbody':'soft_body'}.get(self.manifest['engine'],self.manifest['engine'])
        if self.manifest['engine']!=self.selected_scene['engine']:
            raise ValueError('Scene and SWR engine differ')
        tracking_path = path(self.sources['tracking_asset_id'])
        self.tracking = json.loads(tracking_path.read_text())
        if 'tracking_directory' in self.tracking:
            self.paths['tracking'] = (tracking_path.parent/self.tracking['tracking_directory']).resolve()
            if not tracking_only:
                self.paths['reconstruction'] = (tracking_path.parent/self.tracking['reconstruction_manifest']).resolve().parent
        capture = cv2.VideoCapture(str(self.video))
        try:
            if not capture.isOpened():raise ValueError('Original video is unavailable')
            self.width, self.height = int(capture.get(3)), int(capture.get(4))
            self.fps, self.frame_count = float(capture.get(5)), int(capture.get(7))
        finally: capture.release()
        self.end = (self.frame_count-1)/self.fps
        self.rollouts, self.identity_images, self.identity_mask_observations = {}, [], {}
        self.grounded_appearance = None
        self.limitations = self.manifest.get('limitations',[])
        self.cautions, self.review_status = [], 'retained_stage_one'
        self.summary = dict(status='completed',engine=self.engine)
        self.coordinate_system = self.tracking.get('coordinate_frame','') if tracking_only else self.manifest.get('coordinate_frame','')
        self.route = dict(scene_route='3d_world' if self.tracking['route']=='3D+SAM3' else '2d_world',
                          segmentation='pixel' if self.engine=='fluid' else 'sam3')
        self.world_frame = dict(route=self.route,gravity={},physical_boundary={})
        self.pixel = self.engine=='fluid'
        self.objects = {o['object_id']:dict(o,target_id=o['object_id']) for o in self.tracking['objects']}
        self.scene = SimpleNamespace(object_ids=tuple(self.objects),fps=self.fps,frame_count=self.frame_count)
        self.mask_index, self.best_frames, self.visibility, self.termination = {}, {}, {}, {}
        self.background_object_ids = set()
        self.mask_archive = None
        self._pixel_archives = {}
        if not self.pixel:
            segpath = path(self.sources['segmentation_asset_id'])
            seg = json.loads(segpath.read_text())
            self.paths['segmentation'] = segpath.parent
            self.mask_archive_path = segpath.parent/seg['mask_archive']
            self.mask_index_path = segpath.parent/seg['mask_index']
            self.mask_index = json.loads((segpath.parent/seg['mask_index']).read_text())
            self.mask_archive = np.load(segpath.parent/seg['mask_archive'],allow_pickle=False)
            self._mask_keys = {(v['track_id'],v['frame_index']):k for k,v in self.mask_index.items()}
            for name,obj in self.objects.items():
                rows = [(info['frame_index'],int(np.unpackbits(self.mask_archive[k])[:info['num_bits']].sum()))
                        for k,info in self.mask_index.items() if info['track_id']==obj['track_id']]
                nonempty = [(f,area) for f,area in rows if area]
                if nonempty:self.best_frames[name] = max(nonempty,key=lambda row:row[1])[0]
                self.visibility[name] = mask_visibility(self.mask_index,self.mask_archive,obj['track_id'],self.fps,-1)
        else:
            self._load_native_pixels()
        self._rigid, self._contphy = None, None
        if tracking_only:
            self.simulation_ready = False
            return
        if self.engine=='rigid':
            self._rigid = NativeRigid(self,self.paths['tracking'],self.paths['reconstruction'],tools.scene.device)
        else:
            from atw.world_probing.engines.native_deformable import NativeExperiments
            self._contphy = NativeExperiments(self, tools.scene.device)
        self.simulation_ready = True

    def _mask(self, name, frame):
        if self.pixel:
            obj = self.objects[name]
            archive = self._pixel_archives[obj['mask_asset_id']]
            return archive[f'frame_{frame:06d}']==obj['mask_label']
        key = self._mask_keys[(self.objects[name]['track_id'],frame)]
        info = self.mask_index[key]
        return np.unpackbits(self.mask_archive[key])[:info['num_bits']].reshape(info['height'],info['width']).astype(bool)

    def _load_native_pixels(self):
        assets = self.tracking['assets']
        binding = json.loads(self.artifact_path(assets['source_binding_asset_id']).read_text())
        for key in ('masks_asset_id','source_masks_asset_id'):
            self._pixel_archives[assets[key]] = np.load(self.artifact_path(assets[key]),allow_pickle=False)
        self.objects = {o['object_id']:dict(o,target_id=o['object_id'],physical_type='rigid',
            mask_asset_id=assets['masks_asset_id']) for o in self.tracking['objects'] if o['object_kind']!='fluid'}
        for source in binding['sources']:
            if source['first_visible_frame'] is None:continue
            name = source['source_id']
            self.objects[name] = dict(object_id=name,target_id=name,physical_type='fluid',
                mask_asset_id=assets['source_masks_asset_id'],mask_label=source['mask_label'])
        self.source_binding = binding
        for name in self.objects:
            sizes = [(f,int(self._mask(name,f).sum())) for f in range(self.frame_count)]
            nonempty = [(f,n) for f,n in sizes if n]
            if nonempty:self.best_frames[name] = max(nonempty,key=lambda row:row[1])[0]
        self.scene.object_ids = tuple(self.objects)

    def close(self):
        if self.mask_archive is not None:self.mask_archive.close()
        for archive in self._pixel_archives.values():archive.close()
