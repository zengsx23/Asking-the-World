"""Rigid prediction tools over retained observations, without loading SWR."""
from pathlib import Path

from atw.world_probing.engines.common import EngineTools, fields, TEXT, SECONDS
from atw.world_probing.engines.native import NativeEvidence
from atw.world_probing.engines.tracking_readout import TrackingReadout
from atw.world_modeling import sam3_protocol as s

NUMBER = dict(type='number')
COUNT = dict(type='integer', minimum=0)
VECTOR = dict(type='array', items=NUMBER, minItems=3, maxItems=3)
XY = dict(type='array', items=NUMBER, minItems=2, maxItems=2)
INTERVAL = dict(type='array', items=SECONDS, minItems=2, maxItems=2)


def nullable(schema):
    return dict(schema, type=[schema['type'], 'null'])


OBSERVATION_NOTE = ('These are retained observations within the original video, not simulated future events. '
    'Infer future behavior from recent motion and geometry; missing coverage is not rest or absence. '
    'World distances use the retained tracking scale, with x right, y down, z forward; '
    'mask distances are pixels, x right and y down. Times are seconds from video start. '
    'Recent motion uses the last approximately 0.5 seconds of available track samples, which may end before the video cutoff.')


class NativeTracking(TrackingReadout, NativeEvidence):
    """Observed tracking readouts with explicit asset loading."""


class TrackingTools(EngineTools):
    prompt_name = 'stage_two_predictive_tracking.txt'

    def context(self):
        return dict(engine='rigid', evidence_mode='tracking_only',
                    observed_end_seconds=self.backend.end, object_ids=list(self.backend.objects))

    def definitions(self):
        time_args = dict(
            start_seconds=dict(SECONDS, description='Observed interval start in seconds from video start; default 0.'),
            end_seconds=dict(SECONDS, description='Observed interval end in seconds; default video cutoff, never future.'))
        base = dict(availability=dict(type='string', enum=['observed', 'unavailable']),
                    interval_seconds=INTERVAL)
        trajectory = s.obj(**base, object_id=TEXT,
            coverage=nullable(s.obj(first_time_s=SECONDS, last_time_s=SECONDS, observed_frame_count=COUNT)),
            net_displacement_world=nullable(VECTOR), net_distance_world=nullable(NUMBER),
            path_length_world=nullable(NUMBER), recent_velocity_world_per_s=nullable(VECTOR),
            keyframes=s.array(s.obj(frame_index=COUNT, time_s=SECONDS, center_world=VECTOR)))
        relation = s.obj(**base, object_ids=dict(type='array', items=TEXT, minItems=2, maxItems=2, uniqueItems=True),
            tracking_relation=nullable(s.obj(common_frame_count=COUNT, first_time_s=SECONDS, last_time_s=SECONDS,
                start_distance_world=NUMBER, end_distance_world=NUMBER, minimum_distance_world=NUMBER,
                minimum_time_s=SECONDS, recent_distance_change_world=nullable(NUMBER))),
            mask_relation=nullable(s.obj(sampled_frame_count=COUNT, minimum_gap_px=NUMBER,
                minimum_time_s=SECONDS, contact_candidate_times_s=s.array(SECONDS))))
        for key in ('minimum_distance_world', 'minimum_time_s'):
            relation['properties']['tracking_relation']['properties'].pop(key)
            relation['properties']['tracking_relation']['required'].remove(key)
        relation['properties']['mask_relation'] = nullable(s.obj(
            last_gap_px=NUMBER, last_time_s=SECONDS))
        motion = s.obj(**base, object_id=TEXT,
            centroid_displacement_px=nullable(XY),
            visible_area_px=nullable(s.obj(first=COUNT, last=COUNT, minimum=COUNT, maximum=COUNT)),
            keyframes=s.array(s.obj(frame_index=COUNT, time_s=SECONDS, centroid_px=XY, visible_pixels=COUNT)))
        mask = s.obj(object_id=TEXT, frame_index=nullable(COUNT), mask_available=dict(type='boolean'),
            mask_pixels=nullable(COUNT), bbox_xyxy=nullable(dict(type='array', items=COUNT, minItems=4, maxItems=4)),
            mask_wh=nullable(dict(type='array', items=COUNT, minItems=2, maxItems=2)))
        return [
            self.definition('query_trajectory', 'Read observed 3D positions, displacement and recent velocity to infer events strictly after the video cutoff. These are past observations, not simulated future trajectories; check the last available sample time before extrapolating.',
                fields(['object_id'], object_id=TEXT, **time_args,
                    sample_count=dict(type='integer', minimum=3, maximum=8, default=5,
                                      description='Number of evenly spaced observed keyframes.')),
                trajectory, self.trajectory),
            self.definition('query_observed_relation', 'Read one observed pair: distance change, recent approach or separation, and the latest available mask gap with its time. Historical minimum distances and past contact candidates are not returned. Infer events strictly after the video cutoff; proximity alone is not a future collision verdict.',
                fields(['object_ids'], object_ids=relation['properties']['object_ids'], **time_args,
                    sample_count=dict(type='integer', minimum=3, maximum=12, default=8,
                                      description='Number of observed mask samples searched for the latest available pair gap. Tracking summaries use common observed frames; recent distance change describes motion near their last sample, which can precede the video cutoff.')),
                relation, self.relation),
            self.definition('query_mask_motion', 'Read observed mask centroid displacement and visible area changes; these are not material-point velocity.',
                fields(['object_id'], object_id=TEXT, **time_args), motion, self.mask_motion),
            self.definition('query_mask_assets', 'Read compact mask geometry for supplied identities at one observed frame or each clearest retained frame.',
                fields(['object_ids'], object_ids=dict(type='array', items=TEXT, minItems=1, maxItems=6, uniqueItems=True),
                    frame_index=dict(COUNT, description='Optional zero-based original frame; default maximum-area frame per object.')),
                s.obj(objects=s.array(mask)), self.masks),
            self.definition('inspect_frames', 'View exact original frames, optionally highlighting one identity. Does not simulate or infer future frames.',
                fields(['frame_indices'], frame_indices=dict(type='array', items=COUNT, minItems=1, maxItems=6, uniqueItems=True,
                    description='Zero-based original video frames, all before the video cutoff.'), object_id=TEXT),
                s.obj(frame_indices=s.array(COUNT), highlighted_frames=s.array(COUNT), unavailable_mask_frames=s.array(COUNT)),
                self.frames),
        ]

    def observed(self, context, evidence, raw, note='', images=()):
        retained = dict(raw, source_tracking_asset_id=self.backend.sources['tracking_asset_id'],
                        source_segmentation_asset_id=self.backend.sources['segmentation_asset_id'])
        from atw.world_probing.prediction_scope import PREDICTION_GUIDANCE
        note = PREDICTION_GUIDANCE + ' ' + note
        data, artifacts = self.emit(context, evidence, OBSERVATION_NOTE + (' ' + note if note else ''),
                                    raw=retained, images=[Path(path) for path in images])
        for item in artifacts:
            if item['media_type'].startswith('image/'):
                item['source_kind'] = 'observation'
                item['media_type'] = 'image/png'
        return data, artifacts

    def trajectory(self, args, context):
        raw = self.backend.query_trajectory(**args)
        motion = raw.get('motion', {})
        evidence = dict(availability=raw['status'], object_id=raw['object_id'], interval_seconds=raw['interval_seconds'],
            coverage=raw.get('coverage'), keyframes=raw.get('keyframes', []),
            **{key:motion.get(key) for key in ('net_displacement_world','net_distance_world',
                                             'path_length_world','recent_velocity_world_per_s')})
        return self.observed(context, evidence, raw, raw.get('reason', ''))

    def relation(self, args, context):
        raw = self.backend.query_observed_relation(**args)
        evidence = dict(availability=raw['status'], object_ids=raw['object_ids'], interval_seconds=raw['interval_seconds'],
                        tracking_relation=raw.get('tracking_relation'), mask_relation=raw.get('mask_relation'))
        if evidence['tracking_relation'] is not None:
            evidence['tracking_relation'] = {k:v for k,v in evidence['tracking_relation'].items()
                if k not in {'minimum_distance_world', 'minimum_time_s'}}
        # Only the most recent available pair geometry is exposed, never a
        # historical minimum or a list of past contact candidates.
        evidence['mask_relation'] = None
        start, end = raw['interval_seconds']
        for frame in reversed(self.backend._frames(start, end, args.get('sample_count', 8))):
            try:
                first, second = [self.backend._mask(oid, int(frame)) for oid in args['object_ids']]
            except (KeyError, OSError, ValueError):
                continue
            if first.any() and second.any():
                evidence['mask_relation'] = dict(last_gap_px=round(self.backend._mask_gap(first, second), 2),
                    last_time_s=round(float(frame / self.backend.fps), 3))
                break
        note = 'The last mask gap is observed geometry, not a future collision verdict. '
        return self.observed(context, evidence, raw, note + raw.get('reason', ''))

    def mask_motion(self, args, context):
        raw = self.backend.query_mask_motion(**args)
        evidence = dict(availability=raw['status'], object_id=raw['object_id'], interval_seconds=raw['interval_seconds'],
                        keyframes=raw.get('keyframes', []),
                        **{k:raw.get('motion', {}).get(k) for k in ('centroid_displacement_px','visible_area_px')})
        return self.observed(context, evidence, raw, 'Occlusion, clipping and perspective affect mask centroid and area. '
                             + raw.get('reason', ''))

    def masks(self, args, context):
        raw = self.backend.query_mask_assets(**args)
        evidence = dict(objects=[{k:row.get(k) for k in ('object_id','frame_index','mask_available',
            'mask_pixels','bbox_xyxy','mask_wh')} for row in raw['objects']])
        return self.observed(context, evidence, raw, 'Bounding boxes use exclusive upper bounds.')

    def frames(self, args, context):
        raw = self.backend.inspect_frames(**args)
        highlighting = raw.get('highlighting', {})
        evidence = dict(frame_indices=args['frame_indices'], highlighted_frames=highlighting.get('highlighted_frames', []),
                        unavailable_mask_frames=highlighting.get('unavailable_frames', []))
        return self.observed(context, evidence, raw, images=raw['images'])
