"""Compact selected-world Cloth pose and rigid-box contact evidence."""
import numpy as np

from atw.polyworld.swr_measurements import SWRMeasurements
from atw.polyworld.evidence_measurements import box_separation
from atw.world_probing.engines.common import TEXT, SECONDS, fields
from atw.world_modeling import sam3_protocol as s

NUMBER=dict(type=['number','null'])
BOOL=dict(type=['boolean','null'])
SCOPE=dict(fit_object_id=dict(type=['string','null'],description='Cloth identity selecting the retained simulated world; null if no shared fit was selected.'),
    interval_seconds=dict(type='array',items=dict(type='number'),minItems=2,maxItems=2,
        description='Inclusive queried interval, in seconds from original video start.'),
    coverage_complete=dict(type='boolean',description='All requested retained frames contain the required measurements.'))
POSE=s.obj(object_id=TEXT,**SCOPE,initial_tilt_degrees=NUMBER,final_tilt_degrees=NUMBER,maximum_tilt_degrees=NUMBER)
CONTACT=s.obj(object_ids=dict(type='array',items=TEXT,minItems=2,maxItems=2),**SCOPE,
    minimum_gap_m=NUMBER,final_gap_m=NUMBER,any_box_contact=BOOL,final_box_contact=BOOL,first_box_contact_seconds=NUMBER)
ASSIGNMENT_NOTE='The fit is selected from observed-window fitted cloth-to-body proximity; independent cloth worlds are not combined. '
POSE_NOTE=('Tilt follows the body axis initially closest to reconstructed up: 0 degrees is upright, '
    '90 degrees is horizontal. Use final tilt for final pose and maximum tilt for toppling. '
    'Zero is a measured upright orientation, not a missing value. ')
CONTACT_NOTE=('Signed separating-axis gap of the simulated rigid boxes, in meters: positive means separation, '
    'zero touching, negative overlap. Any contact tests every retained frame; final contact tests only the '
    'last frame. This is a geometric contact proxy, not a collision impulse log; subframe events are not sampled. '
    'Separated centers alone do not establish non-contact. ')


def definitions(engine):
    query=dict(world_id=TEXT,
        start_seconds=dict(SECONDS,description='Seconds from original video start; default 0, clamped to the first future frame for prediction.'),
        end_seconds=dict(SECONDS,description='Seconds from original video start; defaults to the returned world end. Must lie inside that world.'))
    return [engine.definition('query_cloth_pose',
        'Read initial, final and maximum rigid-body tilt in its automatically selected cloth world. No new simulation.',
        fields(['world_id','object_id'],**query,object_id=TEXT),POSE,engine.cloth),
        engine.definition('query_cloth_contact',
        'Measure any-time and final rigid-box contact plus signed gaps for two bound bodies in their shared cloth world. No new simulation.',
        fields(['world_id','object_ids'],**query,object_ids=dict(type='array',items=TEXT,minItems=2,maxItems=2,uniqueItems=True)),
        CONTACT,engine.cloth_contact)]


def selected(source, ids):
    assignments=source.cloth_fit_assignments()
    rows=[assignments.get(name,{}) for name in ids]
    if any(row.get('status')!='assigned' or not row.get('variant') for row in rows):
        return None,'The retained cloth-to-body measurements do not select one influencing cloth for every target.'
    if len({row['variant'] for row in rows})!=1:
        return None,'The targets belong to different independent cloth worlds; no joint trajectory was exported.'
    return rows[0]['variant'],None


def measure(backend,rid,ids,start,end,contact=False):
    source=SWRMeasurements(backend,rid)
    variant,missing=selected(source,ids)
    scope=dict(fit_object_id=variant,interval_seconds=[start,end],coverage_complete=False)
    if contact:
        data=dict(object_ids=ids,**scope,minimum_gap_m=None,final_gap_m=None,
            any_box_contact=None,final_box_contact=None,first_box_contact_seconds=None)
        note=CONTACT_NOTE
    else:
        data=dict(object_id=ids[0],**scope,initial_tilt_degrees=None,final_tilt_degrees=None,maximum_tilt_degrees=None)
        note=POSE_NOTE
    if missing:return data,missing+' Measurements omitted. Use original-video reasoning.',dict(assignments=source.cloth_fit_assignments())
    frames=np.arange(int(np.ceil(start*backend.fps-1e-8)),int(np.floor(end*backend.fps+1e-8))+1)
    if contact:
        boxes=[source.boxes(name).get(variant,{}) for name in ids]
        values=np.array([box_separation(boxes[0][int(f)],boxes[1][int(f)])
            if all(int(f) in b for b in boxes) else np.nan for f in frames])
    else:
        rows=source.orientations(ids[0]).get(variant,[])
        if not rows:return data,'No rigid orientation samples were exported for the selected cloth world. Use original-video reasoning.',{}
        initial=rows[0].get('axis_up_alignment')
        if initial is None:return data,'The initial rigid axes relative to up were not exported. Use original-video reasoning.',{}
        axis=int(np.argmax(initial));lookup={r['frame_index']:r for r in rows}
        tilt=lambda a:float(np.degrees(np.arccos(np.clip(abs(a[axis]),0,1))))
        data['initial_tilt_degrees']=round(tilt(initial),3)
        values=np.array([tilt(lookup[int(f)]['axis_up_alignment'])
            if int(f) in lookup and 'axis_up_alignment' in lookup[int(f)] else np.nan for f in frames])
    valid=np.isfinite(values);data['coverage_complete']=bool(len(values) and valid.all())
    if valid.any():
        if contact:
            hits=valid & (values<=0)
            data.update(minimum_gap_m=round(float(values[valid].min()),6),
                final_gap_m=round(float(values[-1]),6) if valid[-1] else None,
                any_box_contact=True if hits.any() else False if valid.all() else None,
                final_box_contact=bool(values[-1]<=0) if valid[-1] else None,
                first_box_contact_seconds=round(float(frames[np.flatnonzero(hits)[0]]/backend.fps),3) if hits.any() else None)
        else:
            data.update(final_tilt_degrees=round(float(values[-1]),3) if valid[-1] else None,
                maximum_tilt_degrees=round(float(values[valid].max()),3))
    if not data['coverage_complete']:note+='Some requested frame measurements are missing; do not infer absence from those frames. '
    return data,note+ASSIGNMENT_NOTE,dict(frames=frames.tolist(),values=[float(v) if np.isfinite(v) else None for v in values],
        assignments=source.cloth_fit_assignments())
