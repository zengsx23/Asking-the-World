"""Locate observed-window cloth contact in retained 3D SWR geometry."""
import numpy as np


def contact_window(fit, object_id, last_frame):
    from atw.polyworld.cloth_swr import DEFAULT_THICKNESS_M
    cloth=fit.get('objects',{}).get(object_id,{})
    points=np.asarray(cloth.get('predicted_positions_world',[]),float)
    bodies=cloth.get('predicted_rigid_trajectories',{})
    frames=np.asarray(cloth.get('frames',[]),int)
    centers=np.asarray(bodies.get('centers_world',[]),float)
    axes=[np.asarray(bodies.get(f'axis_{axis}_world',[]),float) for axis in 'xyz']
    ids=bodies.get('object_ids',[])
    result=dict(source_kind='simulation',fit_object_id=object_id,
        contact_shell_m=DEFAULT_THICKNESS_M,first_contact_frame=None,
        contact_object_ids=[],samples=[],reason='')
    if points.ndim!=3 or centers.ndim!=3 or any(a.ndim!=3 for a in axes):
        result['reason']='The fitted cloth nodes or rigid poses were not exported.'
        return result
    for k,frame in enumerate(frames):
        if frame>last_frame:continue
        if any(k>=len(a) for a in (points,centers,*axes)):continue
        hits=[]
        for j,name in enumerate(ids):
            half=np.asarray(cloth.get('obstacle_half_extents_m',{}).get(name),float)
            if half.shape!=(3,) or any(j>=a.shape[1] for a in (centers,*axes)):continue
            rotation=np.column_stack([a[k,j] for a in axes])
            local=(points[k]-centers[k,j])@rotation
            # Same axis-expanded box used by the solver's particle projection.
            # One micrometer only absorbs stored float32 projection roundoff.
            gap=np.max(np.abs(local)-half-DEFAULT_THICKNESS_M,axis=1)
            gap=gap[np.isfinite(gap)]
            if not len(gap):continue
            minimum=float(gap.min())
            result['samples'].append(dict(frame=int(frame),object_id=name,shell_gap_m=minimum))
            if minimum<=1e-6:hits.append(name)
        if hits and result['first_contact_frame'] is None:
            result.update(first_contact_frame=int(frame),contact_object_ids=hits)
    if result['first_contact_frame'] is None:
        result['reason']='No retained observed-window cloth node reaches a rigid contact shell.'
    return result
