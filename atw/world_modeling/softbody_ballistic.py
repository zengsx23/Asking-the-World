"""Ballistic soft-body fitting with observation-only warm starts."""
from dataclasses import replace
import time
import numpy as np
from atw.polyworld import soft_body_swr as swr
from atw.world_modeling import softbody_center_initialization as center
from atw.world_modeling.softbody_ballistic_utils import encode, parameter_payload, stage_two_scene, projected, losses, write, make_engine, projection_tolerance, surface_residual
from atw.world_modeling.softbody_local_initialization import fit_local


def ballistic_gravity(scene):
    per_object=[];records=[]
    for i,name in enumerate(scene.object_ids):
        observed=scene.observed_mask[:,i]>0;centers=scene.target_centers[:,i];distances=[]
        for obstacle in scene.obstacles:
            ax=np.asarray(obstacle.axis_x);ay=np.array([-ax[1],ax[0]])
            rel=centers-obstacle.center;local=np.stack((rel@ax,rel@ay),-1)
            d=np.abs(local)-obstacle.half_extents
            distances.append(np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0))
        clearance=np.min(distances,axis=0)-scene.radii[i]
        free=observed&(clearance>.3*scene.radii[i])
        for j in range(scene.object_count):
            if i!=j:free&=(np.linalg.norm(centers-scene.target_centers[:,j],axis=-1)>1.3*(scene.radii[i]+scene.radii[j]))|(scene.observed_mask[:,j]<=0)
        estimates=[]
        for start in range(0,scene.frame_count-8,3):
            idx=np.arange(start,start+9)
            if not free[idx].all():continue
            coef=np.polyfit(idx/scene.fps,centers[idx],2)
            residual=np.sqrt(np.mean((np.stack([np.polyval(coef[:,axis],idx/scene.fps) for axis in (0,1)],-1)-centers[idx])**2))
            g=float(2*coef[0,1]/9.81)
            estimates.append(dict(start_frame=start,gravity_scale=g,residual_reconstruction_units=float(residual)))
        accepted=[r['gravity_scale'] for r in estimates if r['gravity_scale']>0 and r['residual_reconstruction_units']<.005]
        if accepted:per_object.append(float(np.median(accepted)))
        records.append(dict(object_id=name,windows=estimates))
    if not per_object:raise ValueError('No usable nine-frame free-flight gravity evidence')
    return float(np.clip(np.median(per_object),.03,1.5)),records


def raw_windows(scene,base,population,device):
    theta=encode(scene,base);engine=make_engine(stage_two_scene(scene),capacity=population,device=device,substeps=8)
    windows=[]
    for begin in range(6,scene.frame_count-24,24):
        engine.rollout(theta,end_offset=begin)
        position=engine.positions.numpy()[0].copy();velocity=engine.velocities.numpy()[0].copy()
        strain=engine.strains.numpy()[0].copy();strain_velocity=engine.strain_velocities.numpy()[0].copy()
        stop=begin+25;changes={}
        for key in ('target_centers','target_subset_centers','target_covariances','target_deformations','reference_subset_offsets','reference_subset_covariances','shape_observed_mask','shape_contact_recovered_mask','observed_mask'):
            changes[key]=getattr(scene,key)[begin:stop].copy()
        for i in range(scene.object_count):
            idx=np.flatnonzero(scene.observed_mask[:begin+1,i]>0);idx=idx[idx>=begin-5]
            if len(idx)>=3 and idx[-1]==begin:
                position[i]=scene.target_centers[begin,i];velocity[i]=np.polyfit(idx/scene.fps,scene.target_centers[idx,i],1)[0]
            else:changes['observed_mask'][:,i]=0
            if scene.first_offsets[i]>begin:
                position[i]=scene.initial_centers[i];velocity[i]=theta[2*i:2*i+2];strain[i]=0;strain_velocity[i]=0
        changes.update(frames=tuple(range(25)),first_offsets=tuple(max(0,f-begin) for f in scene.first_offsets),initial_centers=position)
        target=replace(scene,**changes)
        if not target.observed_mask.any():continue
        e=make_engine(stage_two_scene(target),initial_strains=strain,initial_strain_velocities=strain_velocity,capacity=population,device=device,substeps=8)
        windows.append((target,e,velocity))
    return engine,windows


def search(scene,base,starts,gravity,device,seed,generations,scales,output):
    population=128;engine,windows=raw_windows(scene,base,population,device)
    layout=swr.make_parameter_layout(scene.object_count)
    def score(values):
        full=losses(scene,engine.rollout(values).detach().cpu().numpy())
        if not windows:return full
        local=np.zeros(len(values))
        for target,e,velocity in windows:
            v=values.copy();v[:,:2*scene.object_count]=velocity.reshape(-1)
            local+=losses(target,e.rollout(v).detach().cpu().numpy())
        return .5*full+.5*local/len(windows)
    low,high=swr.parameter_bounds(layout);high[layout.log_frequency_slice]=np.log(120.)
    _,std=swr.initial_distribution(layout)
    groups=[np.array([2*i,2*i+1]+[getattr(layout,key).start+i for key in ('log_radius_scale_slice','log_frequency_slice','damping_ratio_slice','friction_slice')]) for i in range(scene.object_count)]
    def constrain(values):
        values=np.clip(values,low,high).astype(np.float32)
        values[:,layout.log_gravity_scale_slice]=np.log(gravity);values[:,layout.drag_slice]=0
        return values
    starts=constrain(np.stack([encode(scene,p) for p in starts]));scores=score(starts)
    if not np.isfinite(scores).any():raise ValueError('No finite softbody warm-start score')
    best=starts[np.nanargmin(scores)].copy();best_score=float(np.nanmin(scores));initial=best_score
    trace=[];rng=np.random.default_rng(seed)
    for gen in range(generations):
        group=groups[gen%len(groups)];samples=np.tile(best,(population,1))
        samples[:,group]+=rng.normal(size=(population,len(group)))*std[group]*scales[gen//(generations//3)]
        samples=constrain(samples);samples[0]=best
        if gen%12==0:samples[1:1+len(starts)]=starts
        scores=score(samples);scores[~np.isfinite(scores)]=np.inf;winner=int(np.argmin(scores))
        if scores[winner]<best_score:best=samples[winner].copy();best_score=float(scores[winner])
        trace.append(dict(generation=gen+1,loss=best_score))
        if (gen+1)%24==0:print('ballistic search',seed,gen+1,best_score,flush=True)
    result=dict(substeps_per_frame=8,parameters=parameter_payload(scene,best),optimization=dict(seed=seed,generations=generations,population=population,
        initial_loss=initial,final_loss=best_score,local_windows=len(windows),global_weight=.5,trace=trace))
    write(output,result)
    return result,best


def fit(inputs,context,*,gravity_calibration=None):
    start=time.perf_counter();out=context.output_dir;device=context.scene.device
    scene,source=swr.load_softbody_scene(inputs['tracking_directory'],reconstruction_dir=inputs['reconstruction_directory'])
    expected={o['object_id'] for o in inputs['objects'] if o['physical_type']=='soft_body'}
    if set(scene.object_ids)!=expected:raise ValueError('Softbody tracks do not cover classified identities')
    obstacles={o['object_id'] for o in inputs['objects'] if o['physical_type']=='rigid'}
    if {o.object_id for o in scene.obstacles}!=obstacles:raise ValueError('Static obstacle geometry does not cover classified rigids')
    gravity,evidence=ballistic_gravity(scene) if gravity_calibration is None else gravity_calibration
    write(out/'free_flight_gravity.json',dict(gravity_scale=gravity,effective_acceleration=gravity*9.81,objects=evidence))
    print('softbody original observation-only warm start',flush=True)
    original,_,_=swr.fit_softbody_scene(scene,device=device,substeps=8,projection_tolerance=projection_tolerance(scene),strict_projection=False)
    original=dict(substeps_per_frame=8,parameters=parameter_payload(scene,original['best']))
    write(out/'warm_original.json',original)
    print('softbody shared-center warm start',flush=True)
    shared,_,_=center.fit_softbody_scene(scene,device=device,substeps=8,projection_tolerance=projection_tolerance(scene),strict_projection=False)
    # Expand only the shared gravity parameter into the native per-object layout.
    native_layout=swr.make_parameter_layout(scene.object_count);shared_layout=center.make_parameter_layout(scene.object_count)
    theta=np.zeros(native_layout.size,np.float32)
    for field in native_layout.__dataclass_fields__:
        if field.endswith('_slice'):theta[getattr(native_layout,field)]=shared['best'][getattr(shared_layout,field)]
    base=dict(substeps_per_frame=8,parameters=parameter_payload(scene,theta));write(out/'warm_shared_center.json',base)
    local=fit_local(scene,base,out/'warm_local_response',device)
    bootstrap,_=search(scene,base,[base,original,local],gravity,device,123,144,(.7,.3,.1),out/'bootstrap.json')
    selected,theta=search(scene,base,[base,original,local,bootstrap],gravity,device,7,96,(.4,.15,.05),out/'selected_parameters.json')
    engine=make_engine(stage_two_scene(scene),capacity=1,device=device,substeps=8)
    raw=engine.rollout(theta).detach().cpu().numpy()[0]
    predicted=projected(engine,theta)
    np.savez_compressed(out/'softbody.npz',raw=raw,predicted=predicted,theta=theta)
    metrics=dict(raw=swr.softbody_metrics(scene,raw),projected=swr.softbody_metrics(scene,predicted))
    payload=dict(protocol_version=1,model='agentic-softbody-ballistic-seed7-v1',recipe='ballistic_seed7',source=source,
        object_ids=list(scene.object_ids),prediction_npz='softbody.npz',parameters=selected['parameters'],substeps_per_frame=8,
        gravity_scale=gravity,linear_drag_per_s=0,projection=dict(policy='best_effort',nonconverged_frames=engine.projection_diagnostics,tolerance=projection_tolerance(scene),radius_fraction=.01,max_remaining_penetration=surface_residual(scene,predicted)),optimization=selected['optimization'],metrics=metrics,elapsed_seconds=time.perf_counter()-start)
    write(out/'softbody.json',payload)
    return out/'softbody.json',metrics,['Reduced-order soft spheres and static wall-plane obstacles; not a volumetric material model.',
        'Effective gravity uses reconstruction scale, not calibrated SI acceleration.',
        'Final surface projection is retained separately from raw dynamics used for optimization.']
