"""Retained observation-initialized local-response warm start."""
from atw.world_modeling.softbody_ballistic_utils import replace, np, swr, write, encode, parameter_payload, stage_two_scene, projected, losses, batch_project, make_engine
import time

def fit_local(scene, base, dest, device):
    started=time.time();dest.mkdir(parents=True,exist_ok=False)
    theta=encode(scene,base);layout=swr.make_parameter_layout(scene.object_count)
    reference=make_engine(stage_two_scene(scene),capacity=1,device=device,substeps=base['substeps_per_frame'])
    population=96;windows=[];window_records=[];rejected=[]
    for start in range(6,scene.frame_count-12,12):
        reference.rollout(theta,end_offset=start)
        centers=reference.positions.numpy()[0].copy();velocity=reference.velocities.numpy()[0].copy()
        strain=reference.strains.numpy()[0].copy();strain_velocity=reference.strain_velocities.numpy()[0].copy()
        used=[]
        for i,name in enumerate(scene.object_ids):
            if scene.first_offsets[i]>start:
                centers[i]=scene.initial_centers[i];velocity[i]=theta[2*i:2*i+2];strain[i]=0;strain_velocity[i]=0
                continue
            valid=np.flatnonzero(scene.observed_mask[:start+1,i]>0);frames=valid[valid>=start-5]
            if len(frames)<3 or frames[-1]!=start:continue
            centers[i]=scene.target_centers[start,i]
            velocity[i]=np.polyfit(frames/scene.fps,scene.target_centers[frames,i],1)[0]
            used.append(name)
        stop=start+13;changes={}
        for key in ('target_centers','target_subset_centers','target_covariances','target_deformations','reference_subset_offsets','reference_subset_covariances','shape_observed_mask','shape_contact_recovered_mask','observed_mask'):
            changes[key]=getattr(scene,key)[start:stop].copy()
        changes.update(frames=tuple(range(13)),first_offsets=tuple(max(0,f-start) for f in scene.first_offsets),initial_centers=centers)
        target=replace(scene,**changes)
        # Only refreshed identities are scored in this local window.
        target_mask=target.observed_mask.copy()
        for i,name in enumerate(scene.object_ids):
            if name not in used:target_mask[:,i]=0
        target=replace(target,observed_mask=target_mask)
        if not target_mask.any():continue
        engine=make_engine(stage_two_scene(target),initial_strains=strain,initial_strain_velocities=strain_velocity,capacity=population,device=device,substeps=base['substeps_per_frame'])
        local=theta.copy();local[:2*scene.object_count]=velocity.reshape(-1)
        _,converged=batch_project(engine.scene,engine.rollout(local).detach().cpu().numpy())
        if not converged[0]:
            rejection=dict(start=start,stop=stop,scored_objects=used,reason='baseline output projection did not converge within the 96-iteration batch cap')
            try:
                projected(engine,local)
                rejection['full_projection']='converged; window still excluded by fixed batch eligibility cap'
            except RuntimeError as error:rejection['full_projection']=str(error)
            rejected.append(rejection)
            continue
        windows.append((target,engine,velocity));window_records.append(dict(start=start,stop=stop,scored_objects=used))
    assert windows,(scene.object_ids,rejected)
    groups=[np.array([getattr(layout,f).start+i for f in ('log_frequency_slice','damping_ratio_slice','friction_slice','drag_slice')]) for i in range(scene.object_count)]
    active=np.concatenate(groups);lower,upper=swr.parameter_bounds(layout);_,std=swr.initial_distribution(layout)
    def factual(values):
        total=np.zeros(len(values));valid=np.ones(len(values),bool)
        for target,engine,velocity in windows:
            local=values.copy();local[:,:2*scene.object_count]=velocity.reshape(-1)
            pred,ok=batch_project(engine.scene,engine.rollout(local).detach().cpu().numpy())
            total+=losses(target,pred);valid&=ok
        total/=len(windows);total[~valid]=np.inf
        return total
    initial_loss=float(factual(theta[None])[0]);assert np.isfinite(initial_loss),(scene.object_ids,initial_loss)
    def score(values):
        prior=np.mean(((values[:,active]-theta[active])/std[active])**2,axis=1)
        return factual(values)+.05*initial_loss*prior
    best=theta.copy();best_score=initial_loss;rng=np.random.default_rng(109);trace=[]
    for gen in range(48):
        group=groups[gen%len(groups)];scale=(.2,.06,.02)[gen//16]
        samples=np.tile(best,(population,1));samples[:,group]+=rng.normal(size=(population,len(group)))*std[group]*scale
        samples=np.clip(samples,np.maximum(lower,theta-std),np.minimum(upper,theta+std)).astype(np.float32);samples[0]=best
        scores=score(samples);winner=int(np.argmin(scores))
        if scores[winner]<best_score:best=samples[winner].copy();best_score=float(scores[winner])
        trace.append(dict(generation=gen+1,score=best_score,finite=int(np.isfinite(scores).sum())))
        if (gen+1)%8==0:print('local_response',scene.object_ids,gen+1,best_score,'seconds',time.time()-started,flush=True)
    # Re-evaluate each local window with the scene projection.
    def exact_score(values):
        total=0.
        for target,engine,velocity in windows:
            local=values.copy();local[:2*scene.object_count]=velocity.reshape(-1)
            pred=projected(engine,local)
            total+=float(losses(target,pred[None])[0])
        return total/len(windows)+.05*initial_loss*float(np.mean(((values[active]-theta[active])/std[active])**2))
    prior_score=exact_score(theta)
    try: final_score=exact_score(best)
    except RuntimeError:final_score=float('inf')
    if final_score>prior_score:best=theta.copy();final_score=prior_score;selected='baseline'
    else:selected='refined'
    unchanged=np.ones(layout.size,bool);unchanged[active]=False;assert np.array_equal(best[unchanged],theta[unchanged])
    write(dest/'parameters.json',dict(model='agentic-softbody-local-response-initialization',substeps_per_frame=base['substeps_per_frame'],parameters=parameter_payload(scene,best),selection=dict(name=selected,score=final_score),
        experiment=dict(mode='local_response',population=population,generations=48,seed=109,initial_loss=initial_loss,
            objective='mean 13-frame observation-initialized window position/displacement/shape score + 0.05 relative prior',
            constraints='only contact frequency, damping, friction, drag; gravity/radii/shape/initial velocities frozen to shared baseline',
            note='window initial centers and past-six-frame velocities come from observations; all balls remain jointly simulated; no QA labels or interventions enter fitting')))
    write(dest/'windows.json',window_records);write(dest/'trace.json',trace);write(dest/'rejected_windows.json',rejected)
    print('SELECTED local_response',scene.object_ids,selected,final_score,'baseline',prior_score,flush=True)

    return dict(substeps_per_frame=base['substeps_per_frame'],parameters=parameter_payload(scene,best))
