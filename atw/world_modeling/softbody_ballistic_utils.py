"""Observation-only utilities for the retained ballistic softbody recipe."""
from dataclasses import replace
import numpy as np
from atw.polyworld import soft_body_swr as swr
from atw.world_modeling.scene_assets import write

FIELDS=[('log_gravity_scale_slice','gravity_scale',True),('log_radius_scale_slice','collision_radius_scale',True),
 ('log_frequency_slice','natural_frequency_rad_per_s',True),('damping_ratio_slice','damping_ratio',False),
 ('friction_slice','contact_friction',False),('deformation_gain_slice','deformation_gain',False),
 ('log_shape_frequency_slice','shape_frequency_rad_per_s',True),('shape_damping_ratio_slice','shape_damping_ratio',False),
 ('drag_slice','linear_drag_per_s',False)]

def encode(scene,fit):
    layout=swr.make_parameter_layout(scene.object_count);theta=np.zeros(layout.size,np.float32)
    for i,name in enumerate(scene.object_ids):
        row=fit['parameters'][name]
        theta[2*i:2*i+2]=row['initial_velocity_plane_m_per_s']
        for field,key,log in FIELDS:
            theta[getattr(layout,field).start+i]=np.log(row[key]) if log else row[key]
    return theta


def parameter_payload(scene,theta):
    layout=swr.make_parameter_layout(scene.object_count);out={}
    for i,name in enumerate(scene.object_ids):
        row=dict(initial_velocity_plane_m_per_s=theta[2*i:2*i+2].tolist())
        for field,key,log in FIELDS:
            value=theta[getattr(layout,field).start+i]
            row[key]=float(np.exp(value) if log else value)
        out[name]=row
    return out


def stage_two_scene(scene):
    return replace(scene,observed_mask=np.zeros_like(scene.observed_mask),target_centers=np.zeros_like(scene.target_centers))


def projected(engine,theta):
    scene=engine.scene
    active=np.arange(scene.frame_count)[:,None]>=np.asarray(scene.first_offsets)[None]
    engine.scene=replace(scene,observed_mask=active.astype(float))
    try: return engine.rollout_numpy(theta)[0]
    finally: engine.scene=scene

def losses(scene,pred,end=None,start=0,events=True):
    """Average per object; displacement at 0.1s and 0.2s avoids frame derivatives."""
    end=scene.frame_count if end is None else end
    mask=(scene.observed_mask>0).copy();mask[:start]=False;mask[end:]=False
    delta=pred[...,:2]-scene.target_centers[None]
    # Squared position error is capped smoothly by a 3 cm Huber transition.
    huber=lambda d: np.where(np.abs(d)<=.03,.5*d*d,.03*(np.abs(d)-.015)).mean(-1)
    center=(huber(delta)*mask[None]).sum(1)/np.maximum(mask.sum(0),1)
    displacement=np.zeros_like(center)
    if events:
        for lag in (3,6):
            valid=mask[lag:]&mask[:-lag]
            residual=(pred[:,lag:,:,:2]-pred[:,:-lag,:,:2])-(scene.target_centers[lag:]-scene.target_centers[:-lag])[None]
            displacement+=(huber(residual)*valid[None]).sum(1)/np.maximum(valid.sum(0),1)/2
    shape_mask=mask & (scene.shape_observed_mask>0)
    shape_error=(scene.radii[None,None,:,None,None]*(swr._numpy_deformation(pred)-scene.target_deformations[None]))**2
    shape=(shape_error.mean((-1,-2))*shape_mask[None]).sum(1)/np.maximum(shape_mask.sum(0),1)
    return (center+displacement+0.05*shape).mean(1)

def batch_project(scene,raw):
    out=np.asarray(raw,dtype=np.float64).copy()
    B,F,N=out.shape[:3];xy=out[...,:2].reshape(B*F,N,2)
    # The slice is not contiguous; keep an explicit work array and write back.
    xy=xy.copy()
    D=swr._numpy_deformation(out).reshape(B*F,N,2,2)
    active=np.tile(np.arange(F)[:,None]>=np.asarray(scene.first_offsets)[None],(B,1))
    normals={}
    for i in range(N):
        for j in range(i+1,N):
            delta=xy[:,i]-xy[:,j];length=np.linalg.norm(delta,axis=-1)
            normals[i,j]=np.where((length>1e-8)[:,None],delta/np.maximum(length[:,None],1e-8),[1.,0.])
    pending=np.arange(B*F);converged=np.zeros(B*F,bool)
    for iteration in range(96):
        x=xy[pending].copy();d=D[pending];a=active[pending]
        maximum=np.zeros(len(pending))
        for i in range(N):
            for obstacle in scene.obstacles:
                ax=np.asarray(obstacle.axis_x);ay=np.array([-ax[1],ax[0]])
                offset=x[:,i]-obstacle.center
                local=np.stack([offset@ax,offset@ay],-1)
                dist=np.abs(local)-obstacle.half_extents
                outside=np.maximum(dist,0);length=np.linalg.norm(outside,axis=-1)
                signs=np.where(local>=0,1.,-1.);choose=dist[:,0]>dist[:,1]
                inner=np.stack([np.where(choose,signs[:,0],0),np.where(choose,0,signs[:,1])],-1)
                n=np.where((length>1e-8)[:,None],np.sign(local)*outside/np.maximum(length[:,None],1e-8),inner)
                n=n[:,:1]*ax+n[:,1:]*ay
                signed=length+np.minimum(dist.max(-1),0)
                support=scene.radii[i]*np.linalg.norm(np.einsum('kji,kj->ki',d[:,i],n),axis=-1)
                penetration=np.maximum(support-signed,0)*a[:,i]
                x[:,i]+=penetration[:,None]*n;maximum=np.maximum(maximum,penetration)
        for i in range(N):
            for j in range(i+1,N):
                delta=x[:,i]-x[:,j];length=np.linalg.norm(delta,axis=-1)
                n0=normals[i,j][pending]
                n=np.where((length>1e-8)[:,None],delta/np.maximum(length[:,None],1e-8),n0)
                support=sum(scene.radii[k]*np.linalg.norm(np.einsum('kji,kj->ki',d[:,k],n),axis=-1) for k in (i,j))
                penetration=np.maximum(support-length,0)*(a[:,i]&a[:,j])
                correction=.5*penetration[:,None]*n0
                x[:,i]+=correction;x[:,j]-=correction;maximum=np.maximum(maximum,penetration)
        xy[pending]=x
        done=maximum<=1e-9;converged[pending[done]]=True;pending=pending[~done]
        if not len(pending):break
    out[...,:2]=xy.reshape(B,F,N,2)
    return out,converged.reshape(B,F).all(1)


def projection_tolerance(scene):
    """Accept residual corrections up to one percent of the smallest ball radius."""
    return .01 * float(np.min(scene.radii))


def make_engine(scene, **kwargs):
    return swr.BatchWarpSoftBodyEngine(scene, projection_tolerance=projection_tolerance(scene), strict_projection=False, **kwargs)


def surface_residual(scene, predicted):
    """Measure remaining static and pair penetration after projection."""
    deformation=swr._numpy_deformation(predicted);maximum=0.0
    for frame in range(scene.frame_count):
        active=[i for i in range(scene.object_count) if frame>=scene.first_offsets[i]]
        for i in active:
            for obstacle in scene.obstacles:
                distance,normal=swr._obb_signed_distance_and_normal(predicted[frame,i,:2],obstacle)
                support=scene.radii[i]*np.linalg.norm(deformation[frame,i].T@normal)
                maximum=max(maximum,float(support-distance))
            for j in active:
                if j>=i:continue
                delta=predicted[frame,i,:2]-predicted[frame,j,:2];distance=float(np.linalg.norm(delta))
                normal=delta/max(distance,1e-12) if distance>1e-12 else np.array([1.,0.])
                support=sum(scene.radii[k]*np.linalg.norm(deformation[frame,k].T@normal) for k in (i,j))
                maximum=max(maximum,float(support-distance))
    return maximum
