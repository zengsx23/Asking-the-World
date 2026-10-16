"""Retained route-fitting recipe on explicit Agentic pixel/source observations."""
from dataclasses import asdict
import cv2
import numpy as np
import torch
from scipy.ndimage import binary_closing
from skimage.morphology import skeletonize
from atw.polyworld.fluid_swr import FluidScene, FluidEmitter
from atw.polyworld.fluid_pbf import BatchPBF, PBFConfig, fit_pbf, parameter_bounds, coverage_metrics
from atw.polyworld.fluid_distribution_loss import FluidDistributionLoss
from atw.world_modeling.scene_assets import write


class RouteLoss(FluidDistributionLoss):
    def __init__(self, *args, regions, region_names, **kwargs):
        super().__init__(*args, **kwargs)
        s = self.scene; device = self.target.device
        self.region_names = region_names
        self.regions = torch.as_tensor(np.asarray(regions, dtype=bool).reshape(-1,s.image_height,s.image_width), device=device)
        distances=[];samples=[];fractions=[]
        for phase in range(len(s.fluid_ids)):
            ds=[];ps=[];fs=[]
            for frame in self.frames:
                mask=s.target_masks[frame,phase]
                ds.append(cv2.distanceTransform((~mask).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE))
                y,x=np.where(skeletonize(mask));xy=np.column_stack((x+.5,y+.5))
                ps.append(xy[np.linspace(0,len(xy)-1,32).astype(int)] if len(xy) else np.zeros((32,2)))
                fs.append([(mask&r).sum()/max(mask.sum(),1) for r in regions])
            distances.append(ds);samples.append(ps);fractions.append(fs)
        self.path_dist=torch.as_tensor(np.asarray(distances),device=device)
        self.path_points=torch.as_tensor(np.asarray(samples),dtype=torch.float32,device=device)
        self.region_fraction=torch.as_tensor(np.asarray(fractions),dtype=torch.float32,device=device)

    def __call__(self,positions,groups,births,particle_area,substeps,*,diagnostics=True):
        total,details=super().__call__(positions,groups,births,particle_area,substeps,diagnostics=diagnostics)
        length=positions.shape[1];path=total.new_zeros(total.shape);region=path.clone();s=self.scene
        for phase in range(len(s.fluid_ids)):
            xy=positions[:,:,groups==phase];valid=torch.isfinite(xy).all(-1)&(xy>=0).all(-1)
            safe=torch.nan_to_num(xy);x=safe[...,0].long().clamp(0,s.image_width-1);y=safe[...,1].long().clamp(0,s.image_height-1)
            t=torch.arange(length,device=positions.device)[None,:,None]
            valid&=self.visible[t,y,x]
            observed=self.present[:length,phase][None]
            distance=self.path_dist[phase,t,y,x]
            forward=(distance.clamp(max=20).square()*valid).sum(-1)/valid.sum(-1).clamp_min(1)
            target=self.path_points[phase,:length][None].expand(xy.shape[0],-1,-1,-1)
            pair=torch.cdist(target,safe).masked_fill(~valid[:,:,None,:],20)
            backward=pair.min(-1).values.clamp(max=20).square().mean(-1)
            path+=((forward+backward)/50*observed).sum(-1)/observed.sum().clamp_min(1)
            if len(self.regions):
                inside=self.regions[:,y,x].permute(1,2,3,0)&valid[:,:,:,None]
                frac=inside.sum(-2)/valid.sum(-1).clamp_min(1)[:,:,None]
                err=(frac-self.region_fraction[phase,:length][None]).abs().sum(-1)
                region+=(err*observed).sum(-1)/observed.sum().clamp_min(1)
        path/=len(s.fluid_ids);region/=len(s.fluid_ids)
        return total+.15*path+.20*region, details|dict(path_distance=path,regional_occupancy=region)

    def metadata(self):
        return super().metadata()|dict(path_weight=.15,region_weight=.20,region_names=self.region_names,
            path_description='Symmetric clipped particle/mask distance and 32 skeleton samples; observation penalties, not trajectory forces.')


def route_bounds(engine):
    low,high,mean=parameter_bounds(engine)
    high[1]=3000;high[3:3+engine.fluids]=160
    return low,high,mean


def prepare(inputs, output):
    m=inputs['tracking'];binding=inputs['source_binding'];paths=inputs['pixel_paths']
    size=(int(binding['coordinates']['width']),int(binding['coordinates']['height']))
    resize=lambda a:cv2.resize(a.astype(np.uint16),size,interpolation=cv2.INTER_NEAREST)
    with np.load(paths['static_masks_asset_id']) as z:
        static=resize(z['labels']);domain=resize(z['domain'])>0
    with np.load(paths['source_masks_asset_id']) as z:
        masks=np.stack([z[f'frame_{f:06d}'] for f in range(m['frame_count'])])
    if masks.shape[1:]!=size[::-1]:raise ValueError('Source-mask coordinates differ from source binding')
    emitter_static={s['static_object_id'] for s in binding['sources'] if s['static_object_id'] is not None}
    emitter_labels=[o['mask_label'] for o in m['objects'] if o['object_id'] in emitter_static]
    obstacle=(static!=0)|~domain
    collision=(obstacle&~np.isin(static,emitter_labels))|~domain
    sources=[];targets=[];emitters=[];ids=[]
    for s in binding['sources']:
        if s['fluid_object_id'] is None:continue
        mask=(masks==s['mask_label'])&~obstacle[None]
        active=binary_closing(np.asarray(s['outlet_pixel_counts'])>0,structure=np.ones(3),border_value=0)
        active=tuple(int(x) for x in np.flatnonzero(active))
        if not active or not mask.any():continue
        x0,y0,x1,y1=s['outlet_box']
        xy=((x0+x1-1)/2,float(y0)+1.)
        sources.append(s);targets.append(mask);ids.append(s['source_id'])
        emitters.append(FluidEmitter(s['source_id'],s['source_id'],xy,active))
    if not emitters:raise ValueError('No observed active source has usable liquid masks')
    regions=[];names=[];basins={}
    for o in m['objects']:
        if o['object_kind']!='static':continue
        mask=static==o['mask_label'];label=(o.get('name','')+' '+o.get('description','')).lower()
        if not mask.any():continue
        if any(w in label for w in ('container','funnel','basin','bucket','cup','bowl')):
            yy,xx=np.where(mask);bounds=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
            basins[o['object_id']]=bounds;reg=np.zeros_like(mask);reg[bounds[1]:bounds[3],bounds[0]:bounds[2]]=True
        elif any(w in label for w in ('stick','rod')):
            reg=cv2.dilate(mask.astype('uint8'),np.ones((7,7),np.uint8))>0
        else:continue
        regions.append(reg&~obstacle);names.append(o['object_id'])
    scene=FluidScene(output,m['video_path'],m['fps'],*size,tuple(ids),np.stack(targets,axis=1),obstacle,tuple(emitters),
        np.empty((0,2),np.float32),np.empty(0,np.int32),np.empty(0,np.int32),1,
        particle_emitter=np.empty(0,np.int32),collision_obstacle_mask=collision,basin_bounds=basins)
    write(output/'observations.json',dict(video_path=m['video_path'],frame_count=m['frame_count'],fps=m['fps'],image_wh=list(size),
        fluid_ids=ids,source_records=sources,emitters=[asdict(e) for e in emitters],region_ids=names,
        inputs={k:str(v) for k,v in paths.items()},coordinate_frame='Source-binding pixel grid; x right, y down',
        material_identity='One fitted phase per observed source; color group preserved in source_records.'))
    np.savez_compressed(output/'observations.npz',target_masks=scene.target_masks,obstacle_mask=obstacle,collision_obstacle_mask=collision,static_labels=static)
    return scene,regions,names


def fit(inputs, context):
    output=context.output_dir;scene,regions,names=prepare(inputs,output)
    class RoutePBF(BatchPBF):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            c=self.config
            self.distribution_loss=RouteLoss(self.scene,self.frames,self.device,c.swd_directions,c.swd_quantiles,
                c.stable_volume_weight,c.settled_distribution_weight,c.loss_mode,regions=regions,region_names=names)
    config=PBFConfig(seed=7,population=24,restarts=2,generations=10,substeps=4,iterations=4,settled_distribution_weight=0)
    result=fit_pbf(scene,config,context.scene.device,engine_factory=RoutePBF,bounds_fn=route_bounds)
    fitted=result['scene'];values=result['parameters'];f=len(scene.fluid_ids);n=len(scene.emitters)
    np.savez_compressed(output/'fluid.npz',predicted_xy=result['predicted'],particle_birth=fitted.particle_birth,
        particle_fluid=fitted.particle_fluid,particle_emitter=fitted.particle_emitter,birth_substeps=result['birth_substeps'],
        birth_age_seconds=result['birth_age_seconds'],density_ratio=result['densities'])
    metrics=coverage_metrics(scene,result['predicted'],fitted.particle_fluid,area=result['particle_area'],h=result['smoothing_length'])
    metrics.update(mean_chamfer_pixels=result['mean_chamfer_pixels'],mean_objective=result['mean_loss'])
    payload=dict(protocol_version=1,model='agentic-fluid-route-v1',recipe='route',observations='observations.json',prediction_npz='fluid.npz',
        parameter_vector=values.tolist(),parameter_layout='gravity-cohesion-drag-viscosity-density-flux-width',fluid_ids=list(scene.fluid_ids),
        parameters=dict(gravity_pixels_per_s2=float(values[0]),cohesion_acceleration_pixels_per_s2=float(values[1]),boundary_drag_per_s=float(values[2]),
            viscosity_per_s=dict(zip(scene.fluid_ids,values[3:3+f].tolist())),relative_density=dict(zip(scene.fluid_ids,values[3+f:3+2*f].tolist())),
            emitter_flux_pixels2_per_s=values[-2*n:-n].tolist(),emitter_width_pixels=values[-n:].tolist()),
        physical_support_pixels=result['smoothing_length'],render_support_pixels=result['smoothing_length'],particle_radius_pixels=config.spacing*.5,lattice_rest_density=result['lattice_rest_density'],
        optimization=dict(config=asdict(config),loss=result['loss_details'],trace=result['trace'],seconds=result['elapsed_seconds']),metrics=metrics,
        predictive_material_rule='For a later predictive source, copy fitted viscosity and relative density from the earlier same-color source; preserve emitter parameters. Stage-two replay is not part of this fit.')
    write(output/'fluid.json',payload)
    return output/'fluid.json',metrics,['2D effective material coefficients, not SI material measurements.',
        'One phase per observed source; provenance and static geometry errors affect fitting.',
        'Observed baseline only; stage-two interventions and programmatic QA are not executed by this Engineer.']
