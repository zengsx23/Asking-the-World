"""Engine-owned observation tools and numerical fixed-world queries."""
from pathlib import Path
import math

import cv2
import numpy as np

from atw.world_probing.engines.common import EngineTools, fields, TEXT, IDS, SECONDS
from atw.world_probing.engines.measurements import fluid_static, pit_crossings
from atw.world_probing.evidence import write
from atw.world_modeling import sam3_protocol as s

NUMBER=dict(type=['number','null'])
BOOL=dict(type=['boolean','null'])
NUMBERS=dict(type='array',items=dict(type='number'))
INTERVAL=dict(type='array',items=dict(type='number'),minItems=2,maxItems=2)
FACT=s.obj(object_id=TEXT,quantity=TEXT,value=NUMBER,units=TEXT,status=TEXT)
FACTS=s.obj(facts=s.array(FACT),details=s.array(TEXT),interval_seconds=INTERVAL)
PAIR=dict(type='array',items=TEXT,minItems=2,maxItems=2,uniqueItems=True)
WORLD_RECEIPT=s.obj(object_ids=IDS,interval_seconds=INTERVAL,assumptions=s.array(TEXT))
MOTION_SAMPLE=fields(['object_id','start_seconds','time_seconds'],object_id=TEXT,
    start_seconds=SECONDS,time_seconds=SECONDS,rope_id=TEXT,
    net_right_displacement_pixels=NUMBER,net_down_displacement_pixels=NUMBER,net_rotation_radians=NUMBER)
ROPE_MOTION=s.obj(end_to_end=s.array(MOTION_SAMPLE),intermediate_samples=s.array(MOTION_SAMPLE),
    interval_seconds=INTERVAL)
ROPE_MOTION_NOTE = (
    'Movement means center translation only; spin around a fixed center is not translation. '
    'Image x right, y down; angles use a clockwise-positive sign convention. '
    'Each value is the net change from start_seconds to time_seconds. Intermediate samples use up to three '
    'evenly spaced retained times. Rotation is inferred from taut no-slip rope contact. '
    'Null means unavailable measurement.')


class DeformableTools(EngineTools):
    def context(self):
        value=super().context()
        if self.backend.engine=='fluid':
            from atw.world_probing.engines.fluid_readout import LIMITATIONS
            value['limitations']=LIMITATIONS
        if self.backend.engine=='soft_body':
            from atw.polyworld.swr_queries import scene_regions
            self.regions=scene_regions(self.backend)
            value['regions']=[dict(region_id=r['region_id'],horizontal_order=r['horizontal_order'],
                boundary_ids=r['boundary_ids']) for r in self.regions]
        return value

    def definitions(self):
        kind=self.backend.engine
        if self.mode=='property':
            if kind in {'cloth','soft_body'}:
                names=['bendability','elasticity'] if kind=='cloth' else ['elasticity','plasticity']
                return [self.definition('inspect_material_response',
                    ('Read original full-clip mask-guided timelines for two bound objects to compare deformation and recovery.'
                     if kind=='soft_body' else 'Read original mask-guided panels before/around/after interaction and at the end for two cloths. Select elasticity for stretch response, bendability for bending or wrinkles.'),
                    s.obj(object_ids=PAIR,property_name=dict(type='string',enum=names)),
                    s.obj(status=TEXT,object_ids=IDS),self.material)]
            return [self.definition('query_physical_property',
                ('Compare fitted relative density after sustained mutual contact; otherwise report that density cannot be compared.'
                 if kind=='fluid' else 'Read identifiable relative mass; rope tension returns a focused original mask video.'),
                s.obj(object_ids=PAIR,property_name=dict(type='string',enum=['density'] if kind=='fluid' else ['mass','tension'])),
                FACTS,self.property)]
        definitions=[]
        if kind=='rope':
            from atw.world_probing.engines.rope_controls import definition
            definitions.append(definition(self))
            definitions.append(self.definition('query_rope_motion',
                'Read endpoint and intermediate net translation/rotation for selected objects and times in an existing world. No new simulation.',
                fields(['world_id','object_ids'],world_id=TEXT,object_ids=dict(IDS,minItems=1),
                    measurement=dict(type='string',enum=['translation','rotation','both'],default='both',
                        description='Default both: translation for each target and rotation for modeled pulleys. '
                        'rotation requires pulley targets; translation reads centers only.'),
                    start_seconds=dict(SECONDS,description='Seconds from video start, default 0; clamped after observation for prediction and after the intervention readout start for edited worlds.'),
                    end_seconds=dict(SECONDS,description='Seconds from video start; defaults to the existing world end. Must lie inside that world.')),
                ROPE_MOTION,self.rope_motion))
        if kind=='rope' and self.mode in {'counterfactual','goal'}:
            props=dict(object_id=TEXT,direction=dict(type='string',enum=['increase','decrease']))
            required=['object_id','direction']
            if self.mode=='goal':props['option_id']=TEXT;required.append('option_id')
            definitions.append(self.definition('change_mass','Increase or decrease the exact bound body mass by a finite, system-dependent amount; return end-to-end and intermediate net motion. No refitting. Reuse the returned world_id with query_rope_motion for focused readouts.',
                fields(required,**props),ROPE_MOTION,self.mass))
            definitions.append(self.definition('drive_rope_endpoint','Prescribe up/down movement of a bound endpoint. Only for explicit pulling/moving interventions, never heavier/lighter. Reuse the returned world_id with query_rope_motion for focused readouts.',
                fields(required,**dict(props,direction=dict(type='string',enum=['up','down']))),ROPE_MOTION,self.drive))
            return definitions
        props=dict(extension_seconds=dict(type='number',minimum=0,maximum=10,
            description='Seconds beyond the observed clip; fixed fitted parameters.'))
        required=['extension_seconds']
        if self.mode!='predictive':props['remove_object_ids']=IDS
        if kind=='fluid':
            props['emission_fluid_id']=dict(type='string',description='Optional bound E source to emit for 0.5 seconds beyond cutoff, only when the question explicitly asks about further release. Omit to preserve observed births.')
        if self.mode=='goal':props['option_id']=TEXT;required.append('option_id')
        definitions.append(self.definition('simulate_world','Create the requested fixed-parameter world. Query its numerical evidence before answering.',
            fields(required,**props),WORLD_RECEIPT,self.simulate))
        query=dict(world_id=TEXT,start_seconds=SECONDS,end_seconds=SECONDS)
        if kind=='fluid':
            from atw.world_probing.engines.fluid_readout import EVIDENCE, MEASUREMENT
            definitions.append(self.definition('query_fluid_destination',
                'Read whether bound E-liquid enters a container or passes a stick in an existing simulated world. Returns presence and particle fraction; no new simulation.',
                fields(['world_id','object_id','target_id','measurement'],**query,object_id=TEXT,target_id=TEXT,
                    measurement=MEASUREMENT),EVIDENCE,self.fluid))
        elif kind=='soft_body':
            definitions.append(self.definition('query_softbody_outcome','Measure contact events or entry into a selected pit, including rolling into its mouth followed by whole-body descent. Leaving the image does not identify a pit.',
                fields(['world_id','object_id'],**query,object_id=TEXT,region_id=TEXT),
                s.obj(object_id=TEXT,interval_seconds=INTERVAL,contacts=s.array(s.obj(object_id=TEXT,time_seconds=dict(type='number'))),
                    region_id=dict(type=['string','null']),downward_entry_times_seconds=dict(NUMBERS,
                        description='Entry times into this region within the queried interval, from downward center crossing or whole-body descent through its mouth after rolling entry.'),
                    fully_below_image=dict(BOOL,description='Whether the entire body is below the image at interval end; independent of entry into the queried pit.'),
                    coverage_complete=dict(type='boolean',description='Whether finite retained samples cover consecutive frames in the queried interval.')),self.softbody))
        elif kind=='cloth':
            from atw.world_probing.engines.cloth_readout import definitions as cloth_definitions
            definitions.extend(cloth_definitions(self))
        return definitions

    def bound(self,ids):
        if set(ids)-set(self.bound_ids):raise ValueError('Targets must be bound question identities')

    def material(self,args,context):
        self.bound(args['object_ids'])
        if self.backend.engine=='cloth':
            from atw.world_probing.engines.cloth_material import inspect
            raw,note=inspect(self.backend,args['object_ids'],args['property_name'],context.output_dir/'observations')
            return self.emit(context,dict(status=raw['status'],object_ids=args['object_ids']),note,
                raw=raw,images=[Path(p) for p in raw['images']])
        if self.backend.engine=='soft_body':
            from atw.world_probing.engines.material_timeline import inspect, MATERIAL_NOTE
            raw=inspect(self.backend,args['object_ids'],args['property_name'],context.output_dir/'observations')
            return self.emit(context,dict(status=raw['status'],object_ids=args['object_ids']),MATERIAL_NOTE,
                raw=raw,images=[Path(p) for p in raw['images']])

    def property(self,args,context):
        self.bound(args['object_ids'])
        self.backend._contphy._load()
        if args['property_name']=='tension':
            from atw.polyworld.rope_tension_evidence import render_rope_tension_video
            raw=render_rope_tension_video(self.backend,[dict(object_id=i,label=i) for i in args['object_ids']],context.output_dir/'masks')
            return self.emit(context,dict(facts=[],details=['Compare original rope-mask behavior; no calibrated tension coefficient.'],interval_seconds=[0,self.backend.end]),
                'Original observed mask video, not a simulation.',raw=raw,images=[Path(raw['video'])])
        from atw.polyworld.swr_properties import property_evidence
        raw=property_evidence(self.backend,args['property_name'],args['object_ids'])
        values=(raw.get('value') or {}).get('relative_mass',{})
        facts=[dict(object_id=i,quantity='relative_mass',value=values.get(i),units='relative_to_first_bound_object',status=raw['status']) for i in args['object_ids']]
        next_step=raw.get('next_action')
        if raw['status']=='not_comparable':
            note=('The compared systems have no shared identifiable scale. This is a supported non-comparability '
                  'conclusion, not a failed measurement: use the output convention for non-comparability, '
                  'rather than yes/no or unresolved. Motion in independent systems cannot rank their masses.')
            if self.backend.engine!='rope':
                note=('No sustained mutual contact: relative density cannot be compared. Answer "Can not answer" '
                      'using the dataset output convention; this is a completed comparison, not a yes/no result or a request failure.')
            next_step='Finish evidence collection with this supported non-comparability conclusion.'
        else:
            note=(raw['interpretation'] if raw['status']=='model_relative' else
                  'This source does not identify the property; missing values do not imply zero or absence. '
                  'Missing measurements alone do not establish non-comparability.')
        return self.emit(context,dict(facts=facts,details=[raw.get('reason',raw.get('comparison_scope',''))],interval_seconds=[0,self.backend.end]),
            note,raw=raw,next_step=next_step)

    def simulate(self,args,context):
        removed=args.get('remove_object_ids',[]);self.bound(removed)
        if 'emission_fluid_id' in args:self.bound([args['emission_fluid_id']])
        key=self.experiment_key(args,bool(removed))
        if self.mode=='predictive' and args['extension_seconds']<=0:raise ValueError('Prediction requires positive continuation')
        if key not in self.experiments:
            raw=self.backend._contphy.run(args['extension_seconds'],remove_objects=removed,
                emission_fluid_id=args.get('emission_fluid_id'))
            rid=f'experiment_{len(self.backend.rollouts)+1}'
            directory=self.backend.output/rid;directory.mkdir()
            np.savez_compressed(directory/'geometry.npz',**raw['geometry'])
            if self.backend.engine=='fluid':
                np.savez_compressed(directory/'particles.npz',birth=raw['particle_birth'],fluid=raw['particle_fluid'])
            raw['end_seconds']=self.backend._rollout_end(raw)
            write(directory/'result.json',{k:v for k,v in raw.items() if k not in {'geometry','particle_birth','particle_fluid'}})
            self.backend.rollouts[rid]=raw;self.experiments[key]=rid
        rid=self.experiments[key];raw=self.backend.rollouts[rid]
        limitations=self.backend.limitations
        if self.backend.engine=='fluid':
            from atw.world_probing.engines.fluid_readout import LIMITATIONS
            limitations=LIMITATIONS
        return self.emit(context,dict(object_ids=list(raw['trajectories']),interval_seconds=[0,raw['end_seconds']],assumptions=raw['assumptions']),
            'Fixed-parameter simulated world. '
            +('Future liquid depends on the stated emission schedule. ' if self.backend.engine=='fluid' else '')
            +' '.join(limitations),rid,next_step='Query outcomes in this world.')

    def fluid(self,args,context):
        self.bound([args['object_id'],args['target_id']])
        rid=self.native_world(args['world_id']);start,end=self.interval(args,rid)
        raw=self.backend.rollouts[rid];source=self.backend._contphy._load()
        name=args['object_id']
        if name not in source.fluid_ids or args['target_id'] in source.fluid_ids:raise ValueError('Use an E liquid and a static target')
        selected=raw['particle_fluid']==source.fluid_ids.index(name)
        mask=self.backend._mask(args['target_id'],self.backend.best_frames[args['target_id']])
        mask=cv2.resize(mask.astype(np.uint8),(source.image_width,source.image_height),
            interpolation=cv2.INTER_NEAREST).astype(bool)
        value=fluid_static(raw['geometry'][name],raw['particle_birth'][selected],mask,
            math.ceil(start*self.backend.fps),min(len(raw['geometry'][name])-1,math.floor(end*self.backend.fps)),
            self.backend.fit['particle_radius_pixels'])
        source_ids={r['static_object_id'] for r in self.backend._contphy.fluid_metadata['source_records']}
        if args['target_id'] in source_ids:
            value.update(container_status='not_applicable',entered_fraction=None,final_fraction=None)
        from atw.world_probing.engines.fluid_readout import summarize
        evidence,note=summarize(value,args['measurement'])
        return self.emit(context,dict(object_id=name,target_id=args['target_id'],interval_seconds=[start,end],**evidence),
            note,rid,raw=value)

    def softbody(self,args,context):
        self.bound([args['object_id']]);rid=self.native_world(args['world_id'])
        start,end=self.interval(args,rid);raw=self.backend.rollouts[rid];name=args['object_id']
        from atw.polyworld.experiment_views import image_projection
        points=raw['geometry'][name];frames=np.arange(len(points))+self.backend._contphy._load().frames[0]
        first=self.backend._contphy._load().first_offsets[self.backend._contphy._load().object_ids.index(name)]
        select=(frames/self.backend.fps>=start)&(frames/self.backend.fps<=end)&(np.arange(len(frames))>=first)
        project=image_projection(self.backend,raw)
        pixels=project(points[select].reshape(-1,3)).reshape((-1,points.shape[1],2))
        with np.load(self.backend.paths['swr']/self.backend.fit['observations_npz']) as obs:
            offset=obs['offsets'][self.backend._contphy._load().object_ids.index(name)]
        pixels+=offset
        result=dict(downward_entry_times_seconds=[],fully_below_image=None,coverage_complete=bool(len(pixels)))
        region_id=args.get('region_id')
        if region_id is not None:
            from atw.polyworld.swr_queries import scene_regions
            region=next((r for r in scene_regions(self.backend) if r['region_id']==region_id),None)
            if region is None:raise ValueError('Use an advertised region_id')
            result=pit_crossings(pixels,frames[select],region['region']['box'],self.backend.height,self.backend.fps)
        contacts=[]
        for event in raw.get('events',[]):
            involved=[event.get('soft_body_id'),event.get('contact_id')]
            if name in involved and start<=event['time_s']<=end:
                contacts.append(dict(object_id=next(i for i in involved if i!=name),time_seconds=event['time_s']))
        if region_id is None:
            finding='No pit was selected; the empty entry list is not a pit measurement. '
        elif not result['coverage_complete']:
            finding=f'Entry into {region_id} has incomplete measurement coverage. '
        else:
            finding=f"{region_id}: {'entry detected' if result['downward_entry_times_seconds'] else 'no entry detected'} in the queried interval. "
        return self.emit(context,dict(object_id=name,interval_seconds=[start,end],contacts=contacts,region_id=region_id,**result),
            finding+'Pit entry requires downward center crossing or whole-body descent through the mouth after rolling in, '
            'without the center returning above the mouth. Passage wholly below the mouth is not entry. '
            'fully_below_image only reports leaving the screen bottom; it does not establish entry into this pit or success of an option targeting it. '
            'coverage_complete describes frame coverage. Coordinates use retained observation-calibrated image offsets.',rid)

    def mass(self,args,context):
        self.bound([args['object_id']]);key=self.experiment_key(args,True)
        if key not in self.experiments:
            raw=self.rope_control('change_mass',args)
            self.experiments[key]=raw
        raw=self.experiments[key]
        return self.rope_receipt(raw,context,intervention='mass')

    def drive(self,args,context):
        self.bound([args['object_id']]);key=self.experiment_key(args,True)
        if key not in self.experiments:self.experiments[key]=self.rope_control('drive_rope_endpoint',args)
        return self.rope_receipt(self.experiments[key],context,intervention='endpoint')

    def rope_control(self,operation,args):
        try:
            return getattr(self.backend,operation)(**args)
        except ValueError as exc:
            if 'query_rope_controls' not in str(exc):raise
            target='load mass' if operation=='change_mass' else 'endpoint attachment'
            raise ValueError(f'No unique modeled {target} for this identity. Check query_rope_controls for an '
                             'unambiguous control on the same bound identity; if unavailable, report evidence_unavailable.') from exc

    def rope_receipt(self,raw,context,*,intervention=None):
        rid=raw['rollout_id'];end_to_end=[];intermediate=[];stationary=[]
        usable=raw.get('simulation_stability',{}).get('usable_for_downstream_reasoning',False)
        interval=raw.get('interval_seconds',[raw.get('readout_start_seconds',0),self.backend._rollout_end(self.backend.rollouts[rid])])
        def add_samples(name,row,key,rotation=False):
            samples=row.get(key,[])
            scope=row.get('interval_seconds') or interval
            # Missing native measurements remain explicit nulls in the compact receipt.
            selected=samples[1:] if len(samples)>1 else [dict(time_seconds=scope[1])]
            for index,sample in enumerate(selected):
                item=dict(object_id=name,start_seconds=scope[0],time_seconds=sample['time_seconds'])
                if rotation:
                    if 'rope_id' in row:item['rope_id']=row['rope_id']
                    item['net_rotation_radians']=sample.get('net_angle_rad') if usable else None
                else:
                    delta=sample.get('net_displacement_xy_px') if usable else None
                    item.update(net_right_displacement_pixels=delta[0] if delta is not None else None,
                                net_down_displacement_pixels=delta[1] if delta is not None else None)
                (end_to_end if index==len(selected)-1 else intermediate).append(item)
        for name,row in raw.get('center_translation',{}).items():
            if name not in self.bound_ids:continue
            add_samples(name,row,'net_displacement_samples')
        for name,rows in raw.get('pulley_rotation',{}).get('objects',{}).items():
            if name not in self.bound_ids:continue
            if not isinstance(rows,list):rows=[rows]
            for row in rows:
                add_samples(name,row,'net_angle_samples',rotation=True)
            if usable and rows and all(row.get('rotation_occurred') is False for row in rows):
                stationary.append(name)
        note=''
        if stationary:
            note=('Rotation conclusion for the full queried interval: '+
                  '; '.join(name+' does not rotate' for name in stationary)+'. '
                  'All contact intervals are covered and below the engine stationary tolerance. '
                  'Their small nonzero net angles are sub-tolerance drift, not directional rotation. ')
        note+=ROPE_MOTION_NOTE
        if intervention=='mass':
            note+=(' Mass edits use finite numerically selected amounts, zero initial velocity and a matched '
                   'unedited control with unchanged geometry and constraints. An unidentified member mass cannot stand for the whole assembly.')
        elif intervention=='endpoint':
            note+=' Endpoint driving prescribes motion; it does not change mass.'
        if self.backend.engine=='rope' and self.mode=='goal':
            from atw.world_probing.engines.rope_goal import unchanged_goal_note
            note+=unchanged_goal_note(self.backend,raw,self.bound_ids)
        return self.emit(context,dict(end_to_end=end_to_end,intermediate_samples=intermediate,
            interval_seconds=interval),note,rid,raw)

    def rope_motion(self,args,context):
        self.bound(args['object_ids']);rid=self.native_world(args['world_id'])
        scoped=dict(args,start_seconds=max(args.get('start_seconds',0),self.backend.rollouts[rid].get('readout_start_seconds',0)))
        start,end=self.interval(scoped,rid)
        measurement=args.get('measurement','both')
        raw=dict(rollout_id=rid,center_translation={},pulley_rotation=dict(objects={}),assumptions=self.backend.rollouts[rid]['assumptions'],
            simulation_stability=self.backend.rollouts[rid].get('simulation_stability',{}),interval_seconds=[start,end])
        for name in args['object_ids']:
            if measurement in {'translation','both'}:
                value=self.backend.query_trajectory(name,rid,start,end)
                raw['center_translation'][name]=value['center_translation']
            if measurement=='rotation' or (measurement=='both' and any(
                    b.object_id==name and b.role=='pulley' for scene in self.backend._contphy._load() for b in scene.bodies)):
                raw['pulley_rotation']['objects'][name]=self.backend.query_pulley_rotation(name,rid,start,end)['results']
        return self.rope_receipt(raw,context)

    def cloth(self,args,context):
        self.bound([args['object_id']]);rid=self.native_world(args['world_id']);start,end=self.interval(args,rid)
        from atw.world_probing.engines.cloth_readout import measure
        data,note,raw=measure(self.backend,rid,[args['object_id']],start,end)
        return self.emit(context,data,note,rid,raw)

    def cloth_contact(self,args,context):
        self.bound(args['object_ids']);rid=self.native_world(args['world_id']);start,end=self.interval(args,rid)
        from atw.world_probing.engines.cloth_readout import measure
        data,note,raw=measure(self.backend,rid,args['object_ids'],start,end,contact=True)
        return self.emit(context,data,note,rid,raw)
