"""Question-scoped world identities, compact evidence and canonical tool wrappers."""
import json

from atw.world_modeling import sam3_protocol as s
from atw.world_modeling.scene_assets import artifact
from atw.world_probing.evidence import write
from atw.world_probing.tool_definitions import tool

TEXT = s.TEXT
IDS = dict(type='array',items=TEXT,uniqueItems=True)
SECONDS = dict(type='number',minimum=0)
WORLD = dict(type=['string','null'])


def fields(required=(), **properties):
    return dict(type='object',properties=properties,required=list(required),additionalProperties=False)


def receipt_schema(evidence):
    return fields(['world_id','evidence','note'],world_id=WORLD,evidence=evidence,note=TEXT,
                  next_step=dict(**TEXT,description='Optional collection-only action hint; excluded from final-answer input.'))


class EngineTools:
    def __init__(self, backend, runner, question, mode, binding, output):
        self.backend, self.runner, self.question = backend, runner, question
        self.mode, self.binding, self.output = mode, binding, output
        self.worlds, self.reverse, self.experiments = {}, {}, {}
        self.used_worlds = set()
        self.bound_ids = sorted({oid for row in binding['identity_evidence'] for oid in row['object_ids']})

    def definition(self, name, description, arguments, evidence, execute):
        def scoped(args, context):
            if context.question_id != self.question['id']:
                raise ValueError('This engine session belongs to another question')
            return execute(args,context)
        return tool(name,description,arguments,receipt_schema(evidence),scoped)

    def native_world(self, world_id):
        if world_id not in self.worlds:
            raise ValueError('Use an existing world_id returned for this question')
        return self.worlds[world_id]

    def interval(self, args, rid):
        raw = self.backend.rollouts[rid]
        end = float(raw['end_seconds'] if 'end_seconds' in raw else self.backend._rollout_end(raw))
        start = float(args.get('start_seconds',0))
        if self.mode=='predictive':start=max(start,self.backend.end+1/self.backend.fps)
        stop = float(args.get('end_seconds',end))
        if not 0<=start<=stop<=end:raise ValueError('Query interval must lie inside this world; predictive queries read future only')
        return start,stop

    def experiment_key(self, args, edited):
        if self.mode=='property':raise ValueError('Property questions do not create simulations')
        if self.mode=='predictive' and edited:raise ValueError('Prediction requires the unmodified world')
        if self.mode in ('counterfactual','goal') and not edited:
            raise ValueError('An explicit requested intervention is required')
        if self.mode=='goal' and args.get('option_id') not in {o['id'] for o in self.question['options']}:
            raise ValueError('Goal experiments require a supplied concrete option_id')
        physical = {k:v for k,v in args.items() if k!='option_id'}
        key = json.dumps(physical,sort_keys=True)
        if self.mode in ('predictive','counterfactual') and self.experiments and key not in self.experiments:
            raise ValueError('Reuse the existing question world; do not change its experiment')
        return key

    def register_world(self, rid, context, items):
        if rid in self.reverse:return self.reverse[rid]
        index = getattr(self.runner,'_world_index',0)+1
        self.runner._world_index = index
        wid = f'world_{index:04d}'
        self.worlds[wid], self.reverse[rid] = rid,wid
        raw = self.backend.rollouts[rid]
        # Native arrays stay in backend files; this is the canonical world receipt.
        metadata = dict(protocol_version=1,world_id=wid,question_id=self.question['id'],
            engine=self.backend.manifest['engine'],source_swr_asset_id=self.backend.sources['swr_asset_id'],
            source_kind='simulation',observed_end_seconds=self.backend.end,
            native_record=str((self.backend.output/'rollouts'/f'{rid}.json').relative_to(context.run_dir)) if self.backend.engine=='rigid'
                          else str((self.backend.output/rid/'result.json').relative_to(context.run_dir)),
            intervention={key:raw[key] for key in ('remove_objects','mass_scales','emission_fluid_id','simulation_request') if key in raw})
        path = context.output_dir/f'{wid}.json'
        write(path,metadata)
        item = artifact(context,wid,path); item['source_kind']='simulation'; items.append(item)
        return wid

    def emit(self, context, evidence, note, rid=None, raw=None, images=(), *, next_step=None):
        items=[]
        wid=self.register_world(rid,context,items) if rid is not None else None
        if raw is not None:
            path=context.output_dir/'measurement.json';write(path,raw)
            items.append(artifact(context,'measurement',path))
        for index,path in enumerate(images,1):
            items.append(artifact(context,f'observation_{index:03d}',path,
                'video/mp4' if str(path).endswith('.mp4') else 'image/jpeg'))
        if wid:self.used_worlds.add(wid)
        data=dict(world_id=wid,evidence=evidence,note=note)
        if next_step is not None:data['next_step']=next_step
        return data,items

    def context(self):
        return dict(engine=self.backend.manifest['engine'],observed_end_seconds=self.backend.end,
                    object_ids=list(self.backend.objects),limitations=self.backend.limitations)
