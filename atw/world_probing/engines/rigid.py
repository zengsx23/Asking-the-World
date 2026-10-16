"""Rigid-world simulation and event queries."""
from copy import deepcopy

from atw.polyworld import clevrer_evidence as events
from atw.world_probing.engines.common import EngineTools, fields, TEXT, IDS, SECONDS
from atw.world_modeling import sam3_protocol as s
from atw.world_probing.evidence import write

EVENT = s.obj(event_id=TEXT,kind=dict(type='string',enum=['collision','entry','exit']),
              object_ids=IDS,time_seconds=SECONDS)
INTERVAL = dict(type='array',items=SECONDS,minItems=2,maxItems=2)
WORLD_EVIDENCE = s.obj(removed_object_ids=IDS,end_seconds=SECONDS,observed_end_seconds=SECONDS)
EVENT_IDENTITY_NOTE = ('Compare the specific target event across worlds. Use the five-frame event identity rule: '
    'matching participants within five frames retain the original event; a larger shift is a different event. '
    'Do not treat a later different event as preservation of the target.')


class RigidTools(EngineTools):
    def definitions(self):
        rows=[]
        if self.mode=='explanatory':
            rows.append(self.definition('load_baseline_world','Read the retained fitted baseline for historical event analysis; does not run fitting or simulation.',
                s.obj(),WORLD_EVIDENCE,self.baseline))
        if self.mode!='property':
            args=fields(['extension_seconds'],extension_seconds=SECONDS)
            if self.mode!='predictive':args['properties']['remove_object_ids']=IDS
            if self.mode=='goal':args['properties']['option_id']=TEXT;args['required'].append('option_id')
            rows.append(self.definition('simulate_world','Create the requested fixed-parameter rigid world. Removal applies from the modeled beginning. Reuse returned world_id for every query.',
                args,WORLD_EVIDENCE,self.simulate))
            query=fields(['world_id'],world_id=TEXT,start_seconds=SECONDS,end_seconds=SECONDS)
            if self.mode in {'explanatory','counterfactual'}:
                rows.append(self.definition('query_event_timeline',
                    'List collision, entry or exit events with stable event IDs. ' + events.ENTRY_GUIDANCE +
                    ' Exit queries use projected center crossings of the image boundary.',
                    fields(['world_id'],**query['properties'],kind=dict(type='string',enum=['collision','entry','exit']),
                        object_ids=dict(IDS,description='Optional intersection filter: each returned event must contain EVERY supplied ID. Omit to list all events. Do not combine different candidate pairs into one list.')),
                    s.obj(events=s.array(EVENT),interval_seconds=INTERVAL,complete=dict(type='boolean')),self.timeline))
            rows.append(self.definition('query_event_outcome','Check one collision pair. If absent with end_seconds omitted, extend the same edit up to three video durations. Returns occurrence, times and coverage.',
                fields(['world_id','object_ids'],**query['properties'],object_ids=dict(type='array',items=TEXT,minItems=2,maxItems=2,uniqueItems=True)),
                fields(['object_ids','occurred','event_times_seconds','interval_seconds','complete'],object_ids=IDS,
                    occurred=dict(type=['boolean','null']),event_times_seconds=s.array(SECONDS),interval_seconds=INTERVAL,
                    complete=dict(type='boolean'),same_historical_event=dict(type=['boolean','null']),
                    baseline_first_frame=dict(type=['number','null']),edited_first_frame=dict(type=['number','null'])),self.outcome))
        if self.mode=='explanatory':
            rows.append(self.definition('query_event_path','Read a strictly time-increasing shared-object path between two events in the same world. A path supports possible transmission, not a causal verdict.',
                s.obj(world_id=TEXT,candidate_event_id=TEXT,target_event_id=TEXT),
                s.obj(event_ids=IDS,complete=dict(type='boolean')),self.path))
        return rows

    def world_receipt(self,context,rid):
        raw=self.backend.rollouts[rid]
        return self.emit(context,dict(removed_object_ids=raw['remove_objects'],end_seconds=raw['end_seconds'],
            observed_end_seconds=self.backend.end),
            'A simulation receipt alone does not establish an absent collision. '
            "Each object's first usable 3D state is its appearance in this modeled world. It is absent beforehand, "
            'not subject to an early-coverage uncertainty. Baseline and edited collision logs use the same rule. '
            'Removal removes all interactions of a body, not one chosen collision. '
            +' '.join(self.backend.limitations),rid,next_step='Read this world using event queries.')

    def baseline(self,args,context):
        rid='baseline'
        if rid not in self.backend.rollouts:
            fit=self.backend.fit
            raw=dict(rollout_id=rid,source_kind='simulation',remove_objects=[],extension_seconds=0.,
                events=deepcopy(fit['events']),predicted_trajectories=deepcopy(fit['predicted_trajectories']),
                end_seconds=self.backend.end,observed_end_seconds=self.backend.end,event_capacity_reached=len(fit['events'])>=32)
            self.backend.rollouts[rid]=raw
            write(self.backend.output/'rollouts'/f'{rid}.json',raw)
        return self.world_receipt(context,rid)

    def simulate(self,args,context):
        removed=args.get('remove_object_ids',[])
        if set(removed)-set(self.bound_ids):raise ValueError('Removal targets must be bound question identities')
        if self.mode=='predictive' and args['extension_seconds']<=0:raise ValueError('Prediction requires positive continuation')
        key=self.experiment_key(args,bool(removed))
        if key not in self.experiments:
            result=self.backend._rigid.simulate(args['extension_seconds'],removed)
            self.experiments[key]=result['rollout_id']
        return self.world_receipt(context,self.experiments[key])

    def timeline(self,args,context):
        rid=self.native_world(args['world_id']);start,end=self.interval(args,rid)
        raw=events.timeline(self.backend,rid,args.get('kind','collision'),args.get('object_ids'),start,end)
        evidence=dict(events=[dict(event_id=e['event_id'],kind=e['kind'],object_ids=e['object_ids'],time_seconds=e['event_time_s'])
                              for e in raw.get('events',[])],interval_seconds=[start,end],complete=self.complete(raw))
        note=raw.get('interpretation','Read event order in this modeled world, not as a causal verdict.')
        if self.mode=='explanatory':
            note+=(' Historical targets are baseline events identified by participants and time. Later extrapolated '
                'collisions cannot substitute for a missing historical target. A candidate strictly after the target '
                'cannot cause it; equal or near times do not establish order. For presence/entrance choices, removal '
                'tests an enabling role; a separate immediate-collision choice does not invalidate that role.')
        return self.emit(context,evidence,note,rid,raw)

    @staticmethod
    def complete(raw):
        return raw.get('event_log_complete') is True and not raw.get('event_capacity_reached') and not raw.get('coverage_limitations')

    def outcome(self,args,context):
        rid=self.native_world(args['world_id'])
        start=float(args.get('start_seconds',0))
        if self.mode=='predictive':start=max(start,self.backend.end+1/self.backend.fps)
        raw=events.event_outcome(self.backend,rid,args['object_ids'],start,args.get('end_seconds'))
        actual=raw.get('rollout_id',rid)
        if self.mode=='explanatory':
            raw=events.attach_event_identity(self.backend,'query_event_outcome',dict(object_ids=args['object_ids']),raw)
        evidence=dict(object_ids=args['object_ids'],occurred=raw.get('event_occurs'),
            event_times_seconds=[r['event_time_s'] for r in raw.get('events',[])],
            interval_seconds=raw.get('interval_seconds',[start,start]),complete=self.complete(raw))
        if 'target_comparison' in raw:
            comparison=raw['target_comparison']
            evidence.update(same_historical_event=comparison.get('same_event_with_base'),
                baseline_first_frame=comparison.get('base_first_frame'),edited_first_frame=comparison.get('edited_first_frame'))
        note=('A present event establishes occurrence. With complete coverage, absence at the stated horizon counts '
              'as nonoccurrence under the scene convention; this is not an infinite-time physical theorem. '
              'Incomplete coverage cannot establish nonoccurrence; obtain the missing evidence before deciding.')
        if raw.get('reason'):note+=' '+raw['reason']
        if 'target_comparison' in raw:note+=' '+EVENT_IDENTITY_NOTE
        next_step=('Correct or complete the query; finish with evidence_unavailable if the required measurement remains unresolved.'
                   if evidence['occurred'] is None else None)
        return self.emit(context,evidence,note,actual,raw,next_step=next_step)

    def path(self,args,context):
        rid=self.native_world(args['world_id'])
        raw=events.event_path(self.backend,rid,args['candidate_event_id'],args['target_event_id'])
        return self.emit(context,dict(event_ids=[e['event_id'] for e in raw['path']] if raw['path'] else [],
            complete=not raw.get('coverage_limitations') and not raw.get('event_capacity_reached')),
            raw['interpretation'],rid,raw)
