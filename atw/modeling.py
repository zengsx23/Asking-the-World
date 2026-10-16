"""Programmatic stage-one scheduling; semantic decisions belong to Engineers."""
import json

from atw.config import StageContext
from atw.contracts import StageOneInput, StageResult, stage_result, validate_stage_one_scene


class _ToolFailure(Exception):
    def __init__(self, error):
        self.error=error


def run(inputs: StageOneInput, context: StageContext) -> StageResult:
    if context.tools is None:
        return stage_result('stage_one','blocked',error=dict(code='missing_tools',message='Stage one requires its registered ToolRunner.'))
    scene=dict(route=None,sources={},engine=None,swr_availability='pending',
               ready_for_stage_two=False,direct_video_fallback=False)
    tasks=[dict(id='routing',tool_name='world_representation_routing',status='pending',call_id=None,error=None)]
    artifacts=[]
    checklist_path=context.output_dir/'checklist.json'

    def persist():
        checklist_path.write_text(json.dumps(dict(protocol_version=1,tasks=tasks),indent=2)+'\n')

    def append_task(name,tool):
        tasks.append(dict(id=name,tool_name=tool,status='pending',call_id=None,error=None))

    def call(name,arguments):
        task=next(t for t in tasks if t['id']==name)
        task['status']='running'
        persist()
        try:
            result=context.tools.call(task['tool_name'],arguments,stage='stage_one')
        except Exception as exc:
            task.update(status='failed',error=dict(code='execution_failed',message=f'{type(exc).__name__}: {exc}'))
            persist()
            raise _ToolFailure(task['error']) from exc
        task['call_id']=result['call_id']
        artifacts.extend(result['artifacts'])
        if result['status']=='error':
            task.update(status='failed',error=result['error'])
            persist()
            raise _ToolFailure(result['error'])
        task['status']='completed'
        persist()
        return result

    def read_asset(asset_id):
        item=next((a for a in artifacts if a['artifact_id']==asset_id),None)
        if item is None:raise ValueError(f'Missing returned scene artifact: {asset_id}')
        path=(context.run_dir/item['path']).resolve()
        if not path.is_relative_to(context.run_dir.resolve()):
            raise ValueError('Scene artifact must belong to this run')
        return json.loads(path.read_text())

    def finish(status,error=None):
        if status!='completed':scene['ready_for_stage_two']=False
        for task in tasks:
            if task['status']=='running':
                task.update(status='failed',error=error)
            elif task['status']=='pending':
                task['status']='blocked'
        persist()
        record=validate_stage_one_scene(scene)
        checklist=dict(artifact_id='stage_one_checklist',path=str(checklist_path.relative_to(context.run_dir)),
                       media_type='application/json',source_kind='derived')
        return stage_result('stage_one',status,data=dict(scene=record,checklist_artifact_id=checklist['artifact_id']),
                            artifacts=[*artifacts,checklist],error=error)

    persist()
    try:
        routed=call('routing',{})
        route=scene['route']=routed['data']['route']
        sources=scene['sources']
        sources['routing_call_id']=routed['call_id']
        if route=='2D+PIXEL':
            append_task('tracking','track_pixel_scene')
        elif route in ('2D+SAM3','3D+SAM3'):
            append_task('segmentation','sam3_video_segmentation')
            append_task('physical_types','classify_object_physical_types')
            append_task('gravity','estimate_scene_gravity')
            append_task('tracking','reconstruct_and_track_scene')
        else:
            raise ValueError('Unsupported route returned by routing Engineer')
        append_task('swr','fit_scene_swr')
        append_task('handoff',None)
        persist()
        if route=='2D+PIXEL':
            tracked=call('tracking',dict(routing_call_id=sources['routing_call_id']))
        else:
            segmented=call('segmentation',{})
            sources['segmentation_asset_id']=segmented['data']['segmentation_asset_id']
            classified=call('physical_types',dict(segmentation_asset_id=sources['segmentation_asset_id']))
            # The v1 data holds objects; the returned artifact owns their labels.
            classification_id=classified['call_id']+'_physical_types'
            if not any(a['artifact_id']==classification_id for a in classified['artifacts']):
                raise ValueError('Physical classification did not return its physical_types artifact')
            sources['physical_types_asset_id']=classification_id
            arguments=dict(routing_call_id=sources['routing_call_id'])
            if route=='3D+SAM3':
                arguments.update({k:sources[k] for k in ('segmentation_asset_id','physical_types_asset_id')})
            gravity=call('gravity',arguments)
            sources['gravity_asset_id']=gravity['data']['gravity_asset_id']
            # Unavailable gravity is explicitly allowed by native tracking. SWR
            # admission, not this scheduler, decides whether fitting can proceed.
            tracked=call('tracking',{k:sources[k] for k in ('gravity_asset_id','segmentation_asset_id','physical_types_asset_id')})
        sources['tracking_asset_id']=tracked['data']['tracking_asset_id']
        fit=call('swr',dict(tracking_asset_id=sources['tracking_asset_id']))
        sources['swr_asset_id']=fit['data']['swr_asset_id']
        scene.update(engine=fit['data']['engine'],swr_availability=fit['data']['availability'])
        if fit['data']['availability']!='fitted':
            failure=context.tools.stage_one_failure
            if failure is not None:
                scene['direct_video_fallback']=failure['data']['scene']['direct_video_fallback']
                error=failure['error']
            else:
                error=dict(code='swr_unavailable',message=fit['data']['reason'])
            tasks[-2].update(status='failed',error=error)
            return finish('failed',error)
        tasks[-1]['status']='running'
        persist()
        swr=read_asset(sources['swr_asset_id'])
        if swr.get('availability')!='fitted' or not swr.get('result_asset_id'):
            raise ValueError('Fitted SWR requires a retained fitted simulation artifact')
        fit_id=swr['result_asset_id']
        fit_art=next((a for a in artifacts if a['artifact_id']==fit_id),None)
        if fit_art is None or not (context.run_dir/fit_art['path']).is_file():
            raise ValueError('Fitted simulation artifact is missing')
        effective=swr.get('effective_sources',{})
        if effective:
            sources['tracking_asset_id']=effective['tracking_asset_id']
            tracking=read_asset(sources['tracking_asset_id'])
            for key in ('segmentation_asset_id','physical_types_asset_id','gravity_asset_id'):
                sources[key]=tracking['sources'][key]
        tasks[-1]['status']='completed'
        scene['ready_for_stage_two']=True
        return finish('completed')
    except _ToolFailure as exc:
        return finish('failed',exc.error)
    except (ValueError,KeyError,OSError,TypeError) as exc:
        return finish('failed',dict(code='invalid_scene_handoff',message=f'{type(exc).__name__}: {exc}'))
