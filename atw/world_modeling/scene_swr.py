"""Stage-one SWR Engineer with programmatic route/type dispatch."""
import importlib
from copy import deepcopy
import json
from pathlib import Path
from atw.world_modeling import sam3_protocol as c
from atw.world_modeling.scene_assets import asset_path, artifact, classified_assets, write
from atw.world_modeling.request_scope import agentic_requests

# New engines add one rule and one backend; the public tool remains unchanged.
ENGINES = (
    dict(name='rigid', route='3D+SAM3', required={'rigid'}, allowed={'rigid'}),
    dict(name='cloth', route='3D+SAM3', required={'cloth'}, allowed={'cloth','rigid'}),
    dict(name='softbody', route='3D+SAM3', required={'soft_body'}, allowed={'soft_body','rigid'}),
    dict(name='fluid', route='2D+PIXEL', required={'fluid'}, allowed={'fluid','static','boundary'}),
    dict(name='rope', route='2D+SAM3', required={'rope'}, allowed={'rope','rigid'}),
)
DEFINITION = dict(protocol_version=1, name='fit_scene_swr', stages=['stage_one'],
    description='Fit a supported physics model to retained scene tracks; select the SWR engine from the route and physical types.',
    input_schema=c.obj(tracking_asset_id={**c.TEXT, 'description':'Same-run observed tracking manifest artifact ID; upstream assets are read-only.'}),
    output_schema=c.obj(engine=dict(type=['string','null'], enum=['rigid','cloth','rope','fluid','softbody',None]),
        availability=dict(type='string',enum=['fitted','unavailable']),
        swr_asset_id={**c.TEXT,'description':'Retained SWR manifest with input references, eligibility, baseline fitted simulation, metrics, units and limitations.'},
        reason=c.TEXT))


def validate_arguments(value):return c.validate(DEFINITION['input_schema'], value, 'arguments')
def validate_output(value):return c.validate(DEFINITION['output_schema'], value, 'data')


def select_engine(route, types):
    types = set(types)
    for rule in ENGINES:
        if route==rule['route'] and rule['required']<=types<=rule['allowed']:
            return rule['name']
    return None


def local_path(context, parent, value):
    path = (parent/value).resolve()
    if not path.is_relative_to(context.run_dir.resolve()) or not path.exists():
        raise ValueError('Referenced SWR input must exist within this run')
    return path


def load_inputs(arguments, context):
    path = asset_path(context, arguments['tracking_asset_id'])
    tracking = json.loads(path.read_text())
    if tracking.get('protocol_version')!=1:
        raise ValueError('Expected a protocol_version=1 tracking manifest')
    if Path(tracking['video_path']).resolve()!=Path(context.inputs['video_path']).resolve():
        raise ValueError('Tracking video differs from the original input')
    route = tracking['route']
    if route=='2D+PIXEL':
        from atw.world_modeling.scene_assets import route_result
        if route_result(context, tracking['sources']['routing_call_id']) != route:
            raise ValueError('Pixel tracking route differs from the referenced routing result')
        if tracking.get('representation') != 'observed_pixel_masks':
            raise ValueError('Fluid SWR requires observed pixel masks')
        keys=('masks_asset_id','static_masks_asset_id','source_masks_asset_id','source_binding_asset_id')
        paths={k:asset_path(context,tracking['assets'][k]) for k in keys}
        binding=json.loads(paths['source_binding_asset_id'].read_text())
        if Path(binding['source_video_path']).resolve()!=Path(context.inputs['video_path']).resolve():
            raise ValueError('Source binding video differs from tracking')
        if binding['frame_scope']!=tracking['frame_scope'] or binding['fps']!=tracking['fps']:
            raise ValueError('Source binding temporal scope differs from tracking')
        for k in ('masks_asset_id','static_masks_asset_id','source_masks_asset_id'):
            if binding['source_assets'][k]!=tracking['assets'][k]:
                raise ValueError('Source binding references different pixel observations')
        object_ids={o['object_id'] for o in tracking['objects'] if o['object_kind']=='fluid'}
        if any(s['fluid_object_id'] is not None and s['fluid_object_id'] not in object_ids for s in binding['sources']):
            raise ValueError('Source binding references unknown fluid identities')
        engine=select_engine(route,[o['object_kind'] for o in tracking['objects']])
        return dict(tracking=tracking,engine=engine,pixel_paths=paths,source_binding=binding,
                    reason='No supported fluid/static pixel inventory.')
    if route not in ('2D+SAM3','3D+SAM3'):
        raise ValueError('Unknown tracking route')
    sources = tracking['sources']
    bundle = classified_assets(context, sources['segmentation_asset_id'], sources['physical_types_asset_id'])
    objects = bundle[-1]
    if tracking['objects'] != [{k:o[k] for k in ('object_id','track_id','physical_type')} for o in objects]:
        raise ValueError('Tracking identities differ from the referenced classification')
    gravity = json.loads(asset_path(context, sources['gravity_asset_id']).read_text())
    if gravity['route']!=route or Path(gravity['video_path']).resolve()!=Path(context.inputs['video_path']).resolve():
        raise ValueError('Gravity route/video differs from tracking')
    if route=='3D+SAM3' and any(gravity['sources'][k]!=sources[k] for k in ('segmentation_asset_id','physical_types_asset_id')):
        raise ValueError('Gravity references different observation assets')
    engine = select_engine(route, [o['physical_type'] for o in objects])
    inputs = dict(tracking=tracking, bundle=bundle, objects=objects, engine=engine, gravity=gravity,
                  segmentation_path=asset_path(context,sources['segmentation_asset_id']), reason='No supported SWR engine for this route/type combination.')
    if route=='3D+SAM3':
        inputs['tracking_directory'] = local_path(context,path.parent,tracking['tracking_directory'])
        reconstruction = local_path(context,path.parent,tracking['reconstruction_manifest']).parent
        inputs['reconstruction_directory'] = reconstruction
        if inputs['tracking_directory'].parent/'reconstruction' != reconstruction:
            raise ValueError('Native tracking and reconstruction directories must share their declared parent')
    return inputs


def eligibility(inputs):
    engine=inputs['engine']
    if engine is None:return inputs['reason']
    if engine=='fluid':
        if not any(s['fluid_object_id'] is not None and s['first_visible_frame'] is not None for s in inputs['source_binding']['sources']):
            return 'Fluid SWR requires observed source bindings and emission.'
        return None
    gravity=inputs['gravity']['gravity']
    if engine=='rope':
        if gravity.get('direction')!='screen_down':return 'Rope SWR requires screen-down gravity.'
        return None
    if gravity.get('status')!='completed':return 'Surface gravity is unavailable.'
    surface='wall_plane' if engine=='softbody' else 'support_plane'
    if gravity.get('scene_surface_type')!=surface:return f'{engine} SWR requires a '+('wall plane.' if engine=='softbody' else 'support plane.')
    if engine in ('cloth','softbody'):
        import numpy as np
        for obj in inputs['objects']:
            if obj['physical_type']!='rigid':continue
            with np.load(inputs['tracking_directory']/'rigid_trajectories'/f"{obj['object_id']}.npz",allow_pickle=False) as z:
                if str(z['motion_model'].item())!='static':return f'{engine.capitalize()} SWR requires tracked-static rigid obstacles.'
    return None


@agentic_requests
def execute(arguments, context):
    result=dict(protocol_version=1,call_id=context.call_id,tool_name=DEFINITION['name'],status='error',data={},artifacts=[],error=None)
    try:validate_arguments(arguments)
    except ValueError as exc:
        result['error']=dict(code='invalid_arguments',message=str(exc));return result
    try:
        inputs=load_inputs(arguments,context)
        reason=eligibility(inputs)
    except (ValueError,KeyError,OSError,TypeError) as exc:
        result['error']=dict(code='invalid_prerequisite',message=str(exc));return result
    review_artifacts=[]
    gravity_correction=None
    if reason and inputs['tracking']['route']=='2D+SAM3' and inputs['engine']=='rope':
        from atw.world_modeling.gravity_2d import estimate
        original_gravity=deepcopy(inputs['gravity'])
        corrected_gravity=deepcopy(original_gravity)
        try:
            corrected_gravity['gravity']=estimate(context,admission_review=dict(
                original_gravity=original_gravity['gravity'],rejection=reason,
                instruction='原来的重力方向很可能有问题，请结合原视频重新判断并修改；如果证据支持原方向，请保留并说明原因。'))
        except Exception as exc:
            result['error']=dict(code='gravity_admission_review_failed',message=f'{type(exc).__name__}: {exc}')
            return result
        candidate_inputs=dict(inputs,gravity=corrected_gravity)
        candidate_reason=eligibility(candidate_inputs)
        adopted=(candidate_reason is None and corrected_gravity['gravity']['direction']!=original_gravity['gravity'].get('direction'))
        correction_path=context.output_dir/'gravity_admission_correction.json'
        write(correction_path,dict(original_gravity=original_gravity,
            corrected_gravity=corrected_gravity,original_rejection=reason,
            candidate_rejection=candidate_reason,adopted=adopted,
            method='vlm_admission_review'))
        correction_artifact=artifact(context,'gravity_admission_correction',correction_path)
        review_artifacts.append(correction_artifact)
        gravity_correction=dict(artifact_id=correction_artifact['artifact_id'],adopted=adopted)
        if adopted:inputs=candidate_inputs;reason=None
    motion_review_adopted=False
    admission_review_record=None
    if reason and inputs['tracking']['route']=='3D+SAM3':
        from atw.world_modeling.rigid_motion_review import review, prepare_reviewed_tracks
        try:
            reviewed=review(inputs,context)
            if reviewed is not None:
                review_path,decisions=reviewed
                review_artifacts.append(artifact(context,'rigid_motion_review',review_path))
                candidate_inputs,tracking_path=prepare_reviewed_tracks(inputs,context,decisions)
                candidate_reason=eligibility(candidate_inputs)
                review_artifacts.append(artifact(context,'motion_candidate_tracking',tracking_path))
                motion_review_adopted=candidate_reason is None
                write(context.output_dir/'motion_review_selection.json',dict(
                    adopted=motion_review_adopted,original_rejection=reason,
                    candidate_rejection=candidate_reason))
                if motion_review_adopted:
                    inputs=candidate_inputs
                    inputs['effective_tracking_asset_id']=review_artifacts[-1]['artifact_id']
                    reason=None
        except Exception as exc:
            result['error']=dict(code='motion_review_failed',message=f'{type(exc).__name__}: {exc}')
            return result
    if reason and inputs['tracking']['route']=='3D+SAM3':
        from atw.world_modeling.swr_admission_review import review as review_admission
        try:
            inputs,reason,retained,admission_review_record=review_admission(inputs,context,reason)
            review_artifacts.extend(retained)
        except Exception as exc:
            result['error']=dict(code='admission_review_failed',message=f'{type(exc).__name__}: {exc}')
            return result
    engine=inputs['engine'];available=reason is None
    manifest=dict(protocol_version=1,engine=engine,route=inputs['tracking']['route'],availability='unavailable',
        video_path=context.inputs['video_path'],sources=arguments,reason=reason or 'Eligible for fitting.',
        world=dict(kind='observed_baseline',intervention=None),result_asset_id=None,metrics={},limitations=[],
        observation_scope=dict(frame_count=inputs['bundle'][0]['frame_count'],fps=inputs['bundle'][0]['fps']) if 'bundle' in inputs else {k:inputs['tracking'][k] for k in ('frame_count','fps')})
    if gravity_correction is not None:manifest['gravity_correction']=gravity_correction
    artifacts=list(review_artifacts)
    motion_artifacts=[a for a in review_artifacts if a['artifact_id'] in (
        context.call_id+'_rigid_motion_review',context.call_id+'_motion_candidate_tracking')]
    if motion_artifacts:
        manifest['motion_review']={a['artifact_id'].removeprefix(context.call_id+'_'):a['artifact_id'] for a in motion_artifacts}
        manifest['motion_review']['adopted']=motion_review_adopted
    if admission_review_record is not None:
        manifest['admission_review']=dict(artifact_id=context.call_id+'_admission_review',
            outcome=admission_review_record['outcome'],next_action=admission_review_record['next_action'],
            stage_one_status=admission_review_record['stage_one_status'])
    if inputs.get('effective_tracking_asset_id'):
        manifest['effective_sources']=dict(tracking_asset_id=inputs['effective_tracking_asset_id'])
    write(context.output_dir/'eligibility.json',dict(engine=engine,eligible=available,reason=manifest['reason']))
    if available:
        try:
            backend=getattr(importlib.import_module('.swr_backends',__package__),engine)
            path,metrics,limitations=backend(inputs,context)
            fit_artifact=artifact(context,'swr_fit',path);fit_artifact['source_kind']='simulation';artifacts.append(fit_artifact)
            manifest.update(availability='fitted',reason='Fitted the retained observed baseline; fit quality is reported in metrics.',
                result_asset_id=fit_artifact['artifact_id'],metrics=metrics,limitations=limitations,
                coordinate_frame=('Source-binding pixel grid; x right, y down' if engine=='fluid' else inputs['tracking']['coordinate_frame']),
                units=('source-binding pixels and seconds' if engine=='fluid' else 'image_height and seconds; gravity in image_height/s^2' if engine=='rope' else 'reconstruction length units and seconds; length scale is model-estimated'))
            if gravity_correction is not None and gravity_correction['adopted']:
                manifest['limitations'].append('Gravity direction was revised by VLM and adopted after passing SWR admission; see gravity_admission_correction.json for the original estimate.')
            if engine=='rope':
                manifest['fixed_inputs']=dict(gravity_normalized_per_s2=context.scene.rope_gravity_normalized_per_s2)
                coverage_path=context.output_dir/'rope_coverage.json'
                if coverage_path.exists():
                    coverage=json.loads(coverage_path.read_text())
                    artifacts.append(artifact(context,'rope_coverage',coverage_path))
                    if coverage['failed_groups']:
                        manifest['reason']='Partial SWR fit; uncontrolled rope groups: '+', '.join(coverage['failed_groups'])
        except Exception as exc:
            result['error']=dict(code='execution_failed',message=f'{type(exc).__name__}: {exc}');return result
    path=context.output_dir/'swr.json';write(path,manifest)
    art=artifact(context,'swr',path);artifacts.append(art)
    if admission_review_record is not None and not available:
        from atw.contracts import stage_result
        failure_path=context.output_dir/'stage_one_failure.json'
        write(failure_path,stage_result('stage_one','failed',
            data=dict(scene=dict(swr_asset_id=art['artifact_id'],direct_video_fallback=True)),
            artifacts=list(artifacts),error=dict(code='swr_admission_unresolved',message=manifest['reason'])))
        artifacts.append(artifact(context,'stage_one_failure',failure_path))
    result.update(status='ok',data=validate_output(dict(engine=engine,availability=manifest['availability'],swr_asset_id=art['artifact_id'],reason=manifest['reason'])),artifacts=artifacts)
    return result
