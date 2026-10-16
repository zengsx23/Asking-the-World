"""Conditional 3D admission repair; reviewed observations remain read-only."""
import copy
import json
from pathlib import Path

from atw.world_modeling import sam3_protocol as c
from atw.world_modeling.scene_assets import artifact, write
from atw.world_modeling.physical_classification import classification_schema, selection_schema

ROOT = Path(__file__).resolve().parents[2]
PROMPT = Path(__file__).resolve().parents[1]/'prompts/swr_admission_physical_review.txt'


def gravity_key(gravity):
    value = gravity['gravity']
    decision = value.get('rigid_surface_decision', {})
    return dict(scene_surface_type=value.get('scene_surface_type', decision.get('scene_surface_type')),
        all_rigid_objects_share_one_surface=decision.get('all_rigid_objects_share_one_surface', value.get('status')=='completed'))


def requirements(inputs):
    from atw.world_modeling.scene_swr import ENGINES
    key = gravity_key(inputs['gravity'])
    return [dict(engine=r['name'], required_physical_types=sorted(r['required']),
                 allowed_physical_types=sorted(r['allowed']),
                 rigid_obstacles_must_be_tracked_static=r['name'] in ('cloth','softbody'))
            for r in ENGINES if r['route']==inputs['tracking']['route']
            and key['all_rigid_objects_share_one_surface']
            and key['scene_surface_type']==('wall_plane' if r['name']=='softbody' else 'support_plane')]


def media(path):
    from atw.perception.sam3_agent import _media_data_url
    kind, url = _media_data_url(Path(path))
    return dict(type=kind, **{kind:dict(url=url)})


def render(inputs, context, output):
    from atw.world_modeling.physical_classification_impl import render_evidence
    _, inventory, index, archive, _ = inputs['bundle']
    output.mkdir(parents=True)
    paths = render_evidence(Path(context.inputs['video_path']), inventory, index, archive, output)
    return {obj['track_id']:path for obj,path in zip(inventory['objects'], paths)}


def repeat_surface(inputs, context, images, output):
    from PIL import Image
    from atw.perception.sam3_agent import GeminiSam3Agent, RIGID_SURFACE_SYSTEM, RIGID_SURFACE_INSTRUCTION
    output.mkdir(parents=True)
    tiles = []
    for path in images.values():
        with Image.open(path) as image:
            tile = image.convert('RGB')
            tile.thumbnail((640,200))
            tiles.append(tile)
    sheet = Image.new('RGB', (1280,200*((len(tiles)+1)//2)), 'white')
    for i,tile in enumerate(tiles):
        sheet.paste(tile, ((i%2)*640,(i//2)*200))
    contact = output/'contact_sheet.png'
    sheet.save(contact)
    inventory = dict(objects=[dict(physical_type=o['physical_type'], category_noun=o['category_noun']) for o in inputs['objects']])
    write(output/'input.json', inventory)
    agent = GeminiSam3Agent(ROOT)
    agent.client.trace_dir = output/'requests'
    try:
        raw = agent.request_json(operation='rigid_surface_initial', call_id=context.call_id,
            system=RIGID_SURFACE_SYSTEM,
            instruction=RIGID_SURFACE_INSTRUCTION.replace('{inventory}',json.dumps(inventory)),
            media_paths=[Path(context.inputs['video_path']), contact], validate=agent._normalize_rigid_surface_decision)
        decision = agent._normalize_rigid_surface_decision(raw)
        write(output/'decision.json',decision)
        return decision
    finally:
        write(output/'usage.json', agent.usage_report())


def select_questions(context, output):
    from atw.perception.object_material_classifier import ObjectMaterialClassifier
    from atw.world_modeling.physical_classification_impl import request
    output.mkdir(parents=True)
    questions = [{k:q[k] for k in ('id','text','options')} for q in context.inputs['questions']]
    write(output/'input.json',dict(questions=questions))
    if not questions:
        write(output/'selected_questions.json',[])
        return []
    client = ObjectMaterialClassifier(ROOT).client
    client.trace_dir = output/'requests'
    schema = selection_schema([q['id'] for q in questions])
    try:
        selected = request(client,schema,operation='physical_question_selection',call_id='selection',
            prompt_name='physical_question_selection_system.txt',
            content=[dict(type='text',text=json.dumps(dict(questions=questions))+c.prompt(schema))],
            output=output,filename='selection.json',key='selected',id_key='question_id')
    finally:
        write(output/'usage.json',client.usage_report())
    ids = {q['question_id'] for q in selected['selected']}
    kept = [q for q in questions if q['id'] in ids]
    write(output/'selected_questions.json',kept)
    return kept


def classify_batch(inputs, context, labels, updates, batch, questions, images, output):
    from atw.perception.object_material_classifier import ObjectMaterialClassifier
    output.mkdir(parents=True)
    ids = [o['track_id'] for o in batch]
    schema = classification_schema(ids)
    table = [dict(track_id=o['track_id'],physical_type=o['physical_type'],
                  **({'review_update':updates[o['track_id']]} if o['track_id'] in updates else {})) for o in labels]
    payload = dict(route=inputs['tracking']['route'],gravity=gravity_key(inputs['gravity']),objects=table,
                   batch_track_ids=ids,possible_engines=requirements(inputs),questions=questions)
    write(output/'input.json',payload)
    write(output/'schema.json',schema)
    prompt = PROMPT.read_text()
    (output/'system.txt').write_text(prompt)
    content = [dict(type='text',text=json.dumps(payload)+c.prompt(schema)),media(context.inputs['video_path'])]
    content.extend(media(images[tid]) for tid in ids)
    def validate(value):
        value,_ = c.normalize_response(schema,value)
        c.validate(schema,value,'reclassification')
        c.unique_ids(value['objects'],'track_id','reclassification')
    client = ObjectMaterialClassifier(ROOT).client
    client.trace_dir = output/'requests'
    try:
        raw = client.request_json(operation='physical_type_admission_review',call_id=output.name,
            system=prompt,content=content,validate=validate)
        value,_ = c.normalize_response(schema,raw)
        validate(value)
        write(output/'output.json',value)
        return value['objects']
    finally:
        write(output/'usage.json',client.usage_report())


def preparation_rejection(inputs, labels, gravity):
    """Check known prerequisites before rebuilding changed representations."""
    import numpy as np
    from atw.world_modeling.scene_swr import select_engine, eligibility
    engine = select_engine(inputs['tracking']['route'], [o['physical_type'] for o in labels])
    probe = dict(inputs,objects=labels,gravity=gravity,engine=engine)
    # Native gate first checks route, surface and gravity without opening tracks.
    reason = eligibility(dict(probe,objects=[]))
    if reason or engine not in ('cloth','softbody'):
        return reason
    old = {o['object_id']:o['physical_type'] for o in inputs['objects']}
    for obj in labels:
        if obj['physical_type']=='rigid' and old[obj['object_id']]=='rigid':
            with np.load(inputs['tracking_directory']/'rigid_trajectories'/f"{obj['object_id']}.npz",allow_pickle=False) as z:
                if str(z['motion_model'].item())!='static':
                    return f'{engine.capitalize()} SWR requires tracked-static rigid obstacles.'
    return None


def attempt_candidate(inputs, context, labels, output, surface_decision=None):
    from atw.world_modeling.scene_swr import eligibility
    from atw.world_modeling.swr_review_inputs import prepare_candidate
    try:
        candidate, artifacts = prepare_candidate(inputs,context,labels,output,surface_decision)
        return candidate, eligibility(candidate), artifacts
    except Exception as exc:
        # A failed candidate never invalidates or replaces its retained source.
        path = output/'candidate_failure.json'
        reason = f'{type(exc).__name__}: {exc}'
        write(path,dict(code='candidate_reconstruction_failed',message=reason,adopted=False))
        return inputs,reason,[artifact(context,'admission_'+output.name+'_failure',path)]


def review(inputs, context, reason):
    """Run after the motion checkpoint; adopt only fully rebuilt, admitted inputs."""
    if not reason or inputs['tracking']['route']!='3D+SAM3':
        return inputs, reason, [], None
    output = context.output_dir/'admission_review'
    output.mkdir()
    original = inputs
    labels = copy.deepcopy(inputs['objects'])
    artifacts, steps = [], []
    images = render(inputs,context,output/'evidence')
    decision = repeat_surface(inputs,context,images,output/'gravity')
    key = {k:decision[k] for k in ('scene_surface_type','all_rigid_objects_share_one_surface')}
    same = key==gravity_key(inputs['gravity'])
    gravity_record = dict(step='gravity',same_key=same,adopted=False,candidate_key=key)
    if not same:
        # A semantic probe is never published or used by a fitter as numerical gravity.
        probe = copy.deepcopy(inputs['gravity'])
        probe['gravity'].update(scene_surface_type=key['scene_surface_type'],rigid_surface_decision=decision,
                                status='completed' if key['all_rigid_objects_share_one_surface'] else 'skipped')
        candidate_reason = preparation_rejection(inputs,labels,probe)
        if candidate_reason is None:
            candidate,candidate_reason,retained = attempt_candidate(inputs,context,labels,output/'gravity_candidate',decision)
            artifacts.extend(retained)
            if candidate_reason is None:
                inputs,reason = candidate,None
                gravity_record['adopted'] = True
        gravity_record['candidate_rejection'] = candidate_reason
    steps.append(gravity_record)
    if reason:
        questions = select_questions(context,output/'question_selection')
        updates = {}
        for start in range(0,len(labels),5):
            batch = labels[start:start+5]
            index = start//5+1
            answer = classify_batch(inputs,context,labels,updates,batch,questions,images,output/f'physical_{index:02d}')
            by_track = {o['track_id']:o for o in answer}
            for obj in batch:
                proposal = by_track[obj['track_id']]
                changed = proposal['physical_type']!=obj['physical_type']
                updates[obj['track_id']] = proposal['reason'] if changed else 'consistent'
                obj.update(physical_type=proposal['physical_type'],physical_type_reason=proposal['reason'])
            candidate_reason = preparation_rejection(inputs,labels,inputs['gravity'])
            if candidate_reason is None:
                candidate,candidate_reason,retained = attempt_candidate(inputs,context,labels,output/f'physical_candidate_{index:02d}')
                artifacts.extend(retained)
                if candidate_reason is None:
                    inputs,reason = candidate,None
            steps.append(dict(step='physical',batch=index,track_ids=[o['track_id'] for o in batch],
                              candidate_rejection=candidate_reason,adopted=reason is None))
            if reason is None:
                break
    record = dict(protocol_version=1,route='3D+SAM3',steps=steps,
        outcome='admission_recovered' if reason is None else 'no_eligible_engine',
        objects=[{k:o[k] for k in ('object_id','track_id','physical_type')} for o in inputs['objects']],
        unadopted_objects=None if reason is None else [{k:o[k] for k in ('object_id','track_id','physical_type')} for o in labels],
        original_sources=original['tracking']['sources'],effective_sources=inputs['tracking']['sources'],
        next_action='fit_scene_swr' if reason is None else 'direct_video_answer',
        stage_one_status='continue' if reason is None else 'failed')
    path = output/'review.json'
    write(path,record)
    artifacts.append(artifact(context,'admission_review',path))
    return inputs,reason,artifacts,record
