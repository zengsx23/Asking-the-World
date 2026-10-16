"""Visual evidence for pulley roles."""
import json
from pathlib import Path
from atw.world_modeling import sam3_protocol as c
from atw.world_modeling.scene_assets import write


def identify(context, inventory, index, archive, objects):
    from atw.shared.scene_review import SceneReviewer, media_content, PAGE_SIZE
    from atw.world_modeling.physical_classification_impl import render_evidence
    rows = [o for o in objects if o['physical_type']=='rigid']
    output = context.output_dir/'pulley_roles'
    output.mkdir()
    if not rows:
        result = dict(pulley_ids=[], objects=[])
        write(output/'roles.json', result)
        return result
    selected = {o['object_id'] for o in rows}
    evidence = output/'evidence'; evidence.mkdir()
    images = render_evidence(Path(context.inputs['video_path']),
        dict(objects=[o for o in inventory['objects'] if o['object_id'] in selected]), index, archive, evidence)
    strands = [o for o in objects if o['physical_type']=='rope']
    strand_images = []
    if strands:
        strand_evidence = output/'strand_evidence'; strand_evidence.mkdir()
        strand_ids = {o['object_id'] for o in strands}
        strand_images = render_evidence(Path(context.inputs['video_path']),
            dict(objects=[o for o in inventory['objects'] if o['object_id'] in strand_ids]),
            index, archive, strand_evidence)
    root = Path(__file__).resolve().parents[2]
    reviewer = SceneReviewer.from_project(root, output/'requests')
    client = reviewer.client
    client.trace_dir = output/'traces'
    decisions = []
    try:
        for start in range(0, len(rows), PAGE_SIZE):
            page = rows[start:start+PAGE_SIZE];ids = [o['object_id'] for o in page]
            schema = c.obj(objects=c.array(c.obj(object_id={**c.TEXT, 'enum': ids},
                role=dict(type='string', enum=['pulley','other','uncertain']),
                shape=dict(type='string', enum=['box','sphere','cylinder','irregular']), reason=c.TEXT), minItems=len(ids), maxItems=len(ids)))
            write(output/f'page_{start//PAGE_SIZE+1}_schema.json', schema)
            content = media_content('original_video', Path(context.inputs['video_path']))
            if strands:
                content.append(dict(type='text', text='Previously classified flexible strands (context only; do not return role entries for them): '+json.dumps(
                    [dict(track_id=o['track_id'],physical_type='rope') for o in strands])))
                for image in strand_images:content.extend(media_content('flexible_strand_'+image.stem, image))
            content.append(dict(type='text', text=json.dumps(dict(objects=[dict(object_id=o['object_id'],track_id=o['track_id']) for o in page]))+c.prompt(schema)))
            for image in images[start:start+PAGE_SIZE]:content.extend(media_content(image.stem, image))
            prompt = (Path(__file__).resolve().parents[1]/'prompts/swr_rope_roles.txt').read_text()
            def validate(value):
                c.validate(schema, value);c.unique_ids(value['objects'], 'object_id', 'pulley_roles')
            value = client.request_json(operation='rope_pulley_roles', call_id=f'{context.call_id}_page_{start//PAGE_SIZE+1}',
                system=prompt, content=content, validate=validate)
            validate(value)
            write(output/f'page_{start//PAGE_SIZE+1}_response.json', value)
            decisions.extend(value['objects'])
    finally:
        write(output/'usage.json', client.usage_report())
    result = dict(objects=decisions, pulley_ids=[o['object_id'] for o in decisions if o['role']=='pulley'])
    write(output/'roles.json', result)
    if any(o['role']=='uncertain' for o in decisions):
        raise ValueError('Unresolved pulley roles; inspect retained role evidence before fitting')
    return result
