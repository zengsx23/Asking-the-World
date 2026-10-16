"""Explicit, same-run asset references for scene Engineers."""
import json
from atw.world_modeling.physical_classification import validate_output
from atw.contracts import tool_call_directory_name


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def asset_path(context, asset_id):
    row = next((a for a in context.artifacts if a['artifact_id']==asset_id), None)
    if row is None: raise ValueError(f'Unknown asset ID: {asset_id}')
    path=(context.run_dir/row['path']).resolve()
    if not path.is_relative_to(context.run_dir.resolve()): raise ValueError('Asset must belong to this run')
    return path


def artifact(context, kind, path, media_type='application/json'):
    return dict(artifact_id=f'{context.call_id}_{kind}', path=str(path.relative_to(context.run_dir)),
                media_type=media_type, source_kind='derived')


def route_result(context, call_id):
    import re
    from atw.world_modeling.route_selection import validate_output as validate_route
    if not re.fullmatch(r'call_[0-9]{4,}',call_id): raise ValueError('Invalid routing_call_id')
    path=context.run_dir/'stage_one/tools'/tool_call_directory_name(call_id, 'world_representation_routing')/'result.json'
    result=json.loads(path.read_text())
    if result['call_id']!=call_id or result['tool_name']!='world_representation_routing' or result['status']!='ok':
        raise ValueError('routing_call_id must reference a successful routing call')
    return validate_route(result['data'])['route']


def classified_assets(context, segmentation_id, classification_id):
    from atw.world_modeling.physical_classification_impl import load_assets
    manifest,inventory,index,archive=load_assets(segmentation_id,context)
    types=json.loads(asset_path(context,classification_id).read_text());validate_output(types)
    expected={(o['object_id'],o['track_id']) for o in inventory['objects']}
    if {(o['object_id'],o['track_id']) for o in types['objects']}!=expected:
        raise ValueError('Physical classification must match the segmentation inventory exactly')
    byid={o['object_id']:o for o in types['objects']}
    objects=[dict(o,physical_type=byid[o['object_id']]['physical_type'],
                  physical_type_reason=byid[o['object_id']]['reason']) for o in inventory['objects']]
    return manifest,inventory,index,archive,objects
