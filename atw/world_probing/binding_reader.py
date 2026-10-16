"""Normalize answer-preserving presentation differences before canonical validation."""
from atw.world_probing.identities import BINDING_SCHEMA
from atw.world_modeling.sam3_protocol import validate


def binding_record(value):
    """Accept a bare array of complete binding rows at the provider boundary."""
    fields={'reference','status','object_ids','evidence'}
    if isinstance(value,list) and all(isinstance(row,dict) and fields <= set(row) for row in value):
        return dict(identity_evidence=value)
    return value


def read_binding(value, allowed_ids):
    changes=[]
    if set(value)=={'data'} and isinstance(value['data'],dict):
        value=value['data'];changes.append('unwrap_data')
    if 'identity_evidence' not in value:
        raise ValueError('Missing identity_evidence')
    if set(value)!={'identity_evidence'}:
        changes.append('retain_answer_fields')
    rows=[];seen={}
    for original in value['identity_evidence']:
        row={k:original[k] for k in ('reference','status','object_ids','evidence')}
        row['reference']=row['reference'].strip()
        row['status']=row['status'].strip().lower()
        if isinstance(row['object_ids'],str):
            row['object_ids']=[row['object_ids']];changes.append('singleton_id_array')
        if not isinstance(row['object_ids'],list) or any(not isinstance(i,str) for i in row['object_ids']):
            raise ValueError('Object IDs must be strings')
        ids=list(dict.fromkeys(row['object_ids']))
        if ids!=row['object_ids']:changes.append('deduplicate_ids')
        row['object_ids']=ids
        if not set(ids)<=allowed_ids:
            raise ValueError('Binding references an unknown object ID')
        if (row['status']=='found')!=bool(ids):
            raise ValueError('found requires IDs; not_found_in_scene requires an empty list')
        key=' '.join(row['reference'].casefold().split())
        if key.startswith('the '):
            key=key[4:]
        if key in seen:
            prior=seen[key]
            if prior['status']!=row['status'] or set(prior['object_ids'])!=set(ids):
                raise ValueError('Repeated reference has conflicting decisions')
            changes.append('merge_equivalent_reference');continue
        seen[key]=row;rows.append(row)
    result=dict(identity_evidence=rows)
    validate(BINDING_SCHEMA,result,'response')
    return result,changes
