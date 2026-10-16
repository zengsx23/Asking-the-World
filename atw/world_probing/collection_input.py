"""Compact model views of retained calls; canonical tool records stay unchanged."""
import re

from atw.contracts import json_copy


def model_error(error):
    """Keep actionable errors, with local filesystem locations confined to logs."""
    value=json_copy(error)
    value['message']=re.sub(r'''(?<![\w:/])/(?:[^\s'";,\)\]\}]+)''', '[local asset]', value['message'])
    return value


def identity_bindings(rows):
    return [{key:row[key] for key in ('reference','status','object_ids')} for row in rows]


def handoff_instruction(schema):
    statuses=schema['properties']['status']['enum']
    return ('Call finish_evidence_collection alone with arguments {"status":"'+statuses[0]+'","reason":"brief coverage or missing evidence"}. '
            'Allowed status values: '+', '.join(statuses)+'. No extra fields.')


class ToolFeedback:
    """Keep the first complete note and explicit references from later equal notes."""
    def __init__(self):
        self.notes={}

    def view(self,receipt):
        value=dict(call_id=receipt['call_id'],status=receipt['status'])
        if receipt['status']=='error':
            value['error']=model_error(receipt['error'])
            return value
        data=json_copy(receipt['data'])
        note=data.get('note')
        if note:
            if note in self.notes:
                del data['note']
                data['note_from_call_id']=self.notes[note]
            else:self.notes[note]=receipt['call_id']
        value['data']=data
        attachments=[a for a in receipt['artifacts'] if a['media_type'].startswith(('image/','video/'))]
        if attachments:
            value['artifacts']=[{k:a[k] for k in ('artifact_id','media_type','source_kind')} for a in attachments]
        return value
