"""Expose retained Rope control topology without simulating or changing identities."""
from atw.world_probing.engines.common import TEXT, IDS, fields
from atw.world_modeling import sam3_protocol as s

NULLABLE_TEXT = dict(type=['string', 'null'])
SYSTEM = s.obj(rope_id=TEXT, intervention_supported=dict(type='boolean'),
               reason=NULLABLE_TEXT, mass_model=TEXT, massless_object_ids=IDS)
ENDPOINT = s.obj(rope_id=TEXT, endpoint_index=dict(type='integer', enum=[0, 1]),
                 pinned=dict(type='boolean'), object_ids=IDS)
MASS = fields(['rope_id', 'object_id', 'members', 'role', 'mass_scope'],
              rope_id=TEXT, object_id=NULLABLE_TEXT, members=IDS, role=TEXT,
              mass_scope=dict(type='string', enum=['body', 'assembly']), mass_source=TEXT,
              relative_mass=dict(type='number'), motion_constraint=TEXT, parent_id=NULLABLE_TEXT)
PULLEY = s.obj(rope_id=TEXT, object_id=TEXT, center_dynamic=dict(type='boolean'))
EVIDENCE = s.obj(systems=s.array(SYSTEM), endpoints=s.array(ENDPOINT),
                 mass_controls=s.array(MASS), pulleys=s.array(PULLEY), unmodeled_object_ids=IDS)


def definition(engine):
    def execute(args, context):
        raw = engine.backend.query_rope_controls()
        systems, endpoints, masses, pulleys = [], [], [], []
        for rope in raw['ropes']:
            rid = rope['rope_id']
            systems.append(dict(rope_id=rid, intervention_supported=rope['intervention_supported'],
                reason=rope['reason'], mass_model=rope['mass_model'], massless_object_ids=rope['massless_objects']))
            endpoints.extend(dict(rope_id=rid, **{k:row[k] for k in ENDPOINT['properties'] if k!='rope_id'})
                             for row in rope['endpoints'])
            masses.extend({k:v for k,v in row.items() if k in MASS['properties']} for row in rope['mass_controls'])
            pulleys.extend(dict(rope_id=rid, **{k:row[k] for k in PULLEY['properties'] if k!='rope_id'})
                           for row in rope['pulleys'])
        evidence = dict(systems=systems, endpoints=endpoints, mass_controls=masses,
                        pulleys=pulleys, unmodeled_object_ids=raw['unmodeled_object_ids'])
        return engine.emit(context, evidence,
            'Retained fitted topology, not motion or outcome evidence. Endpoint 0/1 follows material-node order, '
            'not left/right. A pinned endpoint remains fixed unless explicitly driven. '
            'Use the exact question-bound body for mass edits; a combined assembly mass cannot replace an '
            'unidentified member mass. default_unidentified is an assumed fixed-load mass, not a fitted measurement. '
            'Massless pulley inertia is not an editable load mass. Unsupported coupled systems require missing-evidence handling. '
            'Choosing a control does not revise identity bindings.', raw=raw)
    return engine.definition('query_rope_controls',
        'List retained rope endpoints, attached-object bindings, effective mass controls and pulley IDs. '
        'Read before choosing an intervention; no simulation. This catalog does not establish motion or rotation.',
        s.obj(), EVIDENCE, execute)
