"""Compact, explicit Fluid outcomes from retained numerical measurements."""
from atw.world_probing.engines.common import fields, TEXT

MEASUREMENT = dict(type='string', enum=['container_entry', 'surface_contact'],
    description='container_entry for flow into a container; surface_contact for passing a stick.')
EVIDENCE = fields(['object_id', 'target_id', 'interval_seconds', 'measurement', 'present', 'fraction'],
    object_id=TEXT, target_id=TEXT,
    interval_seconds=dict(type='array', items=dict(type='number'), minItems=2, maxItems=2),
    measurement=MEASUREMENT, present=dict(type='boolean'),
    fraction=dict(type='number', minimum=0, maximum=1),
    final_fraction=dict(type='number', minimum=0, maximum=1),
    tolerance_pixels=dict(type='number', minimum=0))

LIMITATIONS = [
    '2D effective fluid model with one phase per observed source; source bindings and static masks affect outcomes.',
    'Fixed fitted parameters; predictions depend on the specified emission schedule.']


def summarize(value, measurement):
    """Keep existence separate from quantity; never discard a small positive result."""
    if measurement == 'container_entry':
        if value['container_status'] != 'measured':
            raise ValueError('Container entry requires a target with measurable interior geometry')
        fraction = value['entered_fraction']
        evidence = dict(measurement=measurement, present=fraction > 0, fraction=fraction,
                        final_fraction=value['final_fraction'])
        note = ('present reports any particle entry; fraction counts distinct entered particles / all target particles born by interval end. '
                'final_fraction counts only contents at interval end. For "which containers", include every positive entry, even a small fraction. '
                'Only "most" requires fraction > 0.5; it does not mean the largest option.')
    elif measurement == 'surface_contact':
        contact = value['upper_surface_contact'][0]
        evidence = dict(measurement=measurement, present=contact['unique_particles'] > 0,
                        fraction=contact['fraction'], tolerance_pixels=contact['tolerance_pixels'])
        note = ('present reports any upper-surface contact within particle radius + 0.5 pixels, including transient contact; '
                'side, underside and endpoint grazing are excluded. fraction counts distinct contacting particles / all target particles born by interval end. '
                'For "which sticks", include every present contact; no majority or duration requirement.')
    else:
        raise ValueError('Select container_entry or surface_contact')
    finding = 'Entry' if measurement == 'container_entry' else 'Contact'
    return evidence, f"{finding}: {'present' if evidence['present'] else 'absent'} in the queried simulated interval. " + note
