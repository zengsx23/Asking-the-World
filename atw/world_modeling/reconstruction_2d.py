"""Retain planar SAM3 observations; geometric targets belong to SWR preparation."""


def reconstruct(context, bundle):
    manifest, inventory, index, archive, objects = bundle
    return dict(
        representation='observed_planar_masks',
        coordinate_frame='original video pixels; x right, y down',
        fps=manifest['fps'],
        limitations=[
            'Observed masks are retained in sources.segmentation_asset_id; no new tracks are inferred.',
            'Rigid centers and rope centerlines are deferred to rope SWR input preparation.',
            'No SWR fitting was performed.',
        ],
    )
