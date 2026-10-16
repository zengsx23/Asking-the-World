"""Fit physical worlds from explicit observation inputs."""
from atw.world_modeling.scene_assets import write

MASSLESS_UNSUPPORTED = (
    'No explicit tangent route supports every dynamic pulley for massless fitting',
    'A free massless rope end needs an explicit load',
    'Unloaded massless moving pulley requires a coupled algebraic solver',
)


def fit_rope_with_fallback(scenes, source, args, calibration, output, fitter):
    """Keep usable massless groups; retry particles only after total capability failure."""
    from copy import copy
    import json
    attempts = []
    for model in ('massless', 'linear-density'):
        directory = output/'rope_attempts'/model
        directory.mkdir(parents=True)
        run_args = copy(args)
        run_args.mass_model = model
        run_args.output = directory/'rope.json'
        try:
            fitter(scenes, source, args=run_args, calibration=calibration)
        except ValueError as exc:
            attempts.append(dict(mass_model=model, status='failed', reason=str(exc)))
            write(output/'rope_attempts.json', dict(attempts=attempts))
            if model=='massless' and any(message in str(exc) for message in MASSLESS_UNSUPPORTED):
                continue
            raise
        payload = json.loads(run_args.output.read_text())
        if not isinstance(payload.get('ropes'), list) or not payload['ropes']:
            raise ValueError('Rope fitter produced no usable groups')
        failure_path = run_args.output.with_suffix('.failures.json')
        failures = json.loads(failure_path.read_text()) if failure_path.exists() else {}
        attempts.append(dict(mass_model=model, status='partial' if failures else 'completed',
            result=str(run_args.output.relative_to(output)), failed_groups=failures))
        write(output/'rope_attempts.json', dict(attempts=attempts))
        controlled = {b['object_id'] for r in payload['ropes'] for b in r['bodies']}
        affected = {name for row in failures.values() for name in row.get('object_ids', [])}
        write(output/'rope_coverage.json', dict(selected_result=str(run_args.output.relative_to(output)),
            failed_groups=failures, unmodeled_rope_ids=list(failures),
            uncontrolled_object_ids=sorted(affected-controlled), attempts=attempts))
        # A usable primary-model subset already satisfies partial-success semantics.
        # Do not replace it with a particle model merely to increase group coverage.
        return run_args.output, attempts


def rigid(inputs, context):
    from atw.polyworld.swr_sysid import CEMConfig, DEFAULT_PORTFOLIO_SEEDS, RefinementConfig, fit_scene, load_rotational_scene, result_payload
    scene, source = load_rotational_scene(inputs['tracking_directory'], reconstruction_dir=inputs['reconstruction_directory'])
    expected = {o['object_id'] for o in inputs['objects']}
    if set(scene.object_ids) != expected:
        raise ValueError('Rigid fitting assets do not cover the selected identities')
    configs = tuple(CEMConfig(seed=seed) for seed in DEFAULT_PORTFOLIO_SEEDS)
    refinement = RefinementConfig()
    mode = 'circle-shape-physical-response'
    fit, selected, elapsed = fit_scene(scene, cem_configs=configs, refinement_config=refinement,
                                      device=context.scene.device, proxy_mode=mode)
    payload = result_payload(source, scene, selected, refinement, fit, elapsed, context.scene.device, mode)
    path = context.output_dir/'rigid.json'
    write(path, payload)
    metrics = {k: payload[k] for k in ('overall_point_rmse_m', 'overall_translation_rmse_m', 'overall_yaw_rmse_degrees')}
    return path, metrics, ['Fixed object heights; no airborne motion.', 'Collision proxies, not full mesh contact.']


def cloth(inputs, context):
    from atw.polyworld.cloth_swr import DEFAULT_EXECUTION_MODE, DEFAULT_OBSTACLE_MODE, DEFAULT_SEEDS, ClothCEMConfig, fit_cloth_scene, load_cloth_scenes, result_payload
    scenes, source = load_cloth_scenes(inputs['tracking_directory'], grid_resolution=16, target_depth_mode='raw')
    expected = {o['object_id'] for o in inputs['objects'] if o['physical_type']=='cloth'}
    obstacles = {o['object_id'] for o in inputs['objects'] if o['physical_type']=='rigid'}
    if {s.object_id for s in scenes} != expected:
        raise ValueError('Cloth fitting assets do not cover the selected identities')
    options = dict(device=context.scene.device, substeps=3, constraint_iterations=4, support_mode='initial-above',
                   boundary_mode='edge-drive', obstacle_mode=DEFAULT_OBSTACLE_MODE, edge_stop_frame=None, edge_release_frame=None)
    configs = tuple(ClothCEMConfig(seed=seed) for seed in DEFAULT_SEEDS)
    results = {}
    for scene in scenes:
        if {o.object_id for o in scene.obstacles} != obstacles:
            raise ValueError('Cloth obstacle assets do not cover the selected rigid identities')
        fit, selected, elapsed = fit_cloth_scene(scene, configs=configs, execution_mode=DEFAULT_EXECUTION_MODE, **options)
        results[scene.object_id] = result_payload(source, scene, selected, fit, elapsed, **options)
    path = context.output_dir/'cloth.json'
    write(path, dict(model='physmind-v2-warp-reduced-cloth-swr-v14', source=source, object_ids=list(results), objects=results))
    metrics = {name: {key: item[key] for key in ('vertex_rmse_m', 'raw_vertex_rmse_m', 'centroid_rmse_m', 'shape_rmse_m', 'edge_length_rmse_m')}
               for name, item in results.items()}
    return path, metrics, ['Cloths fitted separately; no cloth self-collision or cloth-cloth contact.',
        'Shared obstacles have separate per-cloth predictions, not a joint scene rollout.',
        'Tracked-static obstacle initialization; predicted obstacles may move under cloth contact.', 'Fitted edge-drive boundary condition.']


def rope(inputs, context):
    import argparse
    import json
    from atw.polyworld.rope_swr import prepare_rope_scenes
    from atw.polyworld.rope_gravity import require_gravity
    from atw.polyworld.rope_fitting import build_parser, fit_prepared_scenes
    from atw.world_modeling.swr_rope_roles import identify
    manifest, inventory, index, archive, objects = inputs['bundle']
    value = require_gravity(context.scene.rope_gravity_normalized_per_s2)
    roles = identify(context, inventory, index, archive, objects)
    calibration = dict(gravity_normalized_per_s2=value, units='image_height/s^2', source='fixed_engine_input',
                       image_wh=[manifest['width'], manifest['height']])
    write(context.output_dir/'rope_gravity.json', calibration)
    shapes = {o['object_id']:o['shape'] for o in roles['objects']}
    source = dict(video=dict(path=manifest['video_path'], frame_count=manifest['frame_count'],
                            width=manifest['width'], height=manifest['height'], fps=manifest['fps']),
                  objects=[dict(target_id=o['object_id'], track_id=o['track_id'], physical_type=o['physical_type'],
                                **(dict(rigid_shape=shapes[o['object_id']]) if o['physical_type']=='rigid' else {})) for o in objects])
    from atw.perception.object_reconstruction import read_packed_mask_archive
    from atw.world_modeling.rope_pulleys import constrain_static_pulleys
    tracks = read_packed_mask_archive(archive, inputs['segmentation_path'].parent/manifest['mask_index'])
    overrides, proxy_records = constrain_static_pulleys(source, tracks, roles)
    source['pulley_radius_overrides_px'] = overrides
    source['pulley_proxy_constraints'] = proxy_records
    write(context.output_dir/'rope_pulley_proxies.json', dict(constraints=proxy_records))
    scenes, source = prepare_rope_scenes(source, mask_archive=archive,
        mask_index=inputs['segmentation_path'].parent/manifest['mask_index'], node_count=96, roles=roles)
    from atw.world_modeling.rope_connectors import find_local_attachment_connectors, without_attachment_connectors
    connectors = find_local_attachment_connectors(scenes, source['objects'], tracks)
    if connectors:
        source = without_attachment_connectors(source, connectors)
        # Rebuild endpoints and fitting observations without connector masks as
        # bodies or occluders, exactly as when no separate connector was detected.
        scenes, source = prepare_rope_scenes(source, mask_archive=archive,
            mask_index=inputs['segmentation_path'].parent/manifest['mask_index'],
            node_count=96, roles=source['rope_roles'])
    write(context.output_dir/'rope_connectors.json',
          dict(ignored_local_connectors=source.get('ignored_local_connectors', [])))
    from atw.world_modeling.rope_endpoints import stabilize_static_terminals
    scenes, terminals = stabilize_static_terminals(scenes, tracks)
    source['static_terminal_regions'] = terminals
    write(context.output_dir/'rope_static_terminals.json', dict(terminals=terminals))
    if not scenes:
        raise ValueError('No usable rope observations')
    write(context.output_dir/'rope_observations.json', source)
    # Read solver defaults without parsing command-line arguments.
    parser = build_parser()
    args = argparse.Namespace(**{a.dest: a.default for a in parser._actions if a.dest!='help'})
    args.output = context.output_dir/'rope.json'
    args.device = context.scene.device
    args.gravity = value
    args.quiet = True
    result_path, attempts = fit_rope_with_fallback(scenes, source, args, calibration, context.output_dir, fit_prepared_scenes)
    payload = json.loads(result_path.read_text())
    metrics = {r['rope_id']: r['metrics'] for r in payload['ropes']}
    return result_path, metrics, payload['limitations'] + ([
        'Static pulley radii were jointly reduced to remove overlap between circular proxies of disjoint masks; these are approximate contact radii.'
    ] if proxy_records else []) + ([
        'Massless capability fallback: linear-density particle model used; see rope_attempts.json for the audit.'
    ] if payload.get('mass_model')=='linear-density' else []) + ([
        'Local attachment connectors retain observed identities but have no independent dynamics or occlusion role: '
        + ', '.join(record['object_id'] for record in connectors)
    ] if connectors else []) + [
        'Image-space parameters are not SI material constants.',
        'Arbitrary enclosing-boundary contact is not modeled; no boundary asset was supplied.',
    ] + (['Unbound endpoints with persistent local mask evidence are held at observed fixed terminal regions.'] if terminals else [])


def fluid(inputs, context):
    from atw.world_modeling.fluid_route import fit
    return fit(inputs, context)


def softbody(inputs, context):
    from atw.world_modeling.softbody_observed_radius import fit
    return fit(inputs, context)
