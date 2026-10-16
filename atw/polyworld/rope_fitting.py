#!/usr/bin/env python3
"""Fit rope routing and dynamics from raw observations (the default rope backend)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atw.polyworld.rope_swr import DEFAULT_EXECUTION_MODE, DEFAULT_MASSLESS_SUBSTEPS, ROPE_EXECUTION_MODES, RopeCEMConfig, result_payload
from atw.polyworld.rope_topology import VisibleRopeLoss, DEFAULT_LENGTH_PRIOR_WEIGHT, DEFAULT_MASS_MODEL, available_pulleys, fit_route_candidates, load_observations, propose_routes, read_observation_tracks, select_observed_route, screen_route_candidates


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("segmentation_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--roles-json", type=Path, help="Visual pulley-role decisions from the pre-SWR check")
    parser.add_argument("--device", default="cuda:0")
    gravity = parser.add_mutually_exclusive_group()
    gravity.add_argument("--gravity", type=float, help="Fixed calibrated acceleration in image heights/s^2; never optimized")
    gravity.add_argument("--gravity-calibration", type=Path, help="Calibration JSON; otherwise look beside segmentation for gravity_calibration.json")
    parser.add_argument("--execution-mode", choices=ROPE_EXECUTION_MODES, default=DEFAULT_EXECUTION_MODE,
                        help="auto (default): ideal-warp for massless models, guarded-chunked for particle models")
    parser.add_argument("--nodes", type=int, default=96)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument(
        "--route-selection",
        choices=("screen", "geometry", "fit-all"),
        default="screen",
        help="Fixed-bank forward screening then one CEM fit (default); geometry skips screening; fit-all fits every route.",
    )
    parser.add_argument("--screen-samples", type=int, default=64)
    parser.add_argument("--fit-initial-speed", action="store_true",
                        help="Explicit moving-initial-state hypothesis; default initial rope/load velocity is zero")
    parser.add_argument("--max-visits", type=int, default=5)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--substeps", type=int, default=DEFAULT_MASSLESS_SUBSTEPS)
    parser.add_argument("--screen-substeps", type=int,
                        help="route-screening resolution; defaults to --substeps")
    parser.add_argument("--observation-stride", type=int, default=6)
    parser.add_argument(
        "--length-prior-weight", type=float, default=DEFAULT_LENGTH_PRIOR_WEIGHT
    )
    parser.add_argument(
        "--mass-model",
        choices=("linear-density", "uniform-node", "massless"),
        default=DEFAULT_MASS_MODEL,
        help="Massless rope/pulleys with load-only inertia (default); particle mass models require explicit selection.",
    )
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--rope-id")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def retain_independent_fits(outputs, failures):
    """A missing coupled dynamic group cannot silently unlock an independent control."""
    unavailable = {name for row in failures.values() for name in row['dynamic_object_ids']}
    retained = list(outputs)
    while True:
        affected = [item for item in retained if unavailable.intersection(
            b['object_id'] for b in item['bodies'] if b['dynamic'])]
        if not affected:
            return retained
        for item in affected:
            dynamic = [b['object_id'] for b in item['bodies'] if b['dynamic']]
            failures[item['rope_id']] = dict(
                reason='Coupled to a failed rope group through dynamic bodies: ' +
                       ', '.join(sorted(unavailable.intersection(dynamic))),
                object_ids=[b['object_id'] for b in item['bodies']],
                dynamic_object_ids=dynamic)
            unavailable.update(dynamic)
            retained.remove(item)


def fit_prepared_scenes(scenes, source, *, args, calibration=None):
    """Fit rope topology to explicitly prepared observations."""
    screen_substeps = args.substeps if args.screen_substeps is None else args.screen_substeps
    tracks = read_observation_tracks(scenes[0])
    baseline = json.loads(args.baseline_json.read_text()) if args.baseline_json else {}
    config = RopeCEMConfig(
        population=args.population,
        generations_per_prefix=args.generations,
        seed=args.seed,
    )
    outputs, reports, failures = [], {}, {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for scene in scenes:
        try:
            rope_start = time.perf_counter()
            obs = load_observations(scene, tracks, source, stride=args.observation_stride)
            circles = available_pulleys(scene, tracks, source)
            candidates, search = propose_routes(
                scene, circles, obs, keep=args.keep, max_visits=args.max_visits
            )
            report = {
                "search": search,
                "candidates": [
                    {"name": c.name, "geometry_loss": c.geometry_loss, "length": c.length}
                    for c in candidates
                ],
            }
            fit_candidates = candidates
            if args.route_selection == "geometry":
                selected, selection = select_observed_route(
                    scene,
                    circles,
                    candidates,
                    obs,
                    length_prior_weight=args.length_prior_weight,
                )
                fit_candidates = [selected]
                report["geometry_selection"] = selection
            elif args.route_selection == "screen" and not args.geometry_only:
                selected, screening = screen_route_candidates(
                    scene,
                    circles,
                    candidates,
                    obs,
                    config=config,
                    device=args.device,
                    substeps=screen_substeps,
                    sample_count=args.screen_samples,
                    length_prior_weight=args.length_prior_weight,
                    mass_model=args.mass_model,
                    execution_mode=args.execution_mode,
                    gravity_normalized_per_s2=args.gravity,
                    fit_initial_speed=args.fit_initial_speed,
                )
                fit_candidates = [selected]
                screening["substeps"] = screen_substeps
                report["forward_screening"] = screening
            if not args.quiet:
                print(json.dumps({"rope": scene.rope_id, **report}), flush=True)
            # Diagnostic gallery: raw mask and inferred routes, never relabelled GT.
            tiles = []
            for candidate in candidates:
                canvas = np.full((scene.image_height, scene.image_width, 3), 248, np.uint8)
                mask = tracks[scene.rope_track_id][0]
                canvas[mask] = (160, 160, 160)
                for body in circles:
                    cv2.circle(
                        canvas,
                        tuple(
                            np.rint(
                                np.asarray(body.initial_center) * scene.image_height
                            ).astype(int)
                        ),
                        round(body.radius * scene.image_height),
                        (80, 80, 80),
                        2,
                    )
                points = np.rint(candidate.positions * scene.image_height).astype(np.int32)
                cv2.polylines(canvas, [points], False, (40, 70, 210), 3, cv2.LINE_AA)
                tile = cv2.resize(canvas, (960, 540))
                cv2.putText(
                    tile,
                    candidate.name[:110],
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                tiles.append(tile)
            cv2.imwrite(
                str(
                    args.output.with_name(f"{args.output.stem}_{scene.rope_id}_routes.jpg")
                ),
                np.concatenate(tiles, axis=0),
            )
            if not args.geometry_only:
                fit, selected_scene, fits = fit_route_candidates(
                    scene,
                    circles,
                    fit_candidates,
                    obs,
                    config=config,
                    device=args.device,
                    substeps=args.substeps,
                    length_prior_weight=args.length_prior_weight,
                    mass_model=args.mass_model,
                    verbose=not args.quiet,
                    execution_mode=args.execution_mode,
                    gravity_normalized_per_s2=args.gravity,
                    fit_initial_speed=args.fit_initial_speed,
                )
                item = result_payload(
                    selected_scene,
                    config,
                    fit,
                    time.perf_counter() - rope_start,
                    device=args.device,
                    substeps=args.substeps,
                )
                item["routing"] = fit["route"]
                item["initial_centerline_normalized"] = (
                    selected_scene.initial_positions.tolist()
                )
                item["metrics"].update(
                    {key: value for key, value in fit["route"].items() if "rmse" in key}
                )
                item["optimization"]["observation_loss"] = fit["best_loss"]
                item["optimization"]["selection_loss"] = fit["selection_loss"]
                item["optimization"]["length_prior_weight"] = args.length_prior_weight
                item["optimization"]["mass_model"] = args.mass_model
                item["optimization"]["gravity_calibration"] = calibration
                item["optimization"]["initial_velocity_policy"] = 'fitted' if args.fit_initial_speed else 'fixed-zero'
                item["parameters"]["rope_linear_density"] = fit["route"][
                    "rope_linear_density"
                ]
                item["optimization"]["cem_config"] = asdict(config)
                item["optimization"]["observation_stride"] = args.observation_stride
                item["optimization"]["route_selection"] = args.route_selection
                item["optimization"]["physics_fit_count"] = len(fits)
                if args.route_selection == "geometry":
                    item["optimization"]["geometry_selection"] = selection
                elif args.route_selection == "screen":
                    item["optimization"]["forward_screening"] = screening
                item["optimization"]["simulate_static_system"] = True
                item["optimization"]["out_of_view_policy"] = "unobserved"
                item["optimization"][
                    "objective"
                ] = "raw-visible-coverage-and-fixed-normalization-free-space-integral-plus-observed-body-huber"
                outputs.append(item)
                report["fits"] = fits
                report["fit_comparison_scope"] = "fitted-routes-only"
                report["selected"] = fit["route"]["name"]
                report["observation_only_winner"] = min(
                    fits, key=lambda row: row["observation_loss"]
                )["name"]
                report["prior_sensitivity"] = {
                    str(weight): min(
                        fits,
                        key=lambda row: row["observation_loss"]
                        + weight * config.huber_delta**2 * row["length_excess_ratio"],
                    )["name"]
                    for weight in (0.0, 0.025, 0.05, 0.1, 0.2)
                }
                old = next(
                    (
                        item
                        for item in baseline.get("ropes", [])
                        if item["rope_id"] == scene.rope_id
                    ),
                    None,
                )
                if old:
                    old_positions = np.asarray(
                        [row["xy_normalized"] for row in old["predicted_centerline"]]
                    )
                    report["previous_result_raw_metrics"] = VisibleRopeLoss(
                        obs, args.device
                    ).metrics(old_positions)
                    report["previous_result_reported_metrics"] = old["metrics"]
            reports[scene.rope_id] = report
            args.output.with_suffix(".audit.json").write_text(
                json.dumps(reports, indent=2) + "\n"
            )
        except ValueError as error:
            known = ('no finite route screening rollout',
                     'No explicit tangent route supports every dynamic pulley',
                     'A free massless rope end needs an explicit load',
                     'Unloaded massless moving pulley requires a coupled algebraic solver',
                     'Invalid initial tangent geometry', 'no finite route fit',
                     'no nonpenetrating route for forward screening', 'no route candidates',
                     'no nonpenetrating initial rope route', 'routing needs visible rope in the initial frame')
            if not any(reason in str(error) for reason in known):
                raise
            failures[scene.rope_id] = dict(reason=str(error),
                object_ids=[body.object_id for body in scene.bodies],
                dynamic_object_ids=[body.object_id for body in scene.bodies if body.dynamic])
            reports[scene.rope_id] = dict(status='failed', **failures[scene.rope_id])
            args.output.with_suffix('.audit.json').write_text(json.dumps(reports, indent=2)+'\n')
    outputs = retain_independent_fits(outputs, failures)
    for name, failure in failures.items():
        reports.setdefault(name, {}).update(status='failed', **failure)
    args.output.with_suffix('.audit.json').write_text(json.dumps(reports, indent=2)+'\n')
    args.output.with_suffix('.failures.json').write_text(json.dumps(failures, indent=2)+'\n')
    if not outputs and not args.geometry_only:
        raise ValueError('No rope group fitted: ' + '; '.join(
            f'{name}: {row["reason"]}' for name, row in failures.items()))
    if not args.geometry_only:
        payload = {
            "schema": "physmind.rope_swr_scene",
            "schema_version": 1,
            "model": ("physmind-v2-ideal-massless-rope-v1" if args.mass_model == "massless"
                      else "physmind-v2-warp-rope-swr-v1"),
            "mass_model": args.mass_model,
            "source": source,
            "routing_method": "raw-visible-topology-search",
            "route_selection": args.route_selection,
            "execution_mode": "ideal-massless" if args.mass_model == "massless" else args.execution_mode,
            "selected_rope_ids": [item["rope_id"] for item in outputs],
            "observation_mode": "raw-visible",
            "topology_hints": {},
            "ropes": outputs,
            "elapsed_seconds": time.perf_counter() - started,
            "limitations": [
                *(["Partial fit: uncontrolled rope groups: " + ", ".join(failures)] if failures else []),
                "bounded route search; at most one extra turn on one pulley",
                "fixed initial endpoints and inherited attachment/pinning hypotheses",
                "no rope-rope friction or self-contact; crossings are projected strands",
                "separate rope fits; not a shared multi-rope rigid simulation",
                *(["tension-only rope with load-guide contacts and zero-wrap guide release; loose shape, load-load collision and arbitrary re-routing are not modeled",
                   "no endpoint actuator or unloaded massless moving-pulley algebraic coupling"]
                  if args.mass_model == "massless" else []),
            ],
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "elapsed_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
