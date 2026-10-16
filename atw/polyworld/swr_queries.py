"""Scene-owned semantic readouts over retained SWR; no fitting or VLM calls."""
from __future__ import annotations

import cv2
import numpy as np

from atw.polyworld.evidence_measurements import measure_results as measure
from atw.polyworld.swr_measurements import SWRMeasurements


MOTION_MEANING = "Move/moving/not moving refers to center translation only. Rotation about a stationary center does NOT count as moving. Rotation is reported separately."
MASS_FACTORS = {"increase": 10.0, "decrease": 0.1}


def mask_geometry(backend):
    """Original-pixel solid components, keeping physical identities separate."""
    rows = []
    for name, obj in backend.objects.items():
        if name in getattr(backend, "background_object_ids", ()):
            continue
        if obj.get("physical_type") not in {None, "rigid"}:
            continue
        if name in backend.fit.get("fluid_ids", []) or obj.get("is_origin"):
            continue
        try:
            frame = 0 if backend.engine == "soft_body" else backend.best_frames[name]
            mask = backend._mask(name, frame).astype(np.uint8)
        except (KeyError, ValueError, OSError):
            continue
        scale = np.array([backend.width / mask.shape[1], backend.height / mask.shape[0]])
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for x, y, w, h, area in stats[1:]:
            if area < max(4, mask.sum() * .03):
                continue
            rows.append(dict(object_id=name, box=(np.array([x,y,x+w,y+h])*np.tile(scale,2)).tolist(),
                             frame=frame, mask=mask, scale=scale))
    return rows


def scene_regions(backend):
    """Model-world regions, defined once from retained static geometry, never QA labels."""
    if backend.engine != "soft_body":
        return []
    solids = mask_geometry(backend)
    regions = []
    # Bottom support blocks define open gaps. Floating ramps and
    # the selected background wall are not support blocks in this world.
    horizontal = [r for r in solids if r['box'][1] >= backend.height*.5]
    if not horizontal:
        return []
    bottom = max(r['box'][3] for r in horizontal)
    bases = sorted([r for r in horizontal if abs(r['box'][3]-bottom) <= backend.height*.08
                    and r['box'][1] >= backend.height*.5], key=lambda r:r['box'][0])
    for left, right in zip(bases, bases[1:]):
        x0, x1 = left['box'][2], right['box'][0]
        y0, y1 = max(left['box'][1],right['box'][1]), min(left['box'][3],right['box'][3])
        if x1 <= x0 or y1 <= y0:
            continue
        regions.append(dict(region_id=f"gap_{len(regions)+1}", kind="pit_gap",
            boundary_ids=list(dict.fromkeys([left['object_id'],right['object_id']])),
            region=dict(box=[x0,y0,x1,float(backend.height)]), horizontal_order=len(regions)+1,
            definition="Open vertical chute between adjacent bottom solid blocks, beginning at their upper side. Crossing into this chute is entry; later falling below the image does not undo entry.",
            source_kind="retained_static_masks"))
    return regions


def catalog(backend):
    if not hasattr(backend, '_world_regions'):
        backend._world_regions = scene_regions(backend)
    regions = backend._world_regions if backend.engine == "soft_body" else []
    queries = ["contact"] if backend.engine in {"soft_body","rigid"} else []
    if backend.engine == "cloth":
        queries += ["pose", "box_overlap"]
    if backend.engine == "rope":
        queries += ["motion"]
    if regions:
        queries += ["region"]
    result = dict(queries=queries, regions=regions)
    if "motion" in queries:
        result["motion_meaning"] = MOTION_MEANING
    if "region" in queries:
        result["region_meaning"] = "Region IDs name empty space, not an object. Bind their boundary entities separately. Region query reports entry and occupancy; enter uses any, remain/final uses final."
    return result


def tool(backend):
    names = catalog(backend)['queries']
    if not names:
        return None
    fields = "{id,kind:" + '|'.join(names) + ",object_ids:[IDs],temporal:any|final|all,start_seconds,end_seconds"
    rules = []
    if "contact" in names:
        rules.append("Contact events require temporal=any.")
    if "pose" in names:
        rules.append("pose requires exactly ONE rigid object per query and defaults final.")
    if "box_overlap" in names:
        rules.append("box_overlap requires exactly TWO rigid objects; temporal=any tests future overlap, temporal=final tests endpoint overlap.")
    if "region" in names:
        fields += ",region_id:catalog ID,relation:enter|occupy|most"
        rules.append("Regions are empty destinations, not boundary bodies. relation=enter requires temporal=any; "
                     "for final contents use relation=occupy|most with temporal=final.")
    if backend.engine in {"cloth", "rope"}:
        fields += ",variant:optional independent fitted system"
    rules.append("Other queries default any.")
    description = ("Read scene-defined " + ', '.join(names) + " measurements. "
                   "Use only advertised kinds. No hand-selected proximity thresholds. Return source, coverage, units and explicit query-repair requirements when unsupported.")
    if "region" in names:
        description += " Use catalog regions, not free pixel boxes or hull operations."
    if "motion" in names:
        description += " " + MOTION_MEANING
    return dict(name="query_world", arguments=dict(
        queries="1..6 query objects with DISTINCT nonempty id labels (id is a query label, not the target entity): "
                + fields + "}. " + " ".join(rules),
        rollout_id="Existing rollout ID; omit for saved SWR. No new simulation."), description=description)


def _softbody_projection_offset(backend, name):
    """Median observation-to-SWR image offset on the unedited shared frames."""
    rows = SWRMeasurements(backend).points(name, "center")
    if not rows:
        return np.zeros(2), 0
    deltas = []
    for frame, points in zip(rows[0].frames, rows[0].points):
        try:
            mask = backend._mask(name, int(frame))
        except (KeyError, ValueError, OSError):
            continue
        yy, xx = np.where(mask)
        if not len(xx) or not len(points):
            continue
        observed = np.array([xx.mean()*backend.width/mask.shape[1], yy.mean()*backend.height/mask.shape[0]])
        deltas.append(observed-np.asarray(points)[0])
    return (np.median(deltas, axis=0), len(deltas)) if deltas else (np.zeros(2), 0)


def softbody_pit_entry(backend, source, ids, region, start, end, temporal):
    if temporal != "any":
        raise ValueError("Entry is an event: use temporal=any")
    if len(ids) != 1:
        raise ValueError("Pit entry requires exactly one soft-body object ID")
    series = source.points(ids[0], "center")
    if not series:
        return dict(status="needs_query", condition_met=None, reason="Projected soft-body trajectory unavailable")
    offset, calibration_count = _softbody_projection_offset(backend, ids[0])
    x0, y0, x1, y1 = region["region"]["box"]
    variants, verdicts = {}, []
    for values in series:
        samples = [(int(frame), np.asarray(points)[0]+offset) for frame, points in zip(values.frames, values.points)
                   if len(points) and start <= frame/backend.fps <= end]
        crossings = []
        for (frame0, point0), (frame1, point1) in zip(samples, samples[1:]):
            if point0[1] < y0 <= point1[1] and point1[1] != point0[1]:
                alpha = (y0-point0[1])/(point1[1]-point0[1])
                crossing_x = point0[0] + alpha*(point1[0]-point0[0])
                if x0 <= crossing_x <= x1:
                    crossings.append(dict(frame=frame1, time_s=frame1/backend.fps, x_px=float(crossing_x)))
            elif point0[1] >= y0 and not (x0 <= point0[0] <= x1) and x0 <= point1[0] <= x1:
                crossings.append(dict(frame=frame1, time_s=frame1/backend.fps, x_px=float(point1[0])))
        inside = [bool(x0 <= point[0] <= x1 and y0 <= point[1] <= y1) for _, point in samples]
        verdict = True if crossings else (None if not samples or inside[0] else False)
        variants[values.variant] = dict(condition_met=verdict, entry_count=len(crossings), entries=crossings,
            first=dict(frame=samples[0][0], value=float(inside[0])) if samples else None,
            final=dict(frame=samples[-1][0], value=float(inside[-1])) if samples else None,
            coverage=dict(complete=bool(samples)),
            samples=[dict(frame=frame, image_xy_px=point.tolist()) for frame, point in samples])
        verdicts.append(verdict)
    condition = verdicts[0] if verdicts and all(value == verdicts[0] for value in verdicts) else None
    return dict(status="measured" if condition is not None else "needs_query", condition_met=condition, variants=variants,
        next_action="If no entry verdict is available, query an interval covering the boundary crossing and resolve the relevant fitted system. If unavailable, request per-question original-video reasoning.",
        projection_calibration=dict(offset_xy_px=offset.tolist(), shared_observed_frames=calibration_count,
            source="median difference between retained SWR centers and first-stage identity-mask centroids"),
        interpretation="Entry is a trajectory crossing into the mask-defined vertical opening. A per-object offset estimated only from the unedited observation aligns the retained SWR projection; no answer label or edited outcome is used for calibration.")


def cloth_query_variant(source, object_ids, requested=None):
    """Resolve rigid queries to the cloth fit that reached every bound rigid body."""
    assignments = source.cloth_fit_assignments()
    selected = {name: assignments.get(name) for name in object_ids}
    unresolved = [name for name, row in selected.items()
                  if not row or row.get("status") != "assigned" or not row.get("variant")]
    membership = {name: dict(status=(row or {}).get("status", "unavailable"),
                            variant=(row or {}).get("variant"))
                  for name, row in selected.items()}
    if unresolved:
        return None, dict(status="needs_query", condition_met=None, fallback_required=True,
            fit_assignment=membership,
            reason="Cloth-fit ownership is unavailable or ambiguous for: " + ", ".join(unresolved),
            next_action="Call request_video_fallback for this question; do not combine or vote across cloth fits.")
    variants = {row["variant"] for row in selected.values()}
    if len(variants) != 1:
        return None, dict(status="needs_query", condition_met=None, fallback_required=True,
            fit_assignment=membership,
            reason="The queried rigid bodies belong to different independently fitted cloth worlds.",
            next_action="Call request_video_fallback for this question; the independent fits cannot establish their joint future interaction.")
    variant = next(iter(variants))
    if requested is not None and requested != variant:
        return None, dict(status="invalid_query", condition_met=None,
            fit_assignment=membership,
            reason=f"variant {requested!r} does not own the queried rigid body or pair; use {variant!r}.",
            next_action="Repeat query_world on the same rollout with the assigned variant.")
    return variant, dict(status="assigned", variant=variant, members=list(object_ids),
        basis="Observed-window fitted cloth-node proximity to each rigid OBB; no question or answer labels.")


def query_world(backend, queries, rollout_id=None):
    if not isinstance(queries,list) or not 1 <= len(queries) <= 6:
        raise ValueError("queries must contain 1..6 scene queries")
    labels = [q.get('id') for q in queries if isinstance(q,dict)]
    if len(labels) != len(queries) or any(not isinstance(label,str) or not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Each query needs a unique nonempty id label; put entity IDs in object_ids")
    source = SWRMeasurements(backend,rollout_id)
    available = catalog(backend)
    results = []
    for request in queries:
        qid = request['id']; kind = request['kind']; ids = backend._ids(request['object_ids'])
        if kind not in available['queries']:
            raise ValueError(f"Unsupported world query {kind}; use scene capabilities")
        temporal = request.get('temporal','final' if kind=='pose' else 'any')
        if temporal not in {'any','final','all'}:
            raise ValueError("temporal must be any, final or all")
        start,end = float(request.get('start_seconds',0)),float(request.get('end_seconds',source.end))
        if not np.isfinite([start,end]).all() or not 0 <= start <= end <= source.end:
            raise ValueError("Query interval must be inside the modeled source")
        row = dict(id=qid,metric=kind,object_ids=ids,interval_seconds=[start,end],temporal=temporal)
        if kind == 'contact':
            if temporal != 'any':
                raise ValueError('Contact event logs support temporal=any within the requested interval')
            row.update(source.events(ids,start,end))
            row.update(condition_met=row.get('event_occurs'),status=row.get('event_status','needs_query'))
        elif kind == 'pose':
            if len(ids)!=1:
                raise ValueError("pose requires one rigid object ID")
            variant, assignment = cloth_query_variant(source, ids, request.get('variant'))
            if variant is None:
                row.update(assignment)
                results.append(row)
                continue
            result = measure(backend,[dict(id=qid,metric='tilt',object_ids=ids,start_seconds=start,end_seconds=end,
                variant=variant)],rollout_id)
            row.update(result['results'][0])
            row['fit_assignment'] = assignment
            row['poses'] = {}
            for candidate, samples in source.orientations(ids[0]).items():
                if candidate != variant:
                    continue
                samples = [sample for sample in samples if start <= sample['frame_index']/backend.fps <= end]
                if samples:
                    keys = ('frame_index','center_world','rotation_world')
                    row['poses'][candidate] = {edge:{k:sample[k] for k in keys if k in sample}
                        for edge,sample in [('first',samples[0]),('final',samples[-1])]}
            row['pose_units'] = 'center_world: meters; rotation_world: body-to-world matrix; frame_index: original frame'
            row['interpretation'] = "Tilt of initially upright body axis: 0 degrees upright, 90 horizontal. Read final for final pose; maximum for whether it toppled. Result is conditional on the automatically assigned cloth fit."
        elif kind == 'box_overlap':
            if len(ids) != 2:
                raise ValueError("box_overlap requires two rigid object IDs")
            variant, assignment = cloth_query_variant(source, ids, request.get('variant'))
            if variant is None:
                row.update(assignment)
                results.append(row)
                continue
            reduce = {'any':'any', 'final':'final', 'all':'all'}[temporal]
            result = measure(backend,[dict(id=qid,metric='box_overlap',object_ids=ids,
                start_seconds=start,end_seconds=end,variant=variant,
                condition=dict(op='le',value=0.,reduce=reduce))],rollout_id)
            row.update(result['results'][0])
            row['fit_assignment'] = assignment
            row['interpretation'] = ("Signed rigid OBB gap in the assigned cloth fit: positive is separated, "
                "zero touches and negative overlaps. This is a modeled geometric contact proxy, not an event log.")
        elif kind == 'region':
            region = next((r for r in available['regions'] if r['region_id']==request.get('region_id')),None)
            if region is None:
                raise ValueError("Use a region_id from the SWR region catalog")
            if set(region['boundary_ids']) & set(source.data.get('remove_objects',[])):
                row.update(status='needs_query',condition_met=None,reason='A boundary of this region was removed; its old geometry cannot define the edited destination')
            else:
                relation = request.get('relation','enter')
                if backend.engine == 'soft_body' and region.get('kind') == 'pit_gap' and relation == 'enter':
                    row.update(softbody_pit_entry(backend,source,ids,region,start,end,temporal),
                        region_id=region['region_id'],region_definition=region['definition'],relation=relation)
                    results.append(row)
                    continue
                if relation == 'enter' and temporal != 'any':
                    raise ValueError('Entry is an event within the interval: use temporal=any; for final occupancy use relation=occupy')
                if relation not in {'enter','occupy','most'}:
                    raise ValueError('region relation must be enter, occupy or most')
                raw=measure(backend,[dict(id=qid,metric='region',object_ids=ids,region=region['region'],
                    start_seconds=start,end_seconds=end,condition=dict(op='gt',value=.5 if relation=='most' else 0.,reduce=temporal))],rollout_id)['results'][0]
                row.update(raw,region_id=region['region_id'],region_definition=region['definition'],relation=relation)
                if relation=='enter':
                    row.pop('condition',None)
                    # Occupancy in the very first sample alone does not prove entry.
                    verdicts=[]
                    for variant in row.get('variants',{}).values():
                        count=variant.get('entry_count',0)
                        value=True if count else False if variant.get('coverage',{}).get('complete') and (variant.get('first') or {}).get('value')==0 else None
                        variant['condition_met']=value
                        verdicts.append(value)
                    row['condition_met']=verdicts[0] if verdicts and all(v is verdicts[0] for v in verdicts) else None
        else:
            motion={}
            for name in ids:
                variants={}
                for variant,rows in source.groups(name).items():
                    rows=[r for r in rows if start <= r['frame_index']/backend.fps <= end]
                    variants[variant]=backend._rope_center_translation(rows, rollout_id)
                motion[name]=dict(translation=variants)
                try:
                    motion[name]['rotation']=backend.query_pulley_rotation(name,rollout_id,start,end)['results']
                except ValueError:
                    motion[name]['rotation']=dict(status='unavailable',reason='No modeled pulley rotation for this object')
            row.update(status='measured',objects=motion,interpretation=MOTION_MEANING)
        results.append(row)
    return source.provenance() | dict(semantic_queries=True,results=results)
