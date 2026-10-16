"""Generic geometric queries over engine-exported evidence, with explicit query requirements."""
from __future__ import annotations

import operator

import cv2
import numpy as np
from scipy.spatial import cKDTree

from atw.polyworld.swr_measurements import SWRMeasurements


MEASUREMENT_TOOL = dict(name="measure_results", arguments=dict(
    queries='REQUIRED array: [{"id":"m1","metric":"region|proximity|tilt|contact|box_overlap","object_ids":["existing_id"],...}]. 1..6 unique ids; optional start_seconds/end_seconds default full source',
    rollout_id="optional existing rollout; omit for saved SWR"), description=
    "Read full exported samples, without simulation. Query fields: region requires region={box:[x0,y0,x1,y1]} or "
    "{polygon:[[x,y],...]} or {object_id,mask_frame,shape:'hull_interior'} (explicit convex-hull-minus-mask approximation). "
    "A gap can use {between:[id1,id2],axis:'x'|'y',mask_frame:0}: between mask bounds, with overlap on the other axis. "
    "Coordinates are ORIGINAL image pixels, x right/y down. point_mode='center' or 'members'; default comes from engine capabilities. "
    "proximity requires target={object_id} for engine geometry (centers when no shape is exported), or {object_id,mask_frame} for an explicitly frozen observed mask; "
    "tolerance_px is required, proximity is not contact. tilt uses body_axis='initial_up' (default), x, y or z against "
    "reconstructed up or explicit reference_world=[x,y,z]. contact queries engine events. box_overlap uses two exported rigid boxes; "
    "signed separating-axis gap in meters, <=0 means overlap. Optional condition={op:gt|ge|lt|le|eq," 
    "value:number,reduce:final|any|all} compares region fraction or tilt degrees. Results preserve coverage, provenance and query requirements. "
    "Use members for particle proportions; specify actual destinations and the literal question condition.")
MEASUREMENT_TOOL["description"] += " Optional variant selects one returned fitted-system ID; otherwise all variants remain separate."


def interval_frames(start, end, fps):
    return np.arange(int(np.ceil(start * fps - 1e-7)), int(np.floor(end * fps + 1e-7)) + 1)


def spans(frames, fps):
    groups = []
    for frame in frames:
        frame = int(frame)
        if groups and frame == groups[-1][1] + 1:
            groups[-1][1] = frame
        else:
            groups.append([frame, frame])
    return [dict(frames=[a,b], seconds=[a/fps,b/fps]) for a,b in groups]


def compare(values, valid, condition):
    if condition is None:
        return None
    operations = dict(gt=operator.gt, ge=operator.ge, lt=operator.lt, le=operator.le, eq=operator.eq)
    if (not isinstance(condition, dict) or condition.get("op") not in operations
            or condition.get("reduce") not in {"final", "any", "all"}
            or isinstance(condition.get("value"), bool) or not isinstance(condition.get("value"), (int,float))
            or not np.isfinite(condition["value"])):
        raise ValueError("condition requires op gt/ge/lt/le/eq, finite value and reduce final/any/all")
    truth = operations[condition["op"]](np.asarray(values), condition["value"])
    valid = np.asarray(valid, bool)
    if not len(valid):
        return None
    mode = condition["reduce"]
    if mode == "final":
        return bool(truth[-1]) if valid[-1] else None
    if mode == "any" and np.any(truth & valid):
        return True
    if mode == "all" and np.any(~truth & valid):
        return False
    return bool(mode == "all") if valid.all() else None


def summarize(frames, values, valid, fps, condition=None):
    valid = np.asarray(valid, bool) & ~np.isnan(values)
    finite = valid & np.isfinite(values)
    indices = np.flatnonzero(finite)
    sample = lambda i: dict(frame_index=int(frames[i]), time_s=float(frames[i]/fps), value=float(values[i]))
    coverage = dict(complete=bool(len(frames) and valid.all()),
        requested_sample_count=len(frames), measured_sample_count=int(valid.sum()),
        missing_intervals=spans(frames[~valid], fps)[:8],
        missing_interval_count=len(spans(frames[~valid], fps)),
        resolution_seconds=1/fps, scope="Retained discrete samples; subframe events and real-world accuracy are not guaranteed.")
    return dict(status="measured" if valid.any() else "needs_query", coverage=coverage,
        first=sample(indices[0]) if len(indices) else None,
        last=sample(indices[-1]) if len(indices) else None,
        final=sample(len(frames)-1) if len(frames) and finite[-1] else None,
        minimum=sample(indices[np.argmin(np.asarray(values)[indices])]) if len(indices) else None,
        maximum=sample(indices[np.argmax(np.asarray(values)[indices])]) if len(indices) else None,
        condition=condition, condition_met=compare(values, valid, condition))


def mask_points(backend, object_id, frame):
    backend._ids([object_id])
    if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < backend.frame_count:
        raise ValueError("mask_frame must be an observed integer frame")
    mask = backend._mask(object_id, frame)
    ys, xs = np.where(mask)
    if not len(xs):
        raise LookupError("Requested mask is empty")
    scale = np.array([backend.width/mask.shape[1], backend.height/mask.shape[0]])
    return mask, np.column_stack((xs,ys)), scale


def region_test(backend, region):
    if not isinstance(region, dict):
        raise ValueError("region must define box, polygon or explicit hull_interior")
    if "box" in region:
        box = np.asarray(region["box"], float)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            raise ValueError("box requires finite x0,y0,x1,y1 with positive area")
        return lambda p: ((p >= box[:2]) & (p <= box[2:])).all(1), dict(box=box.tolist())
    if "polygon" in region:
        polygon = np.asarray(region["polygon"], np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or not 3 <= len(polygon) <= 64 or not np.isfinite(polygon).all():
            raise ValueError("polygon requires 3..64 finite [x,y] vertices")
        return lambda p: np.array([cv2.pointPolygonTest(polygon, tuple(map(float,xy)), False) >= 0 for xy in p]), dict(polygon=polygon.tolist())
    if "between" in region:
        ids = region["between"]
        if not isinstance(ids,list) or len(ids) != 2 or len(set(ids)) != 2 or region.get("axis") not in {"x","y"}:
            raise ValueError("between requires two distinct object IDs and axis x or y")
        bounds = []
        for name in ids:
            _, pixels, scale = mask_points(backend,name,region["mask_frame"])
            bounds.append((pixels.min(0)*scale,(pixels.max(0)+1)*scale))
        axis = "xy".index(region["axis"])
        low, high = sorted(bounds,key=lambda b:b[0][axis])
        start, end = np.maximum(low[0],high[0]), np.minimum(low[1],high[1])
        start[axis], end[axis] = low[1][axis], high[0][axis]
        if np.any(end <= start):
            raise LookupError("Mask bounds define no nonempty gap with transverse overlap; specify the intended polygon")
        box = [*start.tolist(),*end.tolist()]
        contains, _ = region_test(backend,dict(box=box))
        return contains, dict(**region,box=box,assumption="Frozen image gap between observed mask bounds; not automatic identification of a destination or its physical volume.")
    if region.get("shape") != "hull_interior":
        raise ValueError("Use box, polygon or shape=hull_interior with object_id and mask_frame")
    mask, pixels, scale = mask_points(backend, region["object_id"], region["mask_frame"])
    hull = cv2.convexHull(pixels.astype(np.int32))
    inside = cv2.fillConvexPoly(np.zeros_like(mask, dtype=np.uint8), hull, 1).astype(bool) & ~mask
    if not inside.any():
        raise LookupError("Convex hull minus mask has no interior; supply the intended region explicitly")
    def contains(points):
        xy = np.rint(points/scale).astype(int)
        valid = ((xy >= 0) & (xy < [mask.shape[1], mask.shape[0]])).all(1)
        result = np.zeros(len(xy), bool)
        result[valid] = inside[xy[valid,1], xy[valid,0]]
        return result
    return contains, dict(**region, polygon=(hull[:,0]*scale).tolist(), resolution_px=scale.tolist(),
        assumption="Explicitly chosen convex enclosure approximation minus its observed solid mask; frozen at mask_frame, not automatic semantic recognition.")


def point_groups(source, ids, mode):
    by_object = {name:source.points(name,mode) for name in ids}
    if any(not rows for rows in by_object.values()):
        raise LookupError("Requested point representation is not exported for every selected object")
    variants = {s.variant for rows in by_object.values() for s in rows} - {"joint"}
    result = {}
    for variant in sorted(variants or {"joint"}):
        series = [next((s for s in rows if s.variant == variant), next((s for s in rows if s.variant == "joint"), None))
                  for rows in by_object.values()]
        if any(s is None for s in series):
            continue
        result[variant] = [{int(f):p for f,p in zip(s.frames,s.points)} for s in series]
    return result


def measure_region(backend, source, query, frames):
    contains, definition = region_test(backend, query["region"])
    groups = point_groups(source, query["object_ids"], query.get("point_mode", "center"))
    variants = {}
    for variant, series in groups.items():
        counts, totals, valid, values = [], [], [], []
        for frame in frames:
            parts = [s.get(int(frame)) for s in series]
            ok = all(p is not None and np.isfinite(p).all() for p in parts)
            points = np.concatenate(parts) if ok else np.empty((0,2))
            count, total = int(contains(points).sum()), len(points)
            counts.append(count); totals.append(total); valid.append(ok and total > 0)
            values.append(count/total if ok and total else np.nan)
        summary = summarize(frames, values, valid, backend.fps, query.get("condition"))
        for key in ("first", "last", "final", "minimum", "maximum"):
            if summary[key]:
                i = int(np.searchsorted(frames, summary[key]["frame_index"]))
                summary[key].update(inside_count=counts[i], total_count=totals[i])
        occupied = np.array(valid) & (np.array(counts) > 0)
        starts = [i for i in range(1,len(frames)) if valid[i] and valid[i-1] and occupied[i] and not occupied[i-1]]
        summary.update(occupied_intervals=spans(frames[occupied], backend.fps)[:8],
            entry_times_seconds=[float(frames[i]/backend.fps) for i in starts[:8]],
            entry_count=len(starts), interpretation="Occupancy is not entry. Entry requires consecutive outside/inside samples; count weights are equal.")
        variants[variant] = summary
    return dict(region=definition, value_kind="fraction_of_" + ("members" if query["point_mode"] == "members" else "centers") + "_in_region", variants=variants)


def measure_proximity(backend, source, query, frames):
    tolerance = query["tolerance_px"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (float,int)) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_px must be a finite nonnegative distance in original image pixels")
    target = query["target"]
    backend._ids([target["object_id"]])
    if target["object_id"] in source.data.get("remove_objects", []):
        raise LookupError("Target was removed from the requested modeled world")
    if "mask_frame" in target:
        mask, xy, scale = mask_points(backend, target["object_id"], target["mask_frame"])
        tree = cKDTree(xy*scale)
        targets = None
        reference = dict(**target, source_kind="mask_observation", resolution_px=scale.tolist(),
                         assumption="Selected observed mask is treated as fixed throughout the requested modeled interval.")
    else:
        static = source.static_proximity_target(target["object_id"])
        if static is not None:
            points, reference = static
            tree, targets = cKDTree(points), None
        else:
            targets = point_groups(source, [target["object_id"]], "center")
            reference = dict(**target, source_kind="predicted_center")
    variants = {}
    for variant, series in point_groups(source, query["object_ids"], query.get("point_mode","center")).items():
        values, valid = [], []
        target_series = targets.get(variant, targets.get("joint")) if targets is not None else None
        for frame in frames:
            parts = [s.get(int(frame)) for s in series]
            ok = all(p is not None and np.isfinite(p).all() for p in parts)
            if targets is not None:
                target_points = None if target_series is None else target_series[0].get(int(frame))
                ok = ok and target_points is not None and np.isfinite(target_points).all()
                if ok:
                    tree = cKDTree(target_points)
            points = np.concatenate(parts) if ok else np.empty((0,2))
            values.append((float(tree.query(points)[0].min()) if len(points) else np.inf) if ok else np.nan)
            valid.append(bool(ok))
        condition = dict(op="le", value=float(tolerance), reduce="any")
        result = summarize(frames, values, valid, backend.fps, condition)
        near = np.array(valid) & (np.array(values) <= tolerance)
        result.update(near_intervals=spans(frames[near],backend.fps)[:8],
                      known_empty_sample_count=int(np.isinf(values).sum()),
                      near_sample_count=int(near.sum()), near_interval_count=len(spans(frames[near],backend.fps)))
        variants[variant] = result
    return dict(target=reference, value_kind="minimum_image_distance_px", variants=variants,
        interpretation="Projected point-to-mask or point-to-center proximity, not engine contact or surface collision; camera depth is ignored. Known empty point sets cannot be near a target.")


def measure_tilt(backend, source, query, frames):
    if len(query["object_ids"]) != 1:
        raise ValueError("tilt requires one object_id")
    down = backend.world_frame.get("gravity", {}).get("gravity_down_world")
    reference = query.get("reference_world", -np.asarray(down) if down is not None else None)
    if reference is None:
        raise LookupError("No world reference direction; supply reference_world explicitly")
    reference = np.asarray(reference, float)
    if reference.shape != (3,) or not np.isfinite(reference).all() or np.linalg.norm(reference) < 1e-8:
        raise ValueError("reference_world must be a finite nonzero 3-vector")
    reference /= np.linalg.norm(reference)
    axis = query.get("body_axis", "initial_up")
    if axis not in {"initial_up", "x", "y", "z"}:
        raise ValueError("body_axis must be initial_up, x, y or z")
    variants = {}
    for variant, rows in source.orientations(query["object_ids"][0]).items():
        rotations = {}
        for row in rows:
            rotation = row.get("rotation_world")
            if rotation is None and "pose_4x4_world" in row:
                rotation = np.asarray(row["pose_4x4_world"])[:3,:3]
            if rotation is not None:
                rotations[row["frame_index"]] = np.asarray(rotation)
        if not rotations:
            continue
        index = int(np.argmax(np.abs(rotations[min(rotations)].T@reference))) if axis == "initial_up" else "xyz".index(axis)
        values = [float(np.degrees(np.arccos(np.clip(abs(rotations[int(f)][:,index]@reference),0,1))))
                  if int(f) in rotations else np.nan for f in frames]
        variants[variant] = summarize(frames, values, np.isfinite(values), backend.fps, query.get("condition")) | dict(body_axis="xyz"[index])
    if not variants:
        raise LookupError("Engine does not export rigid orientation for this object and source")
    return dict(value_kind="axis_tilt_degrees", reference_world=reference.tolist(), variants=variants,
                interpretation="Unsigned axis tilt: 0 parallel, 90 perpendicular. This does not itself identify a semantic pose or fall event.")


def box_separation(a, b):
    """Signed largest separating-axis gap; positive values are not exact distance."""
    ca, ra, ha = a
    cb, rb, hb = b
    if not all(np.isfinite(v).all() for v in (*a,*b)):
        return np.nan
    axes = np.vstack((ra.T,rb.T,np.cross(ra.T[:,None,:],rb.T[None,:,:]).reshape(-1,3)))
    axes = axes[np.linalg.norm(axes,axis=1) > 1e-8]
    axes /= np.linalg.norm(axes,axis=1)[:,None]
    gaps = np.abs(axes@(cb-ca)) - np.abs(axes@ra)@ha - np.abs(axes@rb)@hb
    return float(np.max(gaps))


def measure_boxes(backend, source, query, frames):
    if len(query["object_ids"]) != 2:
        raise ValueError("box_overlap requires two object_ids")
    left,right = [source.boxes(name) for name in query["object_ids"]]
    variants = {}
    for variant in left.keys() & right.keys():
        values = [box_separation(left[variant][int(f)],right[variant][int(f)])
                  if int(f) in left[variant] and int(f) in right[variant] else np.nan for f in frames]
        condition = query.get("condition",dict(op="le",value=0.,reduce="any"))
        variants[variant] = summarize(frames, values, np.isfinite(values), backend.fps, condition)
    if not variants:
        raise LookupError("No exported rigid boxes for both objects in a common fitted system")
    return dict(value_kind="signed_separating_axis_gap_m",variants=variants,
        interpretation="Overlap of engine obstacle boxes, not observed collision or an event log. Positive gap is a distance lower bound; zero touches, negative overlaps. Independent fits stay separate.")


def measure_results(backend, queries, rollout_id=None):
    if not isinstance(queries, list) or not 1 <= len(queries) <= 6:
        raise ValueError("queries must contain 1..6 measurements")
    if rollout_id is not None and rollout_id not in backend.rollouts:
        raise ValueError("Use an existing rollout_id")
    if any(not isinstance(q,dict) or not isinstance(q.get("id"),str) or not q["id"] for q in queries) or len({q["id"] for q in queries}) != len(queries):
        raise ValueError('Each query requires unique id and object_ids as an array: {"id":"m1","metric":"...","object_ids":["existing_id"],...}')
    source, results = SWRMeasurements(backend,rollout_id), []
    for query in queries:
        query = dict(query)
        query.setdefault("point_mode", source.capabilities()["default_point_mode"])
        result = dict(id=query["id"], metric=query.get("metric"), object_ids=query.get("object_ids"),point_mode=query["point_mode"])
        try:
            backend._ids(query["object_ids"])
            if not 1 <= len(query["object_ids"]) <= 6 or len(set(query["object_ids"])) != len(query["object_ids"]):
                raise ValueError("Select 1..6 distinct object_ids")
            start, end = float(query.get("start_seconds",0)), float(query.get("end_seconds",source.end))
            if not np.isfinite([start,end]).all() or not 0 <= start <= end:
                raise ValueError("Use a finite nonnegative ordered interval")
            result["interval_seconds"] = [start,end]
            if end > source.end:
                raise LookupError("Requested interval exceeds the exported horizon; continue the experiment if needed")
            if query.get("point_mode", "center") not in {"center", "members"}:
                raise ValueError("point_mode must be center or members")
            frames = interval_frames(start,end,backend.fps)
            if query["metric"] == "contact":
                events = source.events(query["object_ids"],start,end)
                result.update(events, condition_met=events["event_occurs"], status=events["event_status"])
            else:
                method = {"region":measure_region,"proximity":measure_proximity,"tilt":measure_tilt,"box_overlap":measure_boxes}.get(query["metric"])
                if method is None:
                    raise ValueError("metric must be region, proximity, tilt, box_overlap or contact")
                result.update(method(backend,source,query,frames))
                if "variant" in query:
                    variant = query["variant"]
                    if not isinstance(variant,str) or variant not in result["variants"]:
                        raise ValueError("variant must identify an exported fitted system: " + ", ".join(result["variants"]))
                    result["variants"] = {variant:result["variants"][variant]}
                    result["selected_variant"] = variant
                    result["variant_scope"] = "Conditional on the explicitly selected fitted system; other systems were not combined."
                values = [r["condition_met"] for r in result["variants"].values()]
                result["condition_met"] = values[0] if values and all(v is values[0] for v in values) else None
                result["status"] = "measured" if any(r["status"] == "measured" for r in result["variants"].values()) else "needs_query"
        except KeyError as error:
            hint = (' Use object_ids for the source and target={"object_id":"existing_id"}, with tolerance_px.'
                    if query.get("metric") == "proximity" else " Check this metric's advertised fields.")
            result.update(status="invalid_query",condition_met=None,reason=f"Missing required field {error.args[0]!r}.{hint} Correct this query on the existing source; no new simulation is needed.")
        except (LookupError, OSError) as error:
            result.update(status="needs_query",condition_met=None,reason=str(error))
        except (ValueError, TypeError) as error:
            result.update(status="invalid_query",condition_met=None,reason=str(error))
        results.append(result)
    return dict(**source.provenance(), image_coordinates=dict(source_wh=[backend.width,backend.height], axes="x right, y down"), results=results)
