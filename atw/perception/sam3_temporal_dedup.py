"""Conservative geometric identity checks before visible-mask arbitration."""
from itertools import combinations
import numpy as np


def pair_evidence(first, second, anchors):
    common = sorted(set(first) & set(second))
    sampled = set(common[i] for i in np.linspace(0, len(common)-1, min(24, len(common)), dtype=int)) if common else set()
    sampled.update(f for f in anchors if f in common)
    rows = []
    for frame in sorted(sampled):
        a, b = first[frame], second[frame]
        aa, bb = int(a.sum()), int(b.sum())
        intersection = int(np.count_nonzero(a & b))
        union = aa + bb - intersection
        rows.append(dict(frame=frame, first_area=aa, second_area=bb,
                         iou=intersection/union if union else 0.,
                         first_coverage=intersection/aa if aa else 0.,
                         second_coverage=intersection/bb if bb else 0.))
    references = [float(np.percentile([r[key] for r in rows], 90)) if rows else 0.
                  for key in ('first_area','second_area')]
    for r in rows:
        r['first_area_ratio'] = r['first_area']/max(1.,references[0])
        r['second_area_ratio'] = r['second_area']/max(1.,references[1])
        r['informative'] = min(r['first_area_ratio'],r['second_area_ratio']) >= .25
        r['strong_overlap'] = r['iou'] >= .85 and min(r['first_coverage'],r['second_coverage']) >= .85
    informative = [r for r in rows if r['informative']]
    distinct = [r for r in informative if r['iou'] <= .1]
    duration = max(set(first) | set(second), default=0)-min(set(first) | set(second), default=0)
    distinct_span = distinct[-1]['frame']-distinct[0]['frame'] if distinct else 0
    sustained_distinct = len(distinct) >= 3 and distinct_span >= .1*max(1,duration)
    fraction = sum(r['strong_overlap'] for r in informative)/len(informative) if informative else 0.
    common_fraction = len(common)/max(1,len(set(first)|set(second)))
    late_small = [r['frame'] for r in rows if r['strong_overlap'] and not r['informative']
                  and distinct and r['frame'] > distinct[-1]['frame']]
    duplicate = len(informative) >= 5 and fraction >= .8 and common_fraction >= .8 and not sustained_distinct
    by_frame = {r['frame']:r for r in rows}
    # Anchor agreement is diagnostic only; identity is decided from temporal evidence.
    anchor_ok = all(f in by_frame and by_frame[f]['iou'] >= .8 for f in anchors)
    if sustained_distinct:
        reason = 'distinct_while_visible_then_small_overlap' if late_small else 'sustained_distinct_visible_tracks'
    elif len(informative) < 5:
        reason = 'insufficient_visible_support'
    elif common_fraction < .8:
        reason = 'insufficient_common_temporal_support'
    else:
        reason = 'persistent_same_visible_region' if duplicate else 'insufficient_persistent_overlap'
    return dict(duplicate=duplicate, reason=reason, sampled_frames=rows,
                anchor_agreement=anchor_ok, strong_fraction=fraction,
                informative_frame_count=len(informative), common_frame_fraction=common_fraction,
                distinct_visible_frames=[r['frame'] for r in distinct],
                sustained_distinct=sustained_distinct, late_small_overlap_frames=late_small,
                reference_areas=references,
                median_iou=float(np.median([r['iou'] for r in rows])) if rows else 0.)


def find_duplicate_tracks(objects, seeds, tracks):
    """Require complete-link evidence; overlap chains must not merge identities."""
    ids = [o['target_id'] for o in objects]
    evidence, accepted = [], {}
    quality = {key: [] for key in ids}
    for a, b in combinations(ids, 2):
        frames = [int(seeds[k]['frame_index']) for k in (a, b)]
        item = pair_evidence(tracks.get(a, {}), tracks.get(b, {}), frames)
        disjoint_seeds = frames[0] == frames[1] and not np.any(
            np.asarray(seeds[a]['mask'],dtype=bool) & np.asarray(seeds[b]['mask'],dtype=bool))
        item['same_frame_disjoint_seeds'] = bool(disjoint_seeds)
        if disjoint_seeds:
            item['temporal_reason'] = item['reason']
            item['duplicate'] = False
            item['reason'] = 'disjoint_seeds_on_same_frame'
        evidence.append(dict(first=a, second=b, **item))
        accepted[frozenset((a,b))] = item['duplicate']
        if item['duplicate']:
            by_frame = {r['frame']: r['iou'] for r in item['sampled_frames']}
            for key, frame in zip((a,b),frames):
                quality[key].append(by_frame.get(frame,0.))
    ranked = sorted(ids, key=lambda key: (-float(np.mean(quality[key])) if quality[key] else 0., ids.index(key)))
    groups = []
    for key in ranked:
        group = next((g for g in groups if all(accepted[frozenset((key,other))] for other in g)), None)
        if group is None:
            groups.append([key])
        else:
            group.append(key)
    aliases = {key:g[0] for g in groups for key in g[1:]}
    return dict(method='multi_frame_bidirectional_mask_agreement',
                thresholds=dict(iou=.85,coverage=.85,strong_fraction=.8,minimum_informative_frames=5,
                    informative_area_ratio=.25,distinct_iou=.1,minimum_distinct_samples=3,
                    distinct_duration_fraction=.1,common_frame_fraction=.8),
                aliases=aliases, groups=[g for g in groups if len(g)>1], pairs=evidence,
                representative_rule='highest seed-frame agreement, then inventory order')


def discover_duplicate_tracks(*, tracker, tracker_resource, objects, seeds, frame_count):
    """Inspect identity evidence without overlap arbitration; restore tracker settings."""
    from atw.perception import sam3_tool as sam
    settings = {key:getattr(tracker,key) for key in
                ('non_overlap_masks_for_mem_enc','non_overlap_masks_for_output')}
    try:
        for key in settings:
            setattr(tracker,key,False)
        candidates, discovery = sam._run_shared_tracker(tracker=tracker,
            tracker_resource=tracker_resource,objects=objects,seeds=seeds,
            frame_count=frame_count,resolve_overlap=False)
        report = find_duplicate_tracks(objects,seeds,candidates)
        del candidates
    finally:
        for key,value in settings.items():
            setattr(tracker,key,value)
    report['discovery_tracking'] = discovery
    return report


def track_with_deduplication(*, tracker, tracker_resource, objects, seeds, frame_count):
    """Detect identities, then rebuild all memory using retained seeds only."""
    from atw.perception import sam3_tool as sam
    report = discover_duplicate_tracks(tracker=tracker,tracker_resource=tracker_resource,
        objects=objects,seeds=seeds,frame_count=frame_count)
    active = [o for o in objects if o['target_id'] not in report['aliases']]
    retained = {o['target_id']:seeds[o['target_id']] for o in active}
    # _run_shared_tracker creates fresh states, including fresh memory encodings.
    tracks, manifest = sam._run_shared_tracker(tracker=tracker,
        tracker_resource=tracker_resource,objects=active,seeds=retained,frame_count=frame_count)
    report['clean_repropagation'] = True
    report['retained_ids'] = list(retained)
    return active, retained, tracks, manifest, report
