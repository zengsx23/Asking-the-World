"""Same-frame seed containment checks, before tracker initialization only."""
import numpy as np

COVERAGE_THRESHOLD = .9


def contained_objects(target_id, frame_index, mask, seeds):
    """Coverage is intersection / smaller mask area, never pairwise IoU."""
    area = int(np.count_nonzero(mask))
    contained = []
    for other_id in sorted(seeds):
        other = seeds[other_id]
        if other_id == target_id or other['frame_index'] != frame_index:
            continue
        other_area = int(np.count_nonzero(other['mask']))
        if not 0 < other_area < area:
            continue
        coverage = int(np.count_nonzero(mask & other['mask'])) / other_area
        if coverage >= COVERAGE_THRESHOLD:
            contained.append(dict(target_id=other_id, coverage=coverage))
    return contained


def resolve_seed_containment(*, objects, seeds, manifest, reselect):
    """Reselect each conflicting large seed at most once, then remove single subsets.

    Reselection uses the existing bounded seed-selection chain. Exhausted empty
    selections omit that seed while retaining the original inventory evidence.
    Single-subset removal is deferred until all multi-object conflicts are handled.
    """
    seeds = dict(seeds)
    manifest = list(manifest)
    by_id = {o['target_id']: o for o in objects}
    attempted, events = set(), []
    while True:
        relations = {
            key: contained_objects(key, seed['frame_index'], seed['mask'], seeds)
            for key, seed in sorted(seeds.items())
        }
        multiple = [key for key, children in relations.items() if len(children) >= 2]
        if not multiple:
            break
        key = multiple[0]
        conflict = dict(target_id=key, frame_index=seeds[key]['frame_index'],
                        contained=relations[key])
        replacement = None
        if key not in attempted:
            attempted.add(key)
            replacement = reselect(by_id[key], conflict, dict(seeds))
        if replacement is None:
            seeds.pop(key)
            events.append(dict(action='omit_unselected_large_seed', **conflict))
        else:
            obj, seed, record = replacement
            if obj['target_id'] != key:
                raise ValueError('Containment reselection must preserve the target_id')
            by_id[key].update(obj)
            seeds[key] = seed
            manifest = [r for r in manifest if r['target']['target_id'] != key] + [record]
            events.append(dict(action='reselect_large_seed', **conflict))
    # Evaluate the remaining containment graph once; never mutate it while scanning.
    removed = {child['target_id'] for children in relations.values() for child in children}
    for key, children in relations.items():
        if children:
            events.append(dict(action='remove_single_contained_seed', target_id=key,
                               removed_target_id=children[0]['target_id'],
                               frame_index=seeds[key]['frame_index'],
                               coverage=children[0]['coverage']))
    seeds = {key: seed for key, seed in seeds.items() if key not in removed}
    active = [o for o in objects if o['target_id'] in seeds]
    manifest = [r for r in manifest if r['target']['target_id'] in seeds]
    return active, seeds, manifest, dict(
        scope='same_frame_seeds_before_propagation', coverage_threshold=COVERAGE_THRESHOLD,
        events=events, retained_target_ids=[o['target_id'] for o in active])
