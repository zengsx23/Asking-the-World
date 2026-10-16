"""Observed source identities from outlet timing and preceding mask ownership."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.signal import find_peaks

from atw.perception.pixel_segmentation_materials import mask_archive


def first_run(counts):
    """First pair of consecutive frames with at least two observed outlet pixels."""
    present = np.asarray(counts) >= 2
    starts = np.flatnonzero(present[:-1] & present[1:])
    return int(starts[0]) if len(starts) else None


def source_candidates(raw, static, rows, domain, fluids):
    """Find compact top-boundary surfaces; no names or reference labels are used."""
    yy, xx = np.where(domain)
    if not len(yy):
        return []
    top, height = yy.min(), np.ptp(yy)
    candidates = []
    for row in rows:
        if row['is_receiving_structure']:
            continue
        sy, sx = np.where(static == int(row['id'][1:]))
        if not len(sx):
            continue
        x, y, r, b = int(sx.min()), int(sy.min()), int(sx.max()+1), int(sy.max()+1)
        if b-y < .065*height and r-x > 1.25*(b-y) and (y+b)/2 < top+.15*height:
            candidates.append(dict(static_object_id=row['id'], box=[x, y, r, b]))
    if not fluids:
        return candidates
    colors = np.array([o['principal_rgb'][::-1] for o in fluids], float)
    distances = np.linalg.norm(raw[:, :, None].astype(float)-colors[None, None], axis=-1)
    closest, minimum = distances.argmin(-1), distances.min(-1)
    grid_y, grid_x = np.indices(static.shape)
    upper = ((grid_y >= top-.08*height) & (grid_y < top+.15*height)
             & (grid_x >= xx.min()-3) & (grid_x <= xx.max()+3))
    for group in range(len(fluids)):
        region = (closest == group) & (minimum < 65) & upper & (static == 0)
        region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(region, 8)
        for cid in range(1, count):
            x, y, width, local_height, area = map(int, stats[cid])
            r, b = x+width, y+local_height
            if (area < .0002*static.size or local_height >= .065*height
                    or width <= 1.25*local_height or width < .02*static.shape[1]
                    or b >= top+.15*height):
                continue
            duplicate = any(
                max(0, min(r, c['box'][2]+2)-max(x, c['box'][0]-2))
                * max(0, min(b, c['box'][3]+2)-max(y, c['box'][1]-2)) > .3*width*local_height
                for c in candidates)
            if not duplicate:
                candidates.append(dict(static_object_id=None, box=[x, y, r, b]))
    return candidates


def identify_outlets(candidates, masks, fluids, fps):
    """Split observed outlet peaks and bind each source to a fixed color group."""
    if not candidates or not fluids:
        return []
    typical = np.median([c['box'][2]-c['box'][0] for c in candidates])
    height = masks.shape[1]
    sources = []
    for candidate in candidates:
        x, y, r, b = candidate['box']
        bottom = min(b+8, height)
        profile = np.isin(masks[:, b:bottom, x:r], [o['mask_label'] for o in fluids]).sum((0, 1))
        peaks = []
        if r-x > 1.6*typical and profile.max() > 0:
            smooth = gaussian_filter1d(profile.astype(float), 1)
            peaks = find_peaks(smooth, distance=max(4, int(.6*typical)),
                               prominence=max(4, .15*smooth.max()))[0].tolist()
        cuts = ([x]+[x+int((a+b)/2) for a, b in zip(peaks, peaks[1:])]+[r]
                if len(peaks) > 1 else [x, r])
        for left, right in zip(cuts, cuts[1:]):
            center, half = (left+right)/2, max(2, (right-left)*.35)
            l, rr = max(left, int(center-half)), min(right, int(center+half)+1)
            region = masks[:, b:bottom, l:rr]
            counts = np.array([(region == o['mask_label']).sum((1, 2)) for o in fluids])
            group = int(counts.sum(1).argmax())
            onset = first_run(counts[group])
            if candidate['static_object_id'] is None and onset is None:
                continue
            # A static candidate without observed outflow has no supported color binding.
            fluid = fluids[group] if onset is not None else None
            index = len(sources)+1
            sources.append(dict(
                source_id=f'E{index}', mask_label=index,
                static_object_id=candidate['static_object_id'],
                fluid_object_id=fluid['object_id'] if fluid else None,
                group_id=fluid['group_id'] if fluid else None,
                box=[left, y, right, b], outlet_box=[l, b, rr, bottom],
                first_visible_frame=onset,
                first_visible_seconds=None if onset is None else onset/fps,
                split_from_static=len(cuts) > 2,
                outlet_pixel_counts=counts[group].tolist() if fluid else [0]*len(masks)))
    return sources


def assign_frame(current, previous, sources, fluids, frame):
    """Preserve existing ownership; only observed outlet pixels seed new identities."""
    result = np.zeros_like(current)
    for fluid in fluids:
        observed = current == fluid['mask_label']
        active = [s for s in sources if s['group_id'] == fluid['group_id']
                  and s['first_visible_frame'] is not None and frame >= s['first_visible_frame']]
        if not active:
            continue
        if len(active) == 1:
            result[observed] = active[0]['mask_label']
            continue
        _, components = cv2.connectedComponents(observed.astype(np.uint8), 8)
        scores = []
        for source in active:
            prior = previous == source['mask_label']
            seed = prior & observed
            x, y, r, b = source['outlet_box']
            seed[y:b, x:r] |= observed[y:b, x:r]
            if seed.any():
                distance = distance_transform_edt(~seed)
                ids = np.unique(components[seed])
                ids = ids[ids != 0]
                score = np.where(np.isin(components, ids), distance, np.inf)
            else:
                score = np.full(observed.shape, np.inf)
            if prior.any():
                distance = distance_transform_edt(~prior)
                score = np.minimum(score, np.where(distance <= 12, distance, np.inf))
            scores.append(score)
        scores = np.stack(scores)
        winner = scores.argmin(0)
        supported = np.isfinite(scores.min(0)) & observed
        for i, source in enumerate(active):
            result[supported & (winner == i)] = source['mask_label']
    return result


def bind(raw, labels, rows, domain, objects, masks_path, frame_count, fps, video_path, output):
    """Write full temporal source masks and return a JSON-serializable manifest.

    Source mask labels are local to this separate archive. F objects remain color
    groups in the original tracking archive; E source IDs never replace them.
    """
    output.mkdir()
    width = min(480, raw.shape[1])
    height = max(1, round(raw.shape[0]*width/raw.shape[1]))
    size = width, height
    original = cv2.resize(raw, size)
    static = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
    domain = cv2.resize(domain.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    fluids = [o for o in objects if o['object_kind'] == 'fluid']
    candidates = source_candidates(original, static, rows, domain, fluids)
    with np.load(masks_path) as archive:
        masks = np.stack([cv2.resize(archive[f'frame_{f:06d}'], size, interpolation=cv2.INTER_NEAREST)
                          for f in range(frame_count)])
    sources = identify_outlets(candidates, masks, fluids, fps)
    previous = np.zeros((height, width), np.uint16)
    unassigned = []
    with mask_archive(output/'identities.npz') as archive:
        for frame, current in enumerate(masks):
            result = assign_frame(current, previous, sources, fluids, frame)
            previous = result
            observed = np.isin(current, [o['mask_label'] for o in fluids])
            unassigned.append(int((observed & (result == 0)).sum()))
            with archive.open(f'frame_{frame:06d}.npy', 'w') as member:
                np.save(member, result, allow_pickle=False)
    return dict(
        protocol_version=1, identity_semantics='observed_source_provenance',
        source_video_path=str(video_path), frame_scope=[0, frame_count-1], fps=fps,
        coordinates=dict(width=width, height=height, units='pixels', origin='top_left',
                         x_direction='right', y_direction='down', box_format='xyxy_half_open'),
        mask_encoding=dict(dtype='uint16', archive_key='frame_{frame_index:06d}',
                           background_or_unassigned_label=0, source_label_field='mask_label'),
        sources=sources, unassigned_liquid_pixel_counts=unassigned,
        limitations=[
            'Assumes fixed camera, downward flow and compact sources near the upper enclosure.',
            'Source IDs identify observed outlets, not separate emission episodes or material particles.',
            'One static object can support multiple outlets; recovered outlets can have null static_object_id.',
            'Outlet onset is observed visibility, not exact physical emission time; no outflow means null binding.',
            'Wide single outlets can resemble multiple sources; long mixing and occlusion remain uncertain.',
            'Zero labels include non-liquid and unsupported liquid; compare the referenced color-mask archive.',
            'Static and boundary geometry remain unchanged; provenance masks refine only fluid identity.'])
