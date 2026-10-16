"""Deterministic liquid body-color groups and original-color naming evidence."""
from atw.world_modeling.pixel_foreground import changed as foreground_changed
import cv2
import numpy as np
from skimage.color import deltaE_ciede2000

from atw.perception import pixel_segmentation_cv as cv
from atw.perception.pixel_segmentation_materials import board, card
from atw.world_modeling.scene_assets import write

SUPPORT_FRACTION = .01
GROUP_DISTANCE = 12
SHADE_DISTANCE = 20


def cluster(anchors, distances, threshold=GROUP_DISTANCE):
    """Complete linkage bounds the entire group's diameter, preventing color chains."""
    groups = [[i] for i in anchors]
    while len(groups) > 1:
        distance, i, j = min((float(distances[np.ix_(a, b)].max()), i, j)
                            for i, a in enumerate(groups) for j, b in enumerate(groups) if i < j)
        if distance > threshold:
            break
        groups[i] = sorted(groups[i] + groups[j])
        groups.pop(j)
    return sorted(groups, key=lambda group: group[0])


def body_groups(palette, support):
    """Select supported interior colors, then assign each shade to at most one group."""
    minimum = sum(support) * SUPPORT_FRACTION
    anchors = [i for i, count in enumerate(support) if count > 0 and count >= minimum]
    if not anchors:
        raise ValueError('No supported liquid body-color prototypes outside static material')
    lab = palette['centers'].astype(float).copy()
    lab[:, 0] /= 2.55
    lab[:, 1:] -= 128
    distances = deltaE_ciede2000(lab[:, None, :], lab[None, :, :])
    clustered = cluster(anchors, distances)
    nearest = np.stack([distances[:, group].min(1) for group in clustered])
    owners = nearest.argmin(0)
    groups = []
    for i, cores in enumerate(clustered):
        principal = max(cores, key=lambda index: support[index])
        groups.append(dict(group_id=f'G{i+1:02d}', core_palette_ids=cores,
            palette_ids=np.where((owners == i) & (nearest[i] <= SHADE_DISTANCE))[0].tolist(),
            principal_palette_id=principal, principal_rgb=palette['bgr'][principal][::-1].tolist()))
    return groups


def prepare(video, info, palette, reference, domain, wall, output):
    """Use 11 original frames and fixed geometry; no model decisions affect grouping."""
    output.mkdir()
    frames = cv.sample_frames(info.frame_count, np.linspace(0, 1, 11))
    raws = {frame: cv.read_frame(video, frame, info.working_wh) for frame in frames}
    baseline = raws[0]
    allowed = (reference == 0) & domain & ~wall
    labels = {frame: palette['lut'][cv.encode_bgr(raw)] for frame, raw in raws.items()}
    changed = {frame: allowed & foreground_changed(raw, baseline) for frame, raw in raws.items()}
    support = [sum(int(cv2.erode(((labels[frame] == i) & changed[frame]).astype(np.uint8),
                               np.ones((3, 3), np.uint8)).sum()) for frame in frames)
               for i in range(len(palette['centers']))]
    groups = body_groups(palette, support)
    tiles = []
    for group in groups:
        evidence = []
        for frame in frames:
            selected = np.isin(labels[frame], group['palette_ids']) & changed[frame]
            count, _, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), 8)
            if count < 2:
                continue
            index = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
            x, y, width, height, area = map(int, stats[index])
            evidence.append((area, frame, (x, y, width, height), selected))
        panels = []
        for _, frame, (x, y, width, height), selected in sorted(evidence, key=lambda row: row[0], reverse=True)[:2]:
            raw = raws[frame]
            left, top = max(0, x-12), max(0, y-12)
            right, bottom = min(raw.shape[1], x+width+12), min(raw.shape[0], y+height+12)
            isolated = np.zeros_like(raw)
            isolated[selected] = raw[selected]
            pair = np.hstack([raw[top:bottom, left:right], isolated[top:bottom, left:right]])
            panels.append(card(pair, f'{group["group_id"]} frame {frame}: CONTEXT / GROUP ONLY', (640, 280)))
        while len(panels) < 2:
            panels.append(np.zeros((280, 640, 3), np.uint8))
        header = np.zeros((64, 640, 3), np.uint8)
        header[:, :80] = palette['bgr'][group['principal_palette_id']]
        cv2.putText(header, f'{group["group_id"]} color group', (92, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        tile = np.vstack([header, *panels])
        cv2.imwrite(str(output/f'{group["group_id"]}.jpg'), tile)
        tiles.append(tile)
    pages = [board(tiles[start:start+6], output/f'page_{start//6}.jpg', 3)
             for start in range(0, len(tiles), 6)]
    write(output/'groups.json', dict(source_video_path=str(video), frame_scope=[0, info.frame_count-1],
        sampled_frames=frames, working_width=info.working_wh[0], working_height=info.working_wh[1],
        support_units='3x3-eroded working-image pixels summed across sampled frames',
        parameters=dict(foreground_method='chromatic_change_or_brightening', chromatic_threshold=12, positive_mean_threshold=24,
                        interior_kernel_size=3, support_fraction=SUPPORT_FRACTION,
                        color_metric='CIEDE2000', complete_link_distance=GROUP_DISTANCE,
                        nearest_shade_distance=SHADE_DISTANCE),
        core_support=support, groups=groups))
    return groups, pages
