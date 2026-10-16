"""Fixed-scale native-RGB crops for adjacent motion; outputs use scene pixels."""
from pathlib import Path

import cv2
import numpy as np

from atw.perception.track4world_tool import AdjacentTrack4WorldResult, _bilinear_sample, run_adjacent_track4world_tracking, save_adjacent_track4world_cache, load_adjacent_track4world_cache


def crop_transforms(masks, original_wh, size=256):
    count, height, width = masks.shape
    ratio = np.asarray(original_wh, float)/[width, height]
    centers = np.full((count, 2), np.nan)
    sides = []
    for t, mask in enumerate(masks):
        y, x = np.nonzero(mask)
        if len(x):
            centers[t] = (np.array([x.min()+x.max(), y.min()+y.max()])/2+.5)*ratio-.5
            sides.append(max((x.max()-x.min()+1)*ratio[0], (y.max()-y.min()+1)*ratio[1]))
    side = float(np.ceil(max(sides)*1.5))
    valid = np.flatnonzero(np.isfinite(centers[:, 0]))
    for axis in range(2):
        centers[:, axis] = np.interp(np.arange(count), valid, centers[valid, axis])
    scale = ratio*size/side
    offset = ((ratio*.5-.5)[None]-centers)*size/side+(size-1)/2
    return dict(centers_native=centers, side_native=side, scale=scale, offset=offset,
                size=size, native_to_crop_scale=size/side)


def crop_rgb(frames, transforms):
    size = int(transforms['size'])
    y, x = np.mgrid[:size, :size].astype(np.float32)
    crops = []
    for frame, center in zip(frames, transforms['centers_native']):
        xx = (x-(size-1)/2)/transforms['native_to_crop_scale']+center[0]
        yy = (y-(size-1)/2)/transforms['native_to_crop_scale']+center[1]
        crops.append(cv2.remap(frame, xx.astype(np.float32), yy.astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT))
    return np.stack(crops)


def map_crop_motion(motion, transforms, shape, reverse=False, frame_count=None):
    count = motion.frame_count
    total = count if frame_count is None else frame_count
    h, w = shape
    y, x = np.mgrid[:h, :w]
    grid = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    uv = np.full((total-1, h, w, 2), np.nan, np.float32)
    confidence = np.zeros((total-1, h, w), np.float32)
    indices = np.arange(count)[::-1] if reverse else np.arange(count)
    for pair, (source, target) in enumerate(zip(indices[:-1], indices[1:])):
        queries = grid*transforms['scale']+transforms['offset'][source]
        endpoints = _bilinear_sample(motion.source_to_target_uv[pair], queries)
        target_pair = pair+total-count if reverse else pair
        uv[target_pair] = ((endpoints-transforms['offset'][target])/transforms['scale']).reshape(h, w, 2)
        conf = _bilinear_sample(motion.tracking_confidence[pair], queries)
        confidence[target_pair] = np.nan_to_num(conf).reshape(h, w)
    return AdjacentTrack4WorldResult(total, uv, confidence)


def local_soft_body_motion(model, frames, masks, *, video_path, output_dir,
                           device, inference_iterations, cache_dir=None):
    """Reuse one loaded model; cache forward/reverse fields in global pixel space."""
    output_dir = Path(output_dir)
    frame_count, h, w = masks.shape
    end = int(np.flatnonzero(masks.any(axis=(1, 2)))[-1])+1
    original_wh = (frames.shape[2], frames.shape[1])
    transforms = crop_transforms(masks[:end], original_wh)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        with np.load(cache_dir/'crop_transforms.npz') as stored:
            for key, value in transforms.items():
                if not np.array_equal(stored[key], value):
                    raise ValueError(f'local motion cache has different crop {key}: {cache_dir}')
        fields = tuple(load_adjacent_track4world_cache(cache_dir/direction,
            source_video_path=video_path, frame_count=frame_count, image_size_wh=(w, h))
            for direction in ('forward', 'backward'))
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.symlink_to(cache_dir.resolve(), target_is_directory=True)
        return fields
    output_dir.mkdir(parents=True)
    crops = crop_rgb(frames[:end], transforms)
    np.savez_compressed(output_dir/'crop_transforms.npz', **transforms)
    fields = []
    for reverse, direction in ((False, 'forward'), (True, 'backward')):
        if end == 1:
            local = AdjacentTrack4WorldResult(1, np.empty((0, 256, 256, 2), np.float32),
                                              np.empty((0, 256, 256), np.float32))
        else:
            local = run_adjacent_track4world_tracking(model,
                np.ascontiguousarray(crops[::-1]) if reverse else crops,
                device=device, inference_iterations=inference_iterations)
        field = map_crop_motion(local, transforms, (h, w), reverse, frame_count)
        save_adjacent_track4world_cache(output_dir/direction, field,
            source_video_path=video_path, inference_iterations=inference_iterations)
        fields.append(field)
    return tuple(fields)
