"""Independent visual motion judgment, conditional on rejected SWR and unknown tracks."""
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from atw.world_modeling import sam3_protocol as c
from atw.world_modeling.scene_assets import write
from atw.openrouter import OpenRouterClient

PROMPT = Path(__file__).resolve().parents[1] / 'prompts/rigid_motion_review.txt'
DECISION = c.obj(motion=dict(type='string', enum=['static', 'moving']), reason=c.TEXT)


def valid_frames(payload):
    """Observed mask lifetime only; 3D pose fitting quality is irrelevant here."""
    valid = (~payload['border_suppressed'].astype(bool)
             & (payload['mask_pixel_counts'] > 0)
             & ~payload['occlusion_terminated'].astype(bool))
    return np.flatnonzero(valid).tolist()


def render_clips(video, masks, frames, output):
    """Retain original time/color and split gaps, never showing excluded frames."""
    import imageio_ffmpeg
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not frames:
        raise ValueError('No visible tracking frames for independent motion review')
    segments = np.split(np.asarray(frames), np.flatnonzero(np.diff(frames) != 1) + 1)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Cannot open original video')
    width, height = (int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    clips, scope = [], []
    try:
        for i, segment in enumerate(segments, 1):
            path = output / f'clip_{i:03d}.mp4'
            proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '1', '-preset', 'fast',
                '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(segment[0]))
                for f in segment:
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError(f'Cannot decode original frame {f}')
                    mask = masks[int(f)]
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, (0, 0, 255), 3, cv2.LINE_AA)
                    proc.stdin.write(image.tobytes())
            finally:
                proc.stdin.close()
                code = proc.wait()
            if code:
                raise RuntimeError('Motion evidence encoding failed')
            clips.append(path)
            scope.append(dict(clip=path.name, source_frame_start=int(segment[0]), source_frame_end=int(segment[-1]),
                              frame_count=len(segment), fps=fps))
    finally:
        cap.release()
    write(output / 'scope.json', dict(source_video=str(video), frames=frames, clips=scope,
        semantics='Original RGB frames with a derived red mask contour; original playback speed; excluded frames absent.'))
    return clips


def render_local_timeline(video, masks, frames, output):
    """Six original-time crops with one shared window/scale, preserving target motion."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not frames:
        raise ValueError('No visible tracking frames for local motion evidence')
    chosen = [frames[i] for i in sorted(set(np.linspace(0, len(frames)-1, min(6, len(frames))).astype(int)))]
    bounds = []
    for f in frames:
        ys, xs = np.nonzero(masks[f])
        bounds.append((xs.min(), ys.min(), xs.max()+1, ys.max()+1))
    bounds = np.asarray(bounds)
    x0, y0 = bounds[:, :2].min(axis=0)
    x1, y1 = bounds[:, 2:].max(axis=0)
    padding = max(16, round(max(x1-x0, y1-y0)*.15))
    height, width = masks[frames[0]].shape
    x0, y0, x1, y1 = max(0, x0-padding), max(0, y0-padding), min(width, x1+padding), min(height, y1+padding)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError('Cannot open original video for local motion evidence')
    fps = cap.get(cv2.CAP_PROP_FPS)
    tiles = []
    try:
        for f in chosen:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode original frame {f}')
            contours, _ = cv2.findContours(masks[f].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 0, 255), 3, cv2.LINE_AA)
            crop = image[y0:y1, x0:x1]
            scale = min(400/crop.shape[1], 280/crop.shape[0])
            crop = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))))
            tile = np.full((320, 420, 3), 255, np.uint8)
            left = (420-crop.shape[1])//2
            tile[35:35+crop.shape[0], left:left+crop.shape[1]] = crop
            cv2.putText(tile, f'{len(tiles)+1} | t = {f/fps:.2f} s', (12, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(tile)
    finally:
        cap.release()
    while len(tiles) % 3:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.concatenate([np.concatenate(tiles[i:i+3], axis=1) for i in range(0, len(tiles), 3)], axis=0)
    path = output / 'local_timeline.jpg'
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError('Cannot write local motion evidence')
    write(output / 'scope.json', dict(source_video=str(video), frames=chosen,
          fixed_crop_xyxy=list(map(int, (x0, y0, x1, y1))), fps=fps))
    return path


def judge(clips, output, vlm, prompt=None, local_timeline=None):
    """No prior labels, unknown flag, object descriptions, or admission reason."""
    from atw.perception.sam3_agent import _media_data_url
    content = [dict(type='text', text='Judge only the red-outlined target in these observation clips.' + c.prompt(DECISION))]
    for path in [*clips, *([local_timeline] if local_timeline is not None else [])]:
        kind, url = _media_data_url(path)
        content.append(dict(type=kind, **{kind: dict(url=url)}))
    def validate(value):
        c.validate(DECISION, value, 'motion_review')
        return value
    return OpenRouterClient(vlm).complete([
        dict(role='system', content=PROMPT.read_text() if prompt is None else prompt),
        dict(role='user', content=content)], output, validate=validate)


def unknown_objects(inputs):
    if inputs['tracking']['route'] != '3D+SAM3':
        return []
    folder = inputs.get('tracking_directory')
    if folder is None:
        return []
    report = json.loads((folder / 'rigid_motion/decision.json').read_text())
    return [o for o in inputs['objects'] if o['physical_type'] == 'rigid'
            and report['objects'][o['object_id']]['state'] == 'unknown']


def review(inputs, context):
    """Caller must first establish failed admission; masks and source tracks are read-only."""
    targets = unknown_objects(inputs)
    if not targets:
        return None
    from atw.perception.object_reconstruction import read_packed_mask_archive
    manifest, inventory, index, archive, objects = inputs['bundle']
    tracks = read_packed_mask_archive(archive, inputs['segmentation_path'].parent / manifest['mask_index'])
    output = context.output_dir / 'rigid_motion_review'
    output.mkdir()
    decisions = []
    for obj in targets:
        obj_dir = output / obj['object_id']
        with np.load(inputs['tracking_directory'] / 'rigid_trajectories' / (obj['object_id'] + '.npz'), allow_pickle=False) as z:
            frames = valid_frames(z)
        clips = render_clips(Path(context.inputs['video_path']), tracks[obj['track_id']], frames, obj_dir / 'clips')
        local_timeline = render_local_timeline(Path(context.inputs['video_path']), tracks[obj['track_id']], frames, obj_dir / 'local')
        answer = judge(clips, obj_dir / 'request', context.vlm, local_timeline=local_timeline)
        decisions.append(dict(object_id=obj['object_id'], track_id=obj['track_id'], **answer,
                              evidence_clips=[str(p.relative_to(context.run_dir)) for p in clips],
                              local_timeline=str(local_timeline.relative_to(context.run_dir)), source_frames=frames))
    path = output / 'review.json'
    write(path, dict(protocol_version=1, objects=decisions, source_tracking=inputs['tracking']['sources'],
                     scope='Independent observed motion review; no unseen or post-termination motion inference.'))
    return path, decisions


def prepare_reviewed_tracks(inputs, context, decisions):
    """Create a private tracking view; rebuild accepted static poses from the aligned model."""
    from dataclasses import fields
    from atw.perception.rigid_tracking import static_rigid_pose_sequence
    from atw.perception.pipeline.surface import support_constraint_from_gravity_summary
    from atw.world_modeling.scene_geometry import load_geometry
    from atw.world_modeling.scene_assets import asset_path

    root = context.output_dir / 'reviewed_motion_inputs'
    directory = root / 'tracking'
    directory.mkdir(parents=True)
    source = inputs['tracking_directory']
    reconstruction = inputs['reconstruction_directory']
    (root / 'reconstruction').symlink_to(reconstruction, target_is_directory=True)
    for p in source.iterdir():
        if p.name not in ('rigid_trajectories', 'rigid_motion'):
            (directory / p.name).symlink_to(p, target_is_directory=p.is_dir())
    trajectories = directory / 'rigid_trajectories'
    trajectories.mkdir()
    by_id = {d['object_id']: d for d in decisions}
    static_ids = {d['object_id'] for d in decisions if d['motion'] == 'static'}
    if static_ids:
        gravity_path = asset_path(context, inputs['tracking']['sources']['gravity_asset_id'])
        geometry = load_geometry(gravity_path.parent, inputs['gravity']['geometry'])
        masks = {}
        for obj in inputs['objects']:
            with np.load(reconstruction / 'masks' / (obj['object_id'] + '.npz')) as z:
                masks[obj['object_id']] = z['masks'].astype(bool)
        union = np.logical_or.reduce(list(masks.values()))
        support = support_constraint_from_gravity_summary(inputs['gravity']['gravity'])
    for p in (source / 'rigid_trajectories').glob('*.npz'):
        if p.stem not in by_id:
            (trajectories / p.name).symlink_to(p)
            continue
        with np.load(p, allow_pickle=False) as z:
            values = {k: z[k] for k in z.files}
        if p.stem in static_ids:
            with np.load(reconstruction / 'models/rigid' / p.stem / 'canonical_rigid_model.npz') as model:
                points, colors, pose = (model[k] for k in ('canonical_surface_points', 'canonical_surface_colors', 'reference_pose'))
            trajectory = static_rigid_pose_sequence(canonical_points=points, reference_pose=pose,
                reference_frame=int(values['reference_frame']), masks=masks[p.stem], all_object_mask_union=union,
                camera_poses=geometry.camera_poses, intrinsics=geometry.intrinsics, support_constraint=support)
            values.update({f.name: getattr(trajectory, f.name) for f in fields(trajectory)})
            values['canonical_tracking_colors'] = colors[trajectory.canonical_tracking_indices]
            values['motion_model'] = np.asarray('static')
        values['motion_state'] = np.asarray('static_compatible' if p.stem in static_ids else 'moving')
        np.savez_compressed(trajectories / p.name, **values)
    motion = json.loads((source / 'rigid_motion/decision.json').read_text())
    for key, decision in by_id.items():
        motion['objects'][key].update(state='static_compatible' if decision['motion'] == 'static' else 'moving',
            tracking_branch='static' if decision['motion'] == 'static' else 'dynamic', source='independent_visual_review',
            reason=decision['reason'])
    motion['policy'] = 'Retained program decisions, with independently reviewed unknown objects only.'
    write(directory / 'rigid_motion/decision.json', motion)
    updated = dict(inputs['tracking'], tracking_directory='tracking',
                   reconstruction_manifest='reconstruction/reconstruction_manifest.json',
                   motion_review='../rigid_motion_review/review.json')
    # This is an effective fitter input, not an in-place replacement of the observed artifact.
    path = root / 'tracking.json'
    write(path, updated)
    return dict(inputs, tracking_directory=directory, reconstruction_directory=root / 'reconstruction'), path
