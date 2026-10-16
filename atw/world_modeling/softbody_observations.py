"""Mask-space targets and initial state calibration from explicit scene assets."""
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Callable
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from atw.polyworld import soft_body_swr as swr
from atw.polyworld.experiment_views import image_projection


@dataclass
class Observations:
    centers: np.ndarray
    valid: np.ndarray
    offsets: np.ndarray
    project: Callable[[np.ndarray], np.ndarray]
    first_areas: np.ndarray
    image_wh: tuple[int, int]


def pixels(scene, raw, obs):
    world = raw[..., 0, None]*scene.wall_lateral_world + raw[..., 1, None]*scene.gravity_down_world + scene.wall_depths[:, None]*scene.wall_normal_world
    return obs.project(world.reshape(-1, 3)).reshape(*raw.shape[:-1], 2) + obs.offsets


def load(scene, inputs):
    manifest, _, index, archive, objects = inputs['bundle']
    width, height = manifest['width'], manifest['height']
    tracks = {o['object_id']: o['track_id'] for o in objects}
    xy = np.zeros((scene.frame_count, scene.object_count, 2))
    valid = np.zeros(xy.shape[:2], bool)
    areas = np.zeros(scene.object_count)
    with np.load(archive, allow_pickle=False) as masks:
        for i, name in enumerate(scene.object_ids):
            for f, frame in enumerate(scene.frames):
                key = f'{tracks[name]}__frame_{frame:05d}'
                if key not in index or not scene.observed_mask[f, i]:
                    continue
                info = index[key]
                mask = np.unpackbits(masks[key])[:info['num_bits']].reshape(info['height'], info['width'])
                yy, xx = np.where(mask)
                if not len(xx):
                    continue
                xy[f, i] = [xx.mean()*width/info['width'], yy.mean()*height/info['height']]
                valid[f, i] = True
                if f == scene.first_offsets[i]:
                    areas[i] = len(xx)*width/info['width']*height/info['height']
    if np.any(areas <= 0):
        raise ValueError('Observed-radius fitting requires a nonempty first mask for every ball')
    backend = SimpleNamespace(engine='soft_body', paths={'tracking': inputs['tracking_directory']}, width=width, height=height)
    project = image_projection(backend, {})
    offsets = []
    for i in range(scene.object_count):
        frames = np.arange(0, scene.frame_count, 4)
        frames = frames[valid[frames, i]]
        if not len(frames):
            raise ValueError('Missing mask-to-camera calibration observations')
        points = scene.target_centers[frames, i]
        world = points[:, :1]*scene.wall_lateral_world + points[:, 1:]*scene.gravity_down_world + scene.wall_depths[i]*scene.wall_normal_world
        offsets.append(np.median(xy[frames, i] - project(world), axis=0))
    return Observations(xy, valid, np.asarray(offsets), project, areas, (width, height))


def radius_scales(scene, obs):
    angles = np.linspace(0, 2*np.pi, 256, endpoint=False)
    scales = []
    for i in range(scene.object_count):
        plane = scene.initial_centers[i] + scene.radii[i]*np.column_stack([np.cos(angles), np.sin(angles)])
        world = plane[:, :1]*scene.wall_lateral_world + plane[:, 1:]*scene.gravity_down_world + scene.wall_depths[i]*scene.wall_normal_world
        p = obs.project(world)
        area = abs(np.sum(p[:, 0]*np.roll(p[:, 1], 1)-p[:, 1]*np.roll(p[:, 0], 1)))/2
        if not np.isfinite(area) or area <= 0:
            raise ValueError('Degenerate reference-circle camera projection')
        scales.append(np.sqrt(obs.first_areas[i]/area))
    return np.asarray(scales)


def observed_states(scene, obs):
    positions = np.zeros_like(obs.centers)
    for i in range(scene.object_count):
        plane = scene.initial_centers[i] + np.array([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
        world = plane[:, :1]*scene.wall_lateral_world + plane[:, 1:]*scene.gravity_down_world + scene.wall_depths[i]*scene.wall_normal_world
        uv = obs.project(world) + obs.offsets[i]
        homography = cv2.getPerspectiveTransform(uv.astype('float32'), plane.astype('float32'))
        p = np.column_stack([obs.centers[:, i], np.ones(scene.frame_count)]) @ homography.T
        positions[:, i] = p[:, :2]/p[:, 2:]
    velocity = savgol_filter(positions, 7, 2, deriv=1, delta=1/scene.fps, axis=0)
    valid = np.stack([np.convolve(obs.valid[:, i].astype(int), np.ones(7), 'same') == 7 for i in range(scene.object_count)], axis=1)
    return positions, velocity, valid


def calibrate_initial_state(scene, obs, theta):
    layout = swr.make_parameter_layout(scene.object_count)
    count = scene.object_count
    frames = []
    for i, first in enumerate(scene.first_offsets):
        clear = []
        for f in range(first, min(first+12, scene.frame_count)):
            if not obs.valid[f, i]:
                break
            pos = scene.target_centers[f, i]
            if any(swr._obb_signed_distance_and_normal(pos, o)[0] < 1.3*scene.radii[i] for o in scene.obstacles):
                break
            if any(j != i and obs.valid[f, j] and np.linalg.norm(pos-scene.target_centers[f, j]) < 1.3*(scene.radii[i]+scene.radii[j]) for j in range(count)):
                break
            clear.append(f)
        if len(clear) < 7:
            raise ValueError(f'{scene.object_ids[i]} requires seven initial clear observed frames')
        frames.append(np.array(clear))
    initial = np.r_[scene.initial_centers.ravel(), theta[:2*count], np.mean(np.exp(theta[layout.log_gravity_scale_slice]))*9.81]
    def residual(values):
        p0 = values[:2*count].reshape(count, 2)
        v0 = values[2*count:4*count].reshape(count, 2)
        raw = np.zeros((scene.frame_count, count, 2))
        for i in range(count):
            t = (np.arange(scene.frame_count)-scene.first_offsets[i])/scene.fps
            raw[:, i] = p0[i] + t[:, None]*v0[i] + .5*t[:, None]**2*np.array([0, values[-1]])
        pred = pixels(scene, raw, obs)
        return np.concatenate([(pred[f, i]-obs.centers[f, i]).ravel() for i, f in enumerate(frames)])
    low = np.full(len(initial), -np.inf)
    high = -low
    low[-1], high[-1] = .03*9.81, 1.5*9.81
    result = least_squares(residual, initial, bounds=(low, high), max_nfev=1000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    if not result.success:
        raise ValueError('Initial free-flight calibration did not converge')
    theta = theta.copy()
    theta[:2*count] = result.x[2*count:4*count]
    theta[layout.log_gravity_scale_slice] = np.log(result.x[-1]/9.81)
    theta[layout.drag_slice] = 0
    scene = replace(scene, initial_centers=result.x[:2*count].reshape(count, 2))
    return scene, theta, dict(frames=[f.tolist() for f in frames], gravity_scale=float(result.x[-1]/9.81), early_pixel_rmse=float(np.sqrt(np.mean(residual(result.x)**2)*2)))


def metrics(scene, raw, obs):
    squared = ((pixels(scene, raw, obs)-obs.centers)**2).sum(-1)
    return dict(pixel_center_rmse=float(np.sqrt(squared[obs.valid].mean())), objects=[dict(object_id=name, pixel_center_rmse=float(np.sqrt(squared[obs.valid[:, i], i].mean()))) for i, name in enumerate(scene.object_ids)])
