"""Observed rigid trajectories, recent motion and mask relations.
"""
import cv2
import numpy as np


def _rounded(values, digits=5):
    return [round(float(value), digits) for value in values]


class TrackingReadout:
    def _dense_trajectory(self, object_id, start, end):
        self._ids([object_id])
        path = self.paths["tracking"] / "rigid_trajectories" / f"{object_id}.npz"
        if not path.is_file():
            return []
        rows = []
        with np.load(path) as data:
            poses = data["poses"]
            first = max(0, int(np.ceil(start * self.fps)))
            last = min(len(poses) - 1, int(np.floor(end * self.fps)))
            intervals = self.visibility.get(object_id, {}).get("intervals", [])
            for frame in range(first, last + 1):
                if 0 <= self.termination.get(object_id, -1) <= frame:
                    continue
                if intervals and not any(span["first_frame"] <= frame <= span["last_frame"]
                                         for span in intervals):
                    continue
                center = poses[frame, :3, 3]
                if np.isfinite(center).all():
                    rows.append((frame, center.astype(float)))
        return rows

    def query_trajectory(self, object_id, start_seconds=0., end_seconds=None,
                         sample_count=5):
        start, end, count = self._interval(start_seconds, end_seconds, sample_count)
        if not 3 <= count <= 8:
            raise ValueError("sample_count must be an integer from 3 to 8")
        rows = self._dense_trajectory(object_id, start, end)
        if not rows:
            return dict(status="unavailable", source_kind="tracked_observation",
                        object_id=object_id, interval_seconds=[start, end],
                        reason="No observed rigid trajectory samples in the requested interval")
        frames = np.array([row[0] for row in rows])
        centers = np.stack([row[1] for row in rows])
        indices = np.unique(np.linspace(0, len(rows) - 1, min(count, len(rows))).astype(int))
        keyframes = [dict(frame_index=int(frames[index]),
                          time_s=round(float(frames[index] / self.fps), 3),
                          center_world=_rounded(centers[index])) for index in indices]
        displacement = centers[-1] - centers[0]
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        recent_start = max(0, len(rows) - max(3, int(round(.5 * self.fps))))
        recent_times = frames[recent_start:] / self.fps
        recent_centers = centers[recent_start:]
        velocity = None
        if len(recent_times) >= 2 and recent_times[-1] > recent_times[0]:
            velocity = np.array([np.polyfit(recent_times, recent_centers[:, axis], 1)[0]
                                 for axis in range(3)])
        return dict(
            status="observed", source_kind="tracked_observation", object_id=object_id,
            interval_seconds=[start, end],
            coverage=dict(first_time_s=round(float(frames[0] / self.fps), 3),
                          last_time_s=round(float(frames[-1] / self.fps), 3),
                          observed_frame_count=len(rows)),
            motion=dict(net_displacement_world=_rounded(displacement),
                        net_distance_world=round(float(np.linalg.norm(displacement)), 5),
                        path_length_world=round(float(steps.sum()), 5),
                        recent_velocity_world_per_s=_rounded(velocity) if velocity is not None else None),
            keyframes=keyframes,
            coordinate_system=self.coordinate_system,
            interpretation="Observed tracking only; extrapolation beyond the last time is reasoning, not evidence.",
        )

    def query_mask_motion(self, object_id, start_seconds=0., end_seconds=None):
        self._ids([object_id])
        start, end, _ = self._interval(start_seconds, end_seconds, 6)
        rows = []
        for frame in self._frames(start, end, 6):
            try:
                mask = self._mask(object_id, int(frame))
            except (KeyError, OSError, ValueError):
                continue
            yy, xx = np.where(mask)
            if not len(xx):
                continue
            rows.append(dict(frame_index=int(frame), time_s=round(float(frame / self.fps), 3),
                             centroid_px=[round(float(xx.mean()), 2), round(float(yy.mean()), 2)],
                             visible_pixels=int(len(xx))))
        if not rows:
            return dict(status="unavailable", source_kind="mask_observation",
                        object_id=object_id, interval_seconds=[start, end],
                        reason="No retained mask samples in the requested interval")
        first, last = rows[0], rows[-1]
        centroid_delta = np.subtract(last["centroid_px"], first["centroid_px"])
        areas = [row["visible_pixels"] for row in rows]
        return dict(
            status="observed", source_kind="mask_observation", object_id=object_id,
            interval_seconds=[start, end],
            visibility=dict(first_sample_s=first["time_s"], last_sample_s=last["time_s"],
                            sampled_visible_count=len(rows)),
            motion=dict(centroid_displacement_px=_rounded(centroid_delta, 2),
                        visible_area_px=dict(first=areas[0], last=areas[-1],
                                             minimum=min(areas), maximum=max(areas))),
            keyframes=rows,
            interpretation="Observed image masks; occlusion, clipping and perspective can change centroid and area.",
        )

    @staticmethod
    def _mask_gap(first, second):
        if np.logical_and(first, second).any():
            return 0.
        distance = cv2.distanceTransform((~first).astype(np.uint8), cv2.DIST_L2, 3)
        values = distance[second]
        return float(values.min()) if len(values) else float("inf")

    def query_observed_relation(self, object_ids, start_seconds=0., end_seconds=None,
                                sample_count=8):
        if not isinstance(object_ids, list) or len(object_ids) != 2 or len(set(object_ids)) != 2:
            raise ValueError("object_ids must contain exactly two distinct observed IDs")
        self._ids(object_ids)
        start, end, count = self._interval(start_seconds, end_seconds, sample_count)
        if not 3 <= count <= 12:
            raise ValueError("sample_count must be an integer from 3 to 12")
        tracks = [self._dense_trajectory(name, start, end) for name in object_ids]
        maps = [{frame: center for frame, center in rows} for rows in tracks]
        common = sorted(set(maps[0]) & set(maps[1]))
        tracking = None
        if common:
            distances = np.array([np.linalg.norm(maps[0][frame] - maps[1][frame])
                                  for frame in common])
            minimum = int(np.argmin(distances))
            recent = distances[max(0, len(distances) - max(3, int(round(.5 * self.fps)))):]
            tracking = dict(
                common_frame_count=len(common),
                first_time_s=round(float(common[0] / self.fps), 3),
                last_time_s=round(float(common[-1] / self.fps), 3),
                start_distance_world=round(float(distances[0]), 5),
                end_distance_world=round(float(distances[-1]), 5),
                minimum_distance_world=round(float(distances[minimum]), 5),
                minimum_time_s=round(float(common[minimum] / self.fps), 3),
                recent_distance_change_world=round(float(recent[-1] - recent[0]), 5) if len(recent) > 1 else None,
            )
        sampled = self._frames(start, end, count)
        mask_rows = []
        for frame in sampled:
            try:
                first = self._mask(object_ids[0], int(frame))
                second = self._mask(object_ids[1], int(frame))
            except (KeyError, OSError, ValueError):
                continue
            if not first.any() or not second.any():
                continue
            gap = self._mask_gap(first, second)
            mask_rows.append((int(frame), gap))
        mask_relation = None
        if mask_rows:
            minimum_frame, minimum_gap = min(mask_rows, key=lambda row: row[1])
            candidates = [round(frame / self.fps, 3) for frame, gap in mask_rows if gap <= 1.5]
            mask_relation = dict(
                sampled_frame_count=len(mask_rows),
                minimum_gap_px=round(float(minimum_gap), 2),
                minimum_time_s=round(float(minimum_frame / self.fps), 3),
                contact_candidate_times_s=candidates[:4],
            )
        if tracking is None and mask_relation is None:
            return dict(status="unavailable", source_kind="observed_tracking_and_masks",
                        object_ids=object_ids, interval_seconds=[start, end],
                        reason="The pair has no common observed tracking or mask coverage")
        return dict(
            status="observed", source_kind="observed_tracking_and_masks",
            object_ids=object_ids, interval_seconds=[start, end],
            tracking_relation=tracking, mask_relation=mask_relation,
            interpretation=("A small mask gap is only an image-space contact candidate; depth ordering, "
                            "occlusion and physical collision must be judged from the original video."),
        )
