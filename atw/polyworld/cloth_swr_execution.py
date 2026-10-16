"""Default cloth execution optimizations with independent CEM seed states."""

import torch
import warp as wp

from atw.polyworld.cloth_swr import BatchWarpClothEngine, _candidate_loss, _cem_iterations


def run_cem_portfolio(engine, configs, *, verbose=False):
    """Batch independent seeds without merging RNGs, losses or elite sets."""
    if not configs:
        raise ValueError("at least one CEM configuration is required")
    if any(c.generations_per_prefix <= 0 or not c.prefix_frames for c in configs):
        raise ValueError("batched seeds require a nonempty, positive-generation schedule")
    if sum(c.population for c in configs) > engine.capacity:
        raise ValueError("engine capacity is smaller than the combined populations")
    schedules = {(c.prefix_frames, c.generations_per_prefix) for c in configs}
    if len(schedules) != 1:
        raise ValueError("batched seeds require matching prefix/generation schedules")
    iterations = [_cem_iterations(engine, c, verbose=verbose) for c in configs]
    requests = [next(it) for it in iterations]
    while True:
        end = requests[0][1]
        if any(request[1] != end for request in requests):
            raise RuntimeError("CEM seed schedules diverged")
        samples = torch.cat([request[0] for request in requests])
        prediction = engine.rollout(samples, end)
        offset = 0
        results = []
        next_requests = []
        for iterator, config in zip(iterations, configs):
            loss = _candidate_loss(engine, prediction[offset:offset+config.population],
                                   end_offset=end, huber_delta_m=config.huber_delta_m)
            offset += config.population
            try:
                next_requests.append(iterator.send(loss))
            except StopIteration as result:
                results.append(result.value)
        if results:
            if len(results) != len(configs):
                raise RuntimeError("CEM seeds completed at different generations")
            return results
        requests = next_requests


class InterleavedClothEngine(BatchWarpClothEngine):
    """Preserve logical indices while storing adjacent worlds contiguously."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        names = ["positions", "previous_positions", "velocities", "predicted",
                 "rigid_centers", "rigid_axis_x", "rigid_axis_y", "rigid_axis_z",
                 "rigid_velocities", "rigid_angular_velocities",
                 "predicted_rigid_centers", "predicted_rigid_axis_x",
                 "predicted_rigid_axis_y", "predicted_rigid_axis_z"]
        self._interleaved_storage = []
        for name in names:
            old = getattr(self, name)
            storage = torch.empty((*old.shape[1:], old.shape[0], 3),
                                  dtype=torch.float32, device=self.torch_device)
            logical = storage.movedim(-2, 0)
            self._interleaved_storage.append(storage)
            array = wp.from_torch(logical, dtype=wp.vec3)
            setattr(self, name, array)
            if hasattr(self, name + "_torch"):
                setattr(self, name + "_torch", wp.to_torch(array))

    def rollout(self, values, end_offset=None):
        # Keep the original reduction layout and floating-point summation order.
        return super().rollout(values, end_offset).contiguous()
