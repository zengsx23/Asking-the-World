"""Split-kernel cloth rollout with colored or unchanged-order constraints.

Kernels are extracted from the cloth solver. Colored constraints change
execution order; serial constraints retain the original order.
"""

import ast
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import warp as wp

from atw.polyworld.cloth_swr import BatchWarpClothEngine
from atw.polyworld.cloth_swr_execution import InterleavedClothEngine


def edge_color_groups(edges):
    occupied, groups = [], []
    for index, (a, b) in enumerate(edges):
        for used, group in zip(occupied, groups):
            if int(a) not in used and int(b) not in used:
                used.update((int(a), int(b)))
                group.append(index)
                break
        else:
            occupied.append({int(a), int(b)})
            groups.append([index])
    return [np.asarray(g, np.int32) for g in groups]


def generate_split_kernels(directory=None, *, dynamic_frames=False):
    source = Path(__file__).with_name('cloth_swr.py').read_text()
    tree = ast.parse(source)
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_simulate_cloth_worlds')
    frame = original.body[-1]
    assert isinstance(frame, ast.For) and ast.unparse(frame.target) == 'frame'
    step = frame.body[2]
    constraints = step.body[2]
    assert ast.unparse(step.target) == 'substep'
    assert ast.unparse(constraints.target) == '_iteration'
    assert len(original.body) == 11 and len(frame.body) == 5
    assert len(step.body) == 4 and len(constraints.body) == 4
    prelude = original.body[:8]
    release = frame.body[:2]
    pieces = []
    inputs = {}

    def kernel(name, body, tid=None, edges=False):
        function = copy.deepcopy(original)
        function.name = name
        function.body = copy.deepcopy(prelude + release + body)
        if tid:
            function.body[0] = ast.parse(tid).body[0]
        function.args.args += [ast.arg(arg='frame', annotation=ast.Name(id='int', ctx=ast.Load())),
                               ast.arg(arg='substep', annotation=ast.Name(id='int', ctx=ast.Load()))]
        if edges:
            function.args.args += ast.parse('def f(edge_ids: wp.array(dtype=int), edge_offset: int): pass').body[0].args.args
            function.body.insert(1, ast.parse('edge = edge_ids[slot + edge_offset]').body[0])
        # Kernels only carry parameters they actually read, keeping graph nodes small.
        used = {n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        function.args.args = [a for a in function.args.args if a.arg in used]
        if dynamic_frames and 'frame' in used:
            function.args.args = [a for a in function.args.args if a.arg != 'frame']
            function.args.args += ast.parse(
                'def f(frame_base: wp.array(dtype=int), frame_offset: int): pass'
            ).body[0].args.args
            function.body.insert(1, ast.parse('frame = frame_base[0] + frame_offset').body[0])
        inputs[name] = [a.arg for a in function.args.args]
        ast.fix_missing_locations(function)
        pieces.append(ast.unparse(function))

    kernel('initialize', original.body[8:10])
    kernel('rigid_step', [step.body[0]])
    kernel('nodes_step', step.body[1].body, 'world, node = wp.tid()')
    kernel('structural_serial', [constraints.body[0]])
    kernel('bending_serial', [constraints.body[1]])
    kernel('structural_color', constraints.body[0].body, 'world, slot = wp.tid()', edges=True)
    kernel('bending_color', constraints.body[1].body, 'world, slot = wp.tid()', edges=True)
    kernel('contacts', constraints.body[2:])
    kernel('velocities_step', step.body[3].body, 'world, node = wp.tid()')
    kernel('store_frame', frame.body[3:])
    header = '''# Generated from agent/cloth_swr.py. Do not edit.
import warp as wp
from atw.polyworld.cloth_swr import (
    GRAVITY_M_PER_S2, CLOTH_RIGID_FRICTION, RIGID_RIGID_FRICTION,
    _rotate_vector, _solve_box_floor_contact, _project_obb_particle,
    _box_inverse_inertia, _obb_separating_axis, _obb_contact_point,
)
wp.set_module_options({"enable_backward": False})
'''
    text = header + '\n\n'.join(pieces)
    if dynamic_frames:
        inputs.update(set_frame=['frame_base', 'amount'], advance_frame=['frame_base', 'amount'])
        text += '''

@wp.kernel
def set_frame(frame_base: wp.array(dtype=int), amount: int):
    frame_base[0] = amount

@wp.kernel
def advance_frame(frame_base: wp.array(dtype=int), amount: int):
    frame_base[0] += amount
'''
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    name = f'_cloth_split_{digest}'
    if directory is None:
        wp.init()
        directory = Path(wp.config.kernel_cache_dir) / 'physmind-cloth'
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + '.py')
    if not path.exists():
        # Multiple scene workers may initialize the same source cache concurrently.
        with tempfile.NamedTemporaryFile(mode='w', dir=directory, suffix='.py', delete=False) as temporary:
            temporary.write(text)
            temporary_path = temporary.name
        try:
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name], inputs, [a.arg for a in original.args.args]


class SplitClothEngine(BatchWarpClothEngine):
    def __init__(self, *args, kernel_directory=None, colored=False,
                 reuse_release_prefix=False, graph_chunk_frames=0, contact_iteration_stride=1, **kwargs):
        if not isinstance(graph_chunk_frames, int) or graph_chunk_frames < 0:
            raise ValueError('graph_chunk_frames must be a nonnegative integer')
        super().__init__(*args, **kwargs)
        self.colored = colored
        if not isinstance(contact_iteration_stride, int) or contact_iteration_stride < 1:
            raise ValueError('contact_iteration_stride must be a positive integer')
        self.contact_iteration_stride = contact_iteration_stride
        self.reuse_release_prefix = reuse_release_prefix
        self.graph_chunk_frames = graph_chunk_frames
        self.frame_base = wp.zeros(1, dtype=int, device=self.device) if graph_chunk_frames else None
        self.kernels, self.input_names, self.original_input_names = generate_split_kernels(
            kernel_directory, dynamic_frames=bool(graph_chunk_frames))
        self.graphs = {}
        self.color_indices = {}
        self.color_ranges = {}
        for name, edges in [('structural', self.scene.structural_edges), ('bending', self.scene.bending_edges)]:
            groups = edge_color_groups(edges)
            indices = np.concatenate(groups) if groups else np.empty(0, np.int32)
            self.color_indices[name] = wp.array(indices, dtype=int, device=self.device)
            offset, ranges = 0, []
            for group in groups:
                ranges.append((offset, len(group)))
                offset += len(group)
            self.color_ranges[name] = ranges

    def _launch(self, count, final_offset, *, start_offset=1, initialize=True):
        if self.graph_chunk_frames:
            if initialize:
                self._launch_segment(count, 0, initialize=True)
            else:
                wp.launch(self.kernels.set_frame, dim=1, inputs=[self.frame_base, start_offset],
                          device=self.device, block_dim=self.block_dim)
            for start in range(start_offset, final_offset + 1, self.graph_chunk_frames):
                self._launch_segment(count, min(self.graph_chunk_frames, final_offset + 1 - start),
                                     initialize=False)
        else:
            self._launch_segment(count, final_offset, start_offset=start_offset, initialize=initialize)

    def _launch_segment(self, count, final_offset, *, start_offset=1, initialize=True):
        key = (count, final_offset, start_offset, initialize)
        if self.torch_device.type == 'cuda':
            if key not in self.graphs:
                wp.load_module(self.kernels, device=self.device, block_dim=self.block_dim)
                with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
                    self._steps(count, final_offset, start_offset=start_offset, initialize=initialize)
                self.graphs[key] = capture.graph
            wp.capture_launch(self.graphs[key])
        else:
            self._steps(count, final_offset, start_offset=start_offset, initialize=initialize)

    def rollout_release_candidates(self, samples):
        """Share only causally identical states before the earliest release."""
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != self.parameter_count or len(samples) > self.capacity:
            raise ValueError("invalid release candidate batch")
        if not len(samples) or not np.isfinite(samples).all():
            raise ValueError("release candidates must be finite and nonempty")
        if self.boundary_mode != "edge-drive" or self.edge_release_frame is not None:
            return self.rollout(samples)
        final = self.scene.frame_count - 1
        prefix_end = min(final, max(0, math.ceil(float(samples[:, 10].min())) - 1))
        # Parameter 10 is the only excluded coordinate. Material, initial state,
        # drive and rigid parameters must all match exactly before sharing state.
        _, first, inverse = np.unique(np.delete(samples, 10, axis=1), axis=0,
                                      return_index=True, return_inverse=True)
        if prefix_end < 1 or len(first) == len(samples):
            return self.rollout(samples)
        self.rollout(samples[first], prefix_end)
        indices = torch.as_tensor(inverse, dtype=torch.long, device=self.torch_device)
        names = ["positions", "previous_positions", "velocities", "predicted",
                 "rigid_centers", "rigid_axis_x", "rigid_axis_y", "rigid_axis_z",
                 "rigid_velocities", "rigid_angular_velocities",
                 "predicted_rigid_centers", "predicted_rigid_axis_x",
                 "predicted_rigid_axis_y", "predicted_rigid_axis_z"]
        for name in names:
            tensor = wp.to_torch(getattr(self, name))
            if name.startswith("predicted"):
                tensor = tensor[:, :prefix_end + 1]
            tensor[:len(samples)].copy_(tensor[:len(first)].index_select(0, indices))
        self.theta_torch[:len(samples)].copy_(torch.as_tensor(samples, device=self.torch_device))
        if prefix_end < final:
            self._launch(len(samples), final, start_offset=prefix_end + 1, initialize=False)
        return self.predicted_torch[:len(samples), :final + 1].contiguous()

    def _steps(self, count, final_offset, *, start_offset=1, initialize=True):
        values = dict(zip(self.original_input_names, self._kernel_inputs(final_offset)))
        values.update(frame=0, substep=0)
        if self.graph_chunk_frames:
            values.update(frame_base=self.frame_base, frame_offset=0)

        def launch(name, dim):
            wp.launch(getattr(self.kernels, name), dim=dim,
                      inputs=[values[k] for k in self.input_names[name]],
                      device=self.device, block_dim=self.block_dim)

        if initialize:
            if self.graph_chunk_frames:
                values['amount'] = 0
                launch('set_frame', 1)
            launch('initialize', count)
            if self.graph_chunk_frames:
                values['amount'] = 1
                launch('advance_frame', 1)
        frames = range(final_offset) if self.graph_chunk_frames else range(start_offset, final_offset + 1)
        for frame in frames:
            values['frame'] = frame
            values['frame_offset'] = frame
            for substep in range(self.substeps):
                values['substep'] = substep
                if self.dynamic_obstacles:
                    launch('rigid_step', count)
                launch('nodes_step', (count, self.scene.node_count))
                for iteration in range(self.constraint_iterations):
                    for kind in ['structural', 'bending']:
                        if self.colored:
                            values['edge_ids'] = self.color_indices[kind]
                            for offset, size in self.color_ranges[kind]:
                                values['edge_offset'] = offset
                                launch(kind + '_color', (count, size))
                        else:
                            launch(kind + '_serial', count)
                    # Always finish a substep with contact resolution, including
                    # iteration counts not divisible by the contact cadence.
                    if ((iteration + 1) % self.contact_iteration_stride == 0
                            or iteration + 1 == self.constraint_iterations):
                        self._launch_contacts(launch, count, values)
                launch('velocities_step', (count, self.scene.node_count))
            launch('store_frame', count)
        if self.graph_chunk_frames and final_offset:
            values['amount'] = final_offset
            launch('advance_frame', 1)

    def _launch_contacts(self, launch, count, values):
        launch('contacts', count)


class InterleavedSplitClothEngine(SplitClothEngine, InterleavedClothEngine):
    """Split launches with world-interleaved storage and loss layout."""
