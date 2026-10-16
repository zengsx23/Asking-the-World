"""Graph-captured rope solver with optional two-color stretch.

Kernels are extracted from the rope solver equations. Pulley contacts stay serial
within each world: multiple nodes and edges may update the same dynamic pulley.
"""

import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import time

import warp as wp

from atw.polyworld.rope_swr import BatchWarpRopeEngine


def generate_split_kernels(directory=None, *, dynamic_frames=False):
    source = Path(__file__).with_name("rope_swr.py").read_text()
    tree = ast.parse(source)
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_simulate_rope_worlds")
    assert len(original.body) == 8
    frame = original.body[-1]
    assert isinstance(frame, ast.For) and ast.unparse(frame.target) == "frame"
    assert len(frame.body) == 3 and isinstance(frame.body[0], ast.If)
    step = frame.body[0].body[0]
    assert ast.unparse(step.target) == "_substep" and len(step.body) == 5
    constraints = step.body[2]
    assert ast.unparse(constraints.target) == "_iteration" and len(constraints.body) == 3
    assert [ast.unparse(n.target) for n in constraints.body] == ["edge", "node", "edge"]
    prelude = original.body[:5]
    pieces, inputs = [], {}

    def kernel(name, body, tid=None, colored=False):
        function = copy.deepcopy(original)
        function.name = name
        function.body = copy.deepcopy(prelude + body)
        if tid:
            function.body[0] = ast.parse(tid).body[0]
        function.args.args += ast.parse("def f(frame: int, parity: int): pass").body[0].args.args
        if colored:
            # Adjacent chain edges have alternating colors; padded final slots
            # are guarded for both odd and even node counts.
            function.body = function.body[:5] + ast.parse("edge = 2 * slot + parity").body + [
                ast.If(test=ast.parse("edge < node_count - 1", mode="eval").body,
                       body=function.body[5:], orelse=[])
            ]
        used = {n.id for n in ast.walk(function)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        function.args.args = [a for a in function.args.args if a.arg in used]
        if dynamic_frames and "frame" in used:
            function.args.args = [a for a in function.args.args if a.arg != "frame"]
            function.args.args += ast.parse(
                "def f(frame_base: wp.array(dtype=int), frame_offset: int): pass"
            ).body[0].args.args
            function.body.insert(1, ast.parse("frame = frame_base[0] + frame_offset").body[0])
        inputs[name] = [a.arg for a in function.args.args]
        ast.fix_missing_locations(function)
        pieces.append(ast.unparse(function))

    kernel("initialize", original.body[5:7])
    kernel("nodes_step", step.body[0].body, "world, node = wp.tid()")
    kernel("pulleys_step", [step.body[1]])
    kernel("stretch_serial", [constraints.body[0]])
    kernel("stretch_color", constraints.body[0].body, "world, slot = wp.tid()", colored=True)
    kernel("contacts", constraints.body[1:])
    kernel("node_contacts", constraints.body[1].body, "world, node = wp.tid()")
    kernel("edge_contacts_serial", [constraints.body[2]])

    class EdgeContinueToReturn(ast.NodeTransformer):
        # Once an edge gets its own thread, a continue of the removed edge loop
        # becomes a thread return. Continues within the pulley loop stay intact.
        def visit_For(self, node):
            return node

        def visit_Continue(self, node):
            return ast.copy_location(ast.Return(value=None), node)

    edge_contacts = [EdgeContinueToReturn().visit(copy.deepcopy(node))
                     for node in constraints.body[2].body]
    kernel("edge_contacts_color", edge_contacts, "world, slot = wp.tid()", colored=True)
    kernel("nodes_velocity", step.body[3].body, "world, node = wp.tid()")
    kernel("pulleys_velocity", [step.body[4]])
    kernel("store_frame", frame.body[1:])
    text = '# Generated from agent/rope_swr.py. Do not edit.\nimport warp as wp\n'
    text += 'wp.set_module_options({"enable_backward": False})\n\n' + "\n\n".join(pieces)
    if dynamic_frames:
        inputs.update(reset_frame=["frame_base"], advance_frame=["frame_base", "amount"])
        text += '''

@wp.kernel
def reset_frame(frame_base: wp.array(dtype=int)):
    frame_base[0] = 0

@wp.kernel
def advance_frame(frame_base: wp.array(dtype=int), amount: int):
    frame_base[0] += amount
'''
    return _load_generated_module(text, directory), inputs, [a.arg for a in original.args.args]


def _load_generated_module(text, directory=None):
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    name = f"_rope_split_{digest}"
    if directory is None:
        wp.init()
        directory = Path(wp.config.kernel_cache_dir) / "physmind-rope"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".py")
    if not path.exists():
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, suffix=".py", delete=False) as temporary:
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
    return sys.modules[name]


class SplitRopeEngine(BatchWarpRopeEngine):
    def __init__(self, *args, colored=False, parallel_static_contacts=False,
                 parallel_static_nodes=False,
                 preserve_coupled_contacts=False,
                 graph_chunk_frames=0,
                 kernel_directory=None, **kwargs):
        if not isinstance(graph_chunk_frames, int) or graph_chunk_frames < 0:
            raise ValueError("graph_chunk_frames must be a nonnegative integer")
        super().__init__(*args, **kwargs)
        self.graph_chunk_frames = graph_chunk_frames
        self.frame_base = wp.zeros(1, dtype=int, device=self.device) if graph_chunk_frames else None
        self.colored = colored
        self.parallel_static_contacts = (
            parallel_static_contacts and not any(body.dynamic for body in self.pulleys)
        )
        self.parallel_static_nodes = (
            (parallel_static_nodes or parallel_static_contacts)
            and not any(body.dynamic for body in self.pulleys)
        )
        if preserve_coupled_contacts and any(body.dynamic for body in self.scene.bodies):
            self.parallel_static_contacts = False
            self.parallel_static_nodes = False
        self.kernels, self.input_names, self.original_input_names = generate_split_kernels(
            kernel_directory, dynamic_frames=bool(graph_chunk_frames))
        self.graphs = {}
        self.graph_capture_seconds = 0.0

    @property
    def execution_config(self) -> dict:
        return dict(
            solver="split", cuda_block_dim=self.block_dim,
            stretch_order="two-color" if self.colored else "original",
            contact_order=("two-color" if self.parallel_static_contacts else
                           "parallel-nodes-original-edges" if self.parallel_static_nodes else "original"),
            cuda_graphs=self.torch_device.type == "cuda",
            cuda_graph_count=len(self.graphs),
            graph_capture_seconds=self.graph_capture_seconds,
            graph_chunk_frames=self.graph_chunk_frames,
        )

    def _launch(self, count, final):
        if self.graph_chunk_frames:
            self._launch_segment(count, 0, initialize=True)
            for start in range(0, final, self.graph_chunk_frames):
                self._launch_segment(count, min(self.graph_chunk_frames, final - start), initialize=False)
        else:
            self._launch_segment(count, final, initialize=True)

    def _launch_segment(self, count, final, *, initialize):
        key = (count, final, self.colored, self.simulate_static_system,
               self.substeps, self.constraint_iterations, self.parallel_static_contacts,
               self.block_dim)
        key += (self.parallel_static_nodes, initialize)
        key += (getattr(self, "_driven_node", -1), getattr(self, "_drive_speed", 0.))
        key += (self.scene.event_frame_offset,)
        if self.torch_device.type == "cuda":
            if key not in self.graphs:
                wp.load_module(self.kernels, device=self.device, block_dim=self.block_dim)
                started = time.perf_counter()
                with wp.ScopedCapture(device=self.device, force_module_load=False) as capture:
                    self._steps(count, final, initialize=initialize)
                self.graphs[key] = capture.graph
                self.graph_capture_seconds += time.perf_counter() - started
            wp.capture_launch(self.graphs[key])
        else:
            self._steps(count, final, initialize=initialize)

    def _steps(self, count, final, *, initialize=True):
        values = dict(zip(self.original_input_names, self._kernel_inputs(final)))
        values.update(frame=0, parity=0)
        if self.graph_chunk_frames:
            values.update(frame_base=self.frame_base, frame_offset=0)

        def launch(name, dim):
            wp.launch(getattr(self.kernels, name), dim=dim,
                      inputs=[values[k] for k in self.input_names[name]],
                      device=self.device, block_dim=self.block_dim)

        if initialize:
            if self.graph_chunk_frames:
                launch("reset_frame", 1)
            launch("initialize", count)
            launch("store_frame", count)
            if self.graph_chunk_frames:
                values["amount"] = 1
                launch("advance_frame", 1)
        for frame in range(1, final + 1):
            values["frame"] = frame
            values["frame_offset"] = frame - 1
            if values["system_dynamic"]:
                for _ in range(self.substeps):
                    launch("nodes_step", (count, self.scene.node_count))
                    if self.pulleys:
                        launch("pulleys_step", count)
                    for _ in range(self.constraint_iterations):
                        if self.colored:
                            for parity in (0, 1):
                                values["parity"] = parity
                                launch("stretch_color", (count, self.scene.node_count // 2))
                        else:
                            launch("stretch_serial", count)
                        if self.pulleys:
                            if self.parallel_static_nodes:
                                launch("node_contacts", (count, self.scene.node_count))
                                if self.parallel_static_contacts:
                                    for parity in (0, 1):
                                        values["parity"] = parity
                                        launch("edge_contacts_color", (count, self.scene.node_count // 2))
                                else:
                                    launch("edge_contacts_serial", count)
                            else:
                                launch("contacts", count)
                    launch("nodes_velocity", (count, self.scene.node_count))
                    if self.pulleys:
                        launch("pulleys_velocity", count)
            launch("store_frame", count)
        if self.graph_chunk_frames and final:
            values["amount"] = final
            launch("advance_frame", 1)
