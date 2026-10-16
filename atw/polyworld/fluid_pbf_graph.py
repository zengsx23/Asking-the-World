"""Device-clock variants of the existing PBF kernels for reusable frame blocks."""

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile

import warp as wp


def clock_kernels():
    tree = ast.parse(Path(__file__).with_name('fluid_pbf.py').read_text())
    names = {'_predict', '_lambda', '_project', '_velocity', '_viscosity',
             '_record', '_splat', '_store_field'}
    pieces = []
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name not in names:
            continue
        args = {a.arg for a in function.args.args}
        function.args.args += ast.parse(
            'def f(frame_base: wp.array(dtype=int), substeps: int, field_slots: wp.array(dtype=int)): pass'
        ).body[0].args.args
        prefix = []
        if 'step' in args:
            prefix += ast.parse('step = step + frame_base[0] * substeps').body
        if 'frame' in args:
            prefix += ast.parse('frame = frame + frame_base[0]').body
        if function.name == '_splat':
            prefix += ast.parse('if field_slots[step // substeps] < 0: return').body
        if function.name == '_store_field':
            prefix += ast.parse('slot = field_slots[slot + frame_base[0]]\nif slot < 0: return').body
        function.body = prefix + function.body
        ast.fix_missing_locations(function)
        pieces.append(ast.unparse(function))
    text = '''import warp as wp
from atw.polyworld.fluid_pbf import _w, _grad, _sample_sdf, _wall
wp.set_module_options({"enable_backward": False})

@wp.kernel
def set_frame(base: wp.array(dtype=int), amount: int):
    base[0] = amount

@wp.kernel
def advance_frame(base: wp.array(dtype=int), amount: int):
    base[0] += amount

''' + '\n\n'.join(pieces)
    name = '_fluid_clock_' + hashlib.sha256(text.encode()).hexdigest()[:16]
    if name not in sys.modules:
        directory = Path(wp.config.kernel_cache_dir) / 'physmind-fluid'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (name + '.py')
        if not path.exists():
            with tempfile.NamedTemporaryFile(mode='w', dir=directory, suffix='.py', delete=False) as f:
                f.write(text)
                temporary = f.name
            try:
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
