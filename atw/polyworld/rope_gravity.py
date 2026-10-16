"""Independent screen-space gravity calibration, never fitted against rope motion."""
import math


def require_gravity(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Supply a positive calibrated gravity_normalized_per_s2; gravity is not a fitted parameter')
    return float(value)


def fix_engine_gravity(engine, value):
    value = require_gravity(value)
    engine.fixed_parameters = {**getattr(engine, 'fixed_parameters', {}), 0: math.log(value)}
