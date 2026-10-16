"""Liquid change evidence that rejects approximately achromatic darkening."""
import numpy as np


def changed(raw, baseline):
    """Compare observed BGR pixels to the liquid-free reference frame.

    Chromatic residual preserves colored liquid; positive mean change also
    retains pale liquid. Spatial boundary/static priority is applied by callers.
    """
    delta = raw.astype(np.float64) - baseline.astype(np.float64)
    common = delta.mean(axis=2)
    return (np.abs(delta - common[..., None]).max(axis=2) > 12) | (common > 24)
