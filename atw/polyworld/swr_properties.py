"""Physical-property evidence semantics owned by SWR adapters, never benchmark rules."""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np


def _relative_pair(values, ids):
    """Normalize two positive fitted masses to the first object."""
    left = float(values[ids[0]])
    right = float(values[ids[1]])
    if not np.isfinite([left, right]).all() or left <= 0 or right <= 0:
        raise ValueError("Relative masses must be finite and positive")
    return {ids[0]: 1.0, ids[1]: right / left}


def _fluid_has_sustained_contact(backend, ids):
    """Reject isolated splash proximity; require substantial contact for 1 second."""
    fit = backend.fit
    if len(ids) != 2 or any(name not in fit.get("fluid_ids", []) for name in ids):
        raise ValueError("Density comparison requires exactly two modeled fluid IDs")
    prediction = Path(fit["prediction_npz"])
    if not prediction.is_absolute():
        prediction = Path(backend.paths["swr"]) / prediction
    with np.load(prediction) as archive:
        xy = np.asarray(archive["predicted_xy"], float)
        labels = np.asarray(archive["particle_fluid"])
        births = np.asarray(archive["particle_birth"])
    indices = [np.flatnonzero(labels == fit["fluid_ids"].index(name)) for name in ids]
    if any(not len(index) for index in indices):
        raise ValueError("A bound fluid has no modeled particles")
    proximity = float(fit.get("smoothing_length_pixels", 0.))
    fps = float(fit.get("fps", getattr(backend, "fps", 0.)))
    if not math.isfinite(proximity) or proximity <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Fluid contact scale or frame rate is unavailable")
    required_frames = max(3, math.ceil(fps))
    minimum_particles = max(2, min(5, math.ceil(.05 * min(map(len, indices)))))
    run = 0
    for frame, positions in enumerate(xy):
        groups = []
        for index in indices:
            active = positions[index[births[index] <= frame]]
            groups.append(active[np.isfinite(active).all(axis=1)])
        if all(len(group) for group in groups):
            near = np.sum((groups[0][:, None] - groups[1][None]) ** 2, axis=-1) <= proximity ** 2
            substantial = (int(near.any(axis=1).sum()) >= minimum_particles
                           and int(near.any(axis=0).sum()) >= minimum_particles)
        else:
            substantial = False
        run = run + 1 if substantial else 0
        if run >= required_frames:
            return True
    return False


def _fallback_property(result, reason):
    return result | dict(status="not_identified", value=None, fallback_required=True,
        reason=reason, next_action="Use the complete original video for physical reasoning.")


def property_evidence(backend, name, object_ids):
    """Expose what the saved model identifies; no fit, rollout or guessed property."""
    if not isinstance(name,str) or not name.strip():
        raise ValueError("property_name must name the requested physical quantity")
    name = name.strip().lower()
    ids = backend._ids(object_ids)
    result = dict(source_kind="saved_swr_fit",engine=backend.engine,property_name=name,
        object_ids=ids,status="unsupported",value=None,
        interpretation="This source does not identify the property. Follow next_evidence where available, or inspect the original video; do not infer zero or absence.")
    if backend.engine == "fluid" and name == "density":
        values = backend.fit.get("parameters",{}).get("relative_density",{})
        if len(ids) != 2 or any(i not in values for i in ids):
            return _fallback_property(result,
                "The SWR does not provide relative mass for exactly two bound fluids.")
        try:
            contact = _fluid_has_sustained_contact(backend, ids)
        except (OSError, ValueError, KeyError, IndexError) as error:
            return _fallback_property(result, f"The observed-world fluid contact check failed: {error}")
        if not contact:
            return result | dict(status="not_comparable", value=None,
                reason=("The two bound fluids did not maintain substantial contact in the modeled observed "
                        "world; isolated splash contact is excluded, so their relative density cannot be compared."),
                comparison_scope="No sustained mutual contact in the observed world.")
        try:
            relative = _relative_pair(values, ids)
        except ValueError as error:
            return _fallback_property(result, f"The fitted fluid relative mass is invalid: {error}")
        return result | dict(status="model_relative", value=dict(relative_mass=relative),
            comparison_scope="The two fluids maintained sustained contact in one joint observed-world fit.",
            interpretation="Relative mass of equal-area modeled fluid parcels; compare only this bound pair.")
    if name in {"elasticity","plasticity"} and backend.engine in {"cloth","soft_body"}:
        result.update(status="not_identified",comparison_scope="No calibrated material-property comparison from fitted coefficients.",
            meaning=("Large deformation and strong recovery after unloading." if name == "elasticity" else
                     "Large deformation and weak recovery after unloading, not flattening under sustained pressure."),
            material_response_convention=backend._contphy.evidence_catalog()["material_response_convention"],
            evidence_needed="Compare original shapes before loading, under comparable load and after unloading. Missing unloading/recovery coverage is inconclusive. Little visible deformation does not by itself identify a large coefficient.",
            parameter_semantics=backend._contphy.parameter_semantics(),
            next_evidence=dict(tool="inspect_frames",purpose="Observe loading and recovery; query_mask_motion can locate shape-change intervals but is not material strain."))
        return result
    if backend.engine == "rope" and name in {"mass","tension"}:
        if name == "tension":
            return result | dict(status="visual_evidence_required",
                reason="No calibrated tension parameter is exported; compare the two bound rope masks in the observed video.",
                comparison_scope="Observed behavior of exactly two question-bound ropes.",
                next_action="Review the focused two-rope mask video; do not infer tension from solver parameters.")
        if len(ids) != 2:
            return _fallback_property(result, "Rope mass comparison requires exactly two bound objects.")
        try:
            raw = backend.query_swr("parameters",object_ids=ids)
            systems = raw["parameters"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            return _fallback_property(result, f"The SWR mass mapping could not be queried: {error}")
        if raw.get("mass_parameter_mapping_error"):
            return _fallback_property(result, raw["mass_parameter_mapping_error"])
        queryable = set()
        prepared = {}
        for rope_id, row in systems.items():
            mapping = row.get("mass_parameter_objects", {})
            epochs = {}
            for epoch in ("relative_masses_before", "relative_masses_after"):
                values_by_object = {mapping.get(key): value
                    for key, value in row.get("parameters", {}).get(epoch, {}).items()
                    if mapping.get(key) is not None}
                epochs[epoch] = values_by_object
                queryable.update(values_by_object)
            body_ids = {body.get("object_id") for body in row.get("bodies", [])}
            prepared[rope_id] = dict(body_ids=body_ids, epochs=epochs)
        common = [rope_id for rope_id, row in prepared.items()
                  if set(ids) <= row["body_ids"]]
        located = all(any(object_id in row['body_ids'] for row in prepared.values()) for object_id in ids)
        if located and not common:
            return result | dict(status="not_comparable", value=None,
                reason=("The two bound objects are not bodies in the same rope system, so they are not "
                        "in one pulley system and their relative masses cannot be compared."),
                comparison_scope="Different rope/pulley systems have no shared fitted mass scale.",
                fallback_required=True,
                next_action="Use the complete original video for physical reasoning.")
        missing = [object_id for object_id in ids if object_id not in queryable]
        if missing:
            return _fallback_property(result,
                "The SWR has no queryable mass for: " + ", ".join(missing) + ".")
        if len(common) != 1:
            return _fallback_property(result,
                "The two objects map to multiple rope systems, so the mass comparison is ambiguous.")
        rope_id = common[0]
        try:
            epochs = {name.removeprefix("relative_masses_"): _relative_pair(values, ids)
                      for name, values in prepared[rope_id]["epochs"].items()
                      if set(ids) <= set(values)}
        except ValueError as error:
            return _fallback_property(result, f"The fitted rope relative mass is invalid: {error}")
        if not epochs:
            return _fallback_property(result,
                "The common rope system has no complete queryable mass pair.")
        masses = next(iter(epochs.values())) if all(value == next(iter(epochs.values()))
            for value in epochs.values()) else epochs
        return result | dict(status="model_relative", value=dict(system_id=rope_id,
            **({"relative_mass": masses} if set(masses) == set(ids) else {"relative_mass_by_epoch": masses})),
            comparison_scope="Both objects are queryable bodies in the same rope/pulley system.",
            interpretation="Relative fitted mass on this one shared system; compare within each returned epoch only.")
    return result | dict(reason="This SWR adapter has no identified readout for the requested property. Raw solver parameters remain available through section=parameters; they are not a substitute for a missing physical quantity.")
